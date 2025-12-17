"""Real-time transcription pipeline for EchoScribe."""

import asyncio
import json
import logging
import os
import tempfile
import time
from asyncio import Queue
from typing import Any, Dict, Optional

import ffmpeg
import numpy as np
import torch
import torchaudio
from config_manager import config_data
from fastapi import WebSocketDisconnect
from faster_whisper import WhisperModel
from utils import VAD_CACHE_DIR


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _calculate_audio_duration(
    audio_bytes: bytes,
    sample_rate: int,
    sample_width: int
) -> float:
    """Calculate duration of audio data in seconds.

    Args:
        audio_bytes: Raw audio data.
        sample_rate: Audio sample rate in Hz.
        sample_width: Bytes per sample.

    Returns:
        Duration in seconds.
    """
    num_samples = len(audio_bytes) / sample_width
    return num_samples / sample_rate


class TranscriptionSession:
    """Manages state and pipeline for single WebSocket connection."""

    def __init__(
        self,
        session_id: str,
        websocket,
        config: Dict,
        whisper_model
    ):
        """Initialize transcription session.

        Args:
            session_id: Unique session identifier.
            websocket: WebSocket connection.
            config: Session configuration dict.
            whisper_model: Loaded WhisperModel instance.
        """
        self.session_id = session_id
        self.websocket = websocket
        self.config = config
        self.whisper_model = whisper_model
        self.vad_model, self.vad_utils = self.load_vad_model()
        self.vad_params = config_data.get("vad_parameters", {})
        self.audio_params = config_data.get("audio_parameters", {})
        self.transcription_params = config_data.get("transcription_parameters", {})
        self.raw_audio_queue: Queue[Optional[bytes]] = Queue()
        self.transcription_queue: Queue[Optional[Dict[str, Any]]] = Queue()
        self.results_queue: Queue[Optional[Dict[str, str]]] = Queue()
        self.audio_buffer = bytearray()
        self.utterance_buffer = bytearray()
        self.last_interim_time = 0
        self.interim_interval = 1.0
        self.vad_state = {"h": torch.zeros(2, 1, 64), "c": torch.zeros(2, 1, 64)}
        self.is_speaking = False
        self.silence_start_time: Optional[float] = None
        self.speech_start_time: Optional[float] = None
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
        self.transcription_context = ""
        self._is_running = True
        self.batch_buffer = bytearray()
        self.cumulative_audio_duration: float = 0.0
        logger.info(f"[{session_id}] New session created with config: {config}")

    def _update_transcription_context(
        self,
        new_text: str,
        max_length: int
    ) -> None:
        """Update transcription context with new text.

        Args:
            new_text: New transcription text to add.
            max_length: Maximum character length for context.
        """
        if not new_text or not new_text.strip():
            return

        if self.transcription_context:
            self.transcription_context = (
                f"{self.transcription_context} {new_text.strip()}"
            )
        else:
            self.transcription_context = new_text.strip()

        if len(self.transcription_context) > max_length:
            trimmed = self.transcription_context[-max_length:]
            first_space = trimmed.find(' ')
            if first_space > 0 and first_space < len(trimmed) - 1:
                self.transcription_context = trimmed[first_space + 1:]
            else:
                self.transcription_context = trimmed


    def load_vad_model(self):
        """Load Silero VAD model from cache.

        Returns:
            Tuple of model and utils.

        Raises:
            Exception: If model loading fails.
        """
        try:
            # Set torch hub directory to use our cache
            original_hub_dir = torch.hub.get_dir()
            torch.hub.set_dir(VAD_CACHE_DIR)

            logger.info(f"Loading Silero VAD model from cache dir: {VAD_CACHE_DIR}")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )

            # Restore original hub directory
            torch.hub.set_dir(original_hub_dir)
            logger.info("Silero VAD model loaded successfully")
            return model, utils
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise


    def load_whisper_model(self):
        """Load Faster Whisper model based on session config.

        Returns:
            Loaded WhisperModel instance.

        Raises:
            Exception: If model loading fails.
        """
        model_size = self.config.get("model", "tiny")
        device = self.config.get("device", "cpu")
        compute_type = "int8" if device == "cpu" else "float16"
        logger.info(
            f"[{self.session_id}] Loading Whisper model "
            f"'{model_size}' on '{device}'..."
        )
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info(f"[{self.session_id}] Whisper model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise


    async def run_pipeline(self) -> None:
        """Run all pipeline tasks concurrently."""
        tasks = [
            self.websocket_ingestion_task(),
            self.vad_chunking_task(),
            self.whisper_worker_task(),
            self.websocket_emitter_task(),
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"[{self.session_id}] Pipeline error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources at end of session."""
        logger.info(f"[{self.session_id}] Cleaning up session resources.")
        self._is_running = False
        if self.temp_file:
            self.temp_file.close()
        if hasattr(self, "whisper_model") and self.config.get("device") == "cuda":
            del self.whisper_model
            torch.cuda.empty_cache()
        logger.info(f"[{self.session_id}] Session resources cleaned up.")


    async def websocket_ingestion_task(self) -> None:
        """Receive raw audio from client and queue it."""
        logger.info(f"[{self.session_id}] Starting WebSocket ingestion task.")
        while self._is_running:
            try:
                audio_chunk = await self.websocket.receive_bytes()
                await self.raw_audio_queue.put(audio_chunk)
            except WebSocketDisconnect:
                logger.info(
                    f"[{self.session_id}] "
                    f"Ingestion task stopped due to WebSocket disconnect."
                )
                break
            except Exception as e:
                logger.warning(f"[{self.session_id}] Ingestion task stopped: {e}")
                break
        await self.raw_audio_queue.put(None)


    async def vad_chunking_task(self) -> None:
        """Consume raw audio, run VAD, and chunk into utterances."""
        logger.info(f"[{self.session_id}] Starting VAD & Chunking task.")
        vad_sample_rate = self.vad_params.get("sample_rate", 16000)
        vad_window_size = self.vad_params.get("window_size", 512)
        audio_sample_width = self.audio_params.get("sample_width", 2)
        audio_sample_rate = self.audio_params.get("sample_rate", 16000)
        vad_prob_threshold = self.vad_params.get("prob_threshold", 0.5)
        vad_silence_duration = self.vad_params.get("silence_duration", 0.7)
        vad_min_speech_duration = self.vad_params.get("min_speech_duration", 0.2)

        batch_short_utterances = self.vad_params.get("batch_short_utterances", False)
        batch_min_duration = self.vad_params.get("batch_min_duration", 1.0)

        while True:
            audio_chunk = await self.raw_audio_queue.get()
            if audio_chunk is None:
                if self.batch_buffer:
                    logger.info(
                        f"[{self.session_id}] End of stream, processing batched utterances."
                    )
                    await self.transcription_queue.put(
                        {"audio": bytes(self.batch_buffer), "type": "final"}
                    )
                    self.batch_buffer.clear()
                if self.utterance_buffer:
                    logger.info(
                        f"[{self.session_id}] End of stream, processing final utterance."
                    )
                    await self.transcription_queue.put(
                        {"audio": bytes(self.utterance_buffer), "type": "final"}
                    )
                    self.utterance_buffer.clear()
                break

            self.temp_file.write(audio_chunk)
            self.audio_buffer.extend(audio_chunk)

            while len(self.audio_buffer) >= vad_window_size * audio_sample_width:
                chunk_to_process_bytes = self.audio_buffer[
                    : vad_window_size * audio_sample_width
                ]
                del self.audio_buffer[: vad_window_size * audio_sample_width]

                audio_int16 = np.frombuffer(chunk_to_process_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)

                speech_prob = self.vad_model(audio_tensor, vad_sample_rate).item()

                if speech_prob > vad_prob_threshold:
                    self.silence_start_time = None
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.speech_start_time = time.time()
                    self.utterance_buffer.extend(chunk_to_process_bytes)
                else:
                    if self.is_speaking:
                        self.utterance_buffer.extend(chunk_to_process_bytes)
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()

                        silence_elapsed = time.time() - self.silence_start_time
                        speech_duration = time.time() - (
                            self.speech_start_time or time.time()
                        )

                        if (
                            silence_elapsed > vad_silence_duration
                            and speech_duration > vad_min_speech_duration
                        ):
                            logger.info(
                                f"[{self.session_id}] End of utterance detected (speech: {speech_duration:.2f}s, silence: {silence_elapsed:.2f}s)."
                            )

                            utterance_audio = bytes(self.utterance_buffer)
                            utterance_duration = _calculate_audio_duration(
                                utterance_audio, audio_sample_rate, audio_sample_width
                            )

                            if (batch_short_utterances and
                                utterance_duration < batch_min_duration):
                                self.batch_buffer.extend(utterance_audio)
                                batch_duration = _calculate_audio_duration(
                                    bytes(self.batch_buffer),
                                    audio_sample_rate,
                                    audio_sample_width
                                )
                                logger.info(
                                    f"[{self.session_id}] Batching short utterance "
                                    f"({utterance_duration:.2f}s), "
                                    f"batch total: {batch_duration:.2f}s"
                                )

                                if batch_duration >= batch_min_duration:
                                    logger.info(
                                        f"[{self.session_id}] Sending batched "
                                        f"utterances ({batch_duration:.2f}s)"
                                    )
                                    await self.transcription_queue.put(
                                        {
                                            "audio": bytes(self.batch_buffer),
                                            "type": "final"
                                        }
                                    )
                                    self.batch_buffer.clear()
                            else:
                                if self.batch_buffer:
                                    batch_duration = _calculate_audio_duration(
                                        bytes(self.batch_buffer),
                                        audio_sample_rate,
                                        audio_sample_width
                                    )
                                    logger.info(
                                        f"[{self.session_id}] Flushing batch buffer "
                                        f"({batch_duration:.2f}s) before longer "
                                        f"utterance"
                                    )
                                    await self.transcription_queue.put(
                                        {
                                            "audio": bytes(self.batch_buffer),
                                            "type": "final"
                                        }
                                    )
                                    self.batch_buffer.clear()

                                await self.transcription_queue.put(
                                    {"audio": utterance_audio, "type": "final"}
                                )

                            self.utterance_buffer.clear()
                            self.is_speaking = False
                            self.silence_start_time = None
                            self.speech_start_time = None

            if self.is_speaking and self.utterance_buffer:
                current_time = time.time()
                if current_time - self.last_interim_time > self.interim_interval:
                    logger.info(
                        f"[{self.session_id}] Sending interim utterance for transcription."
                    )
                    await self.transcription_queue.put(
                        {"audio": bytes(self.utterance_buffer), "type": "interim"}
                    )
                    self.last_interim_time = current_time
        await self.transcription_queue.put(None)


    async def whisper_worker_task(self) -> None:
        """Consume audio utterances and transcribe using Whisper."""
        logger.info(f"[{self.session_id}] Starting Whisper worker task.")
        language = self.config.get("language", "en")
        context_max_length = self.transcription_params.get("context_max_length", 224)
        word_timestamps_enabled = self.transcription_params.get(
            "word_timestamps",
            False
        )
        audio_sample_rate = self.audio_params.get("sample_rate", 16000)
        audio_sample_width = self.audio_params.get("sample_width", 2)

        while True:
            task = await self.transcription_queue.get()
            if task is None:
                break

            audio_bytes = task["audio"]
            task_type = task["type"]

            if not audio_bytes:
                continue

            chunk_duration = _calculate_audio_duration(
                audio_bytes, audio_sample_rate, audio_sample_width
            )

            try:
                audio_np = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                segments, info = await asyncio.to_thread(
                    self.whisper_model.transcribe,
                    audio_np,
                    beam_size=5,
                    language=language,
                    initial_prompt=self.transcription_context,
                    vad_filter=False,
                    word_timestamps=word_timestamps_enabled,
                )

                segment_list = list(segments)
                transcription_text = (
                    "".join([seg.text for seg in segment_list]).strip()
                )

                if transcription_text:
                    logger.info(
                        f"[{self.session_id}] Transcription ({task_type}): "
                        f"{transcription_text}"
                    )

                    if word_timestamps_enabled:
                        for segment in segment_list:
                            if hasattr(segment, 'words') and segment.words:
                                for word in segment.words:
                                    word_result = {
                                        "text": word.word.strip(),
                                        "type": "word",
                                        "start": round(
                                            self.cumulative_audio_duration +
                                            word.start,
                                            2
                                        ),
                                        "end": round(
                                            self.cumulative_audio_duration +
                                            word.end,
                                            2
                                        ),
                                    }
                                    await self.results_queue.put(word_result)

                    start_time = segment_list[0].start if segment_list else 0.0
                    end_time = segment_list[-1].end if segment_list else 0.0

                    result = {
                        "text": transcription_text,
                        "type": task_type,
                        "start": round(
                            self.cumulative_audio_duration + start_time,
                            2
                        ),
                        "end": round(
                            self.cumulative_audio_duration + end_time,
                            2
                        ),
                    }
                    await self.results_queue.put(result)

                    if task_type == "final":
                        self._update_transcription_context(
                            transcription_text, context_max_length
                        )

                if task_type == "final":
                    self.cumulative_audio_duration += chunk_duration

            except Exception as e:
                logger.error(
                    f"[{self.session_id}] Error during transcription: {e}",
                    exc_info=True,
                )
        await self.results_queue.put(None)


    async def websocket_emitter_task(self) -> None:
        """Send transcription results back to client."""
        logger.info(f"[{self.session_id}] Starting WebSocket emitter task.")
        while True:
            result = await self.results_queue.get()
            if result is None:
                break
            try:
                await self.websocket.send_text(json.dumps(result))
            except WebSocketDisconnect:
                logger.warning(
                    f"[{self.session_id}] "
                    f"Emitter task stopped due to WebSocket disconnect."
                )
                break
            except Exception as e:
                logger.warning(f"[{self.session_id}] Emitter task stopped: {e}")
                break


def convert_pcm_to_mp3(pcm_filepath: str) -> str:
    """Convert raw PCM audio file to MP3 format.

    Args:
        pcm_filepath: Path to PCM file.

    Returns:
        Path to converted MP3 file.

    Raises:
        ffmpeg.Error: If conversion fails.
    """
    mp3_filepath = pcm_filepath.replace(".pcm", ".mp3")
    logger.info(f"Converting {pcm_filepath} to {mp3_filepath}...")
    audio_sample_rate = config_data.get("audio_parameters", {}).get(
        "sample_rate", 16000
    )
    audio_channels = config_data.get("audio_parameters", {}).get("channels", 1)
    try:
        (
            ffmpeg.input(
                pcm_filepath,
                format="s16le",
                ar=str(audio_sample_rate),
                ac=audio_channels,
            )
            .output(
                mp3_filepath,
                audio_bitrate="128k",
                ar=str(audio_sample_rate),
                ac=audio_channels,
            )
            .run(overwrite_output=True, quiet=True)
        )
        logger.info("Conversion successful.")
        return mp3_filepath
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg conversion error: {e.stderr.decode()}")
        raise
