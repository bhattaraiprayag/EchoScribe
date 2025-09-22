# backend/pipeline.py

import asyncio
import logging
import os
import tempfile
import time
import json
from asyncio import Queue
from typing import Dict

import numpy as np
import torch
import torchaudio
import ffmpeg
from faster_whisper import WhisperModel
from fastapi import WebSocketDisconnect

from config_manager import config_data

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptionSession:
    """Manages the state and pipeline for a single WebSocket connection."""

    def __init__(self, session_id: str, websocket, config: Dict, whisper_model):
        self.session_id = session_id
        self.websocket = websocket
        self.config = config
        self.whisper_model = whisper_model # Use the cached model
        self.vad_model, self.vad_utils = self.load_vad_model()

        # Load settings from config_manager
        self.vad_params = config_data.get('vad_parameters', {})
        self.audio_params = config_data.get('audio_parameters', {})
        self.transcription_params = config_data.get('transcription_parameters', {})

        self.raw_audio_queue = Queue()
        self.transcription_queue = Queue()
        self.results_queue = Queue()

        self.audio_buffer = bytearray()
        self.utterance_buffer = bytearray() # Buffer for the current utterance
        self.vad_state = {'h': torch.zeros(2, 1, 64), 'c': torch.zeros(2, 1, 64)}
        self.is_speaking = False
        self.silence_start_time = None
        self.speech_start_time = None

        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pcm', dir='.')
        
        self.transcription_context = ""
        self._is_running = True

        logger.info(f"[{session_id}] New session created with config: {config}")

    def load_vad_model(self):
        """Loads the Silero VAD model."""
        try:
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          onnx=True)
            return model, utils
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise

    def load_whisper_model(self):
        """Loads the Faster Whisper model based on session config."""
        model_size = self.config.get("model", "tiny")
        device = self.config.get("device", "cpu")
        compute_type = "int8" if device == "cpu" else "float16"
        
        logger.info(f"[{self.session_id}] Loading Whisper model '{model_size}' on '{device}'...")
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info(f"[{self.session_id}] Whisper model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    async def run_pipeline(self):
        """Runs all tasks in the pipeline concurrently."""
        tasks = [
            self.websocket_ingestion_task(),
            self.vad_chunking_task(),
            self.whisper_worker_task(),
            self.websocket_emitter_task()
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"[{self.session_id}] Pipeline error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleans up resources at the end of the session."""
        logger.info(f"[{self.session_id}] Cleaning up session resources.")
        self._is_running = False # Signal tasks to stop
        
        if self.temp_file:
            self.temp_file.close()

        if hasattr(self, 'whisper_model') and self.config.get("device") == "cuda":
            del self.whisper_model
            torch.cuda.empty_cache()
        logger.info(f"[{self.session_id}] Session resources cleaned up.")

    async def websocket_ingestion_task(self):
        """Receives raw audio from the client and puts it in a queue."""
        logger.info(f"[{self.session_id}] Starting WebSocket ingestion task.")
        while self._is_running:
            try:
                audio_chunk = await self.websocket.receive_bytes()
                await self.raw_audio_queue.put(audio_chunk)
            except WebSocketDisconnect:
                logger.info(f"[{self.session_id}] Ingestion task stopped due to WebSocket disconnect.")
                break
            except Exception as e:
                logger.warning(f"[{self.session_id}] Ingestion task stopped: {e}")
                break
        await self.raw_audio_queue.put(None)

    async def vad_chunking_task(self):
        """Consumes raw audio, runs VAD, and chunks audio into utterances."""
        logger.info(f"[{self.session_id}] Starting VAD & Chunking task.")
        
        # VAD Parameters from config
        vad_sample_rate = self.vad_params.get('sample_rate', 16000)
        vad_window_size = self.vad_params.get('window_size', 512)
        audio_sample_width = self.audio_params.get('sample_width', 2)
        vad_prob_threshold = self.vad_params.get('prob_threshold', 0.5)
        vad_silence_duration = self.vad_params.get('silence_duration', 0.7)
        vad_min_speech_duration = self.vad_params.get('min_speech_duration', 0.2)

        while True:
            audio_chunk = await self.raw_audio_queue.get()
            if audio_chunk is None:
                if self.utterance_buffer:
                    logger.info(f"[{self.session_id}] End of stream, processing final utterance.")
                    await self.transcription_queue.put({'audio': bytes(self.utterance_buffer), 'type': 'final'})
                    self.utterance_buffer.clear()
                break

            self.temp_file.write(audio_chunk)
            self.audio_buffer.extend(audio_chunk)

            while len(self.audio_buffer) >= vad_window_size * audio_sample_width:
                chunk_to_process_bytes = self.audio_buffer[:vad_window_size * audio_sample_width]
                del self.audio_buffer[:vad_window_size * audio_sample_width]

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
                        self.utterance_buffer.extend(chunk_to_process_bytes)    # Append the silence chunk to capture the end of the word
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                        
                        silence_elapsed = time.time() - self.silence_start_time
                        speech_duration = time.time() - (self.speech_start_time or time.time())
                        
                        if silence_elapsed > vad_silence_duration and speech_duration > vad_min_speech_duration:
                            logger.info(f"[{self.session_id}] End of utterance detected (speech: {speech_duration:.2f}s, silence: {silence_elapsed:.2f}s).")
                            
                            await self.transcription_queue.put({'audio': bytes(self.utterance_buffer), 'type': 'final'})

                            self.utterance_buffer.clear()   # Reset for next utterance
                            self.is_speaking = False
                            self.silence_start_time = None
                            self.speech_start_time = None
                    else:
                        pass    # If not speaking, do nothing with the silence chunk
                            
        await self.transcription_queue.put(None)

    async def whisper_worker_task(self):
        """Consumes audio utterances and transcribes them using Whisper."""
        logger.info(f"[{self.session_id}] Starting Whisper worker task.")
        language = self.config.get("language", "en")  # Get language from client config
        context_max_length = self.transcription_params.get('context_max_length', 224)

        while True:
            task = await self.transcription_queue.get()
            if task is None:
                break

            audio_bytes = task['audio']
            task_type = task['type']

            if not audio_bytes:
                continue

            try:
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                segments, info = await asyncio.to_thread(
                    self.whisper_model.transcribe,
                    audio_np,
                    beam_size=5,
                    language=language,
                    initial_prompt=self.transcription_context,
                    vad_filter=True
                )
                
                transcription_text = "".join([seg.text for seg in segments]).strip()
                if transcription_text: # Only process if there is text
                    logger.info(f"[{self.session_id}] Transcription: {transcription_text}")

                    # The logic for interim results was flawed. 
                    # Now, we only send final results for each utterance.
                    result = {'text': transcription_text, 'type': 'final'}
                    await self.results_queue.put(result)

            except Exception as e:
                logger.error(f"[{self.session_id}] Error during transcription: {e}", exc_info=True)

        await self.results_queue.put(None)

    async def websocket_emitter_task(self):
        """Sends transcription results back to the client."""
        logger.info(f"[{self.session_id}] Starting WebSocket emitter task.")
        while True:
            result = await self.results_queue.get()
            if result is None:
                break
            
            try:
                await self.websocket.send_text(json.dumps(result))
            except WebSocketDisconnect:
                logger.warning(f"[{self.session_id}] Emitter task stopped due to WebSocket disconnect.")
                break
            except Exception as e:
                logger.warning(f"[{self.session_id}] Emitter task stopped: {e}")
                break

def convert_pcm_to_mp3(pcm_filepath: str) -> str:
    """Converts a raw PCM audio file to MP3 format."""
    mp3_filepath = pcm_filepath.replace(".pcm", ".mp3")
    logger.info(f"Converting {pcm_filepath} to {mp3_filepath}...")
    audio_sample_rate = config_data.get('audio_parameters', {}).get('sample_rate', 16000)
    audio_channels = config_data.get('audio_parameters', {}).get('channels', 1)
    try:
        (
            ffmpeg
            .input(pcm_filepath, format='s16le', ar=str(audio_sample_rate), ac=audio_channels)
            .output(mp3_filepath, audio_bitrate='128k', ar=str(audio_sample_rate), ac=audio_channels)
            .run(overwrite_output=True, quiet=True)
        )
        logger.info("Conversion successful.")
        return mp3_filepath
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg conversion error: {e.stderr.decode()}")
        raise
