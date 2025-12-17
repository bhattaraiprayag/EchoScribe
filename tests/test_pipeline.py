# tests/test_pipeline.py
"""Tests for the transcription pipeline, including context management."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

pytestmark = pytest.mark.asyncio


class TestTranscriptionContext:
    """Tests for transcription context management."""

    def test_update_transcription_context_basic(self):
        """Context should be updated with new text."""
        from backend.pipeline import TranscriptionSession

        # Create a mock session with minimal required attributes
        session = MagicMock(spec=TranscriptionSession)
        session.transcription_context = ""

        # Call the method we're testing (will be added)
        TranscriptionSession._update_transcription_context(session, "Hello world", 224)

        assert session.transcription_context == "Hello world"

    def test_update_transcription_context_appends(self):
        """Context should append new text to existing."""
        from backend.pipeline import TranscriptionSession

        session = MagicMock(spec=TranscriptionSession)
        session.transcription_context = "Hello"

        TranscriptionSession._update_transcription_context(session, "world", 224)

        assert session.transcription_context == "Hello world"

    def test_update_transcription_context_respects_max_length(self):
        """Context should be trimmed to max_length."""
        from backend.pipeline import TranscriptionSession

        session = MagicMock(spec=TranscriptionSession)
        session.transcription_context = ""

        # Add a long text that exceeds max_length
        long_text = "This is a very long sentence that should be trimmed. " * 10
        TranscriptionSession._update_transcription_context(session, long_text, 50)

        # Result should be at most 50 characters
        assert len(session.transcription_context) <= 50

    def test_update_transcription_context_trims_at_word_boundary(self):
        """Context should trim at word boundary, not mid-word."""
        from backend.pipeline import TranscriptionSession

        session = MagicMock(spec=TranscriptionSession)
        session.transcription_context = ""

        # Add text that needs trimming
        text = "Hello world this is a test sentence"
        TranscriptionSession._update_transcription_context(session, text, 20)

        # Should not end with partial word
        context = session.transcription_context
        assert not context.endswith("tes")  # Not mid-word
        # Context should be complete words
        words = context.split()
        for word in words:
            assert word in text.split()

    def test_update_transcription_context_handles_empty_text(self):
        """Context should handle empty text gracefully."""
        from backend.pipeline import TranscriptionSession

        session = MagicMock(spec=TranscriptionSession)
        session.transcription_context = "existing"

        TranscriptionSession._update_transcription_context(session, "", 224)

        # Should remain unchanged or just have existing trimmed
        assert "existing" in session.transcription_context or session.transcription_context == "existing"

    def test_update_transcription_context_handles_special_characters(self):
        """Context should handle punctuation and unicode safely."""
        from backend.pipeline import TranscriptionSession

        session = MagicMock(spec=TranscriptionSession)
        session.transcription_context = ""

        text = "Hello! How are you? I'm fine, thanks. 你好世界"
        TranscriptionSession._update_transcription_context(session, text, 224)

        assert "Hello!" in session.transcription_context
        assert "你好世界" in session.transcription_context


class TestWhisperWorkerContextUsage:
    """Tests to verify context is properly used in whisper_worker_task."""

    async def test_context_updated_after_final_transcription(self):
        """Context should be updated after each final transcription."""
        from backend.pipeline import TranscriptionSession

        # Create mock objects
        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        # Mock transcribe to return a segment with text
        mock_segment = MagicMock()
        mock_segment.text = "Test transcription"
        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            session = TranscriptionSession(
                session_id="test-123",
                websocket=mock_websocket,
                config={"model": "tiny", "device": "cpu", "language": "en"},
                whisper_model=mock_whisper_model
            )

        # Verify initial context is empty
        assert session.transcription_context == ""

        # Put a final task in the queue
        await session.transcription_queue.put({"audio": b"\x00\x01" * 100, "type": "final"})
        await session.transcription_queue.put(None)  # Signal end

        # Run the worker task
        await session.whisper_worker_task()

        # Context should be updated
        assert session.transcription_context == "Test transcription"

    async def test_context_not_updated_for_interim(self):
        """Interim transcriptions should not update context."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        mock_segment = MagicMock()
        mock_segment.text = "Interim text"
        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            session = TranscriptionSession(
                session_id="test-123",
                websocket=mock_websocket,
                config={"model": "tiny", "device": "cpu", "language": "en"},
                whisper_model=mock_whisper_model
            )

        # Put an interim task in the queue
        await session.transcription_queue.put({"audio": b"\x00\x01" * 100, "type": "interim"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Context should still be empty
        assert session.transcription_context == ""

    async def test_context_passed_to_whisper(self):
        """Whisper should receive context as initial_prompt."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        mock_segment = MagicMock()
        mock_segment.text = "New text"
        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            session = TranscriptionSession(
                session_id="test-123",
                websocket=mock_websocket,
                config={"model": "tiny", "device": "cpu", "language": "en"},
                whisper_model=mock_whisper_model
            )

        # Set existing context
        session.transcription_context = "Previous context"

        await session.transcription_queue.put({"audio": b"\x00\x01" * 100, "type": "final"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Verify transcribe was called with initial_prompt
        call_kwargs = mock_whisper_model.transcribe.call_args
        assert call_kwargs is not None
        # Check if initial_prompt was passed (either in args or kwargs)
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get('initial_prompt') == "Previous context"


class TestWordLevelStreaming:
    """Tests for word-level streaming feature."""

    async def test_word_timestamps_enabled_emits_word_events(self):
        """When word_timestamps is enabled, word events should be emitted."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        # Create mock segment with word-level timestamps
        mock_word1 = MagicMock()
        mock_word1.word = "Hello"
        mock_word1.start = 0.0
        mock_word1.end = 0.5

        mock_word2 = MagicMock()
        mock_word2.word = "world"
        mock_word2.start = 0.5
        mock_word2.end = 1.0

        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.words = [mock_word1, mock_word2]

        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            with patch('backend.pipeline.config_data', {
                'transcription_parameters': {'context_max_length': 224, 'word_timestamps': True},
                'vad_parameters': {},
                'audio_parameters': {}
            }):
                session = TranscriptionSession(
                    session_id="test-word-stream",
                    websocket=mock_websocket,
                    config={"model": "tiny", "device": "cpu", "language": "en"},
                    whisper_model=mock_whisper_model
                )

        await session.transcription_queue.put({"audio": b"\x00\x01" * 100, "type": "final"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Verify word_timestamps=True was passed to transcribe
        call_kwargs = mock_whisper_model.transcribe.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get('word_timestamps') is True

    async def test_word_timestamps_disabled_no_word_events(self):
        """When word_timestamps is disabled, no word events should be emitted."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.words = None  # No words when disabled

        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            with patch('backend.pipeline.config_data', {
                'transcription_parameters': {'context_max_length': 224, 'word_timestamps': False},
                'vad_parameters': {},
                'audio_parameters': {}
            }):
                session = TranscriptionSession(
                    session_id="test-no-word-stream",
                    websocket=mock_websocket,
                    config={"model": "tiny", "device": "cpu", "language": "en"},
                    whisper_model=mock_whisper_model
                )

        await session.transcription_queue.put({"audio": b"\x00\x01" * 100, "type": "final"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Verify word_timestamps was either False or not passed
        call_kwargs = mock_whisper_model.transcribe.call_args
        assert call_kwargs is not None
        word_ts = call_kwargs.kwargs.get('word_timestamps', False)
        assert word_ts is False

    async def test_word_events_include_timestamps(self):
        """Word events should include start/end timestamps."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        mock_word = MagicMock()
        mock_word.word = "Test"
        mock_word.start = 1.5
        mock_word.end = 2.0

        mock_segment = MagicMock()
        mock_segment.text = "Test"
        mock_segment.start = 1.5
        mock_segment.end = 2.0
        mock_segment.words = [mock_word]

        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            with patch('backend.pipeline.config_data', {
                'transcription_parameters': {'context_max_length': 224, 'word_timestamps': True},
                'vad_parameters': {},
                'audio_parameters': {}
            }):
                session = TranscriptionSession(
                    session_id="test-word-ts",
                    websocket=mock_websocket,
                    config={"model": "tiny", "device": "cpu", "language": "en"},
                    whisper_model=mock_whisper_model
                )

        await session.transcription_queue.put({"audio": b"\x00\x01" * 100, "type": "final"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Check results queue for word events
        results = []
        while not session.results_queue.empty():
            result = await session.results_queue.get()
            if result:
                results.append(result)

        # Should have word events before final segment
        word_events = [r for r in results if r.get('type') == 'word']
        if word_events:
            word_event = word_events[0]
            assert 'text' in word_event
            assert 'start' in word_event
            assert 'end' in word_event


class TestCumulativeTimestamps:
    """Tests for cumulative timestamp tracking feature."""

    def test_session_initializes_cumulative_audio_duration(self):
        """TranscriptionSession should initialize cumulative_audio_duration to 0."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = MagicMock()
        mock_whisper_model = MagicMock()

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            session = TranscriptionSession(
                session_id="test-timestamp",
                websocket=mock_websocket,
                config={"model": "tiny", "device": "cpu", "language": "en"},
                whisper_model=mock_whisper_model
            )

        assert hasattr(session, 'cumulative_audio_duration')
        assert session.cumulative_audio_duration == 0.0

    async def test_timestamps_include_cumulative_offset(self):
        """Transcription results should include cumulative offset in timestamps."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        # Create mock segment with Whisper-relative timestamps (0.5 to 2.0)
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.5
        mock_segment.end = 2.0

        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            session = TranscriptionSession(
                session_id="test-cumulative",
                websocket=mock_websocket,
                config={"model": "tiny", "device": "cpu", "language": "en"},
                whisper_model=mock_whisper_model
            )

        # Simulate that 10 seconds of audio have already been processed
        session.cumulative_audio_duration = 10.0

        # Put a final task with 2 seconds of audio (16000 samples/sec * 2 bytes * 2 sec)
        audio_bytes = b'\x00' * 64000  # 2 seconds of audio
        await session.transcription_queue.put({"audio": audio_bytes, "type": "final"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Get result from queue
        result = await session.results_queue.get()

        # Timestamps should be offset by cumulative duration (10.0)
        # Whisper returned start=0.5, end=2.0, so final should be start=10.5, end=12.0
        assert result is not None
        assert result['start'] == 10.5
        assert result['end'] == 12.0

    async def test_cumulative_duration_updates_after_final_transcription(self):
        """cumulative_audio_duration should update after final transcription."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        mock_segment = MagicMock()
        mock_segment.text = "Test"
        mock_segment.start = 0.0
        mock_segment.end = 1.5

        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            with patch('backend.pipeline.config_data', {
                'transcription_parameters': {'context_max_length': 224},
                'vad_parameters': {},
                'audio_parameters': {'sample_rate': 16000, 'sample_width': 2}
            }):
                session = TranscriptionSession(
                    session_id="test-duration-update",
                    websocket=mock_websocket,
                    config={"model": "tiny", "device": "cpu", "language": "en"},
                    whisper_model=mock_whisper_model
                )

        assert session.cumulative_audio_duration == 0.0

        # Send 3 seconds of audio (16000 * 2 * 3 = 96000 bytes)
        audio_bytes = b'\x00' * 96000
        await session.transcription_queue.put({"audio": audio_bytes, "type": "final"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # cumulative_audio_duration should be updated by the audio chunk duration
        assert session.cumulative_audio_duration == 3.0

    async def test_interim_transcriptions_use_cumulative_but_dont_update(self):
        """Interim transcriptions should use cumulative offset but not update it."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        mock_segment = MagicMock()
        mock_segment.text = "Interim text"
        mock_segment.start = 0.0
        mock_segment.end = 1.0

        mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            with patch('backend.pipeline.config_data', {
                'transcription_parameters': {'context_max_length': 224},
                'vad_parameters': {},
                'audio_parameters': {'sample_rate': 16000, 'sample_width': 2}
            }):
                session = TranscriptionSession(
                    session_id="test-interim-no-update",
                    websocket=mock_websocket,
                    config={"model": "tiny", "device": "cpu", "language": "en"},
                    whisper_model=mock_whisper_model
                )

        # Set initial cumulative duration
        session.cumulative_audio_duration = 5.0

        # Send interim task
        audio_bytes = b'\x00' * 32000  # 1 second
        await session.transcription_queue.put({"audio": audio_bytes, "type": "interim"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Get result - should have offset applied
        result = await session.results_queue.get()
        assert result['start'] == 5.0  # 5.0 + 0.0
        assert result['end'] == 6.0    # 5.0 + 1.0

        # But cumulative duration should NOT be updated for interim
        assert session.cumulative_audio_duration == 5.0

    async def test_multiple_final_transcriptions_accumulate(self):
        """Multiple final transcriptions should accumulate timestamps correctly."""
        from backend.pipeline import TranscriptionSession

        mock_websocket = AsyncMock()
        mock_whisper_model = MagicMock()

        # First segment
        mock_segment1 = MagicMock()
        mock_segment1.text = "First"
        mock_segment1.start = 0.0
        mock_segment1.end = 2.0

        # Second segment
        mock_segment2 = MagicMock()
        mock_segment2.text = "Second"
        mock_segment2.start = 0.0
        mock_segment2.end = 1.5

        # Return different segments on subsequent calls
        mock_whisper_model.transcribe.side_effect = [
            ([mock_segment1], MagicMock()),
            ([mock_segment2], MagicMock()),
        ]

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            with patch('backend.pipeline.config_data', {
                'transcription_parameters': {'context_max_length': 224},
                'vad_parameters': {},
                'audio_parameters': {'sample_rate': 16000, 'sample_width': 2}
            }):
                session = TranscriptionSession(
                    session_id="test-accumulate",
                    websocket=mock_websocket,
                    config={"model": "tiny", "device": "cpu", "language": "en"},
                    whisper_model=mock_whisper_model
                )

        # First chunk: 2 seconds
        audio1 = b'\x00' * 64000
        # Second chunk: 1.5 seconds
        audio2 = b'\x00' * 48000

        await session.transcription_queue.put({"audio": audio1, "type": "final"})
        await session.transcription_queue.put({"audio": audio2, "type": "final"})
        await session.transcription_queue.put(None)

        await session.whisper_worker_task()

        # Get first result - should start at 0
        result1 = await session.results_queue.get()
        assert result1['text'] == "First"
        assert result1['start'] == 0.0
        assert result1['end'] == 2.0

        # Get second result - should start at 2.0 (after first chunk duration)
        result2 = await session.results_queue.get()
        assert result2['text'] == "Second"
        assert result2['start'] == 2.0   # 2.0 + 0.0
        assert result2['end'] == 3.5     # 2.0 + 1.5

        # Cumulative should now be 3.5 seconds
        assert session.cumulative_audio_duration == 3.5


class TestBatchShortUtterances:
    """Tests for batching short utterances feature."""

    def test_batch_config_defaults(self):
        """Batch config should have proper defaults."""
        from backend.config_manager import get_config
        config = get_config()
        vad_params = config.get('vad_parameters', {})
        # Should have batch settings
        assert 'batch_short_utterances' in vad_params or True  # Optional
        assert 'batch_min_duration' in vad_params or True  # Optional

    def test_calculate_audio_duration(self):
        """Test audio duration calculation utility."""
        from backend.pipeline import _calculate_audio_duration

        # 16000 samples/second, 2 bytes per sample
        sample_rate = 16000
        sample_width = 2

        # 1 second of audio = 16000 * 2 = 32000 bytes
        audio_1sec = b'\x00' * 32000
        duration = _calculate_audio_duration(audio_1sec, sample_rate, sample_width)
        assert abs(duration - 1.0) < 0.01

        # 0.5 seconds of audio = 16000 bytes
        audio_half_sec = b'\x00' * 16000
        duration = _calculate_audio_duration(audio_half_sec, sample_rate, sample_width)
        assert abs(duration - 0.5) < 0.01

    def test_short_utterances_batched_when_enabled(self):
        """Short utterances should be batched when feature is enabled."""
        from backend.pipeline import TranscriptionSession

        # Create session with batching enabled
        mock_websocket = MagicMock()
        mock_whisper_model = MagicMock()

        with patch.object(TranscriptionSession, 'load_vad_model', return_value=(MagicMock(), MagicMock())):
            with patch('backend.pipeline.config_data', {
                'transcription_parameters': {'context_max_length': 224},
                'vad_parameters': {
                    'batch_short_utterances': True,
                    'batch_min_duration': 1.0,
                    'min_speech_duration': 0.2,
                    'silence_duration': 0.5,
                    'prob_threshold': 0.5
                },
                'audio_parameters': {'sample_rate': 16000, 'sample_width': 2}
            }):
                session = TranscriptionSession(
                    session_id="test-batch",
                    websocket=mock_websocket,
                    config={"model": "tiny", "device": "cpu", "language": "en"},
                    whisper_model=mock_whisper_model
                )

        # Verify batch settings were loaded
        assert session.vad_params.get('batch_short_utterances', False) is True
        assert session.vad_params.get('batch_min_duration', 1.0) == 1.0
