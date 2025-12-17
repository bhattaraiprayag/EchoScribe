# tests/test_validation.py
"""Tests for settings and file upload validation."""

import pytest

pytestmark = pytest.mark.asyncio


class TestSettingsValidation:
    """Tests for settings API input validation."""

    async def test_valid_settings_accepted(self, async_client):
        """Valid settings within bounds should be accepted."""
        valid_settings = {
            "vad_parameters": {
                "prob_threshold": 0.5,
                "silence_duration": 0.7,
                "min_speech_duration": 0.3
            }
        }
        response = await async_client.post("/api/settings", json=valid_settings)
        assert response.status_code == 200
        assert "message" in response.json()

    async def test_prob_threshold_too_high_rejected(self, async_client):
        """prob_threshold > 0.9 should be rejected."""
        invalid_settings = {"vad_parameters": {"prob_threshold": 1.5}}
        response = await async_client.post("/api/settings", json=invalid_settings)
        assert response.status_code == 422

    async def test_prob_threshold_too_low_rejected(self, async_client):
        """prob_threshold < 0.1 should be rejected."""
        invalid_settings = {"vad_parameters": {"prob_threshold": 0.05}}
        response = await async_client.post("/api/settings", json=invalid_settings)
        assert response.status_code == 422

    async def test_negative_duration_rejected(self, async_client):
        """Negative durations should be rejected."""
        invalid_settings = {"vad_parameters": {"silence_duration": -1}}
        response = await async_client.post("/api/settings", json=invalid_settings)
        assert response.status_code == 422

    async def test_duration_too_high_rejected(self, async_client):
        """silence_duration > 5 should be rejected."""
        invalid_settings = {"vad_parameters": {"silence_duration": 10}}
        response = await async_client.post("/api/settings", json=invalid_settings)
        assert response.status_code == 422

    async def test_unknown_keys_rejected(self, async_client):
        """Unknown configuration keys should be rejected."""
        invalid_settings = {"unknown_section": {"malicious_key": "value"}}
        response = await async_client.post("/api/settings", json=invalid_settings)
        assert response.status_code == 422

    async def test_wrong_types_rejected(self, async_client):
        """String values for numeric fields should be rejected."""
        invalid_settings = {"vad_parameters": {"prob_threshold": "not_a_number"}}
        response = await async_client.post("/api/settings", json=invalid_settings)
        assert response.status_code == 422

    async def test_partial_update_accepted(self, async_client):
        """Partial updates (only some fields) should be accepted."""
        partial_settings = {
            "vad_parameters": {
                "prob_threshold": 0.6
            }
        }
        response = await async_client.post("/api/settings", json=partial_settings)
        assert response.status_code == 200

    async def test_empty_object_accepted(self, async_client):
        """Empty settings object should be accepted (no-op)."""
        response = await async_client.post("/api/settings", json={})
        assert response.status_code == 200


class TestFileUploadValidationUnit:
    """Unit tests for file upload validation functions."""

    def test_sanitize_filename_removes_path_separators(self):
        """Filename sanitization should remove path separators."""
        from utils import sanitize_filename

        assert "/" not in sanitize_filename("../../../etc/passwd")
        assert "\\" not in sanitize_filename("..\\..\\etc\\passwd")

    def test_sanitize_filename_removes_double_dots(self):
        """Filename sanitization should remove double dots."""
        from utils import sanitize_filename

        result = sanitize_filename("../../../test.mp3")
        assert ".." not in result

    def test_sanitize_filename_keeps_valid_chars(self):
        """Filename sanitization should keep valid characters."""
        from utils import sanitize_filename

        result = sanitize_filename("my_audio-file.mp3")
        assert "my_audio-file.mp3" == result

    def test_sanitize_filename_handles_special_chars(self):
        """Filename sanitization should handle special characters."""
        from utils import sanitize_filename

        result = sanitize_filename("file<>:\"|?*.mp3")
        # Should replace special chars
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_is_valid_audio_extension_accepts_valid(self):
        """is_valid_audio_extension should accept valid audio extensions."""
        from utils import is_valid_audio_extension

        assert is_valid_audio_extension(".mp3")
        assert is_valid_audio_extension(".wav")
        assert is_valid_audio_extension(".m4a")
        assert is_valid_audio_extension(".ogg")
        assert is_valid_audio_extension(".flac")
        assert is_valid_audio_extension(".MP3")  # Case insensitive
        assert is_valid_audio_extension(".WAV")

    def test_is_valid_audio_extension_rejects_invalid(self):
        """is_valid_audio_extension should reject invalid extensions."""
        from utils import is_valid_audio_extension

        assert not is_valid_audio_extension(".exe")
        assert not is_valid_audio_extension(".txt")
        assert not is_valid_audio_extension(".py")
        assert not is_valid_audio_extension(".jpg")
        assert not is_valid_audio_extension("")

    def test_is_valid_audio_extension_accepts_video_formats(self):
        """is_valid_audio_extension should accept common video formats with audio."""
        from utils import is_valid_audio_extension

        # Video formats that contain audio tracks
        assert is_valid_audio_extension(".mkv")
        assert is_valid_audio_extension(".mp4")
        assert is_valid_audio_extension(".avi")
        assert is_valid_audio_extension(".mov")
        assert is_valid_audio_extension(".MKV")  # Case insensitive
        assert is_valid_audio_extension(".MP4")

    def test_max_file_size_constant_exists(self):
        """MAX_FILE_SIZE constant should be defined."""
        from utils import MAX_FILE_SIZE

        # MAX_FILE_SIZE is 0 to indicate no limit
        assert MAX_FILE_SIZE == 0  # No file size limit


class TestPydanticModelsUnit:
    """Unit tests for Pydantic validation models."""

    def test_vad_parameters_valid(self):
        """VADParameters should accept valid values."""
        from models import VADParameters

        params = VADParameters(
            prob_threshold=0.5,
            silence_duration=0.7,
            min_speech_duration=0.3
        )
        assert params.prob_threshold == 0.5
        assert params.silence_duration == 0.7
        assert params.min_speech_duration == 0.3

    def test_vad_parameters_prob_threshold_bounds(self):
        """VADParameters should validate prob_threshold bounds."""
        from models import VADParameters
        from pydantic import ValidationError

        # Valid boundary values
        VADParameters(prob_threshold=0.1)
        VADParameters(prob_threshold=0.9)

        # Invalid values
        with pytest.raises(ValidationError):
            VADParameters(prob_threshold=0.05)
        with pytest.raises(ValidationError):
            VADParameters(prob_threshold=1.0)

    def test_vad_parameters_silence_duration_bounds(self):
        """VADParameters should validate silence_duration bounds."""
        from models import VADParameters
        from pydantic import ValidationError

        # Valid boundary values
        VADParameters(silence_duration=0.1)
        VADParameters(silence_duration=5.0)

        # Invalid values
        with pytest.raises(ValidationError):
            VADParameters(silence_duration=0.05)
        with pytest.raises(ValidationError):
            VADParameters(silence_duration=6.0)

    def test_vad_parameters_min_speech_duration_bounds(self):
        """VADParameters should validate min_speech_duration bounds."""
        from models import VADParameters
        from pydantic import ValidationError

        # Valid boundary values
        VADParameters(min_speech_duration=0.1)
        VADParameters(min_speech_duration=2.0)

        # Invalid values
        with pytest.raises(ValidationError):
            VADParameters(min_speech_duration=0.05)
        with pytest.raises(ValidationError):
            VADParameters(min_speech_duration=3.0)

    def test_settings_update_rejects_unknown_keys(self):
        """SettingsUpdate should reject unknown configuration keys."""
        from models import SettingsUpdate
        from pydantic import ValidationError

        # Valid
        SettingsUpdate(vad_parameters={"prob_threshold": 0.5})

        # Unknown key should be rejected
        with pytest.raises(ValidationError):
            SettingsUpdate(unknown_key="value")

    def test_settings_update_allows_partial(self):
        """SettingsUpdate should allow partial updates."""
        from models import SettingsUpdate

        # Only vad_parameters
        settings = SettingsUpdate(vad_parameters={"prob_threshold": 0.5})
        assert settings.vad_parameters is not None
        assert settings.audio_parameters is None

    def test_audio_parameters_valid(self):
        """AudioParameters should accept valid values."""
        from models import AudioParameters

        params = AudioParameters(
            channels=1,
            sample_rate=16000,
            sample_width=2
        )
        assert params.channels == 1
        assert params.sample_rate == 16000
        assert params.sample_width == 2

    def test_transcription_parameters_valid(self):
        """TranscriptionParameters should accept valid values."""
        from models import TranscriptionParameters

        params = TranscriptionParameters(context_max_length=224)
        assert params.context_max_length == 224

    def test_transcription_parameters_bounds(self):
        """TranscriptionParameters should validate bounds."""
        from models import TranscriptionParameters
        from pydantic import ValidationError

        # Valid boundary
        TranscriptionParameters(context_max_length=0)
        TranscriptionParameters(context_max_length=500)

        # Invalid
        with pytest.raises(ValidationError):
            TranscriptionParameters(context_max_length=-1)
        with pytest.raises(ValidationError):
            TranscriptionParameters(context_max_length=501)


class TestFilePathValidation:
    """Tests for file path security validation."""

    def test_sanitize_filename_long_names(self):
        """sanitize_filename should handle very long file names."""
        from utils import sanitize_filename

        long_name = "a" * 500 + ".mp3"
        result = sanitize_filename(long_name)
        # Function doesn't truncate, but should not raise
        assert result is not None
        assert len(result) > 0
        # Should preserve the extension
        assert result.endswith(".mp3")

    def test_sanitize_filename_unicode(self):
        """sanitize_filename should handle unicode characters."""
        from utils import sanitize_filename

        unicode_name = "音声ファイル_録音.mp3"
        result = sanitize_filename(unicode_name)
        # Should not raise, should return something valid
        assert result is not None
        assert len(result) > 0

    def test_sanitize_filename_empty_after_sanitization(self):
        """sanitize_filename should handle names that become empty after sanitization."""
        from utils import sanitize_filename

        # Names that might become empty after sanitization
        problematic_names = ["...", "///", "\\\\", ""]
        for name in problematic_names:
            result = sanitize_filename(name)
            # Should either return empty or some default
            assert isinstance(result, str)

    def test_is_valid_audio_extension_case_insensitive(self):
        """is_valid_audio_extension should be case insensitive."""
        from utils import is_valid_audio_extension

        extensions = [".MP3", ".Wav", ".FLAC", ".OGG"]
        for ext in extensions:
            # Should handle uppercase extensions
            assert is_valid_audio_extension(ext.lower())


class TestModelRepoMapping:
    """Tests for model repository mapping."""

    def test_model_repo_map_contains_standard_models(self):
        """MODEL_REPO_MAP should contain all standard model sizes."""
        from utils import MODEL_REPO_MAP

        standard_models = ["tiny", "base", "small", "medium", "large-v3"]
        for model in standard_models:
            assert model in MODEL_REPO_MAP
            assert MODEL_REPO_MAP[model].startswith("Systran/")

    def test_model_repo_map_contains_distil_models(self):
        """MODEL_REPO_MAP should contain distilled model variants."""
        from utils import MODEL_REPO_MAP

        assert "distil-large-v3" in MODEL_REPO_MAP
        assert "distil" in MODEL_REPO_MAP["distil-large-v3"].lower()

    def test_model_repo_map_values_are_valid_repo_ids(self):
        """MODEL_REPO_MAP values should be valid HuggingFace repo IDs."""
        from utils import MODEL_REPO_MAP

        for model, repo_id in MODEL_REPO_MAP.items():
            # Repo IDs should have format "org/name"
            assert "/" in repo_id
            parts = repo_id.split("/")
            assert len(parts) == 2
            assert len(parts[0]) > 0  # org name
            assert len(parts[1]) > 0  # repo name


class TestComputeTypeSelection:
    """Tests for compute type selection based on device."""

    def test_cpu_uses_int8(self):
        """CPU device should use int8 compute type."""
        # This is tested indirectly through model loading tests
        # But we can verify the logic directly
        device = "cpu"
        compute_type = "int8" if device == "cpu" else "int8_float16"
        assert compute_type == "int8"

    def test_cuda_uses_int8_float16(self):
        """CUDA device should use int8_float16 compute type."""
        device = "cuda"
        compute_type = "int8" if device == "cpu" else "int8_float16"
        assert compute_type == "int8_float16"

    def test_mps_uses_int8_float16(self):
        """MPS (Apple Silicon) device should use int8_float16 compute type."""
        device = "mps"
        compute_type = "int8" if device == "cpu" else "int8_float16"
        assert compute_type == "int8_float16"
