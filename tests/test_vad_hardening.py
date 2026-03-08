"""Tests for hardened Silero VAD loading strategy."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_runtime_vad_loading_no_torch_hub_remote_fetch():
    """Backend runtime code should avoid torch.hub remote loading."""
    main_code = Path("backend/main.py").read_text(encoding="utf-8")
    pipeline_code = Path("backend/pipeline.py").read_text(encoding="utf-8")
    downloader_code = Path("backend/get_vad.py").read_text(encoding="utf-8")

    assert "torch.hub.load(" not in main_code
    assert "torch.hub.load(" not in pipeline_code
    assert "zipball/master" not in downloader_code


def test_pipeline_loads_vad_from_silero_package():
    """Pipeline should use load_silero_vad(onnx=True)."""
    from backend.pipeline import TranscriptionSession

    fake_session = MagicMock(spec=TranscriptionSession)
    fake_model = MagicMock()
    with patch("backend.pipeline.load_silero_vad", return_value=fake_model) as loader:
        model, utils = TranscriptionSession.load_vad_model(fake_session)

    loader.assert_called_once_with(onnx=True)
    assert model is fake_model
    assert utils is None
