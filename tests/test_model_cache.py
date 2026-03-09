# tests/test_model_cache.py
"""Tests for Whisper model cache validation, repair, and locking."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _create_snapshot(
    root: Path,
    model_cache_name: str,
    *,
    include_config: bool = True,
    include_model_bin: bool = True,
    include_tokenizer: bool = True,
    include_vocabulary: bool = True,
) -> Path:
    """Create a minimal cache snapshot for tests."""
    snapshot_dir = root / model_cache_name / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)

    if include_config:
        (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    if include_model_bin:
        (snapshot_dir / "model.bin").write_bytes(b"model")
    if include_tokenizer:
        tokenizer_payload = {
            "model": {"vocab": {"a": 0, "b": 1}},
            "added_tokens": [{"id": 2, "content": "<|endoftext|>"}],
        }
        (snapshot_dir / "tokenizer.json").write_text(
            json.dumps(tokenizer_payload), encoding="utf-8"
        )
    if include_vocabulary:
        (snapshot_dir / "vocabulary.json").write_text(
            json.dumps(["a", "b", "<|endoftext|>"]),
            encoding="utf-8",
        )

    return snapshot_dir


class TestIsModelCached:
    """Tests for cache validation."""

    def test_complete_model_returns_true(self):
        """A model is cached only when all required load files exist."""
        from backend.utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_snapshot(
                Path(tmpdir), "models--Systran--faster-distil-whisper-large-v3"
            )
            assert is_model_cached("distil-large-v3", tmpdir) is True

    def test_missing_model_returns_false(self):
        """Missing cache roots should not be treated as cached."""
        from backend.utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_model_cached("distil-large-v3", tmpdir) is False

    def test_missing_snapshot_returns_false(self):
        """Missing snapshot directories should not be treated as cached."""
        from backend.utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models--Systran--faster-distil-whisper-large-v3"
            model_dir.mkdir(parents=True)
            assert is_model_cached("distil-large-v3", tmpdir) is False

    def test_missing_model_bin_returns_false(self):
        """Snapshots without model.bin are incomplete."""
        from backend.utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_snapshot(
                Path(tmpdir),
                "models--Systran--faster-distil-whisper-large-v3",
                include_model_bin=False,
            )
            assert is_model_cached("distil-large-v3", tmpdir) is False

    def test_missing_vocabulary_returns_false(self):
        """Snapshots without vocabulary are not load-ready."""
        from backend.utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_snapshot(
                Path(tmpdir),
                "models--Systran--faster-whisper-tiny",
                include_vocabulary=False,
            )
            assert is_model_cached("tiny", tmpdir) is False

    def test_unknown_model_returns_false(self):
        """Unknown models should not be treated as cached."""
        from backend.utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_model_cached("unknown-model", tmpdir) is False

    def test_all_standard_models_have_mapping(self):
        """All exposed models should be mapped to Hugging Face repos."""
        from backend.utils import MODEL_REPO_MAP

        standard_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "distil-large-v3",
        ]
        for model in standard_models:
            assert model in MODEL_REPO_MAP, (
                f"Model '{model}' missing from MODEL_REPO_MAP"
            )


class TestRepairModelSnapshot:
    """Tests for offline vocabulary repair."""

    def test_repair_creates_vocabulary_json_from_tokenizer(self):
        """repair_model_snapshot should synthesize vocabulary.json."""
        from backend.utils import repair_model_snapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = _create_snapshot(
                Path(tmpdir),
                "models--Systran--faster-whisper-tiny",
                include_vocabulary=False,
            )

            repaired_files = repair_model_snapshot(snapshot_dir)

            assert repaired_files == ["vocabulary.json"]
            vocab_payload = json.loads(
                (snapshot_dir / "vocabulary.json").read_text(encoding="utf-8")
            )
            assert vocab_payload == ["a", "b", "<|endoftext|>"]

    def test_repair_returns_empty_when_vocabulary_already_exists(self):
        """Existing vocabulary files should not be rewritten."""
        from backend.utils import repair_model_snapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = _create_snapshot(
                Path(tmpdir), "models--Systran--faster-whisper-tiny"
            )

            assert repair_model_snapshot(snapshot_dir) == []


class TestGetModelStatus:
    """Tests for detailed cache status reporting."""

    def test_get_model_status_cached_model(self):
        """Complete snapshots should report cached=True."""
        from backend.utils import get_model_status

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_snapshot(Path(tmpdir), "models--Systran--faster-whisper-tiny")

            status = get_model_status("tiny", tmpdir)

            assert status["cached"] is True
            assert "model.bin" in status["existing_files"]
            assert "vocabulary.json" in status["existing_files"]
            assert "repo_id" in status

    def test_get_model_status_missing_model(self):
        """Missing snapshots should report required missing files."""
        from backend.utils import get_model_status

        with tempfile.TemporaryDirectory() as tmpdir:
            status = get_model_status("tiny", tmpdir)

            assert status["cached"] is False
            assert "missing_files" in status
            assert "vocabulary.json" in status["missing_files"]

    def test_get_model_status_unknown_model(self):
        """Unknown models should report a descriptive error."""
        from backend.utils import get_model_status

        status = get_model_status("unknown-model-xyz")
        assert status["cached"] is False
        assert "error" in status

    def test_get_model_status_returns_repo_id(self):
        """Known models should include their repo mapping."""
        from backend.utils import MODEL_REPO_MAP, get_model_status

        status = get_model_status("base")
        assert status["repo_id"] == MODEL_REPO_MAP["base"]

    def test_get_model_status_partial_cache_is_not_ready(self):
        """Partial snapshots should report cached=False and repair hints."""
        from backend.utils import get_model_status

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_snapshot(
                Path(tmpdir),
                "models--Systran--faster-whisper-small",
                include_vocabulary=False,
            )

            status = get_model_status("small", tmpdir)

            assert status["cached"] is False
            assert "vocabulary.json" in status["missing_files"]
            assert "vocabulary.json" in status["repairable_files"]


class TestModelCacheModuleImport:
    """Regression coverage for cache-directory setup during import."""

    def test_import_does_not_create_default_cache_dirs(self):
        """Importing backend.utils should not eagerly create cache directories."""
        repo_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        import_code = """
import pathlib

def fail_mkdir(self, *args, **kwargs):
    raise RuntimeError("mkdir called during import")

pathlib.Path.mkdir = fail_mkdir
import backend.utils
"""
        result = subprocess.run(
            [sys.executable, "-c", import_code],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr


class TestModelCacheLocking:
    """Tests for asynchronous model loading and caching."""

    async def test_get_whisper_model_returns_model(self):
        """get_whisper_model should return the loaded WhisperModel instance."""
        from backend.utils import get_whisper_model, model_cache

        model_cache.clear()
        mock_model = MagicMock()

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel", return_value=mock_model),
        ):
            result = await get_whisper_model("tiny", "cpu")
            assert result is mock_model

    async def test_get_whisper_model_caches_result(self):
        """Repeated requests for the same model/device should reuse the instance."""
        from backend.utils import get_whisper_model, model_cache

        model_cache.clear()
        mock_model = MagicMock()

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel", return_value=mock_model) as mock_class,
        ):
            result1 = await get_whisper_model("tiny", "cpu")
            result2 = await get_whisper_model("tiny", "cpu")

            assert mock_class.call_count == 1
            assert result1 is result2

    async def test_different_models_cached_separately(self):
        """Different model keys should keep separate cached model instances."""
        from backend.utils import get_whisper_model, model_cache

        model_cache.clear()
        mock_model_tiny = MagicMock(name="tiny")
        mock_model_base = MagicMock(name="base")

        def create_model(model_path, **kwargs):
            return mock_model_tiny if model_path == "/mock/tiny" else mock_model_base

        with (
            patch(
                "backend.utils.ensure_model_files",
                side_effect=["/mock/tiny", "/mock/base"],
            ),
            patch("backend.utils.WhisperModel", side_effect=create_model),
        ):
            result_tiny = await get_whisper_model("tiny", "cpu")
            result_base = await get_whisper_model("base", "cpu")

            assert result_tiny is not result_base
            assert "tiny_cpu" in model_cache
            assert "base_cpu" in model_cache

    async def test_concurrent_requests_load_model_once(self):
        """Concurrent loads of the same model should instantiate WhisperModel once."""
        from backend.utils import get_whisper_model, model_cache

        model_cache.clear()
        load_count = 0

        def slow_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            return MagicMock()

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel", side_effect=slow_load),
        ):
            results = await asyncio.gather(
                get_whisper_model("tiny", "cpu"),
                get_whisper_model("tiny", "cpu"),
                get_whisper_model("tiny", "cpu"),
            )

            assert results[0] is results[1]
            assert results[1] is results[2]
            assert load_count == 1

    async def test_model_cache_lock_exists(self):
        """model_cache_lock should be an asyncio.Lock."""
        from backend.utils import model_cache_lock

        assert isinstance(model_cache_lock, asyncio.Lock)

    async def test_get_whisper_model_uses_correct_compute_type(self):
        """CPU and CUDA should select the expected compute types."""
        from backend.utils import get_whisper_model, model_cache

        model_cache.clear()

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel") as mock_class,
        ):
            mock_class.return_value = MagicMock()
            await get_whisper_model("tiny", "cpu")
            assert mock_class.call_args.kwargs["compute_type"] == "int8"

        model_cache.clear()

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel") as mock_class,
        ):
            mock_class.return_value = MagicMock()
            await get_whisper_model("tiny", "cuda")
            assert mock_class.call_args.kwargs["compute_type"] == "int8_float16"

    async def test_get_whisper_model_uses_prepared_snapshot_path(self):
        """Prepared snapshot paths should be passed to WhisperModel."""
        from backend.utils import get_whisper_model, model_cache

        model_cache.clear()

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel") as mock_class,
        ):
            mock_class.return_value = MagicMock()
            await get_whisper_model("tiny", "cpu")

            assert mock_class.call_args.args[0] == "/mock/path"


class TestDownloadModelFiles:
    """Tests for download_model_files."""

    def test_download_model_files_calls_snapshot_download(self):
        """download_model_files should delegate to faster-whisper snapshot download."""
        from backend.utils import download_model_files

        with (
            patch(
                "backend.utils.download_model_snapshot",
                return_value="/mock/cache/snapshots/abc123",
            ) as mock_download,
            patch("backend.utils.repair_model_snapshot", return_value=[]),
            patch("backend.utils.get_model_status", return_value={"cached": True}),
        ):
            result = download_model_files("tiny", "/tmp/cache")

            mock_download.assert_called_once_with(
                "tiny", cache_dir="/tmp/cache", local_files_only=False
            )
            assert result == "/mock/cache/snapshots/abc123"

    def test_download_model_files_progress_callback(self):
        """download_model_files should emit progress updates when provided."""
        from backend.utils import download_model_files

        progress_calls = []

        def callback(status, message, progress):
            progress_calls.append((status, message, progress))

        with (
            patch(
                "backend.utils.download_model_snapshot",
                return_value="/mock/cache/snapshots/abc123",
            ),
            patch(
                "backend.utils.repair_model_snapshot",
                return_value=["vocabulary.json"],
            ),
            patch("backend.utils.get_model_status", return_value={"cached": True}),
        ):
            download_model_files("tiny", "/tmp/cache", callback)

            assert any(call[0] == "downloading" for call in progress_calls)
            assert any(call[0] == "loading" for call in progress_calls)

    def test_download_model_files_unknown_model_raises(self):
        """Unknown model names should raise ValueError."""
        from backend.utils import download_model_files

        with pytest.raises(ValueError, match="Unknown model"):
            download_model_files("nonexistent-model-xyz")

    def test_download_model_files_incomplete_cache_raises(self):
        """download_model_files should fail on incomplete refreshed caches."""
        from backend.utils import download_model_files

        with (
            patch(
                "backend.utils.download_model_snapshot",
                return_value="/mock/cache/snapshots/abc123",
            ),
            patch("backend.utils.repair_model_snapshot", return_value=[]),
            patch(
                "backend.utils.get_model_status",
                return_value={
                    "cached": False,
                    "missing_files": ["vocabulary.json"],
                },
            ),
        ):
            with pytest.raises(
                RuntimeError, match="Model cache is incomplete after refresh"
            ):
                download_model_files("tiny", "/tmp/cache")


class TestSyncModelCacheLocking:
    """Tests for synchronous model loading."""

    def test_sync_model_cache_lock_exists(self):
        """sync_model_cache_lock should be a threading.Lock."""
        from backend.utils import sync_model_cache_lock

        assert isinstance(sync_model_cache_lock, type(threading.Lock()))

    def test_get_whisper_model_sync_uses_lock(self):
        """Synchronous loads should only instantiate once across threads."""
        from backend.utils import get_whisper_model_sync, model_cache

        model_cache.clear()
        load_count = 0
        load_count_lock = threading.Lock()

        def slow_load(*args, **kwargs):
            nonlocal load_count
            with load_count_lock:
                load_count += 1
            return MagicMock()

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel", side_effect=slow_load),
        ):
            threads = []
            results = []
            results_lock = threading.Lock()

            def load_model():
                result = get_whisper_model_sync("tiny", "cpu")
                with results_lock:
                    results.append(result)

            for _ in range(3):
                thread = threading.Thread(target=load_model)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            assert load_count == 1
            assert len(results) == 3
            assert results[0] is results[1]
            assert results[1] is results[2]

        model_cache.clear()


class TestProgressCallback:
    """Tests for status callbacks and model-loaded checks."""

    async def test_get_whisper_model_calls_progress_callback(self):
        """get_whisper_model should emit progress updates while loading."""
        from backend.utils import get_whisper_model, model_cache

        model_cache.clear()
        progress_calls = []

        def callback(status, message, progress):
            progress_calls.append((status, message, progress))

        with (
            patch("backend.utils.ensure_model_files", return_value="/mock/path"),
            patch("backend.utils.WhisperModel", return_value=MagicMock()),
        ):
            await get_whisper_model("tiny", "cpu", callback)

            assert len(progress_calls) > 0
            assert any(call[0] == "loading" for call in progress_calls)
            assert any(call[0] == "ready" for call in progress_calls)

    async def test_get_whisper_model_cached_returns_immediately(self):
        """In-memory cache hits should still emit a ready callback."""
        from backend.utils import get_whisper_model, model_cache

        mock_model = MagicMock()
        model_cache["tiny_cpu"] = mock_model
        progress_calls = []

        def callback(status, message, progress):
            progress_calls.append((status, message, progress))

        result = await get_whisper_model("tiny", "cpu", callback)

        assert result is mock_model
        assert any(call[0] == "ready" for call in progress_calls)

        model_cache.clear()

    def test_is_model_loaded_returns_true_for_loaded_model(self):
        """is_model_loaded should reflect the in-memory model cache."""
        from backend.utils import is_model_loaded, model_cache

        model_cache["tiny_cpu"] = MagicMock()
        assert is_model_loaded("tiny", "cpu") is True
        assert is_model_loaded("tiny", "cuda") is False
        model_cache.clear()
