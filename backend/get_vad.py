"""Silero VAD model downloader utility for EchoScribe."""

import torch
from utils import VAD_CACHE_DIR


def download_silero_vad() -> None:
    """Download Silero VAD model to the models_cache directory."""
    print(f"Downloading Silero VAD model to: {VAD_CACHE_DIR}")
    try:
        # Set torch hub directory to use our cache location
        torch.hub.set_dir(VAD_CACHE_DIR)
        torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
            onnx=True,
            trust_repo=True
        )
        print(f"Model downloaded successfully to '{VAD_CACHE_DIR}'.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check your internet connection and PyTorch installation.")


if __name__ == "__main__":
    download_silero_vad()
