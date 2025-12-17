"""Silero VAD model downloader utility for EchoScribe."""

import torch


def download_silero_vad() -> None:
    """Download Silero VAD model to current directory."""
    print("Downloading Silero VAD model...")
    try:
        torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
            onnx=True,
            trust_repo=True
        )
        print("Model downloaded successfully as 'silero_vad.onnx'.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check your internet connection and PyTorch installation.")


if __name__ == "__main__":
    download_silero_vad()
