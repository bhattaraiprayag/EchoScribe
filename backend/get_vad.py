# backend/get_vad.py

import torch

def download_silero_vad():
    """
    Downloads the Silero VAD model to the current directory.
    """
    print("Downloading Silero VAD model...")
    try:
        torch.hub.load(repo_or_dir='snakers4/silero-vad',
                       model='silero_vad',
                       force_reload=True,
                       onnx=True)
        print("Model downloaded successfully as 'silero_vad.onnx'.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check your internet connection and PyTorch installation.")

if __name__ == "__main__":
    download_silero_vad()
