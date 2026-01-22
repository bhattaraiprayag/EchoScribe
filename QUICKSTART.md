# Quickstart Guide

This guide will help you get EchoScribe up and running quickly on your local machine or using Docker.

## Prerequisites

- **Python 3.11+** installed.
- **uv** installed (recommended for dependency management).
  - Installation: `pip install uv` or see [uv docs](https://docs.astral.sh/uv/getting-started/installation/).
- **Git** to clone the repository.
- **FFmpeg**: Required system dependency.
  - Windows: [Download](https://ffmpeg.org/download.html) and add to PATH.
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

## 🚀 Running Locally with `uv`

1.  **Clone the repository:**
    ```powershell
    git clone https://github.com/bhattaraiprayag/echoscribe.git
    cd echoscribe
    ```

2.  **Sync dependencies:**
    This will create a virtual environment (`.venv`) and install all required packages into it.
    ```powershell
    uv sync
    ```

3.  **Download VAD Model (Auto-downloading, but you can pre-fetch if desired):**
    The application will attempt to download `silero_vad.onnx` on first run if missing, but you can verify using:
    ```powershell
    uv run python backend/get_vad.py
    ```

4.  **Run the Server:**
    ```powershell
    uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```

5.  **Access the App:**
    Open [http://localhost:8000](http://localhost:8000) in your browser.

## 🐳 Running with Docker

1.  **Start the container:**
    ```powershell
    docker-compose up --build
    ```

2.  **Access the App:**
    Open [http://localhost:8000](http://localhost:8000).

## 🧪 Running Tests

To verify your installation:

```powershell
uv run pytest tests/
```

## 🔧 Troubleshooting

- **Missing FFmpeg:** If you see errors about audio processing, ensure `ffmpeg` is in your system PATH.
- **Cuda/GPU Issues:** If you have an NVIDIA GPU but it's not being detected, ensure you have the correct NVIDIA drivers installed. The Docker image attempts to use CUDA 12.9 equivalents; local setups will default to your installed drivers or PyTorch's default (CPU) if not configured.
- **VAD Model Error:** If the VAD model fails to load, ensure `silero_vad.onnx` is present in the root directory (where the backend runs).
