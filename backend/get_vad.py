"""Silero VAD model verification utility for EchoScribe."""

from silero_vad import load_silero_vad


def verify_silero_vad() -> None:
    """Verify pinned silero-vad package can load the model artifact."""
    print("Loading Silero VAD model from installed package...")
    try:
        load_silero_vad(onnx=True)
        print("Silero VAD model loaded successfully.")
    except Exception as e:
        print(f"Error loading Silero VAD model: {e}")
        print("Please check your silero-vad installation.")


if __name__ == "__main__":
    verify_silero_vad()
