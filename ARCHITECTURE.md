# System Architecture

## Overview

EchoScribe is designed as a high-performance, low-latency audio transcription system capable of handling both real-time streams and batch file processing. The architecture decouples audio ingestion, activity detection, and transcription to ensure responsiveness.

## Core Components

### 1. Backend Framework

- **FastAPI**: Chosen for its high performance (Starlette-based), native async support, and automatic OpenAPI documentation.
- **Uvicorn**: An ASGI server to run the application, handling WebSockets and HTTP requests efficiently.

### 2. Audio Processing Pipeline

The real-time transcription relies on a multi-stage pipeline coordinated by `asyncio.Queue`s.

#### Stages:

1.  **Ingestion (WebSocket)**:

    - Receives raw 16-bit PCM audio chunks from the client.
    - Requires a short-lived one-time `auth_token` in the initial configuration message.
    - Pushes data independently to a `raw_audio_queue`.
    - **Design Decision**: Decoupling ingestion prevents network latency or client stuttering from blocking the processing logic.

2.  **Voice Activity Detection (VAD)**:

    - **Engine**: Silero VAD (ONNX).
    - **Logic**: Continuously analyzes audio frames. Buffers speech segments until a pause (silence) is detected.
    - **Why Silero?**: Lightweight, fast, and robust against noise compared to WebRTC VAD.
    - **Output**: "Utterances" (complete sentences or phrases) are pushed to the `transcription_queue`.

3.  **Transcription Worker**:

    - **Engine**: `faster-whisper` (CTranslate2 backend).
    - **Execution**: Runs in a separate thread/process executor to avoid blocking the asyncio event loop, as model inference is CPU/GPU intensive.
    - **Optimization**: Uses CTranslate2 for 4x faster inference than OpenAI's vanilla Whisper and reduced memory usage (quantization).

4.  **Result Emitter**:
    - Picks up transcribed text and sends it back via the WebSocket.

### Security Controls

- **Authentication**: API key auth can be enabled in config and is enforced on settings updates, uploads, and WebSocket token issuance.
- **WebSocket Hardening**: The browser exchanges API key auth for a short-lived one-time token via `POST /api/ws-auth-token`, then presents that token during WebSocket session bootstrap.
- **Rate Limiting**: Config-driven in-memory limits are applied for API requests, uploads, and WebSocket connections.
- **Trusted Proxies**: `X-Forwarded-For` is only honored when the direct source IP is inside configured trusted proxy CIDRs.
- **Settings Hygiene**: `GET /api/settings` redacts sensitive values (e.g., `auth.api_key`) before returning payloads.

### 3. Dependency Management (Migration to `uv`)

We have migrated to **uv** for Python package management.

- **Speed**: `uv` resolves and installs dependencies significantly faster than pip.
- **Determinism**: The `uv.lock` file ensures that all developers and CI/CD pipelines use the exact same package versions across platforms (Universal resolution).
- **Workspace**: While currently a single project, `uv` positions us for a monorepo structure if frontend/backend separation grows.

### 4. Docker Deployment

The application is containerized using a **multi-stage build** process to minimize image size and ensure security.

- **Builder Stage**: Installs build tools (gcc, git) and compiles dependencies. Uses `uv sync` to install packages into a virtual environment.
- **Runtime Stage**: A slim Python image that copies _only_ the virtual environment (`.venv`) and application code.
- **Security**: No build tools or credentials exist in the final image.
- **GPU Support**: Configured to utilize NVIDIA GPUs via the NVIDIA Container Toolkit.

### 5. DevOps & CI/CD

We strictly adhere to "GitOps" and "Shift-Left" security principles.

- **CI Pipeline**: GitHub Actions workflow that runs on every Pull Request and Push to main. It enforces:
  - **Linting**: Ruff (Python).
  - **Formatting**: Ruff (Python) and Prettier (Frontend).
  - **Testing**: Full pytest suite and coverage checks.
  - **Build**: Docker image verification.
- **Pre-Commit Hooks**: Local enforcement of code quality and security (secret scanning) to prevent bad commits.
- **Infrastructure**: Dockerfile follows best practices (non-root user, multi-stage builds).

## Directory Structure

```
/
├── backend/            # Python backend source code
│   ├── main.py         # Entry point (FastAPI app)
│   ├── agent.py        # (Legacy/Agentic components)
│   ├── config.yaml     # Application configuration
│   └── ...
├── frontend/           # Static frontend assets (HTML/CSS/JS)
├── tests/              # Pytest suite
├── .venv/              # uv-managed virtual environment (not committed)
├── uv.lock             # Dependency lock file
├── pyproject.toml      # Project metadata and dependencies
└── Dockerfile          # Multi-stage Docker definition
```

## Production Considerations

- **Scaling**: The current architecture is single-node. For horizontal scaling, the WebSocket connection state and queues would need to be managed via a message broker (e.g., Redis Pub/Sub) to handle clients connecting to different replicas.
- **SSL/TLS**: In production, `uvicorn` should sit behind a reverse proxy like Nginx or Traefik handling SSL termination.
- **Authentication**: While basic API key auth is implemented, production environments should integrate with robust Identity Providers (OAuth2/OIDC).
