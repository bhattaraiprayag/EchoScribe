# Docker CI Build Optimization Analysis

## Investigation Output
**Date**: 2026-01-24
**Subject**: CI Build Time Optimization (Currently ~45 mins)

### 1. Root Cause Analysis
The discrepancy between local build times (fast) and GitHub Actions build times (slow) is primarily due to **Cache Invalidation of Heavy Dependencies (PyTorch)**.

- **The Monolithic Dependency Layer**: currently, your `Dockerfile` installs all dependencies in a single `RUN` instruction:
  ```dockerfile
  RUN ... uv sync --frozen --no-install-project --no-editable
  ```
- **The "All-or-Nothing" Cache**: Docker caches build layers based on the command string and input files (`uv.lock`). 
  - If **ANY** dependency in your project changes (e.g., you update `fastapi` or `pydantic`, or even a tiny dev tool), the hash of `uv.lock` changes.
  - This invalidates the **entire** `RUN` layer.
  - Consequently, Docker must re-run the command, which triggers `uv` to re-download **ALL** dependencies from scratch.
- **The Bandwidth Bottleneck**: Your project depends on `torch>=2.9.1` with CUDA 12.9 support. 
  - PyTorch + CUDA binaries are approximately **3GB - 5GB** in size.
  - GitHub Actions runners have decent bandwidth, but downloading 5GB of data from the PyTorch index (which can be slow) takes a significant amount of time (20-40 mins is not uncommon for fresh pulls of these sizes).
- **Cache Persistence Limits**: While you are using `cache-to: type=gha`, GitHub Actions cache has a size limit (10GB total per repo). A 5GB layer + intermediate layers can easily hit eviction thresholds, causing cache misses even if `uv.lock` hasn't changed.

### 2. Optimization Strategies

#### Strategy A: The "Split Layers" Approach (Recommended)
**Concept**: Isolate the installation of heavy, slowly-changing dependencies (PyTorch, Nvidia deps) into their own Docker layer *before* the main `uv sync`.

**Why it works**:
1. You add a `RUN` step to install `torch` explicitly.
2. This creates a cached layer containing the heavy files.
3. When you update other libraries in `uv.lock`, only the subsequent `uv sync` layer is invalidated.
4. The `uv sync` command runs on top of the cached "Torch Layer". It detects `torch` is already installed and valid, skipping the massive download.

**Implementation Plan**:
Modify `Dockerfile` to manually install heavy deps into the virtual environment first.

```dockerfile
# ... setup steps ...

# create venv explicitly
RUN uv venv /app/.venv

# Layer 1: Heavy Dependencies (Cached aggressively)
# This layer only rebuilds if you change this specific line (e.g. upgarding torch)
RUN uv pip install --python /app/.venv "torch==2.9.1+cu129" "torchaudio==2.9.1+cu129" --extra-index-url https://download.pytorch.org/whl/cu129

# Layer 2: The rest of the dependencies
# This layer rebuilds whenever uv.lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-editable
```

#### Strategy B: The "Base Image" Approach (Robust / Enterprise)
**Concept**: Create a separate `Dockerfile.base` that builds an image `echoscribe-base` containing Python + UV + PyTorch.
**Workflow**:
1. Build `echoscribe-base` and push to GitHub Container Registry (GHCR).
2. Change your main `Dockerfile` to `FROM ghcr.io/yourname/echoscribe-base:latest`.
**Pros**: Main CI is lightning fast (seconds/minutes).
**Cons**: Requires managing a second workflow to build the base image.

#### Strategy C: CPU-Only Builds for CI (Partial Solution)
If the CI `docker-build` job is only for verification (not deployment to a GPU server), configuring it to use CPU-only wheels decreases the download size from ~5GB to ~200MB.
However, since you likely deploy this image, this might not be an option.

### 3. Immediate Action Plan
We can implement **Strategy A** immediately without changing your infrastructure or adding workflows. 

1. **Verify Versions**: We need to lock the exact version of `torch` you are using (check `uv.lock`).
2. **Update Dockerfile**: Insert the split installation steps.

This change optimizes the build by ensuring the 45-minute "tax" is only paid when you explicitly upgrade PyTorch, not when you tweak your app code or other libraries.

### 4. Other Notes
- **.dockerignore**: Your `.dockerignore` is good, but ensure `.git` is definitely ignored (it is).
- **Disk Space**: The "Free Disk Space" step in CI is necessary because 5GB+ downloads + extraction can fill up the 14GB standard runner space. Keeping this step is correct.
