# Testing with GPU

Integration tests in `app/tests_with_gpu/` run real WhisperX inference and require a CUDA-capable GPU. This guide covers running them on [RunPod](https://www.runpod.io/).

## Prerequisites

- A RunPod account
- A Hugging Face token with access to the pyannote diarization models

## 1. Launch a RunPod Instance

Create a GPU pod with the following settings:

| Setting        | Value                                                        |
| -------------- | ------------------------------------------------------------ |
| Template image | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`  |
| Exposed port   | `8010` (or any free port)                                    |

Set these environment variables **before starting the pod** (RunPod > Pod Settings > Environment Variables):

| Variable                            | Value               |
| ----------------------------------- | ------------------- |
| `API_KEY`                           | *(choose any value — the tests will read and use this same value)* |
| `HF_TOKEN`                          | *(your HF token)*   |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | `1`                 |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0`                 |

## 2. Install System Dependencies

```bash
apt update && apt install -y ffmpeg
apt-get install -y libcudnn8 libcudnn8-dev
```

## 3. Clone and Install the Project

`uv` is used for package management. Install it first:

```bash
pip install uv
```

Then clone and install:

```bash
cd /workspace
git clone https://github.com/suitenumerique/meet-whisperx.git
cd meet-whisperx
uv sync --group dev --group inference
```

## 4. Run the Integration Tests

```bash
cd /workspace/meet-whisperx
uv run pytest app/tests_with_gpu -m integration
```

> **Note:** `testpaths` in `pyproject.toml` points to `app/tests` (unit tests). Pass the path explicitly to target the GPU tests.

## 5. Run the API Manually (Optional)

Start the server for manual / curl-based testing:

```bash
export LOGGING_CONFIG=logging-config.yaml
export PORT=8010
export RELOAD=true
uv run python app/main.py
```

### Health Check

```bash
curl -X GET "https://<pod-id>-8010.proxy.runpod.net/info" \
  -H "Authorization: Bearer $API_KEY"
```

### Transcription

```bash
curl -X POST "https://<pod-id>-8010.proxy.runpod.net/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@path/to/audio.ogg"
```
