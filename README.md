# Whisper OpenAI API

FastAPI-based ASR (Automatic Speech Recognition) API built on WhisperX. Provides transcription, word-level alignment, and speaker diarization with OpenAI-compatible endpoints.

## Getting Started

### Install uv

NB: This package uses `uv` for package management as a modern alternative to pip. Install instructions [in this link](https://docs.astral.sh/uv/getting-started/installation/).

### API-only Development

Inference libraries (`whisperx`, `pytorch`, etc.) are heavy and may not run on all devices. We provide a `dev` dependency group to allow running API tests locally and IDE autocompletion. To install:

```bash
uv sync --group dev
```

### Full Inference Development

To develop with a fully functional transcription pipeline:

```bash
uv sync --group dev --group inference
```

Run the server locally (port 8010 avoids conflicts with other services):

```bash
export PORT=8010
export RELOAD=true
export LOGGING_CONFIG=logging-config.yaml
python app/main.py
```

## Testing

Tests mock actual inference and can be run locally:

```bash
cd app
python -m pytest tests/ -v
```

## Environment Variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| API_KEY | API key for API access | Required |
| HF_TOKEN | Hugging Face token | Required |
| BATCH_SIZE | Transcription batch size | `16` |
| MODEL | WhisperX model to load | `large-v2` |
| TIMEOUT_KEEP_ALIVE | Keep-alive timeout (seconds) | `60` |
| RETURN_CHAR_ALIGNMENTS | Return character-level alignments | `false` |
| INTERPOLATE_METHOD | WhisperX interpolation method | `nearest` |
| FILL_NEAREST | Fill nearest gaps in alignment | `false` |
| PORT | Server port | `8000` |
| RELOAD | Enable auto-reload | `false` |
| ROOT_PATH | API root path | `None` |
| LOGGING_CONFIG | Path to logging config file | `None` |
| DEBUG | Enable debug logging | `false` |

## Contributing

Please follow [these guidelines](https://suitenumerique.gitbook.io/handbook) when contributing to this repo.
