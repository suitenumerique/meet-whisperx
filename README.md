# Whisper OpenAI API

## Contributing

### Local development
```bash
pip install .
```

Run the server on port 8010 (avoid conflict with the default ports 8000, 8001... with other services)
```bash
export PORT=8010 
export RELOAD=true
export LOGGING_CONFIG=logging-config.yaml
python app/main.py
```

### Docker development

```bash
export MODELS_CACHE_DIR="." && docker compose up --pull always
```

## Environment variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| MODEL | WhisperX model to load | `large-v2` |
| PORT | Server port | `8000` |
| RELOAD | Enable auto-reload | `false` |
| ROOT_PATH | API root path | `None` |
| LOGGING_CONFIG | Path to logging config file | `None` |
| DEBUG | Enable debug logging | `false` |
| API_KEY | API key for API access | Required |
| HF_TOKEN | Hugging Face token | Required |
| BATCH_SIZE | Transcription batch size | `16` |
| TIMEOUT_KEEP_ALIVE | Keep-alive timeout (seconds) | `60` |
| MODELS_CACHE_DIR | Directory to store the models (Docker only) | `/data/models` |
