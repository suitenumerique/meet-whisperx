# Whisper OpenAI API

## Contributing

### Local development
```bash
pip install ".[inference]"
```

Run the server on port 8010 (avoid conflict with the default ports 8000, 8001... with other services)
```bash
export PORT=8010
export RELOAD=true
export LOGGING_CONFIG=logging-config.yaml
python app/main.py
```

## Environment variables

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



