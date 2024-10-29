# Whisper OpenAI API

## Contributing

### Local development
```bash
pip install .
```

```bash
python app/main.py
```

### Docker development

```bash
export MODELS_CACHE_DIR="." && docker compose up --pull always
```

## Environment variables

| Variable | Description |
| -------- | ----------- |
| MODELS_CACHE_DIR | Directory to store the models |
| API_KEY | API key for API access (optional) |
