# Whisper OpenAI API

## Contributing

### Local development
```bash
pip install .
```

Run the server on port 8010 (avoid conflict with the default ports 8000, 8001... with other services)
```bash
python app/main.py --port 8010 --reload
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
