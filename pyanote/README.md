# pyannoteAI transcription

Transcribe with **Whisper large-v2** and pyannoteAI speaker diarization.

## Install

Requires an NVIDIA GPU, driver >=580, CUDA 13.x, cuDNN 9, and ffmpeg 4–9.
Run from the repository root:

```bash
uv venv pyanote/.venv --python 3.12

export PYA_DOWNLOAD_KEY='your-download-key'
uv pip install --python pyanote/.venv/bin/python \
  --extra-index-url "https://${PYA_DOWNLOAD_KEY}@download.pyannote.ai" \
  -r pyanote/requirements.txt
uv pip install --python pyanote/.venv/bin/python \
  requests
```

## Run

Use a separate API key with **Run jobs** permission:

```bash
export HF_TOKEN='HF_token'
export HF_HOME=/data/models
mkdir /data
export PYA_JOBS_KEY='your-jobs-key'

pyanote/.venv/bin/python pyanote/transcribe.py \
  --warmup-file /data/warmup.wav \
  --audio-file /data/meeting.wav \
  --batch-size 32 \
  --stt-batch-size 16
```

- `--audio-file`: file to transcribe.
- `--warmup-file`: optional short clip processed first; its result is discarded.
- `--batch-size`: diarization batch size (default 8, minimum 8).
- `--stt-batch-size`: Whisper batch size (default 8, minimum 1).
- `--mode`: `accuracy` (default), `balance`, or `speed`.
- `-o`: output JSON path (default: `/data/meeting.transcription.json` for this example).

Prints speaker turns and saves JSON with word timestamps and speaker labels.
Timestamped logs show stage starts, completions, and elapsed time.
