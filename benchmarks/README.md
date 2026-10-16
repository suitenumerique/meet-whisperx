# RunPod transcription performance benchmarks

Two independent runners produce the same JSON protocol:

- `run_repo.py`: the repository's real `lifespan` and `transcribe` functions,
  including all four preloaded alignment languages by default.
- `run_premium.py`: local premium `Pipeline.transcribe(path, FasterWhisperSTT())`.
  No hybrid pipeline, HTTP service, remote transcription API, or quality scoring.
- `compare.ipynb`: reconcile results from both environments and both GPUs.
  Analysis needs neither CUDA nor either inference SDK.

Each configuration runs in a fresh subprocess. Models stay loaded across files
and repeats within that configuration. Files run sequentially; this measures
single-worker latency and throughput, not concurrent request capacity.

## Prepare both RunPod machines

Use one L40S pod and one H100 pod with the same dataset mounted under `/data`.
Put model caches in `/data/models`. Use a comparable CPU/RAM allocation and
storage for both pods. H100 variants and CPU allocations can differ; actual GPU
names, UUIDs, VRAM, CPU affinity and cgroup limits are saved with every trial.
Leave the GPU idle except for the benchmark and expose exactly one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/models
```

Install `ffmpeg`/`ffprobe` on each pod and clone this repository. Run the commands
below from its root. Paths under `/workspace/venvs` are examples of persistent
RunPod virtual environments; adapt them to your pod.

### Repository environment

Match the repository Dockerfile's CUDA 12.8 / cuDNN runtime and locked PyTorch
2.8 dependencies. Set `HF_TOKEN` to a token authorized for the community
diarization models and accept the models' access conditions beforehand.

```bash
UV_PROJECT_ENVIRONMENT=/workspace/venvs/repo \
  uv sync --frozen --group inference --no-dev
uv pip install --python /workspace/venvs/repo/bin/python \
  -r benchmarks/requirements-runner.txt

# Match the existing Docker image's checkpoint-loading configuration.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

/workspace/venvs/repo/bin/python benchmarks/run_repo.py \
  --label l40s-repo --data-dir /data \
  --batch-sizes 8 16 32 64 --repeats 3
```

Settings otherwise come from the application's existing environment/`.env`.
The runner overrides `batch_size` for each configuration and sets a dummy
`API_KEY` if absent because it never serves HTTP. Tokens are excluded from saved
settings. Use `--language fr` to force one language or omit it to retain language
detection. Avoid unrelated changes to model settings between GPU runs.

### Premium environment

The supplied premium v4 onboarding material specifies Python >=3.10,
CUDA 13, cuDNN 9 and an NVIDIA driver >=580; its changelog specifies PyTorch
2.11. Use a RunPod image/runtime compatible with those requirements. An isolated
virtual environment alone does not supply a compatible driver/runtime.

Use `PYA_DOWNLOAD_KEY` for the private package index and a separate `PYA_JOBS_KEY`
with Run jobs permission. Set both in the pod environment. The following installs
the v4 local package described in the supplied material:

```bash
uv venv /workspace/venvs/premium --python 3.12
uv pip install --python /workspace/venvs/premium/bin/python \
  --extra-index-url "https://${PYA_DOWNLOAD_KEY}@download.pyannote.ai" \
  'pyannoteai-local[io]==4.0.0' 'pyannoteai-stt[faster-whisper]'
uv pip install --python /workspace/venvs/premium/bin/python \
  -r benchmarks/requirements-runner.txt

/workspace/venvs/premium/bin/python benchmarks/run_premium.py --describe-backend
/workspace/venvs/premium/bin/python benchmarks/run_premium.py \
  --label l40s-premium --data-dir /data \
  --batch-sizes 8 16 32 64 --modes accuracy --repeats 3
```

The supplied documentation does not specify FasterWhisperSTT constructor
options. The runner uses the documented no-argument constructor by default and
records its signature/defaults. To select the same ASR model as the repo
(`large-v2` by default), inspect `--describe-backend` in the installed premium
environment, then pass supported constructor arguments via
`--stt-kwargs '{...}'`. Unsupported arguments fail explicitly. Unspecified options
retain vendor defaults; a run with different ASR models measures the performance
of those complete setups. Do not assume matching model names or precision.

Repeat both commands on H100 using labels `h100-repo` and `h100-premium`.
Keep premium package versions and STT options identical across GPUs; use the
first run's recorded versions to pin subsequent installs. Premium `balance` and
`speed` modes are available through `--modes accuracy balance speed` if wanted.

## Batch sizes and timing boundaries

**The two batch controls affect different stages.** Repository `batch_size`
controls WhisperX ASR, leaving community diarization at its production defaults.
Premium `batch_size` controls diarization (minimum 8); it does not claim to control
Faster Whisper ASR batching. Any supported STT options can be set independently
with `--stt-kwargs` and become part of the recorded configuration.

The primary timer includes audio decoding and complete native inference, ending
when GPU work finishes. It excludes file hashing, ffprobe, model initialization,
warm-up, result normalization, JSON writing and sampler startup/shutdown. It also
excludes HTTP upload and serialization overhead. OS file caches are not flushed;
these are persistent-model measurements with normal OS caching. Model downloads, if needed,
are included only in the separately recorded initialization time.

The default warm-up processes the first manifest file once per fresh worker.
Use a representative first file, `--warmup-files N` to warm more of the sorted
manifest, or `--warmup-files 0` for a deliberately unwarmed run. Lazy initialization
for workloads not exercised by those files can still appear in measured trials.
`--seed` controls configuration and file ordering. It does not promise deterministic
inference. Measured repeats each process the entire corpus in a seeded order.

Repo stage timers wrap its actual decode, ASR, alignment, diarization and speaker
assignment calls. Premium records the entire native call plus the supported
`STT.apply` hook, forwarding the backend's documented preprocessing properties.
Premium native time minus STT time combines decoding, diarization, preparation
and merging; it is **not** a pure diarization measurement. Nested native and STT
events must not be summed together. CUDA is synchronized at every timed boundary;
this adds instrumentation overhead and can inhibit asynchronous overlap.

## Resources

A separate process samples every 0.2 seconds by default (`--sample-interval`).
Each JSON includes the time series and sampled peak/mean values for:

- Process-tree CPU use, where **100% means one fully occupied CPU core**.
- Sum of process-tree RSS; shared pages can be counted in multiple processes.
- Host CPU use and RAM use (host metrics, not container utilization).
- GPU device utilization, device VRAM, and process-tree VRAM when NVML exposes it.

The sampler excludes itself from process-tree metrics. CUDA subprocesses in a
different PID namespace can make per-process VRAM unavailable; it is saved as
null and device VRAM remains available. Device GPU metrics include other GPU
users and resident allocations. RAM/VRAM values are absolute usage, not increases
over idle. Sampling can miss short peaks and CPU use of very short-lived children.
Baseline and final samples outside the timed interval are retained for context
but excluded from summaries. Short trials can have no in-interval samples.
The notebook reports resource errors and missing samples explicitly.

## Files, failures and output

Runners recursively discover common audio/video extensions and hash/probe the
files before timing. Hidden folders, `models`, `benchmark-results`, and the chosen
output directory are excluded. `--limit N` is useful for a smoke run. Failure to
probe a file stops discovery with an error rather than silently omitting it.

Results are written atomically after every trial:

```text
/data/benchmark-results/<label>-<pipeline>-<UTC-time>-<id>/
    manifest.json
    <config-id>/
        job.json
        startup-<attempt>.json
        <file-id>-repeat-01.json
        <file-id>-repeat-02.json
        ...
```

Each trial JSON is self-contained: file identity/duration, configuration,
environment/package versions, success/error, total/stage times, resources and
normalized `segments` / `word_segments`. Premium native turns retain their
boundaries; normalized turns do not duplicate the flat word list into nested
`words` arrays. This is a common inspection format, not strict equality with the
API's response schema. `--save-raw` additionally stores native output, including
premium frame-level scores, and can generate much larger files.

On a file error or OOM, the runner saves the failure and restarts a fresh worker
for the remaining trials of that configuration. It does not retry the failed
trial. Startup/warm-up failures mark the remaining trials as setup errors. A hard
worker crash is detected from the durable `running` record. A run with failures
exits nonzero; completed JSON files remain usable. Each CLI invocation makes a
new run directory; there is no cross-invocation resume or automatic GPU rental.

## Reconcile results

Copy the four run directories under a single results root, preserving their
names. Install analysis dependencies in any CPU environment:

```bash
uv pip install --python /path/to/analysis/python \
  -r benchmarks/requirements-analysis.txt
BENCHMARK_RESULTS_DIR=/path/to/all-results \
  /path/to/analysis/python -m jupyterlab benchmarks/compare.ipynb
```

The notebook shows configuration/coverage tables, failures, weighted corpus
throughput, latency variability, stage timings and resource timelines. Select
two `(run_id, config_id)` pairs for paired comparisons: L40S vs H100 within one
pipeline, or repository vs native premium on the same GPU. Pairing uses relative
path + checksum and repeat, retaining unmatched/failed cases. Batch numbers from
different pipelines are labels for different controls, not equivalent workloads.

Incomplete configurations have no corpus throughput ranking. A partial paired
comparison shows speedups on common successful trials but withholds the corpus
speedup. Duplicate trial copies are rejected. Analysis exports CSV summaries and
PNG charts under the results root's `analysis` directory.

CPU protocol checks, including a synthetic execution of all notebook cells:

```bash
python -m unittest discover -s benchmarks/tests -v
```

Install the analysis requirements to run these checks. They validate orchestration
and reconciliation; real inference and NVML still require a GPU pod and premium access.
