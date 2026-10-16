"""Shared experiment protocol. Importing this module does not import inference SDKs."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".mp4",
    ".webm",
    ".mkv",
}
SCHEMA_VERSION = 1


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            default=json_default,
            allow_nan=False,
            ensure_ascii=False,
        )
        + "\n"
    )
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def error_text(exc):
    message = f"{type(exc).__name__}: {exc}"
    for name in ("HF_TOKEN", "API_KEY", "PYA_JOBS_KEY", "PYA_DOWNLOAD_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def manifest(data_dir, output_dir, limit=None):
    """Hash/probe once outside all measured intervals; exclude caches and results."""
    files = []
    for path in sorted(data_dir.rglob("*")):
        relative = path.relative_to(data_dir)
        if (
            not path.is_file()
            or path.suffix.lower() not in EXTENSIONS
            or path.is_relative_to(output_dir)
            or any(
                p.startswith(".") or p in {"models", "benchmark-results"}
                for p in relative.parts[:-1]
            )
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type,sample_rate,channels,duration",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
        )
        streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
        if not streams:
            raise ValueError(f"No audio stream in {path}")
        duration = probe.get("format", {}).get("duration")
        if duration in (None, "N/A"):
            duration = max(float(s.get("duration", 0)) for s in streams)
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Cannot determine positive duration for {path}")
        entry = {
            "relative_path": relative.as_posix(),
            "sha256": digest.hexdigest(),
            "duration_s": duration,
            "size_bytes": path.stat().st_size,
            "sample_rate": streams[0].get("sample_rate"),
            "channels": streams[0].get("channels"),
        }
        entry["file_id"] = fingerprint([entry["relative_path"], entry["sha256"]])
        files.append(entry)
        if limit and len(files) >= limit:
            break
    if not files:
        raise ValueError(f"No supported media files found under {data_dir}")
    return files


class Clock:
    """Wall-clock timings, with CUDA completion enforced at each stage boundary."""

    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.events = []
        self.origin = None
        self.duration = None

    def start(self):
        self.synchronize()
        self.origin = time.perf_counter()

    def stop(self):
        try:
            self.synchronize()
        finally:
            self.duration = time.perf_counter() - self.origin

    @contextmanager
    def stage(self, name):
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            end = time.perf_counter()
            self.events.append(
                {
                    "name": name,
                    "start_s": start - self.origin,
                    "end_s": end - self.origin,
                    "duration_s": end - start,
                }
            )

    def wrap(self, name, function):
        def timed(*args, **kwargs):
            with self.stage(name):
                return function(*args, **kwargs)

        return timed


def _sample_resources(pid, gpu_uuid, interval, destination, ready, stop):
    """Separate process: native code holding the GIL cannot stall this sampler."""
    samples = []
    warnings = set()
    try:
        import psutil
        import pynvml as nvml

        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        process = psutil.Process(pid)
        previous_cpu = {}
        previous_time = time.perf_counter()
        psutil.cpu_percent()

        def sample():
            nonlocal previous_cpu, previous_time
            now = time.perf_counter()
            tree = [process] + process.children(recursive=True)
            # Exclude the sampler's own process and any of its children.
            excluded = {os.getpid()}
            tree = [p for p in tree if p.pid not in excluded]
            cpu = {}
            rss = 0
            for p in tree:
                try:
                    times = p.cpu_times()
                    cpu[p.pid] = times.user + times.system
                    rss += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            cpu_delta = sum(
                max(0, seconds - previous_cpu[p])
                for p, seconds in cpu.items()
                if p in previous_cpu
            )
            cpu_percent = (
                cpu_delta / max(now - previous_time, 1e-9) * 100
                if previous_cpu
                else None
            )
            previous_cpu, previous_time = cpu, now
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except nvml.NVMLError as exc:
                utilization = None
                warnings.add(f"GPU utilization unavailable: {type(exc).__name__}")
            process_vram = None
            try:
                running = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                values = [p.usedGpuMemory for p in running if p.pid in cpu]
                if values and all(0 <= v <= memory.total for v in values):
                    process_vram = sum(values)
            except nvml.NVMLError as exc:
                warnings.add(f"Process VRAM unavailable: {type(exc).__name__}")
            if process_vram is None:
                warnings.add("Process VRAM unavailable; device VRAM remains recorded.")
            virtual = psutil.virtual_memory()
            samples.append(
                {
                    "monotonic_s": now,
                    "process_cpu_percent": cpu_percent,
                    "process_tree_rss_bytes": rss,
                    "system_cpu_percent": psutil.cpu_percent(),
                    "system_ram_used_bytes": virtual.used,
                    "system_ram_percent": virtual.percent,
                    "gpu_utilization_percent": utilization,
                    "gpu_device_vram_bytes": memory.used,
                    "gpu_process_vram_bytes": process_vram,
                }
            )

        sample()  # prime CPU counters before signalling readiness
        ready.set()
        while not stop.wait(interval):
            sample()
        sample()
        nvml.nvmlShutdown()
        write_json(destination, {"samples": samples, "warnings": sorted(warnings)})
    except Exception as exc:
        write_json(destination, {"samples": samples, "error": error_text(exc)})
        ready.set()


class ResourceMonitor:
    def __init__(self, gpu_uuid, interval):
        self.temporary = tempfile.TemporaryDirectory(prefix="whisperx-benchmark-")
        self.destination = Path(self.temporary.name) / "samples.json"
        context = mp.get_context("spawn")
        self.ready, self.stop = context.Event(), context.Event()
        self.process = context.Process(
            target=_sample_resources,
            daemon=True,
            args=(
                os.getpid(),
                gpu_uuid,
                interval,
                self.destination,
                self.ready,
                self.stop,
            ),
        )

    def start(self):
        self.process.start()
        if not self.ready.wait(20):
            self.process.terminate()
            self.process.join()
            self.temporary.cleanup()
            raise RuntimeError("Resource sampler did not start within 20 seconds")
        if self.destination.exists():
            result = json.loads(self.destination.read_text())
            if "error" in result:
                self.process.join()
                self.temporary.cleanup()
                raise RuntimeError(f"Resource sampler: {result['error']}")

    def finish(self, clock):
        self.stop.set()
        self.process.join(20)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        try:
            if not self.destination.exists():
                return {"samples": [], "error": "Resource sampler exited without data"}
            result = json.loads(self.destination.read_text())
            for sample in result["samples"]:
                sample["elapsed_s"] = sample.pop("monotonic_s") - clock.origin
                sample["in_timed_interval"] = 0 <= sample["elapsed_s"] <= clock.duration
            result["summary"] = resource_summary(result["samples"])
            return result
        finally:
            self.temporary.cleanup()


def resource_summary(samples):
    valid = [s for s in samples if s["in_timed_interval"]]
    result = {"sample_count": len(valid)}
    # Arithmetic sample means approximate interval means at a fixed cadence.
    for key in (
        "process_cpu_percent",
        "process_tree_rss_bytes",
        "system_cpu_percent",
        "system_ram_used_bytes",
        "gpu_utilization_percent",
        "gpu_device_vram_bytes",
        "gpu_process_vram_bytes",
    ):
        values = [s[key] for s in valid if s.get(key) is not None]
        result[key + "_peak"] = max(values) if values else None
        result[key + "_mean"] = sum(values) / len(values) if values else None
    return result


def _command(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    import psutil
    import pynvml as nvml
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU using CUDA_VISIBLE_DEVICES.")
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = str(properties.uuid)
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
    gpu = {
        "name": nvml.nvmlDeviceGetName(handle),
        "uuid": gpu_uuid,
        "vram_bytes": nvml.nvmlDeviceGetMemoryInfo(handle).total,
        "driver": nvml.nvmlSystemGetDriverVersion(),
    }
    nvml.nvmlShutdown()
    limits = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        limits[name] = path.read_text().strip() if path.exists() else None
    return {
        "gpu": gpu,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": _command(["lscpu"]),
        "cpu_logical_count": psutil.cpu_count(),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "ram_bytes": psutil.virtual_memory().total,
        "cgroup_limits": limits,
        "torch_cuda": torch.version.cuda,
        "ffmpeg": _command(["ffmpeg", "-version"]),
        "git_commit": _command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_status": _command(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "packages": {
            d.metadata["Name"]: d.version
            for d in metadata.distributions()
            if d.metadata["Name"]
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_settings": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def result_base(job, task):
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "trial",
        "run_id": job["run_id"],
        "label": job["label"],
        "pipeline": job["pipeline"],
        "config": job["config"],
        "config_id": job["config_id"],
        "dataset_id": job["dataset_id"],
        "file": task["file"],
        "repeat": task["repeat"],
        "planned_trials": job["planned_trials"],
        "sample_interval_s": job["sample_interval_s"],
        "timing_protocol": "synchronized_wall_including_decode_v1",
        "status": "running",
    }


def trial_path(job, task):
    return Path(job["directory"]) / (
        f"{task['file']['file_id'][:20]}-repeat-{task['repeat']:02d}.json"
    )


def run_worker(job, factory):
    """A worker owns one model configuration. Any failed trial triggers a restart."""
    import torch

    attempt = uuid.uuid4().hex[:8]
    startup_path = Path(job["directory"]) / f"startup-{attempt}.json"
    backend = None
    info = {}
    try:
        info = environment()
        monitor = ResourceMonitor(info["gpu"]["uuid"], job["sample_interval_s"])
        monitor.start()
        clock = Clock(torch.cuda.synchronize)
        clock.start()
        try:
            backend = factory(job["config"])
        finally:
            try:
                clock.stop()
            finally:
                startup_resources = monitor.finish(clock)
        write_json(
            startup_path,
            {
                "record_type": "startup",
                "environment": info,
                "initialization_s": clock.duration,
                "resources": startup_resources,
                "backend": backend.details,
            },
        )
        # Warm every new worker; include native STT lazy initialization here.
        for entry in job["files"][: job["warmup_files"]]:
            warm_clock = Clock(torch.cuda.synchronize)
            warm_clock.start()
            backend.transcribe(
                Path(job["data_dir"]) / entry["relative_path"], warm_clock
            )
            warm_clock.stop()
    except Exception as exc:
        failure = error_text(exc)
        write_json(
            startup_path.with_name(f"startup-error-{attempt}.json"),
            {
                "record_type": "startup_error",
                "error": failure,
            },
        )
        for task in job["tasks"]:
            result = result_base(job, task)
            result.update(
                status="setup_error",
                error=failure,
                environment=info,
                backend=getattr(backend, "details", {}),
                startup_record=startup_path.name,
            )
            write_json(trial_path(job, task), result)
        return 2

    for task in job["tasks"]:
        destination = trial_path(job, task)
        result = result_base(job, task)
        result.update(
            environment=info,
            backend=backend.details,
            startup_record=startup_path.name,
            worker_attempt=attempt,
            warmup_files=job["warmup_files"],
        )
        # A durable marker lets the parent identify fatal worker exits.
        write_json(destination, result)
        clock = Clock(torch.cuda.synchronize)
        monitor = ResourceMonitor(info["gpu"]["uuid"], job["sample_interval_s"])
        error = None
        try:
            monitor.start()
            clock.start()
            try:
                raw = backend.transcribe(
                    Path(job["data_dir"]) / task["file"]["relative_path"],
                    clock,
                )
            finally:
                try:
                    clock.stop()
                finally:
                    result["resources"] = monitor.finish(clock)
            result["output"] = backend.normalize(raw)
            if job["save_raw"]:
                result["raw_output"] = raw
            del raw
            result["status"] = "success"
        except Exception as exc:
            error = error_text(exc)
            result.update(status="error", error=error)
        result.update(total_s=clock.duration, stages=clock.events)
        if clock.duration and result["status"] == "success":
            result["real_time_factor"] = clock.duration / task["file"]["duration_s"]
            result["audio_hours_per_wall_hour"] = 1 / result["real_time_factor"]
        try:
            write_json(destination, result)
        except (TypeError, ValueError) as exc:
            # Keep measured data even if a vendor output cannot be JSON encoded.
            result.pop("output", None)
            result.pop("raw_output", None)
            error = error_text(exc)
            result.update(status="output_error", error=error)
            write_json(destination, result)
        print(
            f"{result['status']}: {task['file']['relative_path']} "
            f"repeat={task['repeat']} batch={job['config']['batch_size']} "
            f"total={clock.duration}",
            flush=True,
        )
        if error:
            return 2  # no further trials in a possibly poisoned CUDA context
    backend.close()
    return 0


def main(pipeline, factory):
    parser = argparse.ArgumentParser(description=f"{pipeline} file benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/data/benchmark-results")
    )
    parser.add_argument("--label", required=False, default="runpod")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-files", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-raw", action="store_true")
    if pipeline == "repo":
        parser.add_argument("--language", default=None)
    else:
        parser.add_argument(
            "--modes",
            nargs="+",
            choices=["accuracy", "balance", "speed"],
            default=["accuracy"],
        )
        parser.add_argument("--stt-kwargs", type=json.loads, default={})
        parser.add_argument(
            "--describe-backend",
            action="store_true",
            help="Show SDK signatures and resolved STT configuration without running inference",
        )
        parser.add_argument(
            "--stt-model",
            default="Systran/faster-whisper-large-v2",
            help="FasterWhisperSTT model ID/path (default: %(default)s)",
        )
        parser.add_argument(
            "--stt-model-parameter",
            help="SDK constructor keyword for the model ID, if auto-detection fails",
        )
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        raise SystemExit(run_worker(json.loads(args._worker.read_text()), factory))
    if (
        args.repeats < 1
        or args.warmup_files < 0
        or not math.isfinite(args.sample_interval)
        or args.sample_interval <= 0
        or (args.limit is not None and args.limit < 1)
        or any(b < (8 if pipeline == "premium" else 1) for b in args.batch_sizes)
    ):
        parser.error(
            "Invalid repeats, warmup, interval, limit or batch size (premium >=8)."
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error(
            "Use a label containing only letters, digits, dots, dashes or underscores."
        )
    if pipeline == "premium" and not isinstance(args.stt_kwargs, dict):
        parser.error("--stt-kwargs must be a JSON object")
    stt_config = {}
    if pipeline == "premium":
        stt_config = {
            "stt_kwargs": args.stt_kwargs,
            "stt_model": args.stt_model,
            "stt_model_parameter": args.stt_model_parameter,
        }
        if args.describe_backend:
            print(json.dumps(factory.describe(stt_config), indent=2))
            return
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    files = manifest(data_dir, output_dir, args.limit)
    if args.warmup_files > len(files):
        parser.error("--warmup-files exceeds the number of input files")
    dataset_id = fingerprint([(f["relative_path"], f["sha256"]) for f in files])
    run_id = (
        f"{args.label}-{pipeline}-"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = output_dir / run_id
    write_json(
        run_dir / "manifest.json",
        {
            "record_type": "manifest",
            "dataset_id": dataset_id,
            "files": files,
        },
    )
    batches = list(dict.fromkeys(args.batch_sizes))
    random.Random(args.seed).shuffle(batches)
    tasks = []
    for repeat in range(1, args.repeats + 1):
        order = files.copy()
        random.Random(args.seed + repeat).shuffle(order)
        tasks.extend({"file": f, "repeat": repeat} for f in order)
    modes = list(dict.fromkeys(args.modes)) if pipeline == "premium" else [None]
    failures = 0
    for mode in modes:
        for batch in batches:
            config = {"batch_size": batch, "seed": args.seed}
            if pipeline == "repo":
                config["language"] = args.language
            else:
                config.update(mode=mode, **stt_config)
            config_id = fingerprint(config)[:16]
            job = {
                "run_id": run_id,
                "label": args.label,
                "pipeline": pipeline,
                "config": config,
                "config_id": config_id,
                "dataset_id": dataset_id,
                "directory": str(run_dir / config_id),
                "data_dir": str(data_dir),
                "sample_interval_s": args.sample_interval,
                "warmup_files": args.warmup_files,
                "files": files,
                "tasks": tasks,
                "save_raw": args.save_raw,
                "planned_trials": len(tasks),
            }
            pending = tasks.copy()
            while pending:
                job["tasks"] = pending
                job_path = Path(job["directory"]) / "job.json"
                write_json(job_path, job)
                print(
                    f"Starting {pipeline}: {config}, pending={len(pending)}", flush=True
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(sys.argv[0]).resolve()),
                        "--_worker",
                        str(job_path),
                    ]
                )
                remaining = []
                progress = False
                for task in pending:
                    path = trial_path(job, task)
                    if path.exists():
                        result = json.loads(path.read_text())
                        if result["status"] == "running":
                            result.update(
                                status="worker_crash",
                                error=f"Worker exited with code {completed.returncode}",
                            )
                            write_json(path, result)
                        progress = True
                    else:
                        remaining.append(task)
                if not progress:
                    # A fatal startup/import failure cannot be fixed by retrying forever.
                    for task in remaining:
                        result = result_base(job, task)
                        result.update(
                            status="setup_error",
                            error=(
                                f"Worker exited before writing results (code {completed.returncode}); "
                                "see terminal output."
                            ),
                        )
                        write_json(trial_path(job, task), result)
                    break
                pending = remaining
            for task in tasks:
                if json.loads(trial_path(job, task).read_text())["status"] != "success":
                    failures += 1
    print(f"Results: {run_dir}; failed trials: {failures}", flush=True)
    if failures:
        raise SystemExit(1)
