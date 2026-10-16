"""Read-only reconciliation helpers shared by the notebook and protocol tests."""

import hashlib
import json
from pathlib import Path

import pandas as pd


GROUPS = [
    "run_id",
    "dataset_id",
    "pipeline",
    "gpu",
    "variant",
    "config_id",
    "batch_size",
    "batch_target",
    "mode",
]


def load_results(root):
    records, rows, seen = [], [], set()
    for path in sorted(Path(root).rglob("*.json")):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or record.get("record_type") != "trial":
            continue
        if record.get("schema_version") != 1:
            raise ValueError(f"Unsupported schema in {path}")
        key = (
            record["run_id"],
            record["config_id"],
            record["file"]["file_id"],
            record["repeat"],
        )
        if key in seen:
            raise ValueError(f"Duplicate trial {key}; remove duplicated result copies.")
        seen.add(key)
        records.append(record)
        environment = record.get("environment", {})
        gpu = environment.get("gpu", {})
        gpu_name = gpu.get("name", "unknown")
        if gpu.get("vram_bytes"):
            gpu_name += f" ({gpu['vram_bytes'] / 2**30:.0f} GiB)"
        backend = json.loads(json.dumps(record.get("backend", {})))
        backend.get("settings", {}).pop("batch_size", None)
        # Keep different models/settings/software out of the same plot series.
        software = {
            k: v
            for k, v in environment.get("packages", {}).items()
            if k.lower().replace("_", "-")
            in {
                "torch",
                "torchaudio",
                "whisperx",
                "faster-whisper",
                "ctranslate2",
                "pyannote-audio",
                "pyannoteai-local",
                "pyannoteai-stt",
                "transformers",
            }
        }
        identity = {"backend": backend, "software": software}
        if record["pipeline"] == "repo":
            identity["git_commit"] = environment.get("git_commit")
        variant = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:8]
        config = record["config"]
        resources = record.get("resources", {})
        rows.append(
            {
                "record_index": len(records) - 1,
                "path": str(path),
                "run_id": record["run_id"],
                "label": record["label"],
                "dataset_id": record["dataset_id"],
                "pipeline": record["pipeline"],
                "gpu": gpu_name,
                "variant": variant,
                "config_id": record["config_id"],
                "batch_size": config["batch_size"],
                "planned_trials": record["planned_trials"],
                "batch_target": record.get("backend", {}).get(
                    "batch_target",
                    "whisperx_asr"
                    if record["pipeline"] == "repo"
                    else "premium_diarization",
                ),
                "mode": config.get("mode", "default"),
                "file_id": record["file"]["file_id"],
                "file": record["file"]["relative_path"],
                "repeat": record["repeat"],
                "audio_s": record["file"]["duration_s"],
                "status": record["status"],
                "total_s": record.get("total_s"),
                "error": record.get("error"),
                "resource_error": resources.get("error"),
                "timing_protocol": record["timing_protocol"],
                **resources.get("summary", {}),
            }
        )
    if not rows:
        raise ValueError(f"No trial JSON files found under {root}")
    frame = pd.DataFrame(rows)
    frame["total_s"] = pd.to_numeric(frame["total_s"], errors="coerce")
    frame["rtf"] = frame["total_s"] / frame["audio_s"]
    frame["throughput"] = frame["audio_s"] / frame["total_s"]
    return records, frame


def summarize(frame):
    summaries = []
    for keys, group in frame.groupby(GROUPS, dropna=False):
        ok = group[group.status.eq("success")]
        planned = int(group.planned_trials.max())
        complete = len(ok) == len(group) == planned
        row = dict(zip(GROUPS, keys))
        row.update(
            trials=len(group),
            successes=len(ok),
            failures=len(group) - len(ok),
            planned_trials=planned,
            missing_trials=max(0, planned - len(group)),
            files=group.file_id.nunique(),
            repeats=group["repeat"].nunique(),
            complete=complete,
            audio_s=ok.audio_s.sum(),
            total_s=ok.total_s.sum(),
            # Do not reward a configuration which only completed easy files.
            rtf=ok.total_s.sum() / ok.audio_s.sum() if complete else float("nan"),
            throughput=ok.audio_s.sum() / ok.total_s.sum()
            if complete
            else float("nan"),
        )
        for name in (
            "process_tree_rss_bytes_peak",
            "gpu_device_vram_bytes_peak",
            "gpu_process_vram_bytes_peak",
        ):
            row[name] = ok[name].max() if name in ok else None
        for name in ("gpu_utilization_percent_mean", "process_cpu_percent_mean"):
            available = ok.dropna(subset=[name]) if name in ok else ok.iloc[:0]
            row[name] = (
                (available[name] * available.total_s).sum() / available.total_s.sum()
                if len(available)
                else None
            )
        summaries.append(row)
    return pd.DataFrame(summaries)


def pair_trials(frame, baseline, candidate):
    """Selectors are (run_id, config_id). Preserve failures and unmatched files."""

    def select(selector):
        selected = frame[
            frame.run_id.eq(selector[0]) & frame.config_id.eq(selector[1])
        ].copy()
        if selected.empty:
            raise ValueError(f"No trials for {selector}")
        if selected.duplicated(["file_id", "repeat"]).any():
            raise ValueError("Select exactly one run and configuration per side")
        return selected

    left, right = select(baseline), select(candidate)
    if set(left.timing_protocol) != set(right.timing_protocol):
        raise ValueError("The selected trials use different timing protocols")
    paired = left.merge(
        right,
        on=["file_id", "repeat"],
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
    )
    paired["comparable"] = (
        paired["_merge"].eq("both")
        & paired.status_baseline.eq("success")
        & paired.status_candidate.eq("success")
    )
    matched = paired[paired["_merge"].eq("both")]
    if ((matched.audio_s_baseline - matched.audio_s_candidate).abs() > 0.05).any():
        raise ValueError(
            "Audio duration differs for identical checksums; check decoding/probing."
        )
    paired["speedup"] = (paired.total_s_baseline / paired.total_s_candidate).where(
        paired.comparable
    )
    return paired


def stage_rows(records):
    rows = []
    for index, record in enumerate(records):
        for stage in record.get("stages", []):
            rows.append({"record_index": index, **stage})
    return pd.DataFrame(
        rows,
        columns=[
            "record_index",
            "name",
            "start_s",
            "end_s",
            "duration_s",
        ],
    )
