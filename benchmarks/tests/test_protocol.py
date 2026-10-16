"""Protocol checks run on CPU without either proprietary or GPU dependencies."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import wave

BENCHMARKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARKS))
from common import Clock, manifest, resource_summary, write_json  # noqa: E402


def audio(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16000)


def fixture_record(pipeline="repo", gpu="L40S", file_id="a", repeat=1):
    return {
        "schema_version": 1,
        "record_type": "trial",
        "run_id": pipeline + gpu,
        "label": gpu,
        "pipeline": pipeline,
        "config_id": "batch8",
        "config": {"batch_size": 8},
        "dataset_id": "corpus",
        "planned_trials": 2,
        "file": {
            "file_id": file_id,
            "relative_path": file_id + ".wav",
            "duration_s": 100 if file_id == "a" else 10,
        },
        "repeat": repeat,
        "status": "success",
        "total_s": 10 if file_id == "a" else 5,
        "environment": {
            "gpu": {"name": gpu, "vram_bytes": 48 * 2**30},
            "packages": {},
            "git_commit": "test",
        },
        "backend": {
            "batch_target": "whisperx_asr"
            if pipeline == "repo"
            else "premium_diarization"
        },
        "timing_protocol": "synchronized_wall_including_decode_v1",
        "stages": [{"name": "asr", "start_s": 0.0, "end_s": 10.0, "duration_s": 10.0}],
        "resources": {
            "samples": [
                {
                    "elapsed_s": 1.0,
                    "in_timed_interval": True,
                    "process_cpu_percent": 150,
                    "process_tree_rss_bytes": 2**30,
                    "system_cpu_percent": 20,
                    "system_ram_used_bytes": 4 * 2**30,
                    "gpu_utilization_percent": 90,
                    "gpu_device_vram_bytes": 8 * 2**30,
                    "gpu_process_vram_bytes": None,
                }
            ],
        },
        "output": {"segments": [], "word_segments": []},
    }


class ProtocolTests(unittest.TestCase):
    def test_premium_hook_preserves_native_preprocessing_and_word_output(self):
        from run_premium import PremiumBackend

        class STT:
            sample_rate = 16000
            target_max_duration = None
            min_turn_duration = 0.0
            min_gap_duration = 0.15
            turn_spacing_duration = 0.15

        class FasterWhisperSTT(STT):
            def __init__(self, model_name="large-v2"):
                self.model_name = model_name

            def apply(self, file, cut_points=None):
                self_test.assertEqual(cut_points, [0, 1])
                return [(0, 1, "hello")]

        class Pipeline:
            version = "4.0.0"

            def __init__(self, key, batch_size, mode):
                self_test.assertEqual((batch_size, mode), (8, "accuracy"))

            def transcribe(self, path, stt):
                self_test.assertEqual(stt.min_gap_duration, 0.15)
                self_test.assertEqual(stt.turn_spacing_duration, 0.15)
                self_test.assertEqual(stt.sample_rate, 16000)
                self_test.assertEqual(
                    stt.apply({}, cut_points=[0, 1]), [(0, 1, "hello")]
                )
                return {
                    "word_level_transcription": [
                        {"start": 0, "end": 1, "text": "hello", "speaker": "SPEAKER_00"}
                    ],
                    "turn_level_transcription": [
                        {"start": 0, "end": 1, "text": "hello", "speaker": "SPEAKER_00"}
                    ],
                }

            def drop(self):
                pass

        self_test = self
        local, stt_module = ModuleType("pyannoteai.local"), ModuleType("pyannoteai.stt")
        local.Pipeline = Pipeline
        stt_module.STT = STT
        stt_module.FasterWhisperSTT = FasterWhisperSTT
        with patch.dict(
            sys.modules, {"pyannoteai.local": local, "pyannoteai.stt": stt_module}
        ):
            with patch.dict(os.environ, {"PYA_JOBS_KEY": "test"}):
                backend = PremiumBackend(
                    {"batch_size": 8, "mode": "accuracy", "stt_kwargs": {}}
                )
                clock = Clock(lambda: None)
                clock.start()
                result = backend.transcribe(Path("test.wav"), clock)
                clock.stop()
                output = backend.normalize(result)
                self.assertEqual(output["word_segments"][0]["word"], "hello")
                self.assertEqual(
                    {event["name"] for event in clock.events},
                    {"native_transcribe", "stt_apply"},
                )
                backend.close()

    def test_resource_peaks_ignore_boundary_samples_and_preserve_null(self):
        sample = fixture_record()["resources"]["samples"][0]
        boundary = dict(
            sample, gpu_device_vram_bytes=999 * 2**30, in_timed_interval=False
        )
        result = resource_summary([boundary, sample])
        self.assertEqual(result["gpu_device_vram_bytes_peak"], 8 * 2**30)
        self.assertEqual(result["process_cpu_percent_mean"], 150)
        self.assertIsNone(result["gpu_process_vram_bytes_peak"])
        self.assertEqual(resource_summary([])["sample_count"], 0)

    def test_failed_cuda_synchronization_still_closes_total_timer(self):
        def synchronize():
            if clock.origin is not None:
                raise RuntimeError("GPU error")

        clock = Clock(synchronize)
        clock.start()
        with self.assertRaises(RuntimeError):
            clock.stop()
        self.assertGreaterEqual(clock.duration, 0)

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe required")
    def test_manifest_excludes_results_and_hashes_input_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio(root / "input.wav")
            audio(root / "models" / "cached.wav")
            audio(root / "results" / "output.wav")
            files = manifest(root, root / "results")
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["relative_path"], "input.wav")
            self.assertAlmostEqual(files[0]["duration_s"], 1)
            self.assertEqual(files, manifest(root, root / "results"))

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe required")
    def test_supervisor_restarts_after_failure_and_saves_every_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio(root / "audio" / "good.wav")
            audio(root / "audio" / "bad.wav")
            script = root / "fake_runner.py"
            script.write_text("""
import sys
from types import SimpleNamespace
import common
sys.modules["torch"] = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
common.environment = lambda: {"gpu": {"uuid": "fake", "name": "fake"}}
class Monitor:
    def __init__(self, *args): pass
    def start(self): pass
    def finish(self, clock): return {"samples": [], "summary": {"sample_count": 0}}
common.ResourceMonitor = Monitor
class Backend:
    def __init__(self, config): self.details = {"batch_target": "whisperx_asr"}
    def transcribe(self, path, clock):
        if path.name == "bad.wav": raise RuntimeError("simulated OOM")
        with clock.stage("asr"): return {"segments": [], "word_segments": []}
    def normalize(self, raw): return raw
    def close(self): pass
if __name__ == "__main__": common.main("repo", Backend)
""")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--data-dir",
                    str(root / "audio"),
                    "--output-dir",
                    str(root / "results"),
                    "--batch-sizes",
                    "8",
                    "--repeats",
                    "2",
                    "--warmup-files",
                    "0",
                ],
                env={**os.environ, "PYTHONPATH": str(BENCHMARKS)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            results = [
                json.loads(p.read_text()) for p in (root / "results").rglob("*.json")
            ]
            trials = [r for r in results if r.get("record_type") == "trial"]
            self.assertEqual(len(trials), 4)
            self.assertEqual(sum(r["status"] == "success" for r in trials), 2)
            self.assertEqual(sum(r["status"] == "error" for r in trials), 2)
            self.assertGreaterEqual(
                sum(r.get("record_type") == "startup" for r in results), 2
            )


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        from analysis import load_results, pair_trials, summarize

        self.load = load_results
        self.pair = pair_trials
        self.summarize = summarize
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, record, filename):
        record = copy.deepcopy(record)
        record["resources"]["summary"] = resource_summary(
            record["resources"]["samples"]
        )
        write_json(self.root / filename, record)

    def test_weighted_throughput_and_missing_trial_suppression(self):
        self.write(fixture_record(file_id="a"), "a.json")
        _, partial = self.load(self.root)
        summary = self.summarize(partial).iloc[0]
        self.assertFalse(summary.complete)
        self.assertEqual(summary.missing_trials, 1)
        self.assertNotEqual(summary.throughput, summary.throughput)  # NaN
        self.write(fixture_record(file_id="b"), "b.json")
        _, full = self.load(self.root)
        summary = self.summarize(full).iloc[0]
        self.assertTrue(summary.complete)
        self.assertAlmostEqual(summary.throughput, 110 / 15)

    def test_pairing_preserves_unmatched_and_failed_files(self):
        self.write(fixture_record(file_id="a"), "a.json")
        self.write(fixture_record(file_id="b"), "b.json")
        candidate = fixture_record(pipeline="premium", file_id="a")
        candidate["total_s"] /= 2
        self.write(candidate, "candidate-a.json")
        _, frame = self.load(self.root)
        pairs = self.pair(frame, ("repoL40S", "batch8"), ("premiumL40S", "batch8"))
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs.comparable.sum(), 1)
        self.assertEqual(pairs.loc[pairs.comparable, "speedup"].iloc[0], 2)
        candidate["status"] = "error"
        self.write(candidate, "candidate-a.json")
        _, frame = self.load(self.root)
        pairs = self.pair(frame, ("repoL40S", "batch8"), ("premiumL40S", "batch8"))
        self.assertEqual(pairs.comparable.sum(), 0)

    def test_duplicate_copies_are_rejected(self):
        self.write(fixture_record(), "a.json")
        self.write(fixture_record(), "duplicate.json")
        with self.assertRaisesRegex(ValueError, "Duplicate trial"):
            self.load(self.root)

    @unittest.skipUnless(
        importlib.util.find_spec("IPython"),
        "Install analysis requirements for notebook execution",
    )
    def test_notebook_executes_against_both_pipeline_schemas(self):
        for pipeline in ("repo", "premium"):
            for gpu in ("L40S", "H100"):
                for file_id in ("a", "b"):
                    record = fixture_record(pipeline=pipeline, gpu=gpu, file_id=file_id)
                    if pipeline == "premium":
                        record["stages"] = [
                            {
                                "name": "native_transcribe",
                                "start_s": 0.0,
                                "end_s": 10.0,
                                "duration_s": 10.0,
                            },
                            {
                                "name": "stt_apply",
                                "start_s": 2.0,
                                "end_s": 8.0,
                                "duration_s": 6.0,
                            },
                        ]
                    self.write(record, f"{pipeline}-{gpu}-{file_id}.json")
        notebook = json.loads((BENCHMARKS / "compare.ipynb").read_text())
        scope = {}
        old_cwd = Path.cwd()
        try:
            os.chdir(BENCHMARKS)
            with patch.dict(
                os.environ,
                {
                    "BENCHMARK_RESULTS_DIR": str(self.root),
                    "MPLBACKEND": "Agg",
                },
            ):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        source = "".join(cell["source"])
                        # Exercise the optional paired branch as well.
                        source = source.replace(
                            "BASELINE = None",
                            'BASELINE = ("repoL40S", "batch8")',
                        ).replace(
                            "CANDIDATE = None",
                            'CANDIDATE = ("premiumL40S", "batch8")',
                        )
                        exec(compile(source, f"notebook-cell-{index}", "exec"), scope)
        finally:
            os.chdir(old_cwd)
        self.assertTrue((self.root / "analysis" / "summary.csv").exists())
        self.assertTrue((self.root / "analysis" / "resource-timeline.png").exists())
        self.assertTrue((self.root / "analysis" / "paired-trials.csv").exists())


if __name__ == "__main__":
    unittest.main()
