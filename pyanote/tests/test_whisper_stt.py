"""Adapter checks without model downloads or GPU dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_adapter():
    faster = ModuleType("faster_whisper")
    faster.BatchedInferencePipeline = Mock()
    faster.WhisperModel = Mock()
    stt = ModuleType("pyannoteai.stt")
    stt.STT = object
    spec = importlib.util.spec_from_file_location(
        "whisper_stt", Path(__file__).resolve().parents[1] / "whisper_stt.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"faster_whisper": faster, "pyannoteai.stt": stt}):
        spec.loader.exec_module(module)
    return module.BatchedWhisperSTT


class CutPointTests(unittest.TestCase):
    def setUp(self):
        self.adapter = object.__new__(load_adapter())
        self.adapter.batch_size = 32
        self.adapter._model = SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000, chunk_length=30)
        )
        self.adapter._batched = Mock()

    def test_safe_cuts_and_oversized_interval(self):
        self.assertEqual(self.adapter.target_max_duration, 30)
        self.assertEqual(
            self.adapter._clip_timestamps(90 * 16000, [0, 28.5, 61.2, 90]),
            [
                {"start": 0, "end": 28.5},
                {"start": 28.5, "end": 58.5},
                {"start": 58.5, "end": 61.2},
                {"start": 61.2, "end": 90},
            ],
        )

    def test_missing_cuts_cover_long_audio_and_tail(self):
        for cuts in (None, []):
            with self.subTest(cuts=cuts):
                clips = self.adapter._clip_timestamps(441 * 16000, cuts)
                self.assertEqual(len(clips), 15)
                self.assertEqual(clips[0]["start"], 0)
                self.assertEqual(clips[-1], {"start": 420, "end": 441})
                self.assertTrue(all(0 < c["end"] - c["start"] <= 30 for c in clips))
                self.assertTrue(
                    all(a["end"] == b["start"] for a, b in zip(clips, clips[1:]))
                )

    def test_duplicate_and_missing_endpoints(self):
        self.assertEqual(
            self.adapter._clip_timestamps(20 * 16000, [5, 5, 15]),
            [{"start": 0, "end": 5}, {"start": 5, "end": 15}, {"start": 15, "end": 20}],
        )
        self.assertEqual(self.adapter._clip_timestamps(0, [0, 0]), [])

    def test_invalid_cuts_are_rejected(self):
        for cuts in ([-1], [21], [10, 5], [float("nan")], [float("inf")]):
            with self.subTest(cuts=cuts), self.assertRaises(ValueError):
                self.adapter._clip_timestamps(20 * 16000, cuts)

    def test_apply_passes_clips_and_preserves_word_offsets(self):
        waveform = Mock()
        audio = range(60 * 16000)
        waveform.squeeze.return_value.cpu.return_value.numpy.return_value = audio
        word = SimpleNamespace(start=31.0, end=31.5, word=" hello ")
        self.adapter._batched.transcribe.return_value = (
            iter([SimpleNamespace(words=[word])]),
            None,
        )
        result = self.adapter.apply({"waveform": waveform}, [0, 29, 60])
        self.assertEqual(result, [(31.0, 31.5, "hello")])
        self.adapter._batched.transcribe.assert_called_once_with(
            audio,
            batch_size=32,
            word_timestamps=True,
            vad_filter=False,
            clip_timestamps=[
                {"start": 0, "end": 29},
                {"start": 29, "end": 59},
                {"start": 59, "end": 60},
            ],
        )


if __name__ == "__main__":
    unittest.main()
