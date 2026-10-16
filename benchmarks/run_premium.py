"""Benchmark pyannote.ai's native Pipeline.transcribe with FasterWhisperSTT."""

import inspect
import json
import os
import sys

from common import main


class PremiumBackend:
    def __init__(self, config):
        from pyannoteai.local import Pipeline
        from pyannoteai.stt import FasterWhisperSTT, STT

        # The supplied docs only guarantee the no-argument STT constructor.
        # Forward explicitly requested options, failing rather than ignoring them.
        kwargs = config["stt_kwargs"]
        inspect.signature(FasterWhisperSTT).bind(**kwargs)
        self.pipeline = Pipeline(
            os.environ["PYA_JOBS_KEY"],
            batch_size=config["batch_size"],
            mode=config["mode"],
        )
        delegate = FasterWhisperSTT(**kwargs)

        class TimedSTT(STT):
            # Forward every documented preprocessing property unchanged.
            @property
            def sample_rate(self):
                return delegate.sample_rate

            @property
            def target_max_duration(self):
                return delegate.target_max_duration

            @property
            def min_turn_duration(self):
                return delegate.min_turn_duration

            @property
            def min_gap_duration(self):
                return delegate.min_gap_duration

            @property
            def turn_spacing_duration(self):
                return delegate.turn_spacing_duration

            def apply(self, file, cut_points=None):
                with self.clock.stage("stt_apply"):
                    return delegate.apply(file, cut_points=cut_points)

        self.stt = TimedSTT()
        self.details = {
            "batch_target": "premium_diarization",
            "mode": config["mode"],
            "stt_class": "FasterWhisperSTT",
            "stt_kwargs": kwargs,
            "stt_constructor": str(inspect.signature(FasterWhisperSTT)),
            "pipeline_version": str(Pipeline.version),
            "note": "Unspecified STT options use installed SDK defaults.",
        }

    def transcribe(self, path, clock):
        self.stt.clock = clock
        # Native path loading is deliberately inside the end-to-end timer.
        with clock.stage("native_transcribe"):
            return self.pipeline.transcribe(str(path), self.stt)

    def normalize(self, result):
        def word(item):
            return {
                "word": item["text"],
                "start": item.get("start"),
                "end": item.get("end"),
                "speaker": item.get("speaker"),
            }

        words = [word(w) for w in result["word_level_transcription"]]
        # Native turns need not match WhisperX segment boundaries. Preserve them.
        segments = [
            {
                "start": t["start"],
                "end": t["end"],
                "text": t["text"],
                "speaker": t.get("speaker"),
            }
            for t in result["turn_level_transcription"]
        ]
        return {"segments": segments, "word_segments": words}

    def close(self):
        self.pipeline.drop()


if __name__ == "__main__":
    if "--describe-backend" in sys.argv:
        from pyannoteai.stt import FasterWhisperSTT

        print(
            json.dumps(
                {
                    "FasterWhisperSTT": str(inspect.signature(FasterWhisperSTT)),
                    "apply": str(inspect.signature(FasterWhisperSTT.apply)),
                },
                indent=2,
            )
        )
    else:
        main("premium", PremiumBackend)
