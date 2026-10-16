"""Benchmark pyannote.ai's native Pipeline.transcribe with FasterWhisperSTT."""

import inspect
import json
import os
import sys

from common import main


DEFAULT_STT_MODEL = "Systran/faster-whisper-large-v2"


def stt_options(constructor, config):
    """Select the requested model using the installed SDK's constructor signature."""
    signature = inspect.signature(constructor)
    kwargs = dict(config["stt_kwargs"])
    parameter = config.get("stt_model_parameter")
    if parameter is None:
        # Prefer the SDK's explicit model-ID/path argument. Never assume that an
        # opaque **kwargs accepts a particular spelling, or that `model=None`
        # accepts a string rather than an already constructed WhisperModel.
        for name in ("model_size_or_path", "model_name", "model_id", "model_path"):
            if name in signature.parameters:
                parameter = name
                break
        if parameter is None:
            candidates = [
                p.name
                for p in signature.parameters.values()
                if isinstance(p.default, str) and "faster-whisper" in p.default
            ]
            if len(candidates) == 1:
                parameter = candidates[0]
    if parameter is None:
        raise TypeError(
            f"Cannot identify the STT model argument in FasterWhisperSTT{signature}. "
            "Inspect --describe-backend and specify --stt-model-parameter with "
            "the SDK's model-ID argument. The vendor default will not be used."
        )
    kwargs.setdefault(parameter, config.get("stt_model", DEFAULT_STT_MODEL))
    signature.bind(**kwargs)
    return kwargs, parameter


class PremiumBackend:
    def __init__(self, config):
        from pyannoteai.local import Pipeline
        from pyannoteai.stt import FasterWhisperSTT, STT

        kwargs, model_parameter = stt_options(FasterWhisperSTT, config)
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
            "stt_model": kwargs[model_parameter],
            "stt_model_parameter": model_parameter,
            "stt_kwargs": kwargs,
            "stt_constructor": str(inspect.signature(FasterWhisperSTT)),
            "pipeline_version": str(Pipeline.version),
            "note": "The ASR model is explicit; other unspecified options use SDK defaults.",
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
