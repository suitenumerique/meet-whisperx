"""Benchmark the actual application pipeline, including its production preload policy."""

import asyncio
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from unittest.mock import patch

from common import main


class RepoBackend:
    def __init__(self, config):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
        # No HTTP server is started; Settings still requires an API key.
        os.environ.setdefault("API_KEY", "benchmark-local-only")
        from utils.config import Settings
        from utils.lifespan import lifespan
        import services.transcription as service
        import whisperx

        self.service = service
        self.whisperx = whisperx
        self.settings = Settings(batch_size=config["batch_size"])
        self.language = config["language"]
        self.loop = asyncio.new_event_loop()
        self.context = lifespan(None)
        self.loop.run_until_complete(self.context.__aenter__())
        self.details = {
            "batch_target": "whisperx_asr",
            "settings": self.settings.model_dump(
                include={
                    "transcribe_model",
                    "diarize_model",
                    "batch_size",
                    "preloaded_align_model_languages",
                    "interpolate_method",
                    "return_char_alignments",
                    "fill_nearest",
                }
            ),
            "compute_type": "float16",
            "language": self.language,
        }

    def transcribe(self, path, clock):
        with clock.stage("decode"):
            audio = self.whisperx.load_audio(str(path))
        with ExitStack() as stack:
            # Wrap the production functions; do not reimplement the pipeline.
            for name, stage in (
                ("_transcribe_audio", "asr"),
                ("_align_transcription", "alignment"),
            ):
                stack.enter_context(
                    patch.object(
                        self.service,
                        name,
                        clock.wrap(stage, getattr(self.service, name)),
                    )
                )
            stack.enter_context(
                patch.object(
                    self.service.pipelines,
                    "diarize_model",
                    clock.wrap("diarization", self.service.pipelines.diarize_model),
                )
            )
            stack.enter_context(
                patch.object(
                    self.whisperx,
                    "assign_word_speakers",
                    clock.wrap(
                        "speaker_assignment", self.whisperx.assign_word_speakers
                    ),
                )
            )
            return self.service.transcribe(audio, self.settings, self.language)

    def normalize(self, result):
        return {
            "segments": result.get("segments", []),
            "word_segments": result.get("word_segments", []),
        }

    def close(self):
        self.loop.run_until_complete(self.context.__aexit__(None, None, None))
        self.loop.close()


if __name__ == "__main__":
    main("repo", RepoBackend)
