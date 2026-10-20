"""Batched Whisper large-v2 adapter for pyannoteAI's public STT interface."""

import logging
from math import isfinite
from time import perf_counter

from faster_whisper import BatchedInferencePipeline, WhisperModel
from pyannoteai.stt import STT


logger = logging.getLogger("pyanote")


class BatchedWhisperSTT(STT):
    def __init__(self, batch_size=8):
        if batch_size < 1:
            raise ValueError("STT batch size must be at least 1")
        self.id = "Systran/faster-whisper-large-v2"
        self.batch_size = batch_size
        started = perf_counter()
        logger.info(
            "Loading Whisper model: %s (ASR batch_size=%d)", self.id, batch_size
        )
        self._model = WhisperModel(self.id)
        self._batched = BatchedInferencePipeline(model=self._model)
        logger.info("Whisper model loaded in %.2fs", perf_counter() - started)

    @property
    def sample_rate(self):
        return self._model.feature_extractor.sampling_rate

    @property
    def target_max_duration(self):
        # Ask pyannote to choose safe cuts near Whisper's context limit.
        return self._model.feature_extractor.chunk_length

    def _clip_timestamps(self, num_samples, cut_points):
        """Convert safe cuts in seconds to contiguous, context-sized clips."""
        boundaries = [0]
        for point in () if cut_points is None else cut_points:
            point = float(point)
            if not isfinite(point):
                raise ValueError("Cut points must be finite timestamps in seconds")
            sample = round(point * self.sample_rate)
            if sample < boundaries[-1] or sample > num_samples:
                raise ValueError("Cut points must be ordered and within the waveform")
            boundaries.append(sample)
        boundaries.append(num_samples)

        max_samples = round(self.target_max_duration * self.sample_rate)
        clips = []
        for start, end in zip(boundaries, boundaries[1:]):
            # Safe cuts are approximate; split oversized intervals so Whisper
            # cannot silently truncate them to its context window.
            for chunk_start in range(start, end, max_samples):
                clips.append(
                    {
                        "start": chunk_start / self.sample_rate,
                        "end": min(chunk_start + max_samples, end) / self.sample_rate,
                    }
                )
        return clips

    def apply(self, file, cut_points=None):
        try:
            return self._apply(file, cut_points)
        except Exception:
            # Log here before the pipeline replaces the original exception
            # with a generic RuntimeError("Transcription failed").
            logger.exception("Whisper ASR failed (batch_size=%d)", self.batch_size)
            raise

    def _apply(self, file, cut_points=None):
        started = perf_counter()
        logger.info("Starting Whisper ASR (batch_size=%d)", self.batch_size)
        # pyannote has resampled the waveform to mono at self.sample_rate.
        waveform = file["waveform"].squeeze(0).cpu().numpy()
        clips = self._clip_timestamps(len(waveform), cut_points)
        if not clips:
            return []
        logger.info("Whisper ASR using %d clips with VAD disabled", len(clips))
        segments, _ = self._batched.transcribe(
            waveform,
            batch_size=self.batch_size,
            word_timestamps=True,
            vad_filter=False,
            clip_timestamps=clips,
        )
        # Faster Whisper already offsets timestamps by each clip's start.
        # Inference is lazy: consume all segments before logging completion.
        words = [
            (float(word.start), float(word.end), word.word.strip())
            for segment in segments
            for word in (segment.words or [])
        ]
        logger.info(
            "Whisper ASR finished in %.2fs (words=%d)",
            perf_counter() - started,
            len(words),
        )
        return words
