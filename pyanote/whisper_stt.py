"""Batched Whisper large-v2 adapter for pyannoteAI's public STT interface."""

import logging
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
        logger.info("Loading Whisper model: %s (ASR batch_size=%d)", self.id, batch_size)
        self._model = WhisperModel(self.id)
        self._batched = BatchedInferencePipeline(model=self._model)
        logger.info("Whisper model loaded in %.2fs", perf_counter() - started)

    @property
    def sample_rate(self):
        return self._model.feature_extractor.sampling_rate

    def apply(self, file, cut_points=None):
        started = perf_counter()
        logger.info("Starting Whisper ASR (batch_size=%d)", self.batch_size)
        # Like the vendor Whisper wrapper, process the whole waveform and let
        # Faster Whisper handle chunking. pyannote has resampled it to mono.
        waveform = file["waveform"].squeeze(0).cpu().numpy()
        segments, _ = self._batched.transcribe(
            waveform, batch_size=self.batch_size, word_timestamps=True, vad_filter=True
        )
        # Inference is lazy: consume all segments before logging completion.
        words = [
            (float(word.start), float(word.end), word.word.strip())
            for segment in segments
            for word in (segment.words or [])
        ]
        logger.info(
            "Whisper ASR finished in %.2fs (words=%d)",
            perf_counter() - started, len(words),
        )
        return words
