import logging

import numpy as np
import whisperx

from utils.config import Settings, get_device
from utils.lifespan import pipelines

logger = logging.getLogger("api")


def transcribe(
    audio: np.ndarray,
    settings: Settings,
    language: str | None = None,
) -> dict:
    """Run the full transcription pipeline: transcribe, align, and diarize."""
    result = _transcribe_audio(audio, settings, language)
    result = _align_transcription(audio, result, settings)
    result = _diarize_and_assign_speakers(audio, result, settings)
    return result


def _transcribe_audio(
    audio: np.ndarray,
    settings: Settings,
    language: str | None,
) -> dict:
    """Run whisperx transcription on audio."""
    logger.info("Starting transcription …")
    result = pipelines[settings.model].transcribe(
        audio, batch_size=settings.batch_size, language=language
    )
    logger.info("Transcription done.")
    return result


def _align_transcription(
    audio: np.ndarray,
    transcription_result: dict,
    settings: Settings,
) -> dict:
    """Align transcription segments with audio."""
    logger.info("Aligning transcription …")
    device = get_device()
    model_alignment, metadata = whisperx.load_align_model(
        language_code=transcription_result["language"], device=device
    )
    aligned = whisperx.align(
        transcription_result["segments"],
        model_alignment,
        metadata,
        audio,
        device,
        interpolate_method=settings.interpolate_method,
        return_char_alignments=settings.return_char_alignments,
    )
    logger.info("Alignment done.")
    return aligned


def _diarize_and_assign_speakers(
    audio: np.ndarray,
    alignment_result: dict,
    settings: Settings,
) -> dict:
    """Run diarization and assign speakers to words."""
    logger.info("Diarization …")
    diarize_segments = pipelines["diarize_model"](audio)
    result = whisperx.assign_word_speakers(
        diarize_segments, alignment_result, fill_nearest=settings.fill_nearest
    )
    logger.info("Diarization done.")
    return result
