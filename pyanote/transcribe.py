"""Transcribe one file with pyannoteAI Local and Whisper large-v2."""

import argparse
import json
import logging
import os
from pathlib import Path
from time import perf_counter


logger = logging.getLogger("pyanote")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("audio", nargs="?", type=Path, help="Input audio/video file")
    inputs.add_argument("--audio-file", type=Path, help="File to transcribe")
    parser.add_argument(
        "--warmup-file", type=Path, help="Short audio file to process once before the input"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=8, help="Diarization batch size")
    parser.add_argument(
        "--stt-batch-size", type=int, default=8, help="Whisper ASR batch size"
    )
    parser.add_argument(
        "--mode", choices=("accuracy", "balance", "speed"), default="accuracy"
    )
    args = parser.parse_args()
    args.audio = args.audio_file or args.audio
    if not args.audio.is_file():
        parser.error(f"Input file does not exist: {args.audio}")
    if args.warmup_file is not None and not args.warmup_file.is_file():
        parser.error(f"Warm-up file does not exist: {args.warmup_file}")
    if args.batch_size < 8:
        parser.error("--batch-size must be at least 8")
    if args.stt_batch_size < 1:
        parser.error("--stt-batch-size must be at least 1")
    key = os.environ.get("PYA_JOBS_KEY")
    if not key:
        parser.error("Set PYA_JOBS_KEY to your pyannoteAI Run jobs API key")
    output = args.output or args.audio.with_suffix(".transcription.json")
    if output.resolve() == args.audio.resolve():
        parser.error("Output must differ from the input file")
    if args.warmup_file is not None and output.resolve() == args.warmup_file.resolve():
        parser.error("Output must differ from the warm-up file")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    started = perf_counter()
    logger.info("Starting job: %s", args.audio)
    logger.info("Importing inference libraries")
    from pyannoteai.local import Pipeline
    from whisper_stt import BatchedWhisperSTT

    logger.info("Inference libraries imported")
    stt = BatchedWhisperSTT(batch_size=args.stt_batch_size)
    stage_started = perf_counter()
    logger.info(
        "Loading pyannote pipeline (batch_size=%d, mode=%s)",
        args.batch_size, args.mode,
    )
    pipeline = Pipeline(key, batch_size=args.batch_size, mode=args.mode)
    logger.info("Pyannote pipeline loaded in %.2fs", perf_counter() - stage_started)
    try:
        if args.warmup_file is not None:
            stage_started = perf_counter()
            logger.info("Starting warm-up: %s", args.warmup_file)
            pipeline.transcribe(str(args.warmup_file), stt)
            logger.info("Warm-up finished in %.2fs", perf_counter() - stage_started)
        stage_started = perf_counter()
        logger.info(
            "Starting full pipeline for %s: diarization, ASR, and speaker assignment",
            args.audio,
        )
        result = pipeline.transcribe(str(args.audio), stt)
        logger.info("Full pipeline finished in %.2fs", perf_counter() - stage_started)
        logger.info("Saving transcription to %s", output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Transcription saved")
        for turn in result["turn_level_transcription"]:
            print(
                f"{turn['start']:8.2f}–{turn['end']:8.2f} "
                f"{turn['speaker']}: {turn['text']}"
            )
    finally:
        logger.info("Releasing pyannote pipeline")
        pipeline.drop()
        logger.info("Pyannote pipeline released")
    logger.info("Job finished in %.2fs", perf_counter() - started)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Transcription job failed")
        raise SystemExit(1)
