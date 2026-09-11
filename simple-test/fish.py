#!/usr/bin/env python3
"""Minimal in-process Fish Audio S2 text-to-speech smoke test."""

import os
from pathlib import Path

import soundfile as sf
import torch

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = Path(
    os.getenv(
        "VOCALIZER_MODEL_ID",
        str(PROJECT_DIR / "fish-speech-checkpoints" / "s2-pro"),
    )
).expanduser().resolve()
OUTPUT_PATH = Path(__file__).resolve().parent / "fish-test.wav"
REFERENCE_PATH = Path(__file__).resolve().parent / "garrison.flac"
REFERENCE_TRANSCRIPT = (
    "Choose between the high road and the low, sell your gift to a buyer at a good game, "
    "try to trace the fine lines of the painting"
)
SYNTHESIS_TEXT = (
    "[excited] Hello world! [laughing] This is Fish Audio S2. "
    "[sigh] [excited] I cannot believe how expressive this voice can sound!"
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    codec_path = CHECKPOINT_DIR / "codec.pth"
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"S2 checkpoint directory not found: {CHECKPOINT_DIR}\n"
            "Set VOCALIZER_MODEL_ID to your local fishaudio/s2-pro directory."
        )
    if not codec_path.is_file():
        raise FileNotFoundError(f"S2 codec checkpoint not found: {codec_path}")
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"Reference voice file not found: {REFERENCE_PATH}")

    device = pick_device()
    precision = torch.bfloat16

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(CHECKPOINT_DIR),
        device=device,
        precision=precision,
        compile=False,
    )
    try:
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(codec_path),
            device=device,
        )
        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )

        request = ServeTTSRequest(
            # S2 uses inline natural-language tags for delivery and expressions.
            text=SYNTHESIS_TEXT,
            references=[
                ServeReferenceAudio(
                    audio=REFERENCE_PATH.read_bytes(),
                    text=REFERENCE_TRANSCRIPT,
                )
            ],
            reference_id=None,
            streaming=False,
            format="wav",
        )

        for result in engine.inference(request):
            if result.code == "error":
                error = result.error
                if isinstance(error, BaseException):
                    raise RuntimeError(f"Fish Audio S2 generation failed: {error}") from error
                raise RuntimeError(f"Fish Audio S2 generation failed: {error}")
            if result.code == "final" and isinstance(result.audio, tuple):
                sample_rate, audio = result.audio
                sf.write(OUTPUT_PATH, audio, sample_rate)
                return

        raise RuntimeError("Fish Audio S2 did not generate any audio")
    finally:
        # Stop Fish Speech's local semantic-model worker thread.
        llama_queue.put(None)


if __name__ == "__main__":
    main()
