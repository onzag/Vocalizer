#!/usr/bin/env python3
"""CLI entry point.

    python main.py script.json -o output.wav
    python main.py script.json -o output.mp3
    python main.py script.json -o output.ogg

Renders a JSON script with Vocalizer and writes the resulting audio to the
specified output file (.wav, .mp3, or .ogg). All logging/progress text goes to stderr.
"""

import argparse
import json
import os
import sys

from vocalizer import Vocalizer, VocalizerConfig


def main():
    parser = argparse.ArgumentParser(description="Render a vocalizer JSON script to audio (wav/mp3/ogg).")
    parser.add_argument("json_path", help="Path to the JSON script file")
    parser.add_argument("-o", "--output", required=True, help="Output file path (.wav, .mp3, or .ogg)")
    parser.add_argument("--sounds-dir", default="./sounds", help="Sound library directory (default: ./sounds)")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Output sample rate (default: 48000)")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Default VoxCPM cfg_value (default: 2.0)")
    parser.add_argument("--inference-timesteps", type=int, default=10,
                         help="Default VoxCPM inference_timesteps (default: 10)")
    parser.add_argument("--target-loudness-db", type=float, default=-20.0,
                         help="RMS loudness normalization target in dBFS (default: -20.0)")
    parser.add_argument("--model-id", default="openbmb/VoxCPM2", help="VoxCPM model id or local path")
    parser.add_argument("--denoise", action="store_true", help="Enable VoxCPM's ZipEnhancer denoiser on load")
    args = parser.parse_args()

    _SUPPORTED_EXTS = {".wav", ".mp3", ".ogg"}
    output_ext = os.path.splitext(args.output)[1].lower()
    if output_ext not in _SUPPORTED_EXTS:
        parser.error(f"Output file must be .wav, .mp3, or .ogg (got '{output_ext}')")

    with open(args.json_path, "r", encoding="utf-8") as f:
        json_loaded = json.load(f)

    config = VocalizerConfig(
        sound_library_dir=args.sounds_dir,
        output_sample_rate=args.sample_rate,
        voxcpm_model_id=args.model_id,
        load_denoiser=args.denoise,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        target_loudness_db=args.target_loudness_db,
    )

    print(f"Loading {args.model_id}...", file=sys.stderr)
    vocalizer = Vocalizer(config)

    print(f"Rendering {args.json_path}...", file=sys.stderr)
    vocalizer.render_json_to_file(json_loaded, args.output)

    print(f"Done: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
