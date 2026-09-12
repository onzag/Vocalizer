# Vocalizer

Turn a JSON script into a single mixed WAV: synthesized speech via
[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) or Fish Audio S2,
sound-effect clips,
timed silences, and a looping, volume-automated background bed — all
stitched into one continuous timeline.

```json
{
  "background": { "file": "rain.wav", "volume": 5, "presence": "on", "fade_ms": 300 },
  "segments": [
    { "ref": "narrator.wav", "text": "It started raining just after dusk." },
    { "duration_ms": 300 },
    { "voice_prompt": "hushed, a little conspiratorial", "text": "Come a little closer." }
  ]
}
```

## Features

- **Selectable local text-to-speech engine** with voice cloning (`ref`) and/or free-text style
  control (`voice_prompt`), plus Hi-Fi cloning via reference wav + transcript.
- **Sound-effect playback** from a local library, including `{n}` wildcard
  pools, randomized/cyclic selection, repeats, and per-draw volume jitter.
- **Timed silences**, fixed or randomized (`[min, max]` ms).
- **A looping background track** with crossfaded file switches, smooth
  volume ramps, and presence on/off ducking — driven by lightweight
  "background-update" markers interleaved with the main segments.
- **Consistent perceived loudness**: every clip is RMS-normalized before a
  simple 1-9 volume scale is applied, so "5" always means the same thing
  regardless of the source recording's original level.

## Installation

```bash
git clone https://github.com/rickywoof/vocalizer.git
cd vocalizer
pip install -r requirements-voxcpm.txt
```

For Fish Audio S2, use `install-fishaudio.sh` and run Vocalizer from the
Fish Speech environment it creates. Separate environments are recommended
because the two engines have different Python and PyTorch requirements.

## Selecting and configuring the speech engine

VoxCPM is selected by default. Set either `VOCALIZER_MODE` or
`VOCALIZER_BACKEND` to `fishaudio` to use Fish Audio S2 instead. If both
variables are set, `VOCALIZER_MODE` takes precedence.

```bash
# Default
VOCALIZER_MODE=voxcpm python server.py

# Fish Audio S2
VOCALIZER_MODE=fishaudio python server.py
# VOCALIZER_BACKEND=fishaudio python server.py  # equivalent alias
```

When an engine is initialized for the first time, Vocalizer creates its
configuration beside `vocalizer.py`. Restart Vocalizer after editing it.

`.config-voxcpm.json` defaults:

```json
{
  "model_id": "openbmb/VoxCPM2",
  "load_denoiser": false,
  "cfg_value": 2.0,
  "inference_timesteps": 10
}
```

`.config-fishaudio.json` defaults:

```json
{
  "model_id": "checkpoints/s2-pro",
  "decoder_checkpoint_path": null,
  "decoder_config_name": "modded_dac_vq",
  "device": null,
  "half": false,
  "compile": false,
  "max_new_tokens": 1024,
  "chunk_length": 200,
  "top_p": 0.8,
  "repetition_penalty": 1.1,
  "temperature": 0.8
}
```

Relative Fish Audio checkpoint paths are resolved from the Vocalizer directory.
When `decoder_checkpoint_path` is `null`, Vocalizer uses `codec.pth` inside
`model_id`. A `null` device automatically selects CUDA, XPU, MPS, or CPU in
that order. Keep `compile` disabled on unsupported platforms.

## Usage

```bash
python main.py script.json -o output.wav
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--sounds-dir` | `./sounds` | Directory your `ref`/`file` paths resolve against |
| `--sample-rate` | 48000 | Output sample rate |
| `--target-loudness-db` | -20.0 | RMS normalization target |

Engine-specific settings belong in the selected `.config-<engine>.json`, not
in `VocalizerConfig` or CLI flags.

An example script is in [`examples/basic-scene.json`](examples/basic-scene.json).

## Script format

A script is either a bare list of segments, or:

```json
{ "generation": { ... }, "background": { ... }, "segments": [ ... ] }
```

Each segment's kind (speech / sound-effect file / silence / background
update) is inferred automatically from which keys are present — there's
no explicit `"type"` field. Full schema, the volume/loudness model, and
how the background track's crossfades and gain envelopes work are
documented in [`docs/vocalizer-documentation.md`](docs/vocalizer-documentation.md),
along with script-writing guidelines and an annotated example.

## Sound library

Point `--sounds-dir` at a folder of your own wav files (voice-cloning
references, ambience loops, one-shot effects). None are bundled with
this repo — see `sounds/README.md` for the expected layout.

## License

[LICENSE](LICENSE).
