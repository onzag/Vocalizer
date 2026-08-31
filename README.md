# Vocalizer

Turn a JSON script into a single mixed WAV: synthesized speech (via
[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)), sound-effect clips,
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

- **Text-to-speech** with voice cloning (`ref`) and/or free-text style
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
pip install -r requirements.txt
```

Requires Python 3.9+ and less than 3.12. VoxCPM2 will download on first run unless you point
`--model-id` at a local path.

## Usage

```bash
python main.py script.json > output.wav
# or
python main.py script.json -o output.wav
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--sounds-dir` | `./sounds` | Directory your `ref`/`file` paths resolve against |
| `--sample-rate` | 48000 | Output sample rate |
| `--cfg-value` | 2.0 | Default VoxCPM guidance value |
| `--inference-timesteps` | 10 | Default VoxCPM diffusion steps |
| `--target-loudness-db` | -20.0 | RMS normalization target |
| `--model-id` | `openbmb/VoxCPM2` | VoxCPM model id or local path |
| `--denoise` | off | Enable VoxCPM's ZipEnhancer denoiser |

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
