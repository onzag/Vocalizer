# Vocalizer — Documentation & Script Design Guide

Vocalizer turns a JSON "script" into a single mixed WAV file: synthesized
speech (via VoxCPM2), pre-recorded sound-effect clips, silences, and a
looping background bed, all stitched into one continuous timeline.

This doc covers two things: **how the program actually works internally**
(`main.py` + `vocalizer.py`), and **how to write good scripts** for it.

---

## 1. Program architecture

```
main.py            CLI wrapper — parses args, loads JSON, writes WAV out
vocalizer.py        Vocalizer / VocalizerConfig — the actual rendering engine
```

### 1.1 `main.py` — CLI entry point

```
python main.py script.json > output.wav
python main.py script.json -o output.wav
```

- Loads the JSON script file.
- Builds a `VocalizerConfig` from CLI flags (sound dir, sample rate, VoxCPM
  model, default `cfg_value`/`inference_timesteps`, target loudness, denoiser).
- Instantiates `Vocalizer(config)` — this loads the VoxCPM model.
- Calls `vocalizer.render_script(script)`, which returns a stereo
  `float32` numpy array.
- Writes it either to `-o/--output` or as raw WAV bytes to **stdout**.
  All progress/logging goes to **stderr** specifically so stdout redirection
  (`> output.wav`) stays uncorrupted.

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--sounds-dir` | `./sounds` | Where library clips/refs are resolved from |
| `--sample-rate` | 48000 | Output sample rate; all clips get resampled to this |
| `--cfg-value` | 2.0 | Default VoxCPM guidance value |
| `--inference-timesteps` | 10 | Default VoxCPM diffusion steps |
| `--target-loudness-db` | -20.0 | RMS target every clip is normalized to |
| `--model-id` | `openbmb/VoxCPM2` | VoxCPM model id/path |
| `--denoise` | off | Enables VoxCPM's ZipEnhancer denoiser |

### 1.2 `vocalizer.py` — the engine

Core classes:

- **`VocalizerConfig`** — dataclass holding all the knobs above, plus
  `crossfade_ms` (default 15ms, used for concatenation seams between
  segments and between background file changes).
- **`SoundLibrary`** — resolves file references relative to `sound_library_dir`,
  including glob-style `{n}` wildcard patterns, with caching.
- **`Vocalizer`** — loads the VoxCPM model once, then exposes
  `render_segment`, `render_script`, `render_to_file`.

---

## 2. Script schema

A script is either a **bare list of segments**, or a **dict**:

```json
{
  "generation": { "cfg_value": 2.0, "inference_timesteps": 10, "normalize": true },
  "background": { "file": "background.wav", "volume": 5, "presence": "on", "fade_ms": 250 },
  "segments": [ ... ]
}
```

- `generation` — default VoxCPM parameters applied to every speech segment
  unless a segment overrides them itself.
- `background` — initial state of the looping background bed (see §5).
- `segments` — the ordered list that becomes the main timeline.

### 2.1 How segment "kind" is inferred

There is **no explicit `"type"` key**. `_infer_kind()` looks at which keys
are present, in this priority order:

1. `"text"` present → **speech**
2. else `"duration_ms"` present → **delay**
3. else `"ref"` absent → **background_update**
4. else → **file** (library clip playback)

This ordering matters: e.g. a segment with both `"text"` and `"ref"` is
always speech (the `ref` becomes the voice-cloning reference, not a file
to play back verbatim).

### 2.2 Speech segments

```json
{
  "ref": "emotion.wav",
  "voice_prompt": "warm, gentle female voice, speaking softly",
  "text": "So Ricky",
  "volume": 5,
  "cfg_value": 2.0,
  "inference_timesteps": 10,
  "seed": 42,
  "normalize": false,
  "prompt_ref": "speaker_prompt.wav",
  "prompt_text": "..."
}
```

- `ref` (optional) — a wav in the sound library used as a **voice-cloning
  reference** (sets timbre/voice identity). If omitted, VoxCPM invents a
  voice from scratch.
- `voice_prompt` (optional) — free-text style direction. It's **prepended**
  to the text as `"(voice_prompt)text"` — this is VoxCPM2's Voice
  Design/Style Control syntax. With no `ref`, it defines the whole voice;
  with a `ref`, `ref` sets timbre and `voice_prompt` steers delivery/emotion
  on top of it.
- `text` — required, what gets spoken.
- `volume` — 1-9 scale, see §4.
- `cfg_value`, `inference_timesteps`, `seed`, `normalize` — override the
  script-level `generation` defaults for this segment only.
- `prompt_ref` + `prompt_text` — VoxCPM's **Hi-Fi cloning** path: a
  reference wav *plus* its exact transcript, for higher-fidelity voice
  cloning than `ref` alone.

Rendering pipeline for a speech segment (`_render_speech`):
1. Build the final prompt string (`(voice_prompt)text` or just `text`).
2. Resolve generation params: config defaults → script `generation` dict →
   segment-level overrides (segment wins).
3. Call `VoxCPM.generate(...)` with `reference_wav_path`, `prompt_wav_path`,
   `prompt_text`, and the generation params.
4. Convert to stereo float32, resample to the output sample rate if the
   model's native rate differs.
5. RMS-normalize to `target_loudness_db`, then apply the `volume` gain.

### 2.3 File segments (library clip playback, no synthesis)

```json
{
  "ref": "moan-{n}.wav",
  "randomize": true,
  "repeat": 3,
  "volume": 5,
  "volume_jitter": [3, 8]
}
```

- `ref` — a literal filename, or a pattern containing `{n}` which expands
  to a glob (`moan-{n}.wav` → matches `moan-1.wav`, `moan-2.wav`, ... sorted).
- `randomize` — if true, each draw picks a random match; if false, draws
  cycle through matches **in sorted order** (`index % len(matches)`).
- `repeat` — how many clips to draw and chain together (default 1). Each
  draw gets its own volume if `volume_jitter` is set.
- `volume` — flat gain applied to every drawn clip (used when no jitter).
- `volume_jitter` — `[min, max]`; if present it **overrides** `volume`,
  and each of the `repeat` draws gets its own independently randomized
  gain in that range.
- The `repeat` clips are joined with the same short crossfade
  (`crossfade_ms`) used elsewhere, not concatenated raw.

Every loaded clip is RMS-normalized to `target_loudness_db` before the
volume gain is applied (same as speech).

### 2.4 Delay segments

```json
{ "duration_ms": 500 }
{ "duration_ms": [300, 900] }
```

Pure silence. A single number is a fixed gap; `[min, max]` picks a random
duration uniformly in that range **each time the script is rendered**.
Renders as literal zero-samples (stereo), which also acts as a hard reset
point in the timeline.

### 2.5 Background-update segments (markers)

```json
{ "file": "storm.wav", "volume": 8, "presence": "off", "fade_ms": 400 }
```

Identified by the **absence of `ref`** and absence of `text`/`duration_ms`.
These render **no audio of their own** in the main timeline — they're
pure metadata markers that change the state of the background bed. All
fields are optional and "hold" the previous value until changed again:

- `file` — switches the active background loop (triggers a **file
  crossfade**, not just a volume ramp).
- `volume` — 1-9 scale for the background level from this point on.
- `presence` — `"on"`/`"off"`; `"off"` ramps the background to silence
  without stopping the underlying loop (so it can fade back `"on"` later
  in sync, rather than restarting from sample 0).
- `fade_ms` — ramp/crossfade duration for *this* transition. If omitted,
  falls back to the script's top-level `background.fade_ms`.

**Timing:** a background-update marker takes effect at the timestamp where
the *next real (audio-producing) segment* starts. If it's the last item in
`segments`, it still gets applied, ramping to the end of the render, but
produces no additional audio after the last real segment ends — so a
trailing background-update after the final speech line has no audible
effect beyond ramping under whatever's still playing.

---

## 3. Generation defaults

```json
"generation": { "cfg_value": 2.0, "inference_timesteps": 10, "normalize": true }
```

Applied to every speech segment. Resolution order (`_resolve_generation_params`):

```
VocalizerConfig defaults  →  script["generation"]  →  segment-level keys
```

Later stages override earlier ones. Only these keys are considered for
segment-level override: `cfg_value`, `inference_timesteps`, `normalize`,
`denoise`, `seed`.

---

## 4. Volume system

All volume fields use a **1-9 scale**, foreground and background alike:

```
gain = 1.0 + (volume - 5) * 0.125
```

| volume | gain |
|---|---|
| 1 | 0.50x |
| 5 | 1.00x (natural, default) |
| 9 | 1.50x |

`[min, max]` is also accepted anywhere a volume is expected (speech,
file, background) — it's resolved to a single random value with
`random.uniform()` at render time, then converted through the same
formula.

### Loudness normalization (why "5" means the same thing everywhere)

Every clip — synthesized speech, every library clip, every background
loop — is first **RMS-normalized to `target_loudness_db`** (default
-20 dBFS) *before* the volume gain is applied. This means the 1-9 scale
is a genuine *relative* control: a quiet ambience loop and a loud
recorded effect both land at the same perceived loudness at volume 5,
so you're only ever describing how loud something should be *relative
to natural*, not fighting the source recording's original level.

---

## 5. Background track rendering (the tricky part)

This is the most involved subsystem (`render_background_track`), worth
understanding in detail if you're designing scripts that lean on it.

1. **State timeline construction.** The initial `background` dict is
   state at t=0. Each background-update marker is converted into a state
   change *at the sample offset where the next real segment begins*
   (looked up via `offsets`, the per-segment start times returned by the
   main crossfade-concat pass).
2. **Collapsing.** If multiple markers resolve to the exact same start
   offset (e.g. two background-updates back-to-back before the next real
   segment), only the last one wins for that offset.
3. **Segmenting into "runs".** Consecutive states that share the same
   `file` are grouped into a "run" — a file only gets loaded/decoded once
   per contiguous run, even if volume/presence changes several times
   within it.
4. **Per-run envelope.** Within a run, `build_gain_envelope` builds a
   sample-accurate gain curve: flat segments at each state's gain, with
   linear ramps of length `fade_ms` at each transition point (volume
   change, presence toggle).
5. **Run boundaries and crossfades between different files.** Each run's
   audio contribution extends `fade_ms` samples *past* its natural end
   (borrowing time from the next run) with a linear fade-to-zero tail, so
   that when the next run's file starts, the two overlap and crossfade
   rather than hard-cutting.
6. **Looping.** The background file for a run is loaded once and looped
   (`loop_to_length`) to cover however long that run's contribution needs
   to be — so short ambience loops can underlie arbitrarily long stretches
   of dialogue.
7. **Mixing.** The finished background track is summed with the
   foreground (speech/file/delay) timeline. If the sum clips above 1.0,
   the whole mix is peak-normalized down (not per-track — a single global
   safety limiter at the very end).

If `background` is omitted from the script, or the segments produce zero
audio, no background track is rendered at all.

---

## 6. Full render pipeline (`render_script`)

1. Normalize input: list script → segments only, no background/generation;
   dict script → pull out `generation`, `background`, `segments`.
2. Walk `segments` in order:
   - Background-update → record a marker (`clip_index` = index into the
     *audio-producing* clip list so far, not the raw segment list).
   - Anything else → render it immediately to audio via `render_segment`
     and append to `clips`.
3. Concatenate all `clips` with short crossfades
   (`crossfade_concat_with_offsets`) → get the full foreground timeline
   plus each clip's actual start offset (crossfades shrink total length
   vs. naive concatenation, so offsets aren't just cumulative durations).
4. If there's a `background` config and the timeline isn't empty, render
   the background track against those same offsets and mix it in.
5. Global peak safety limiter, return stereo float32 audio.

`render_to_file` is a convenience wrapper that also calls `sf.write`.

---

## 7. Script design guidelines

### 7.1 Mental model

Think of a script as **two parallel tracks**: a foreground track (built
strictly from speech/file/delay segments, in order, crossfaded together)
and a background track (a state machine driven by background-update
markers, which only "commits" its changes at the moment the *next*
foreground segment starts). Background-update markers are timing-relative
to what comes after them, not what comes before — put them **immediately
before** the segment where you want the change to land, not after.

### 7.2 Pacing with delays

- Use small delays (150-400ms) between distinct lines for natural
  breathing room; crossfading alone (15ms default) is too short to read
  as a pause.
- Use `[min, max]` delay ranges for anything that repeats or could feel
  mechanical (e.g. between repeated ambient/reaction clips) so pacing
  doesn't feel metronomic on replay.
- A delay right after a `presence: off` background marker gives the fade
  room to actually be heard before the next line starts.

### 7.3 Voice consistency

- Reuse the same `ref` across a character's lines for timbre consistency;
  vary `voice_prompt` line-to-line to carry emotional arc (calm → hushed →
  urgent) without re-cloning the voice each time.
- For a one-off or narrator voice with no established reference clip,
  `voice_prompt` alone (no `ref`) is fine — just expect more voice-identity
  drift between generations since nothing anchors the timbre.
- If a line needs to closely match a specific real reference performance
  (exact prosody/timing), use `prompt_ref` + `prompt_text` (Hi-Fi cloning)
  instead of plain `ref`.
- Bump `inference_timesteps` above the default (10) for hero lines where
  quality matters more than render speed; leave it low for filler/ambient
  lines.

### 7.4 Volume levels

- Keep dialogue around 5-6; reserve 7-9 for genuine loud/emphatic moments
  so the scale still reads as meaningful contrast.
- Because loudness-normalization happens before the gain is applied, you
  don't need to compensate for a "loud" source file — set volume purely
  for *how it should feel*, not to correct recording level.
- For repeated file segments (e.g. reaction clips), prefer `volume_jitter`
  over a flat `volume` — a small range (±1-2) avoids every repeat sounding
  identically loud, which reads as artificial.

### 7.5 Background bed usage

- Always set an explicit `fade_ms` on background-update markers that
  change `file` or toggle `presence` — with `fade_ms: 0` you get a hard
  cut/pop, which is rarely what you want for ambience.
- To swap ambience under a scene transition, put the `file`-changing
  marker right before the first line of the new scene; the crossfade
  will straddle the transition naturally.
- Use `presence: off` (not removing the marker) when you want to duck
  the background out and later bring the *same* loop back in sync,
  e.g. for a moment of hushed emphasis — `presence: on` again later
  resumes the same underlying loop position's fade curve rather than
  restarting it.
- A background-update as the very last segment still has an effect (it
  ramps under whatever's already rendered) but adds no new duration —
  don't rely on a trailing marker to add tail silence; use a `duration_ms`
  delay for that instead.

### 7.6 File/library segments

- `{n}` wildcard + `randomize: true` is the right choice for variety pools
  (reaction sounds, ambient one-shots) where any match is acceptable.
- `{n}` wildcard + `randomize: false` cycles deterministically through
  matches in sorted filename order — useful when you want a reproducible,
  specific sequence (e.g. numbered story beats: `beat-1.wav`, `beat-2.wav`…)
  rather than random variety.
- `repeat` on a single ref call chains multiple draws with crossfades —
  useful for building up a longer effect from short one-shots without
  needing a pre-mixed long file.

### 7.7 Generation defaults vs. per-segment overrides

- Set `cfg_value`/`inference_timesteps` once at the script's `generation`
  level for the "default" delivery quality/adherence, and override only
  the handful of segments that need something different (e.g. a hushed
  aside might want higher `cfg_value` for stronger prompt adherence to
  `voice_prompt`).
- `seed` at the segment level is useful for locking in a take you like
  during iteration — otherwise every re-render of that line will vary.

### 7.8 Annotated walkthrough of `script.txt`

The uploaded `script.txt` is a good example of most features in one file:

1. Opens with a normal speech line at natural volume (5), cloned from
   `emotion.wav`.
2. A 400ms delay for breathing room before the next speaker.
3. A `voice_prompt`-only line (no `ref`) — VoxCPM invents an older-man
   voice from the prompt text alone.
4. A line combining `ref` (timbre) + `voice_prompt` (hushed delivery) +
   a segment-level `inference_timesteps: 20` override for extra quality
   on this more nuanced delivery.
5. A **bare background-update** (`volume: 8, fade_ms: 300`) that raises
   the (as-yet-unset) background level — but since no background `file`
   has been set yet at the script level either, this only matters once a
   `background` block or later `file` marker actually establishes one.
   *(Note: in this particular script there's no top-level `"background"`
   key, so background markers here only take effect if a script wrapper
   supplies one — worth double-checking when reusing this pattern.)*
6. A `file`-changing marker to `rain.wav` with a 600ms crossfade, timed to
   land under the "storm's rolling in" line — a good example of pre-empting
   a background swap one segment ahead of the line it should underlie.
7. A randomized `[300, 900]` delay before that line, avoiding a mechanically
   exact gap.
8. A `presence: off` marker to duck the rain out before the closer.
9. A `moan-{n}.wav` pool: randomized, repeated 3x, with `volume_jitter`
   for natural-sounding variation — a textbook use of the variety-pool
   pattern from §7.6.
10. Closes with a short delay and a final line back on the `emotion.wav`
    voice, bringing the scene back to the original speaker.

---

## 8. Common pitfalls

- **Missing `background` block but using background-update markers.**
  Markers only affect a background bed that was declared via the
  top-level `"background"` key; without it, markers are silently inert.
- **`fade_ms: 0` on file-changing markers** causes an audible pop instead
  of a crossfade.
- **Very short delay/file segments combined with the default 15ms
  crossfade** can get visibly/audibly eaten — `crossfade_concat` requires
  both neighboring clips to be longer than `n_fade` samples or it falls
  back to a hard concat, so ultra-short clips may not crossfade at all.
- **Relying on a trailing background-update for silence** — it adds no
  timeline duration; use a `duration_ms` delay instead.
- **Forgetting `ref` vs `file` semantics** — in a speech segment, `ref` is
  a *voice reference*, never played back as-is; in a file segment, `ref`
  (or its `{n}` pattern) is literally the audio that gets played.
