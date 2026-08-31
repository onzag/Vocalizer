import glob
import os
import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import soundfile as sf

try:
    from voxcpm import VoxCPM
except ImportError:
    VoxCPM = None


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class VocalizerConfig:
    sound_library_dir: str = "./sounds"
    output_sample_rate: int = 48000
    voxcpm_model_id: str = "openbmb/VoxCPM2"
    load_denoiser: bool = False
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    crossfade_ms: int = 15
    target_loudness_db: float = -20.0   # RMS target every clip is normalized to before volume gain


# --------------------------------------------------------------------------
# Volume / duration / loudness helpers
# --------------------------------------------------------------------------

def volume_to_gain(volume: float) -> float:
    """5 -> 1.0x (natural), 1 -> 0.5x, 9 -> 1.5x."""
    return 1.0 + (volume - 5) * 0.125


def resolve_volume_spec(volume_spec) -> float:
    if volume_spec is None:
        value = 5
    elif isinstance(volume_spec, (list, tuple)):
        value = random.uniform(*volume_spec)
    else:
        value = volume_spec
    return volume_to_gain(value)


def resolve_duration_ms(duration_spec) -> float:
    if isinstance(duration_spec, (list, tuple)):
        return random.uniform(*duration_spec)
    return duration_spec or 0


def normalize_rms(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-9
    target_rms = 10 ** (target_dbfs / 20)
    return x * (target_rms / rms)


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------

def _to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) if x.dtype != np.float32 else x


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    duration = x.shape[0] / sr_from
    n_target = max(1, int(round(duration * sr_to)))
    resampled = np.interp(np.linspace(0, x.shape[0] - 1, n_target), np.arange(x.shape[0]), x)
    return resampled.astype(np.float32)


def _ensure_stereo_shape(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def load_audio(path: str, target_sr: int, target_dbfs: Optional[float] = None) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    data = _to_float32(data)
    if data.ndim == 1:
        data = _resample(data, sr, target_sr)
        data = _ensure_stereo_shape(data)
    else:
        chans = [_resample(data[:, c], sr, target_sr) for c in range(data.shape[1])]
        data = _ensure_stereo_shape(np.stack(chans, axis=1))
    if target_dbfs is not None:
        data = normalize_rms(data, target_dbfs)
    return data


def loop_to_length(x: np.ndarray, n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros((0, x.shape[1] if x.ndim > 1 else 1), dtype=np.float32)
    if x.shape[0] >= n_samples:
        return x[:n_samples]
    reps = int(np.ceil(n_samples / x.shape[0]))
    return np.tile(x, (reps, 1))[:n_samples]


def crossfade_concat(clips: list, sr: int, crossfade_ms: int) -> np.ndarray:
    audio, _ = crossfade_concat_with_offsets(clips, sr, crossfade_ms)
    return audio


def crossfade_concat_with_offsets(clips: list, sr: int, crossfade_ms: int):
    if not clips:
        return np.zeros((0, 2), dtype=np.float32), []

    n_fade = int(sr * crossfade_ms / 1000)
    out = clips[0]
    offsets = [0]
    for clip in clips[1:]:
        if n_fade > 0 and out.shape[0] > n_fade and clip.shape[0] > n_fade:
            fade_out = np.linspace(1, 0, n_fade, dtype=np.float32)[:, None]
            fade_in = np.linspace(0, 1, n_fade, dtype=np.float32)[:, None]
            head = out[:-n_fade]
            tail = out[-n_fade:] * fade_out + clip[:n_fade] * fade_in
            offsets.append(head.shape[0])
            out = np.concatenate([head, tail, clip[n_fade:]], axis=0)
        else:
            offsets.append(out.shape[0])
            out = np.concatenate([out, clip], axis=0)
    return out, offsets


def build_gain_envelope(n_samples: int, initial_gain: float, events: list) -> np.ndarray:
    env = np.empty(max(n_samples, 0), dtype=np.float32)
    cursor = 0
    current_gain = initial_gain
    for offset, target_gain, fade_samples in events:
        offset = max(0, min(offset, n_samples))
        if offset > cursor:
            env[cursor:offset] = current_gain
            cursor = offset
        fade_end = min(offset + max(fade_samples, 0), n_samples)
        if fade_end > offset:
            env[offset:fade_end] = np.linspace(current_gain, target_gain, fade_end - offset)
        current_gain = target_gain
        cursor = fade_end
    if cursor < n_samples:
        env[cursor:] = current_gain
    return env


def render_background_track(background_cfg: dict, markers: list, offsets: list,
                             total_len: int, sr: int, library: "SoundLibrary",
                             target_dbfs: float):
    default_fade_ms = background_cfg.get("fade_ms", 0)
    cur_file = background_cfg.get("file")
    cur_volume = background_cfg.get("volume", 5)
    cur_presence = background_cfg.get("presence", "on") == "on"

    state = [{"start": 0, "file": cur_file, "volume": cur_volume,
              "presence": cur_presence, "fade_ms": 0}]
    for marker in markers:
        offset = offsets[marker["clip_index"]] if marker["clip_index"] < len(offsets) else total_len
        if marker.get("file") is not None:
            cur_file = marker["file"]
        if marker.get("volume") is not None:
            cur_volume = marker["volume"]
        if marker.get("presence") is not None:
            cur_presence = marker["presence"] == "on"
        fade_ms = marker.get("fade_ms") if marker.get("fade_ms") is not None else default_fade_ms
        state.append({"start": offset, "file": cur_file, "volume": cur_volume,
                      "presence": cur_presence, "fade_ms": fade_ms})

    collapsed = []
    for s in state:
        if collapsed and collapsed[-1]["start"] == s["start"]:
            collapsed[-1] = s
        else:
            collapsed.append(s)
    state = collapsed

    for i, s in enumerate(state):
        s["end"] = state[i + 1]["start"] if i + 1 < len(state) else total_len

    if not any(s["file"] for s in state):
        return None

    runs = []
    for s in state:
        if runs and runs[-1][-1]["file"] == s["file"]:
            runs[-1].append(s)
        else:
            runs.append([s])

    out = np.zeros((total_len, 2), dtype=np.float32)
    for run_idx, run in enumerate(runs):
        file = run[0]["file"]
        if not file:
            continue

        run_start = run[0]["start"]
        run_end = run[-1]["end"]
        natural_len = run_end - run_start
        next_fade_ms = runs[run_idx + 1][0]["fade_ms"] if run_idx + 1 < len(runs) else 0
        tail_samples = int(sr * next_fade_ms / 1000)
        contrib_len = min(natural_len + tail_samples, total_len - run_start)
        if contrib_len <= 0:
            continue

        base_gain = volume_to_gain(run[0]["volume"]) if run[0]["presence"] else 0.0
        events = []
        for s in run[1:]:
            local_offset = s["start"] - run_start
            target_gain = volume_to_gain(s["volume"]) if s["presence"] else 0.0
            fade_samples = int(sr * s["fade_ms"] / 1000)
            events.append((local_offset, target_gain, fade_samples))

        env_natural = build_gain_envelope(natural_len, base_gain, events)
        if tail_samples > 0:
            last_gain = env_natural[-1] if natural_len > 0 else base_gain
            tail_env = np.linspace(last_gain, 0.0, tail_samples, dtype=np.float32)
            env = np.concatenate([env_natural, tail_env])
        else:
            env = env_natural
        env = env[:contrib_len]

        bg_audio = load_audio(library.path_for(file), sr, target_dbfs)
        bg_looped = loop_to_length(bg_audio, contrib_len)
        contribution = bg_looped[:contrib_len] * env[:, None]

        end_idx = run_start + contrib_len
        out[run_start:end_idx] += contribution[: end_idx - run_start]

    return out


# --------------------------------------------------------------------------
# Sound library resolution: "moan-{n}.wav" -> [moan-1.wav, moan-2.wav, ...]
# --------------------------------------------------------------------------

class SoundLibrary:
    def __init__(self, directory: str):
        self.directory = directory
        self._cache = {}

    def resolve(self, pattern: str) -> list:
        if pattern not in self._cache:
            glob_pattern = pattern.replace("{n}", "*") if "{n}" in pattern else pattern
            self._cache[pattern] = sorted(glob.glob(os.path.join(self.directory, glob_pattern)))
        return self._cache[pattern]

    def path_for(self, ref: str) -> str:
        return os.path.join(self.directory, ref)

    def pick(self, pattern: str, randomize: bool, index: int = 0) -> str:
        matches = self.resolve(pattern)
        return random.choice(matches) if randomize else matches[index % len(matches)]


# --------------------------------------------------------------------------
# Main vocalizer
# --------------------------------------------------------------------------

def _infer_kind(segment: dict) -> str:
    if "text" in segment:
        return "text"
    if "duration_ms" in segment:
        return "delay"
    if "ref" not in segment:
        return "background_update"
    return "file"


class Vocalizer:
    def __init__(self, config: VocalizerConfig):
        self.config = config
        self.library = SoundLibrary(config.sound_library_dir)
        self.model = None
        if VoxCPM is not None:
            self.model = VoxCPM.from_pretrained(config.voxcpm_model_id, load_denoiser=config.load_denoiser)

    def _resolve_generation_params(self, segment: dict, script_generation: dict) -> dict:
        params = {"cfg_value": self.config.cfg_value, "inference_timesteps": self.config.inference_timesteps}
        params.update(script_generation or {})
        for key in ("cfg_value", "inference_timesteps", "normalize", "denoise", "seed"):
            if key in segment:
                params[key] = segment[key]
        return params

    def _render_speech(self, segment: dict, script_generation: dict) -> np.ndarray:
        ref = segment.get("ref")
        text = segment["text"]
        voice_prompt = segment.get("voice_prompt")
        if voice_prompt:
            text = f"({voice_prompt}){text}"

        gen_params = self._resolve_generation_params(segment, script_generation)
        ref_path = self.library.path_for(ref) if ref else None
        prompt_ref = segment.get("prompt_ref")
        prompt_wav_path = self.library.path_for(prompt_ref) if prompt_ref else None
        prompt_text = segment.get("prompt_text") if prompt_ref else None

        wav = self.model.generate(
            text=text,
            reference_wav_path=ref_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            **gen_params,
        )
        sr = self.model.tts_model.sample_rate
        wav = _ensure_stereo_shape(_to_float32(np.asarray(wav)))
        if sr != self.config.output_sample_rate:
            wav = np.stack(
                [_resample(wav[:, c], sr, self.config.output_sample_rate) for c in range(wav.shape[1])], axis=1
            )
        wav = normalize_rms(wav, self.config.target_loudness_db)
        wav = wav * resolve_volume_spec(segment.get("volume"))
        return wav

    def _render_library_clip(self, segment: dict) -> np.ndarray:
        pattern = segment["ref"]
        randomize = segment.get("randomize", False)
        repeat = segment.get("repeat", 1)
        volume_jitter = segment.get("volume_jitter")
        volume = segment.get("volume", 5)

        clips = []
        for i in range(repeat):
            path = self.library.pick(pattern, randomize=randomize, index=i)
            clip = load_audio(path, self.config.output_sample_rate, self.config.target_loudness_db)
            clips.append(clip * resolve_volume_spec(volume_jitter if volume_jitter else volume))

        return crossfade_concat(clips, self.config.output_sample_rate, self.config.crossfade_ms)

    def _render_delay(self, segment: dict) -> np.ndarray:
        duration_ms = resolve_duration_ms(segment.get("duration_ms", 0))
        n_samples = max(0, int(self.config.output_sample_rate * duration_ms / 1000))
        return np.zeros((n_samples, 2), dtype=np.float32)

    def render_segment(self, segment: dict, script_generation: Optional[dict] = None) -> np.ndarray:
        kind = _infer_kind(segment)
        if kind == "text":
            return self._render_speech(segment, script_generation or {})
        if kind == "file":
            return self._render_library_clip(segment)
        return self._render_delay(segment)

    def render_json(self, jsonv: Union[list, dict]) -> np.ndarray:
        if isinstance(jsonv, list):
            segments = jsonv
            background_cfg = None
            script_generation = {}
        else:
            segments = jsonv.get("segments", [])
            background_cfg = jsonv.get("background")
            script_generation = jsonv.get("generation", {})

        clips = []
        markers = []
        for seg in segments:
            if _infer_kind(seg) == "background_update":
                markers.append({
                    "clip_index": len(clips),
                    "file": seg.get("file"),
                    "volume": seg.get("volume"),
                    "presence": seg.get("presence"),
                    "fade_ms": seg.get("fade_ms"),
                })
            else:
                clips.append(self.render_segment(seg, script_generation))

        audio, offsets = crossfade_concat_with_offsets(clips, self.config.output_sample_rate, self.config.crossfade_ms)
        total_len = audio.shape[0]

        if not background_cfg or total_len == 0:
            return audio

        bg_track = render_background_track(
            background_cfg, markers, offsets, total_len,
            self.config.output_sample_rate, self.library, self.config.target_loudness_db,
        )
        if bg_track is None:
            return audio

        mixed = audio + bg_track
        peak = float(np.max(np.abs(mixed))) or 1.0
        if peak > 1.0:
            mixed = mixed / peak
        return mixed

    def render_json_to_file(self, jsonv: Union[list, dict], output_path: str) -> str:
        audio = self.render_json(jsonv)
        sf.write(output_path, audio, self.config.output_sample_rate)
        return output_path
