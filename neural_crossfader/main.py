"""
-------------------------------------------------------------------------------
Single-Pair Neural Crossfader 

Generates multiple AI-assisted DJ transition takes for a single track pair.
Constructs the base audio alignment once and runs the inpainting generation 
in a loop to output numbered variations.
-------------------------------------------------------------------------------
"""

import os
import time

import torch
import torchaudio
import numpy as np
import librosa
import pyrubberband as pyrb
from stable_audio_3 import StableAudioModel


# SINGLE PAIR CONFIGURATION
# Set these to the keys of two tracks defined in TRACKS below.
TRACK_A_KEY = "track_1"
TRACK_B_KEY = "track_2"

# Number of distinct variations to generate
NUM_VERSIONS = 5

# Folder where the generated file will be saved
OUTPUT_DIR = r"C:\path\to\output\folder"

# Optional: Override automatic genre prompts with a custom text prompt.
PROMPT_OVERRIDE = None

TARGET_SR = 44100
ENABLE_BEATMATCHING = True

TEMPO_STRATEGY = "SMOOTH_RAMP" 

CROSSFADE_BARS = 2
BEATS_PER_BAR = 4
TARGET_DURATION_SECS = 20
RAMP_MICRO_XFADE_MS = 15


# LIBRARY DEFINITIONS

# Add one entry per track you own. The dict key is an arbitrary short
# identifier you choose (used elsewhere as TRACK_A_KEY / TRACK_B_KEY).
# "path"  -> absolute path to the track's audio file on disk.
# "genre" -> a free-text genre label; tracks sharing a genre combo will use
#            the matching prompt in GENRE_PROMPTS below.
TRACKS = {
    "track_1": {"path": r"C:\path\to\tracks\track_1.wav", "genre": "rap"},
    "track_2": {"path": r"C:\path\to\tracks\track_2.wav", "genre": "rap"},
    "track_3": {"path": r"C:\path\to\tracks\track_3.wav", "genre": "rap"},
    "track_4": {"path": r"C:\path\to\tracks\track_4.wav", "genre": "rap"},
    "track_5": {"path": r"C:\path\to\tracks\track_5.wav", "genre": "electronic"},
    "track_6": {"path": r"C:\path\to\tracks\track_6.wav", "genre": "electronic"},
    "track_7": {"path": r"C:\path\to\tracks\track_7.wav", "genre": "electronic"},
    "track_8": {"path": r"C:\path\to\tracks\track_8.wav", "genre": "electronic"},
    "track_9": {"path": r"C:\path\to\tracks\track_9.wav", "genre": "pop"},
    "track_10": {"path": r"C:\path\to\tracks\track_10.wav", "genre": "pop"},
    "track_11": {"path": r"C:\path\to\tracks\track_11.wav", "genre": "pop"},
    "track_12": {"path": r"C:\path\to\tracks\track_12.wav", "genre": "pop"},
}

# Optional per-track cue points marking the region to use for the transition.
# Use EITHER {"start_bar": X, "end_bar": Y} (requires a TRACK_BEATGRID entry
# for that track, since bar numbers need a BPM + downbeat to convert to
# seconds) OR {"start": X, "end": Y} in raw seconds. Any track without an
# entry here falls back to DEFAULT_TRACK_CUE.
TRACK_CUE_POINTS = {
    "track_1": {"start_bar": 1, "end_bar": 7},
    "track_2": {"start_bar": 1, "end_bar": 7},
    "track_3": {"start_bar": 1, "end_bar": 7},
    "track_4": {"start_bar": 102, "end_bar": 108},

    "track_5": {"start_bar": 10, "end_bar": 16},
    "track_6": {"start_bar": 1, "end_bar": 7},
    "track_7": {"start_bar": 18, "end_bar": 24},
    "track_8": {"start_bar": 120, "end_bar": 126},

    "track_9": {"start_bar": 1, "end_bar": 7},
    "track_10": {"start_bar": 107, "end_bar": 113},
    "track_11": {"start_bar": 6, "end_bar": 12},
    "track_12": {"start_bar": 99, "end_bar": 105},
}

# Optional per-track BPM + downbeat offset (in seconds), used to convert
# bar numbers above into seconds and to drive beatmatching. Get these from
# your DAW/analysis tool of choice. Tracks without an entry here will have
# their BPM auto-detected via librosa instead (see get_bpm()).
TRACK_BEATGRID = {
    "track_1": {"bpm": 95.0, "downbeat_secs": 0.05},
    "track_2": {"bpm": 138.0, "downbeat_secs": 0.27},
    "track_3": {"bpm": 92.0, "downbeat_secs": 0.05},
    "track_4": {"bpm": 98.0, "downbeat_secs": 0.56},

    "track_5": {"bpm": 117.0, "downbeat_secs": 2.07},
    "track_6": {"bpm": 78.0, "downbeat_secs": 0.50},
    "track_7": {"bpm": 120.0, "downbeat_secs": 0.0},
    "track_8": {"bpm": 137.5, "downbeat_secs": 0.37},

    "track_9": {"bpm": 119.0, "downbeat_secs": 0.13},
    "track_10": {"bpm": 128.0, "downbeat_secs": 0.33},
    "track_11": {"bpm": 125.0, "downbeat_secs": 4.56},
    "track_12": {"bpm": 128.0, "downbeat_secs": 0.12},
}

DEFAULT_TRACK_CUE = {"start": 60, "end": 85}
PAIR_OVERRIDES = {}

GENRE_PROMPTS = {
    frozenset({"rap"}): (
        "Clean boom-bap hip-hop beat with crisp kick and snare, rolling 808 sub bass, "
        "and warm synth pads, late-night studio vibe. "
        "Strictly instrumental."
    ),
    frozenset({"rap", "electronic"}): (
        "Sleek electro-trap bridge with driving 808 hi-hats, heavy sub bass, "
        "pulsing synth arpeggiators, and polished percussion fills, futuristic night atmosphere. "
        "Strictly instrumental."
    ),
    frozenset({"rap", "pop"}): (
        "Mid-tempo pop-hip-hop groove with bright kick and snare, warm bassline, "
        "shimmering electric piano chords, and clean rhythmic percussion, sunny summer vibe. "
        "Strictly instrumental."
    ),
    frozenset({"electronic"}): (
        "Driving 4-on-the-floor electronic track with rolling 16th-note hi-hats, deep sub bass, "
        "warm synth swells, and building tension, high-energy club vibe. "
        "Strictly instrumental."
    ),
    frozenset({"electronic", "pop"}): (
        "Upbeat dance-pop groove with punchy kick, sparkling arpeggiators, "
        "deep sidechained bass, and bright synth pads, euphoric festival vibe. "
        "Strictly instrumental."
    ),
    frozenset({"pop"}): (
        "Uplifting pop beat with crisp shaker, warm bassline, "
        "clean strummed acoustic guitar, and smooth piano chords, pristine studio vibe. "
        "Strictly instrumental."
    ),
}

FALLBACK_PROMPT = (
    "Pristine high-fidelity instrumental DJ transition with driving drums, solid sub bass, "
    "warm synth chords, and smooth rhythmic flow. Strictly instrumental, no vocals, no speech."
)

PLAUSIBLE_BPM_RANGE = (70.0, 180.0)


# FUNCTIONS
def bar_number_to_seconds(bar_number, downbeat_secs, bpm, beats_per_bar=4):
    bar_length_secs = beats_per_bar * 60.0 / bpm
    return downbeat_secs + (bar_number - 1) * bar_length_secs


def cue_entry_to_seconds(track_key, cue_entry):
    if "start_bar" in cue_entry and "end_bar" in cue_entry:
        beatgrid = TRACK_BEATGRID.get(track_key)
        if beatgrid is None:
            raise ValueError(
                f"Track '{track_key}' has a bar-based cue point "
                f"(start_bar={cue_entry['start_bar']}, end_bar={cue_entry['end_bar']}) "
                f"but no TRACK_BEATGRID entry."
            )
        start = bar_number_to_seconds(cue_entry["start_bar"], beatgrid["downbeat_secs"], beatgrid["bpm"])
        end = bar_number_to_seconds(cue_entry["end_bar"], beatgrid["downbeat_secs"], beatgrid["bpm"])
        return start, end
    return cue_entry["start"], cue_entry["end"]


def resolve_cue_points(track_a_key, track_b_key):
    pair_key = frozenset({track_a_key, track_b_key})
    override = PAIR_OVERRIDES.get(pair_key, {})
    used_fallback = False

    def cue_for(track_key):
        nonlocal used_fallback
        if track_key in override:
            raw_entry = override[track_key]
        elif track_key in TRACK_CUE_POINTS:
            raw_entry = TRACK_CUE_POINTS[track_key]
        else:
            used_fallback = True
            raw_entry = DEFAULT_TRACK_CUE
        return cue_entry_to_seconds(track_key, raw_entry)

    a_start, a_end = cue_for(track_a_key)
    b_start, b_end = cue_for(track_b_key)
    crossfade_override = override.get("crossfade_secs")

    return a_start, a_end, b_start, b_end, crossfade_override, used_fallback


def resolve_prompt(track_a_key, track_b_key):
    if PROMPT_OVERRIDE:
        return PROMPT_OVERRIDE
    genre_a = TRACKS[track_a_key]["genre"]
    genre_b = TRACKS[track_b_key]["genre"]
    key = frozenset({genre_a, genre_b})
    return GENRE_PROMPTS.get(key, FALLBACK_PROMPT)


def extract_region_lazy(path, start_secs, end_secs, target_sr):
    """Loads only the target audio region from disk and resamples it on-the-fly."""
    info = torchaudio.info(path)
    sr = info.sample_rate
    total_samples = info.num_frames
    
    start_sample = max(0, int(start_secs * sr))
    end_sample = min(total_samples, int(end_secs * sr))
    num_frames = end_sample - start_sample

    if end_secs * sr > total_samples:
        print(f"    Warning: end {end_secs}s exceeds track length ({total_samples/sr:.1f}s) — clamping.")

    waveform, sr = torchaudio.load(path, frame_offset=start_sample, num_frames=num_frames)
    
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    return waveform


def detect_bpm(waveform, sr):
    mono = waveform.mean(dim=0).cpu().numpy().astype(np.float32)
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    lo, hi = PLAUSIBLE_BPM_RANGE
    while bpm < lo:
        bpm *= 2.0
    while bpm > hi:
        bpm /= 2.0
    return bpm


def get_bpm(track_key, path, target_sr):
    """Lazily loads track audio for detection ONLY if beatgrid is missing."""
    if track_key in TRACK_BEATGRID:
        return TRACK_BEATGRID[track_key]["bpm"], "verified"
    waveform = extract_region_lazy(path, 0, 60, target_sr) # sample 1 min for bpm
    return detect_bpm(waveform, target_sr), "detected"


def time_stretch_waveform(waveform, rate, target_sr):
    if waveform.shape[-1] == 0:
        return waveform
    audio = waveform.cpu().numpy().astype(np.float32).T
    stretched = pyrb.time_stretch(audio, sr=target_sr, rate=rate, rbargs={"--crisp": "6"})
    return torch.from_numpy(stretched.T.copy())


def fit_length(waveform, target_samples):
    channels, num_samples = waveform.shape
    if num_samples == target_samples:
        return waveform
    elif num_samples > target_samples:
        return waveform[:, :target_samples]
    else:
        padding = target_samples - num_samples
        return torch.nn.functional.pad(waveform, (0, padding))


def resample_speed_ramp(waveform, speed_start, speed_end, target_samples):
    channels, num_samples = waveform.shape
    device = waveform.device
    
    j = torch.arange(target_samples, dtype=torch.float32, device=device)
    input_indices = speed_start * j + (speed_end - speed_start) * (j ** 2) / (2.0 * target_samples)
    input_indices = torch.clamp(input_indices, 0.0, float(num_samples - 1.0))
    
    idx_floor = input_indices.long()
    idx_ceil = torch.clamp(idx_floor + 1, 0, num_samples - 1)
    frac = (input_indices - idx_floor.float()).unsqueeze(0)
    
    val_floor = torch.gather(waveform, 1, idx_floor.unsqueeze(0).expand(channels, -1))
    val_ceil = torch.gather(waveform, 1, idx_ceil.unsqueeze(0).expand(channels, -1))
    
    return val_floor * (1.0 - frac) + val_ceil * frac


def bars_to_seconds(bars, bpm, beats_per_bar=BEATS_PER_BAR):
    return bars * beats_per_bar * 60.0 / bpm


def micro_crossfade_concat(pieces, xfade_samples):
    pieces = [p for p in pieces if p.shape[-1] > 0]
    if not pieces:
        return pieces[0] if pieces else None
    result = pieces[0]
    for next_piece in pieces[1:]:
        overlap = min(xfade_samples, result.shape[-1], next_piece.shape[-1])
        if overlap <= 0:
            result = torch.cat([result, next_piece], dim=-1)
            continue
        t = torch.linspace(0.0, 1.0, overlap, device=result.device)
        fade_out = torch.cos(t * torch.pi / 2)
        fade_in = torch.sin(t * torch.pi / 2)
        seam = result[:, -overlap:] * fade_out + next_piece[:, :overlap] * fade_in
        result = torch.cat([result[:, :-overlap], seam, next_piece[:, overlap:]], dim=-1)
    return result


def build_transition_audio(track_a_key, track_b_key, a_start, a_end, b_start, b_end,
                            crossfade_override, target_sr, enable_beatmatching,
                            crossfade_bars, target_duration_secs, tempo_strategy):
    path_a = TRACKS[track_a_key]["path"]
    path_b = TRACKS[track_b_key]["path"]

    bpm_a, bpm_a_source = get_bpm(track_a_key, path_a, target_sr)
    bpm_b, bpm_b_source = get_bpm(track_b_key, path_b, target_sr)
    target_bpm = (bpm_a + bpm_b) / 2.0

    rate_a, rate_b = 1.0, 1.0
    if enable_beatmatching:
        rate_a = target_bpm / bpm_a
        rate_b = target_bpm / bpm_b

    if crossfade_override is not None:
        crossfade_secs = crossfade_override
    else:
        crossfade_secs = bars_to_seconds(crossfade_bars, target_bpm)

    target_flank_secs = max(0.0, (target_duration_secs - crossfade_secs) / 2.0)

    if enable_beatmatching:
        if tempo_strategy == "SMOOTH_RAMP":
            s_avg_a = (1.0 + (bpm_b / bpm_a)) / 2.0
            raw_crossfade_dur_a = crossfade_secs * s_avg_a
            raw_flank_dur_a = target_flank_secs

            s_avg_b = ((bpm_a / bpm_b) + 1.0) / 2.0
            raw_crossfade_dur_b = crossfade_secs * s_avg_b
            raw_flank_dur_b = target_flank_secs
        elif tempo_strategy == "CONSTANT":
            raw_crossfade_dur_a = crossfade_secs * rate_a
            raw_flank_dur_a = target_flank_secs * rate_a

            raw_crossfade_dur_b = crossfade_secs * rate_b
            raw_flank_dur_b = target_flank_secs * rate_b
        else: # NATIVE_FLANKS
            raw_crossfade_dur_a = crossfade_secs * rate_a
            raw_flank_dur_a = target_flank_secs

            raw_crossfade_dur_b = crossfade_secs * rate_b
            raw_flank_dur_b = target_flank_secs
    else:
        raw_crossfade_dur_a = crossfade_secs
        raw_flank_dur_a = target_flank_secs
        raw_crossfade_dur_b = crossfade_secs
        raw_flank_dur_b = target_flank_secs

    dynamic_a_start = max(0.0, a_end - raw_crossfade_dur_a - raw_flank_dur_a)
    dynamic_b_end = b_start + raw_crossfade_dur_b + raw_flank_dur_b

    # Efficiently stream and resample ONLY the required windows from disk
    raw_segment_a = extract_region_lazy(path_a, dynamic_a_start, a_end, target_sr)
    raw_segment_b = extract_region_lazy(path_b, b_start, dynamic_b_end, target_sr)

    crossfade_samples = int(crossfade_secs * target_sr)

    if not enable_beatmatching:
        if crossfade_samples > raw_segment_a.shape[-1] or crossfade_samples > raw_segment_b.shape[-1]:
            raise ValueError(f"crossfade_secs ({crossfade_secs:.2f}s) exceeds segment length.")
        xfade_a = raw_segment_a[:, -crossfade_samples:]
        xfade_b = raw_segment_b[:, :crossfade_samples]
        body_a = raw_segment_a[:, :-crossfade_samples]
        body_b = raw_segment_b[:, crossfade_samples:]
    else:
        if tempo_strategy == "SMOOTH_RAMP":
            raw_crossfade_samples_a = int(raw_crossfade_dur_a * target_sr)
            raw_crossfade_samples_b = int(raw_crossfade_dur_b * target_sr)

            body_a = raw_segment_a[:, :-raw_crossfade_samples_a] if raw_crossfade_samples_a > 0 else raw_segment_a
            raw_crossfade_raw_a = raw_segment_a[:, -raw_crossfade_samples_a:]

            raw_crossfade_raw_b = raw_segment_b[:, :raw_crossfade_samples_b]
            body_b = raw_segment_b[:, raw_crossfade_samples_b:]

            xfade_a = resample_speed_ramp(raw_crossfade_raw_a, 1.0, bpm_b / bpm_a, crossfade_samples)
            xfade_b = resample_speed_ramp(raw_crossfade_raw_b, bpm_a / bpm_b, 1.0, crossfade_samples)

            print("Tempo Strategy: SMOOTH_RAMP (Flanks 100% native.)")
            print(f"Glides — A: {bpm_a:.1f} -> {bpm_b:.1f} BPM | B: {bpm_a:.1f} -> {bpm_b:.1f} BPM")

        elif tempo_strategy == "CONSTANT":
            raw_crossfade_samples_a = int(raw_crossfade_dur_a * target_sr)
            raw_crossfade_samples_b = int(raw_crossfade_dur_b * target_sr)

            raw_flank_a = raw_segment_a[:, :-raw_crossfade_samples_a] if raw_crossfade_samples_a > 0 else raw_segment_a
            raw_crossfade_raw_a = raw_segment_a[:, -raw_crossfade_samples_a:]
            raw_crossfade_raw_b = raw_segment_b[:, :raw_crossfade_samples_b]
            raw_flank_b = raw_segment_b[:, raw_crossfade_samples_b:]

            xfade_a = fit_length(time_stretch_waveform(raw_crossfade_raw_a, rate_a, target_sr), crossfade_samples)
            xfade_b = fit_length(time_stretch_waveform(raw_crossfade_raw_b, rate_b, target_sr), crossfade_samples)
            
            body_a = time_stretch_waveform(raw_flank_a, rate_a, target_sr)
            body_b = time_stretch_waveform(raw_flank_b, rate_b, target_sr)
            print(f"Tempo Strategy: CONSTANT (Entire transition stretched to shared tempo {target_bpm:.1f} BPM)")

        else:
            raw_crossfade_samples_a = int(raw_crossfade_dur_a * target_sr)
            raw_crossfade_samples_b = int(raw_crossfade_dur_b * target_sr)

            body_a = raw_segment_a[:, :-raw_crossfade_samples_a] if raw_crossfade_samples_a > 0 else raw_segment_a
            raw_crossfade_raw_a = raw_segment_a[:, -raw_crossfade_samples_a:]
            raw_crossfade_raw_b = raw_segment_b[:, :raw_crossfade_samples_b]
            body_b = raw_segment_b[:, raw_crossfade_samples_b:]

            xfade_a = fit_length(time_stretch_waveform(raw_crossfade_raw_a, rate_a, target_sr), crossfade_samples)
            xfade_b = fit_length(time_stretch_waveform(raw_crossfade_raw_b, rate_b, target_sr), crossfade_samples)
            print("Tempo Strategy: NATIVE_FLANKS")

    t = torch.linspace(0.0, 1.0, xfade_a.shape[-1], device=xfade_a.device)
    fade_out = torch.cos(t * torch.pi / 2)
    fade_in = torch.sin(t * torch.pi / 2)
    blended = xfade_a * fade_out + xfade_b * fade_in

    target_samples = int(target_duration_secs * target_sr)
    target_flank_samples = (target_samples - blended.shape[-1]) // 2

    if target_flank_samples < 0:
        blended = blended[:, :target_samples]
        body_a = body_a[:, :0]
        body_b = body_b[:, :0]
    else:
        symmetric_flank_samples = min(body_a.shape[-1], body_b.shape[-1], target_flank_samples)
        body_a = body_a[:, -symmetric_flank_samples:] if symmetric_flank_samples > 0 else body_a[:, :0]
        body_b = body_b[:, :symmetric_flank_samples] if symmetric_flank_samples > 0 else body_b[:, :0]
        print(f"Adjusted flanks to fit {target_duration_secs}s limit. "
              f"Kept {symmetric_flank_samples / target_sr:.2f}s of native audio on both sides.")

    xfade_samples_for_seams = int((RAMP_MICRO_XFADE_MS / 1000.0) * target_sr)
    seam_overlap_a = min(xfade_samples_for_seams, body_a.shape[-1], blended.shape[-1])
    stitched = micro_crossfade_concat([body_a, blended, body_b], xfade_samples_for_seams)

    gap_start_secs = (body_a.shape[-1] - seam_overlap_a) / target_sr
    gap_end_secs = gap_start_secs + (blended.shape[-1] / target_sr)
    
    stitched = fit_length(stitched, target_samples)
    duration = stitched.shape[-1] / target_sr

    return stitched, gap_start_secs, gap_end_secs, duration, crossfade_secs


# MAIN EXECUTION
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Stable Audio model...")
    torch.set_default_device("cuda")
    model = StableAudioModel.from_pretrained("medium")

    print(f"\nProcessing single pair: '{TRACK_A_KEY}' -> '{TRACK_B_KEY}'")

    a_start, a_end, b_start, b_end, crossfade_override, used_fallback = resolve_cue_points(
        TRACK_A_KEY, TRACK_B_KEY
    )
    prompt = resolve_prompt(TRACK_A_KEY, TRACK_B_KEY)

    if used_fallback:
        print("Note: Using default fallback cue points.")

    # Build underlying transition audio ONCE
    stitched, gap_start_secs, gap_end_secs, duration, crossfade_secs = build_transition_audio(
        TRACK_A_KEY, TRACK_B_KEY, a_start, a_end, b_start, b_end,
        crossfade_override, TARGET_SR, ENABLE_BEATMATCHING,
        CROSSFADE_BARS, TARGET_DURATION_SECS, TEMPO_STRATEGY
    )

    print(f"Total duration: {duration:.2f}s")
    print(f"Inpainting transition region: {gap_start_secs:.2f}s -> {gap_end_secs:.2f}s")
    print(f'Prompt: "{prompt}"')

    inpaint_audio = (TARGET_SR, stitched)

    # Pre-calculate invariant tensors/indices outside the generation loop
    stitched_cpu = stitched.cpu()
    gap_start_sample = int(gap_start_secs * TARGET_SR)
    gap_end_sample = int(gap_end_secs * TARGET_SR)

    flank_a_clean = stitched_cpu[:, :gap_start_sample]
    flank_b_clean = stitched_cpu[:, gap_end_sample:]

    # Generate multiple inpainting variations with Stable Audio
    print(f"\nGenerating {NUM_VERSIONS} variations...")
    with torch.inference_mode():
        for i in range(1, NUM_VERSIONS + 1):
            output_filename = f"{TRACK_A_KEY}__{TRACK_B_KEY}_{i}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            print(f"[{i}/{NUM_VERSIONS}] Generating take -> {output_filename}")
            start_time = time.time()

            audio = model.generate(
                inpaint_audio=inpaint_audio,
                inpaint_mask_start_seconds=gap_start_secs,
                inpaint_mask_end_seconds=gap_end_secs,
                prompt=prompt,
                duration=duration,
            )
            print(f"    Completed in {time.time() - start_time:.2f}s")

            audio_out = audio.cpu().squeeze(0)
            clean_output = audio_out.clone()
            
            # Stitch back original PCM flanks
            clean_output[:, :gap_start_sample] = flank_a_clean
            clean_output[:, gap_end_sample:] = flank_b_clean

            torchaudio.save(output_path, clean_output, sample_rate=TARGET_SR)
            print(f"    Saved: {output_path}\n")

    print("All variations generated successfully!")


if __name__ == "__main__":
    main()