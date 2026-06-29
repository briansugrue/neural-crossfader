"""
-------------------------------------------------------------------------------
Neural Crossfader

Creates a seamless transition between two audio tracks using Stable Audio's
inpainting capabilities.

Overview:
1. Load two input tracks.
2. Extract user-defined segments from each track.
3. Create a temporary linear crossfade.
4. Regenerate the crossfade region using a text prompt.
5. Save the resulting transition as a new audio file.
-------------------------------------------------------------------------------
"""

import torch
import torchaudio
import time
from stable_audio_3 import StableAudioModel

# Input audio files 
TRACK_A     = r"[PATH TO INPUT FILE A]"
TRACK_B     = r"[PATH TO INPUT FILE B]"
OUTPUT_FILE = r"[PATH TO OUTPUT AUDIO FILE]"

PROMPT      = """[Text prompt used by Stable Audio to generate the transition. The model will only 
                  modify the masked (crossfade) region while following this description]"""

# Sections to extract from each track

# Region to take from Track A (the tail going into the transition)
TRACK_A_START = 98
TRACK_A_END   = 118

# Region to take from Track B (the head coming out of the transition)
TRACK_B_START = 0
TRACK_B_END   = 20

# How many seconds of overlap to crossfade over (the inpainted region)
# This sits at the join point and must be <= both segment lengths
CROSSFADE_SECS = 10

TARGET_SR = 44100

# Load Stable Audio model
torch.set_default_device("cuda")
model = StableAudioModel.from_pretrained("medium")

# Functions to load audio and extract region
def load_and_resample(path, target_sr):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    return waveform

def extract_region(waveform, start_secs, end_secs, sr):
    start_sample = int(start_secs * sr)
    end_sample   = int(end_secs   * sr)
    total        = waveform.shape[-1]
    if end_sample > total:
        print(f"Warning: end {end_secs}s exceeds track length ({total/sr:.1f}s) — clamping.")
        end_sample = total
    return waveform[:, start_sample:end_sample]

# Load and extract
wav_a = load_and_resample(TRACK_A, TARGET_SR)
wav_b = load_and_resample(TRACK_B, TARGET_SR)

segment_a = extract_region(wav_a, TRACK_A_START, TRACK_A_END, TARGET_SR)
segment_b = extract_region(wav_b, TRACK_B_START, TRACK_B_END, TARGET_SR)

tail_secs = TRACK_A_END - TRACK_A_START
head_secs = TRACK_B_END - TRACK_B_START

crossfade_samples = int(CROSSFADE_SECS * TARGET_SR)


# Build a linear crossfade blend at the overlap region

# Tail end of A and head start of B, both trimmed to crossfade length
xfade_a = segment_a[:, -crossfade_samples:]   # last CROSSFADE_SECS of A
xfade_b = segment_b[:,  :crossfade_samples]   # first CROSSFADE_SECS of B

# Linear fade: A fades out, B fades in
fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device="cpu")
fade_in  = torch.linspace(0.0, 1.0, crossfade_samples, device="cpu")

blended = xfade_a * fade_out + xfade_b * fade_in   # smooth mix of both tracks

# Stitch: [A without its tail] + [blended overlap] + [B without its head]
body_a = segment_a[:, :-crossfade_samples]   # A up to the crossfade region
body_b = segment_b[:, crossfade_samples:]    # B after the crossfade region

stitched = torch.cat([body_a, blended, body_b], dim=-1)

# Inpaint mask covers just the blended region 
gap_start_secs = tail_secs - CROSSFADE_SECS   # start of overlap within stitched
gap_end_secs   = tail_secs                    # end of overlap

duration = stitched.shape[-1] / TARGET_SR
print(f"Segment A: {tail_secs}s  |  Segment B: {head_secs}s  |  "
      f"Crossfade region: {CROSSFADE_SECS}s")
print(f"Total duration: {duration:.1f}s  |  "
      f"Inpainting from {gap_start_secs}s → {gap_end_secs}s")

inpaint_audio = (TARGET_SR, stitched)

# Generate audio and start timer
start = time.time()

audio = model.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=gap_start_secs,
    inpaint_mask_end_seconds=gap_end_secs,
    prompt=PROMPT,
    duration=duration,
)

print(f"Generation time: {time.time() - start:.2f}s")

# Save output audio file
audio = audio.cpu().squeeze(0)
torchaudio.save(OUTPUT_FILE, audio, sample_rate=TARGET_SR)
print(f"Done! Saved to {OUTPUT_FILE}")