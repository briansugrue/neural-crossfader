"""
-------------------------------------------------------------------------------
Downbeat Onset Detection

Given a track and an approximate time window (e.g. "somewhere between 0 and
2 seconds"), reports the precise timestamp of detected audio onsets within
that window — useful when a DJ tool's display only shows rounded values
(e.g. tenths of a second) but you need a more exact downbeat_secs value for
TRACK_BEATGRID.

Usage:
    python find_downbeat.py "C:\path\to\snoop_dogg.wav" 0.0 2.0

This searches for onsets between 0.0s and 2.0s and prints each one found,
with millisecond precision, strongest-first. The loudest/sharpest onset in
a mostly-silent intro is usually the actual downbeat.

Notes:
- This finds ONSETS, not specifically "beat 1 of the bar." If there's a pickup
  note, a vocal ad-lib, or a fill before the real downbeat, the first
  onset found may not be the one you want — cross-check by ear.
- Always confirm the reported timestamp actually sounds right by playing
  the track from that exact point in an editor with fine-grained seeking
  (e.g. Audacity), since onset detection is an estimate.
-------------------------------------------------------------------------------
"""

import sys
import librosa
import numpy as np

TARGET_SR = 44100


def find_onsets(path, window_start, window_end):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True, offset=window_start,
                          duration=window_end - window_start)

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units="frames", backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr) + window_start

    # Onset strength envelope, used to rank onsets by how sharp/loud they are.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr) + window_start

    if len(onset_times) == 0:
        print(f"No onsets detected between {window_start:.2f}s and {window_end:.2f}s. "
              f"Try widening the search window.")
        return

    print(f"Onsets detected between {window_start:.2f}s and {window_end:.2f}s:\n")

    scored = []
    for t in onset_times:
        nearest_idx = int(np.argmin(np.abs(onset_env_times - t)))
        strength = float(onset_env[nearest_idx])
        scored.append((t, strength))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    for t, strength in scored:
        print(f"  {t:.3f}s   (strength: {strength:.2f})")

    strongest_time, strongest_val = scored[0]
    print(f"\nStrongest onset (likely downbeat): {strongest_time:.3f}s")
    print(f"Suggested TRACK_BEATGRID downbeat_secs: {round(strongest_time, 3)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python find_downbeat.py <path_to_track> <window_start_secs> <window_end_secs>")
        sys.exit(1)

    track_path = sys.argv[1]
    start = float(sys.argv[2])
    end = float(sys.argv[3])
    find_onsets(track_path, start, end)

