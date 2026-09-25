# Neural Crossfader

Neural Crossfader is a Python script that uses Stable Audio 3's inpainting capabilities to generate AI-assisted transitions between two audio tracks.

Instead of relying solely on a traditional crossfade, the script builds a short overlap region between the outgoing and incoming track and asks Stable Audio to regenerate that region from a text prompt. The result is a machine learning-assisted transition that can sound more natural, or more stylistic, than a conventional fade — while the audio on either side of the transition is left completely untouched.

## How it works

Tracks are defined once in a small library (`TRACKS`), rather than passed in as one-off file paths, so you can reuse the same catalog across many transitions. For a given run, you tell the script which two tracks to transition between:

1. Add each track you want to use to the `TRACKS` dictionary, with a short key, its file path, and a genre label.
2. Set `TRACK_A_KEY` and `TRACK_B_KEY` to the two tracks you want to transition between.
3. (Optional) Set cue points (`TRACK_CUE_POINTS`) marking where in each track the transition should happen, and beatgrid info (`TRACK_BEATGRID`) if you want beatmatching. Tracks without cue points fall back to a default window; tracks without a beatgrid have their BPM auto-detected.
4. (Optional) Override the auto-selected prompt with your own via `PROMPT_OVERRIDE`.

The script then:

1. Extracts only the relevant regions of both tracks from disk (not the full files).
2. If beatmatching is enabled, time-stretches the transition region toward a shared tempo, using one of several tempo strategies (a smooth tempo ramp, a constant shared tempo, or native/unstretched tempo).
3. Builds a short linear crossfade between the two beatmatched regions as the seed for inpainting.
4. Picks a text prompt automatically based on the genre(s) of the two tracks (or uses your override), then uses Stable Audio 3 to inpaint the crossfade region according to that prompt.
5. Stitches the regenerated region back between the untouched original audio and saves the result.
6. Repeats the generation step to produce several numbered variations (`NUM_VERSIONS`) from the same underlying alignment, so you can compare takes without rebuilding the crossfade each time.

Because only the transition region is regenerated, the original audio before and after it is byte-for-byte unchanged.

## Configuration

Most behavior is controlled by constants near the top of the script:

| Setting | Purpose |
|---|---|
| `TRACK_A_KEY` / `TRACK_B_KEY` | Which two tracks (by key in `TRACKS`) to transition between |
| `NUM_VERSIONS` | How many inpainted variations to generate per run |
| `OUTPUT_DIR` | Folder the generated `.wav` files are saved to |
| `PROMPT_OVERRIDE` | Skip genre-based prompt selection and use a fixed prompt |
| `ENABLE_BEATMATCHING` | Whether to time-stretch tracks toward a shared tempo before blending |
| `TEMPO_STRATEGY` | `SMOOTH_RAMP`, `CONSTANT`, or `NATIVE_FLANKS` — how tempo is handled across the transition |
| `CROSSFADE_BARS` / `TARGET_DURATION_SECS` | Length of the crossfade and of the exported clip |

## Requirements

* Python 3.10+
* PyTorch
* Torchaudio
* librosa (for automatic BPM detection)
* pyrubberband (requires the `rubberband` CLI installed on your system)
* CUDA-compatible GPU (highly recommended)
* Stable Audio 3

## Example prompt

Prompts are normally chosen automatically based on the genres of the two tracks (see `GENRE_PROMPTS`), but you can also write your own via `PROMPT_OVERRIDE`:

```
infinite reverb tail, slow dissolve, guitar smearing into ambience,
dark resonant wash, fading to silence, fragile piano emerging from stillness,
no attack, no transients, no drums, no bass, no rhythm
```

## Notes

This project is intended as an experimental demonstration of AI-assisted music transitions using Stable Audio's inpainting functionality. Different prompts, overlap lengths, tempo strategies, and source material can produce dramatically different results.
