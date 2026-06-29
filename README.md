# neural-crossfader

**Neural Crossfader** is a Python script that uses Stable Audio 3's inpainting capabilities to generate seamless transitions between two audio tracks.

Instead of relying solely on a traditional crossfade, the script creates a temporary overlap between two selected audio segments and asks Stable Audio to regenerate that region from a text prompt. The result is a machine learning-assisted transition that can sound more natural, or stylistic than a conventional fade.

## How it works

Before running the script, the user:

1. Sets the paths for `TRACK_A`, `TRACK_B`, and `OUTPUT_FILE`.
2. Specifies the segments to transition between (the end of Track A and the beginning of Track B).
3. Provides a text prompt describing the desired transition.

The script then:

4. Creates a temporary linear crossfade between the selected segments.
5. Uses Stable Audio 3 to inpaint (regenerate) the crossfade region according to the prompt.
6. Saves the completed transition as a new audio file.

Because only the overlap region is regenerated, the original audio before and after the transition remains unchanged.

## Requirements

* Python 3.10+
* PyTorch
* Torchaudio
* CUDA-compatible GPU (highly recommended)
* Stable Audio 3

## Example Prompt

```
infinite reverb tail, slow dissolve, guitar smearing into ambience,
dark resonant wash, fading to silence, fragile piano emerging from stillness,
no attack, no transients, no drums, no bass, no rhythm
```

## Notes

This project is intended as an experimental demonstration of AI-assisted music transitions using Stable Audio's inpainting functionality. Different prompts, overlap lengths, and source material can produce dramatically different results.
