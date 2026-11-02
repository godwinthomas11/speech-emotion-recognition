# Feature reference (190 features)

All features are computed in `src/ser/features.py`.

**Pre-processing.** Audio is loaded at 22,050 Hz mono. Leading and trailing silence is trimmed (`top_db = 30`), and the clip is peak-normalised so its loudest sample is 1.

**Frames.** Every frame-level descriptor uses a 2,048-sample window (≈ 93 ms) with a 512-sample hop (≈ 23 ms). Each descriptor is then summarised over all frames of the clip.

Column names follow the pattern `<descriptor>[_<index>]_<statistic>`, for example `mfcc_03_std` is the standard deviation of the 3rd MFCC.

| Family | Columns | Count | Description | Why it relates to emotion |
|---|---|---|---|---|
| MFCC | `mfcc_01…20_mean/std` | 40 | Mel-frequency cepstral coefficients from a 128-band mel spectrogram | Spectral envelope (vocal-tract shape, timbre); changes with tension and articulation |
| MFCC Δ | `mfcc_d1_01…20_mean/std` | 40 | First temporal derivative of the MFCCs (9-frame window) | How fast timbre changes, related to speaking rate and articulation |
| MFCC ΔΔ | `mfcc_d2_01…20_mean/std` | 40 | Second derivative (acceleration) | Abruptness of changes |
| Spectral shape | `centroid`, `bandwidth`, `rolloff`, `flatness` × `mean/std` | 8 | Centre of mass of the spectrum, its spread, the frequency below which 85 % of energy lies, and how noise-like it is | "Bright" and tense voices (anger, fear) shift energy to higher frequencies |
| Spectral contrast | `contrast_01…07_mean/std` | 14 | Peak-minus-valley energy in 7 octave sub-bands | Clear harmonic structure vs breathy or noisy voice |
| Chroma | `chroma_01…12_mean/std` | 24 | Energy in the 12 pitch classes (C, C#, … B) | Tonal and pitch-class distribution |
| Tonnetz | `tonnetz_01…06_mean/std` | 12 | Tonal centroid features (fifths, minor and major thirds) | Harmonic relations, a coarse description of intonation |
| Energy | `rms_mean/std/max`, `zcr_mean/std` | 5 | Root-mean-square energy (relative to the clip's peak) and zero-crossing rate | Energy dynamics (bursty vs sustained) and noisiness/fricatives |
| Pitch | `f0_mean`, `f0_std`, `f0_min`, `f0_max`, `f0_range`, `voiced_ratio` | 6 | Fundamental frequency from pYIN (65–600 Hz), voiced frames only | Intonation: high-arousal emotions raise pitch and its variation |
| Duration | `duration_s` | 1 | Length of the trimmed clip in seconds | Speaking rate: sad speech is slower, angry and happy speech faster |

## Notes on specific features

- `f0_mean` is in Hz. `f0_min` and `f0_max` are the **5th and 95th percentiles**, not the true minimum and maximum, so single tracking errors don't dominate.
- `f0_std` and `f0_range` (95th − 5th percentile) are measured in **semitones**. Spread therefore has the same meaning for low (male) and high (female) voices.
- `voiced_ratio` is the fraction of frames that pYIN classifies as voiced.
- **Pitch fallback.** For 13 of the 1,440 clips (all from male actors; 7 surprised, 3 angry, 2 sad, 1 disgust), pYIN classifies every frame as unvoiced even though the clip is clearly speech. For those clips only, pitch statistics come from plain YIN on the louder half of the frames, after dropping values more than an octave from the clip median. Their `voiced_ratio` stays at its true value of 0.
- **RMS.** Because of peak normalisation, RMS does **not** measure absolute loudness, which depends on microphone gain. It describes how energy is distributed within the clip.

## Libraries

librosa 0.11.0 (McFee et al., 2015), NumPy 2.4, scikit-learn 1.9.
