"""Turn one audio clip into a fixed-length vector of acoustic features.

Each descriptor is first measured on short overlapping frames
(2048-sample window ≈ 93 ms, 512-sample hop ≈ 23 ms at 22.05 kHz) and then
summarised over the whole clip with statistics such as the mean and the
standard deviation. Every clip therefore gives the same number of features,
whatever its length.
"""
import warnings

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import SAMPLE_RATE, TRIM_TOP_DB

N_FFT = 2048
HOP = 512
N_MFCC = 20
F0_MIN, F0_MAX = 65.0, 600.0  # covers adult male and female speaking pitch


def load_clip(path, sr=SAMPLE_RATE):
    """Load as mono, cut silent edges and scale the loudest sample to 1."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def _stats(name, frames, which=("mean", "std")):
    """Summarise a (n_coeffs, n_frames) matrix row by row."""
    frames = np.atleast_2d(frames)
    out = {}
    for i, row in enumerate(frames):
        tag = name if frames.shape[0] == 1 else f"{name}_{i + 1:02d}"
        if "mean" in which:
            out[f"{tag}_mean"] = float(np.mean(row))
        if "std" in which:
            out[f"{tag}_std"] = float(np.std(row))
        if "max" in which:
            out[f"{tag}_max"] = float(np.max(row))
    return out


def _pitch_features(y, sr):
    """Pitch contour with pYIN; statistics are taken over voiced frames only."""
    f0_all, voiced, _ = librosa.pyin(
        y, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=N_FFT, hop_length=HOP
    )
    f0 = f0_all[voiced & ~np.isnan(f0_all)]
    if f0.size < 5:
        # On a few loud, strained clips (13 of 1,440) pYIN's voicing decision
        # rejects every frame although the clip is clearly speech. For those we
        # fall back to plain YIN on the louder half of the frames, where speech is voiced.
        f0_yin = librosa.yin(y, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=N_FFT, hop_length=HOP)
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
        n = min(len(f0_yin), len(rms))
        f0 = f0_yin[:n][rms[:n] >= np.median(rms[:n])]
        med = np.median(f0)
        f0 = f0[(f0 > med / 2) & (f0 < med * 2)]  # drop octave jumps
    # Work in semitones so male and female voices are on a comparable scale
    # for spread, while the mean still keeps the absolute level in Hz.
    semis = 12 * np.log2(f0 / F0_MIN)
    return {
        "f0_mean": float(np.mean(f0)),
        "f0_std": float(np.std(semis)),
        "f0_min": float(np.percentile(f0, 5)),
        "f0_max": float(np.percentile(f0, 95)),
        "f0_range": float(np.percentile(semis, 95) - np.percentile(semis, 5)),
        "voiced_ratio": float(voiced.mean()),
    }


def extract_features(path):
    """Return a dict of named features for one WAV file."""
    y = load_clip(path)
    sr = SAMPLE_RATE
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    feats = {}
    feats |= _stats("mfcc", mfcc)
    feats |= _stats("mfcc_d1", librosa.feature.delta(mfcc))
    feats |= _stats("mfcc_d2", librosa.feature.delta(mfcc, order=2))

    feats |= _stats("centroid", librosa.feature.spectral_centroid(S=S, sr=sr))
    feats |= _stats("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr))
    feats |= _stats("rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr))
    feats |= _stats("flatness", librosa.feature.spectral_flatness(S=S))
    feats |= _stats("contrast", librosa.feature.spectral_contrast(S=S, sr=sr))

    chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)
    feats |= _stats("chroma", chroma)
    with warnings.catch_warnings():  # tonnetz warns on very short clips only
        warnings.simplefilter("ignore")
        feats |= _stats("tonnetz", librosa.feature.tonnetz(chroma=chroma))

    feats |= _stats("rms", librosa.feature.rms(S=S), which=("mean", "std", "max"))
    feats |= _stats("zcr", librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP))
    feats |= _pitch_features(y, sr)
    feats["duration_s"] = len(y) / sr
    return feats


def feature_group(name):
    """Map a feature name to its family, used for plots and the report."""
    prefix_map = [
        ("mfcc_d1", "MFCC Δ"), ("mfcc_d2", "MFCC ΔΔ"), ("mfcc", "MFCC"),
        ("chroma", "Chroma"), ("tonnetz", "Tonnetz"), ("contrast", "Spectral contrast"),
        ("centroid", "Spectral shape"), ("bandwidth", "Spectral shape"),
        ("rolloff", "Spectral shape"), ("flatness", "Spectral shape"),
        ("rms", "Energy"), ("zcr", "Energy"),
        ("f0", "Pitch"), ("voiced", "Pitch"), ("duration", "Duration"),
    ]
    for prefix, group in prefix_map:
        if name.startswith(prefix):
            return group
    return "Other"


def extract_all(meta: pd.DataFrame, n_jobs=4) -> pd.DataFrame:
    """Extract features for every file in the metadata table (in parallel)."""
    rows = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(extract_features)(p) for p in meta["path"]
    )
    feats = pd.DataFrame(rows)
    keep = ["file", "actor", "gender", "emotion_id", "emotion", "intensity",
            "statement", "repetition"]
    return pd.concat([meta[keep].reset_index(drop=True), feats], axis=1)
