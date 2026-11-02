"""Read RAVDESS file names into a tidy metadata table.

A RAVDESS file name such as 03-01-06-01-02-01-12.wav holds seven codes:
modality - vocal channel - emotion - intensity - statement - repetition - actor.
Odd actor numbers are male, even numbers are female.
"""
from pathlib import Path

import pandas as pd

from .config import EMOTIONS, INTENSITY, RAW_DIR, STATEMENT


def parse_filename(path: Path) -> dict:
    """Turn one RAVDESS file name into a dictionary of labels."""
    codes = [int(part) for part in path.stem.split("-")]
    if len(codes) != 7:
        raise ValueError(f"Unexpected RAVDESS file name: {path.name}")
    modality, channel, emotion, intensity, statement, repetition, actor = codes
    if modality != 3 or channel != 1:
        raise ValueError(f"{path.name} is not an audio-only speech file")
    return {
        "file": path.name,
        "path": str(path),
        "actor": actor,
        "gender": "male" if actor % 2 == 1 else "female",
        "emotion_id": emotion,
        "emotion": EMOTIONS[emotion],
        "intensity": INTENSITY[intensity],
        "statement": STATEMENT[statement],
        "repetition": repetition,
    }


def load_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return one row per WAV file, sorted by actor and file name."""
    files = sorted(Path(raw_dir).rglob("*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No WAV files under {raw_dir}. See README.md for download steps."
        )
    rows = [parse_filename(f) for f in files]
    return pd.DataFrame(rows).sort_values(["actor", "file"], ignore_index=True)
