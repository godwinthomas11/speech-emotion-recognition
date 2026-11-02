"""Project-wide settings: paths, random seed and label names.

Everything that more than one notebook needs lives here, so a value is
changed in one place only.
"""
from pathlib import Path

# Folders (resolved from this file, so notebooks work from any directory)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_CSV = DATA_DIR / "features.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# One seed for every random step (CV shuffles, models, the GA)
RANDOM_STATE = 42

# Audio settings
SAMPLE_RATE = 22_050
TRIM_TOP_DB = 30  # how quiet (in dB below peak) counts as silence when trimming

# RAVDESS filename codes -> readable names
EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
INTENSITY = {1: "normal", 2: "strong"}
STATEMENT = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}

EMOTION_ORDER = list(EMOTIONS.values())
