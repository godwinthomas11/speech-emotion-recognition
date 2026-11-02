"""One shared figure style so every plot in the project looks consistent."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import FIGURES_DIR

# Fixed colour per model, used in every comparison figure.
# Baselines are grey so the three solutions stand out.
SOLUTION_COLORS = {
    "S1: MI + SVM": "#2a78d6",
    "S2: RFE + RF": "#eb6834",
    "S3: GA + MLP": "#1baf7a",
    "Baseline": "#9a9893",
}
GREY = "#9a9893"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name):
    """Save to figures/<name>.png at 300 dpi."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------
# Plots reused by the three solution notebooks
# --------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, labels, title, ax=None):
    """Row-normalised confusion matrix (each row sums to 100 %)."""
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=100)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if cm[i, j] > 55 else INK)
    names = [l.capitalize() for l in labels]
    ax.set_xticks(range(len(labels)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(title)
    return ax


def plot_k_curve(inner, outer, color, label, baseline=None, ax=None):
    """Macro-F1 against the number of selected features (mean ± 1 std over folds)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4))
    o = outer.groupby("k").test_macro_f1.agg(["mean", "std"])
    ax.fill_between(o.index, o["mean"] - o["std"], o["mean"] + o["std"], color=color, alpha=0.15, linewidth=0)
    ax.plot(o.index, o["mean"], color=color, lw=2, marker="o", ms=5, label=f"{label}: unseen test actors")
    i = inner.groupby("k").inner_macro_f1.mean()
    ax.plot(i.index, i.values, color=color, lw=1.5, ls="--", marker="o", ms=4, alpha=0.7,
            label=f"{label}: inner validation (used to pick k)")
    if baseline is not None:
        ax.axhline(baseline, color=GREY, lw=1.5, ls=":", label=f"All 190 features (baseline) = {baseline:.3f}")
    ax.set_xlabel("Number of selected features (k)")
    ax.set_ylabel("Macro-F1")
    ax.legend(loc="lower right", fontsize=8.5)
    return ax


def plot_selection_frequency(selected_per_fold, top=25, color="#2a78d6", title="", ax=None):
    """How many of the 6 outer folds picked each feature (selection stability)."""
    import pandas as pd

    counts = pd.Series([f for fold in selected_per_fold for f in fold]).value_counts().head(top)[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.26 * len(counts) + 1))
    ax.barh(counts.index, counts.values, color=color, height=0.65)
    ax.set_xlim(0, len(selected_per_fold) + 0.2)
    ax.set_xlabel(f"Folds that selected the feature (out of {len(selected_per_fold)})")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title)
    return ax
