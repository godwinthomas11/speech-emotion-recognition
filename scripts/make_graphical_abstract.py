"""Draw the graphical abstract for the paper (figures/graphical_abstract.png).

Run after notebooks 03-06; every number is read from results/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ser.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from ser.plots import GREY, INK, INK_2, SOLUTION_COLORS  # noqa: E402

folds = {n: pd.read_csv(RESULTS_DIR / f"{s}_folds.csv") for n, s in [
    ("S1", "s1"), ("S2", "s2"), ("S3", "s3"),
    ("B_SVM", "baseline_svm"), ("B_RF", "baseline_rf"), ("B_MLP", "baseline_mlp")]}
leak = pd.read_csv(RESULTS_DIR / "leakage_check.csv")


def box(ax, x, y, w, h, title, body, color, title_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="white", edgecolor=color, linewidth=1.6))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.075), w, 0.075, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor=color, edgecolor=color, linewidth=1.6))
    ax.text(x + w / 2, y + h - 0.0375, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + (h - 0.075) / 2, body, ha="center", va="center", fontsize=8.6,
            color=INK, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color="#8a8984", linewidth=1.5))


fig = plt.figure(figsize=(13, 6.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(0.5, 0.955, "Speaker-Independent Speech Emotion Recognition with Acoustic Feature Selection",
        ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(0.5, 0.91, "RAVDESS · 1,440 clips · 24 actors · 8 emotions · tested only on actors never seen in training",
        ha="center", fontsize=10, color=INK_2)

# Column 1: data and features
box(ax, 0.02, 0.50, 0.17, 0.34, "Speech clips",
    "RAVDESS speech\n12 male + 12 female actors\n\nneutral · calm · happy · sad\nangry · fearful · disgust · surprised", "#52514e")
box(ax, 0.02, 0.08, 0.17, 0.34, "190 acoustic features",
    "MFCC + Δ + ΔΔ (120)\nchroma · tonnetz (36)\nspectral shape/contrast (22)\npitch (pYIN) · energy (11)\nduration (1)", "#52514e")
arrow(ax, 0.105, 0.50, 0.105, 0.43)

# Column 2: three solutions
labels = [("S1", "S1: MI + SVM", "Filter\nMutual Information ranking\n→ SVM (RBF kernel)"),
          ("S2", "S2: RFE + RF", "Wrapper\nRecursive Feature Elimination\n→ Random Forest"),
          ("S3", "S3: GA + MLP", "Metaheuristic\nGenetic Algorithm subset search\n→ Multilayer Perceptron")]
ys = [0.62, 0.36, 0.10]
for (key, name, body), y in zip(labels, ys):
    box(ax, 0.27, y, 0.22, 0.22, name, body, SOLUTION_COLORS[name])
    arrow(ax, 0.19, 0.25, 0.268, y + 0.11)

# Column 3: results bars
rx, rw = 0.56, 0.24
ax.text(rx + rw / 2, 0.85, "Macro-F1 on unseen actors", ha="center", fontsize=11, fontweight="bold", color=INK)
pairs = [("S1", "B_SVM", "S1: MI + SVM", "SVM"), ("S2", "B_RF", "S2: RFE + RF", "RF"),
         ("S3", "B_MLP", "S3: GA + MLP", "MLP")]
vmax = 0.8
for (s, b, name, short), y in zip(pairs, ys):
    sf, bf = folds[s], folds[b]
    for j, (val, color, lab) in enumerate([
            (bf.macro_f1.mean(), GREY, f"{short}, all 190 features"),
            (sf.macro_f1.mean(), SOLUTION_COLORS[name], f"{name.split(': ')[1]}, {sf.n_features.mean():.0f} features")]):
        yy = y + 0.13 - j * 0.075
        ax.add_patch(FancyBboxPatch((rx, yy), rw * val / vmax, 0.05, boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor=color, edgecolor="none"))
        ax.text(rx + rw * val / vmax + 0.006, yy + 0.025, f"{val:.3f}", va="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(rx, yy + 0.064, lab, fontsize=8, color=INK_2)
    arrow(ax, 0.49, y + 0.11, 0.553, y + 0.11)

# Column 4: key messages
best = max(["S1", "S2", "S3"], key=lambda k: folds[k].macro_f1.mean())
best_name = {"S1": "S1: MI + SVM", "S2": "S2: RFE + RF", "S3": "S3: GA + MLP"}[best]
gap = leak.pivot(index="Model", columns="Protocol", values="Accuracy")
rand_col = [c for c in gap.columns if c.startswith("Random")][0]
spk_col = [c for c in gap.columns if c.startswith("Speaker")][0]
svm_gap = gap.loc["SVM (RBF)", rand_col] - gap.loc["SVM (RBF)", spk_col]
msgs = (f"Best: {best_name}\n"
        f"accuracy {folds[best].accuracy.mean():.1%}, macro-F1 {folds[best].macro_f1.mean():.3f}\n"
        f"(chance = 12.5 %)\n\n"
        f"Feature selection kept\n{min(folds[k].n_features.mean() for k in ['S1', 'S2', 'S3']):.0f}–"
        f"{max(folds[k].n_features.mean() for k in ['S1', 'S2', 'S3']):.0f} of 190 features\n\n"
        f"Random train/test splits\noverstate SVM accuracy by\n{svm_gap * 100:.0f} percentage points")
box(ax, 0.83, 0.10, 0.155, 0.74, "Key findings", msgs, "#0d366b")

FIGURES_DIR.mkdir(exist_ok=True)
out = FIGURES_DIR / "graphical_abstract.png"
fig.savefig(out, dpi=300, facecolor="white")
print("saved", out)
