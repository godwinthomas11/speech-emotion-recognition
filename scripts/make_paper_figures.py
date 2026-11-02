"""Compact, single-column figures for the two-column paper (figures/paper_*.png).

Run after notebooks 03-06. Width 3.4 in, 8-pt text, 300 dpi.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ser.config import EMOTION_ORDER, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from ser.plots import GREY, INK, INK_2, SEQ_CMAP, SOLUTION_COLORS, set_style  # noqa: E402

set_style()
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8})
W = 3.4
S = {"s1": "S1: MI + SVM", "s2": "S2: RFE + RF", "s3": "S3: GA + MLP"}
B = {"s1": "baseline_svm", "s2": "baseline_rf", "s3": "baseline_mlp"}
CLF = {"s1": "SVM", "s2": "RF", "s3": "MLP"}
folds = {k: pd.read_csv(RESULTS_DIR / f"{k}_folds.csv") for k in list(S) + list(B.values())}
oof = {k: pd.read_csv(RESULTS_DIR / f"{k}_oof.csv").sort_values("file", ignore_index=True) for k in list(S) + list(B.values())}


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---- Fig 1: pipeline ----------------------------------------------------
fig = plt.figure(figsize=(W, 2.35))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def node(x, y, w, h, text, color, fill="white", bold=False, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                facecolor=fill, edgecolor=color, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color="#8a8984", lw=0.9))


node(0.01, 0.62, 0.20, 0.30, "RAVDESS\nspeech\n1,440 clips\n24 actors", INK_2, fs=6.0)
node(0.01, 0.10, 0.20, 0.42, "Pre-processing\nmono, 22.05 kHz\ntrim silence\npeak-normalise", INK_2, fs=5.6)
arr(0.11, 0.62, 0.11, 0.525)
node(0.255, 0.10, 0.19, 0.82, "Feature\nextraction\n\nMFCC+Δ+ΔΔ\nspectral\nchroma\ntonnetz\npitch, energy\nduration\n\n190-D", INK_2, fs=5.8)
arr(0.21, 0.31, 0.258, 0.31)
ys = [0.68, 0.39, 0.10]
for (k, name), y, txt in zip(S.items(), ys, ["MI ranking\n(filter)", "RFE with RF\n(wrapper)", "Genetic\nAlgorithm"]):
    node(0.50, y, 0.17, 0.24, txt, SOLUTION_COLORS[name])
    node(0.72, y, 0.12, 0.24, CLF[k], SOLUTION_COLORS[name], fill=SOLUTION_COLORS[name] + "22", bold=True)
    arr(0.44, 0.51, 0.498, y + 0.12)
    arr(0.67, y + 0.12, 0.718, y + 0.12)
    arr(0.84, y + 0.12, 0.878, 0.51)
node(0.875, 0.25, 0.12, 0.52, "Test on\nunseen\nactors\n\nAcc.\nMacro-F1\nUAR", "#0d366b", fs=5.4)
ax.add_patch(FancyBboxPatch((0.475, 0.03), 0.39, 0.93, boxstyle="round,pad=0,rounding_size=0.02",
                            facecolor="none", edgecolor="#c3c2b7", linewidth=0.8, linestyle=(0, (3, 2))))
ax.text(0.67, 0.965, "nested, actor-wise CV (refit per training fold)", ha="center", va="bottom", fontsize=5.8, color=INK_2)
save(fig, "paper_fig1_pipeline")

# ---- Fig 2: solutions vs baselines ---------------------------------------
fig, ax = plt.subplots(figsize=(W, 2.1))
metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("uar", "UAR")]
x = np.arange(3)
for i, (k, name) in enumerate(S.items()):
    for j, (src, color) in enumerate([(B[k], GREY), (k, SOLUTION_COLORS[name])]):
        v = folds[src].macro_f1
        xx = i + (j - 0.5) * 0.38
        ax.bar(xx, v.mean(), width=0.36, color=color, edgecolor="white", linewidth=1)
        ax.errorbar(xx, v.mean(), yerr=v.std(), color=INK_2, capsize=2, lw=0.8)
        ax.text(xx, v.mean() + v.std() + 0.012, f"{v.mean():.3f}", ha="center", fontsize=6.2, color=INK_2)
ax.set_xticks(x, ["SVM", "Random Forest", "MLP"])
ax.set_ylabel("Macro-F1 (unseen actors)")
ax.set_ylim(0, max(folds[k].macro_f1.mean() + folds[k].macro_f1.std() for k in folds) + 0.1)
ax.grid(axis="x", visible=False)
ax.legend(handles=[Patch(color=GREY, label="All 190 features")] +
          [Patch(color=SOLUTION_COLORS[n], label=n) for n in S.values()],
          ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
save(fig, "paper_fig2_vs_baseline")

# ---- Fig 3: k curves (S1 and S2) -----------------------------------------
fig, ax = plt.subplots(figsize=(W, 2.0))
for k in ["s1", "s2"]:
    c = pd.read_csv(RESULTS_DIR / f"{k}_outer_curve.csv").groupby("k").test_macro_f1.agg(["mean", "std"])
    ax.fill_between(c.index, c["mean"] - c["std"], c["mean"] + c["std"], color=SOLUTION_COLORS[S[k]], alpha=0.12, lw=0)
    ax.plot(c.index, c["mean"], color=SOLUTION_COLORS[S[k]], marker="o", ms=3, lw=1.4, label=S[k])
ax.set_xlabel("Number of selected features k")
ax.set_ylabel("Macro-F1 (unseen actors)")
ax.legend(loc="lower right")
save(fig, "paper_fig3_k_curve")

# ---- Fig 4: GA convergence (mean over the 6 folds, band = min-max) -------
from matplotlib.ticker import MaxNLocator  # noqa: E402

h = pd.read_csv(RESULTS_DIR / "s3_ga_history.csv")
gens = range(int(h.generation.max()) + 1)
piv = lambda col: h.pivot(index="generation", columns="fold", values=col).reindex(gens).ffill()  # noqa: E731
best_f, mean_f, size = piv("best_fitness"), piv("mean_fitness"), piv("best_n_features")
fig, ax1 = plt.subplots(figsize=(W, 1.9))
c = SOLUTION_COLORS[S["s3"]]
ax1.fill_between(best_f.index, best_f.min(1), best_f.max(1), color=c, alpha=0.15, lw=0)
ax1.plot(best_f.index, best_f.mean(1), color=c, lw=1.6, label="best individual")
ax1.plot(mean_f.index, mean_f.mean(1), color=GREY, lw=1.3, ls="--", label="population mean")
ax1.set_xlabel("Generation"); ax1.set_ylabel("Fitness (mean of 6 folds)")
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.legend(loc="lower right")
save(fig, "paper_fig4_ga_convergence")

# ---- Fig 5: confusion matrix of best solution -----------------------------
best = max(S, key=lambda k: folds[k].macro_f1.mean())
o = oof[best]
cm = confusion_matrix(o.y_true, o.y_pred, labels=EMOTION_ORDER, normalize="true") * 100
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=100); ax.grid(False)
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=6.3, color="white" if cm[i, j] > 55 else INK)
names = [e[:4].capitalize() + "." if len(e) > 5 else e.capitalize() for e in EMOTION_ORDER]
ax.set_xticks(range(8), names, rotation=45, ha="right"); ax.set_yticks(range(8), names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
save(fig, "paper_fig5_confusion_best")

# ---- Fig 6: per-class F1 ----------------------------------------------------
pc = pd.DataFrame({S[k]: f1_score(oof[k].y_true, oof[k].y_pred, labels=EMOTION_ORDER, average=None) for k in S},
                  index=[e.capitalize() for e in EMOTION_ORDER])
fig, ax = plt.subplots(figsize=(W, 2.0))
xx = np.arange(8)
for i, n in enumerate(pc.columns):
    ax.bar(xx + (i - 1) * 0.26, pc[n], width=0.25, color=SOLUTION_COLORS[n], label=n, edgecolor="white", linewidth=0.5)
ax.set_xticks(xx, pc.index, rotation=35, ha="right")
ax.set_ylabel("F1 score"); ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False)
ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
save(fig, "paper_fig6_per_class_f1")

# ---- Fig 7: leakage ---------------------------------------------------------
leak = pd.read_csv(RESULTS_DIR / "leakage_check.csv")
fig, ax = plt.subplots(figsize=(W, 1.9))
models_ = list(leak.Model.unique())
for j, (prot, color, lab) in enumerate([("Random split (actors shared)", "#eb6834", "Random split (speakers shared)"),
                                        ("Speaker-independent (ours)", "#2a78d6", "Speaker-independent (ours)")]):
    v = leak[leak.Protocol == prot].set_index("Model").loc[models_, "Accuracy"]
    xs = np.arange(len(models_)) + (j - 0.5) * 0.38
    ax.bar(xs, v, width=0.36, color=color, label=lab, edgecolor="white", linewidth=1)
    for a, b in zip(xs, v):
        ax.text(a, b + 0.015, f"{b:.1%}", ha="center", fontsize=6.3, color=INK_2)
ax.set_xticks(range(len(models_)), models_)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right")
save(fig, "paper_fig7_leakage")

# ---- Fig 8: family share -----------------------------------------------------
nums = json.load(open(RESULTS_DIR / "paper_numbers.json"))
fam = pd.DataFrame(nums["family_share"]).fillna(0)
fam = fam[["all", "s1", "s2", "s3"]].sort_values("all")
fig, ax = plt.subplots(figsize=(W, 2.6))
yy = np.arange(len(fam)); hh = 0.2
for i, col in enumerate(fam.columns):
    color = GREY if col == "all" else SOLUTION_COLORS[S[col]]
    ax.barh(yy + (1.5 - i) * hh, fam[col], height=hh, color=color,
            label="All 190 features" if col == "all" else S[col])
ax.set_yticks(yy, fam.index); ax.set_xlabel("Share of selected features (%)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
save(fig, "paper_fig8_family_share")
