"""Collect every number quoted in the paper into results/paper_numbers.json.

Keeping this in one script means the paper text can never drift from the
experiment outputs.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ser.config import EMOTION_ORDER, FEATURES_CSV, RESULTS_DIR  # noqa: E402
from ser.evaluation import scores  # noqa: E402
from ser.features import feature_group  # noqa: E402

RUNS = {"s1": "S1: MI + SVM", "s2": "S2: RFE + RF", "s3": "S3: GA + MLP",
        "baseline_svm": "Baseline SVM", "baseline_rf": "Baseline RF", "baseline_mlp": "Baseline MLP",
        "ablation_mi_rf": "MI + RF", "ablation_rfe_svm": "RFE + SVM"}

out = {"runs": {}, "per_class": {}, "data": {}}
for slug, name in RUNS.items():
    f = pd.read_csv(RESULTS_DIR / f"{slug}_folds.csv")
    o = pd.read_csv(RESULTS_DIR / f"{slug}_oof.csv")
    pooled = scores(o.y_true, o.y_pred)
    params = [json.loads(p) for p in f.params]
    out["runs"][slug] = {
        "name": name,
        "acc": f.accuracy.mean(), "acc_sd": f.accuracy.std(),
        "f1": f.macro_f1.mean(), "f1_sd": f.macro_f1.std(),
        "uar": f.uar.mean(), "uar_sd": f.uar.std(),
        "k": f.n_features.mean(), "k_sd": f.n_features.std(),
        "k_min": int(f.n_features.min()), "k_max": int(f.n_features.max()),
        "sel_s": f.select_seconds.mean(), "fit_s": f.fit_seconds.mean(),
        "pred_ms": f.predict_ms_per_clip.mean(),
        "pooled_acc": pooled["accuracy"], "pooled_f1": pooled["macro_f1"],
        "params": params,
        "fold_acc": f.accuracy.round(3).tolist(),
    }
    if "ga_generations" in f:
        out["runs"][slug].update(ga_gens=f.ga_generations.mean(), ga_evals=f.ga_evaluations.mean())
    out["per_class"][slug] = dict(zip(EMOTION_ORDER, f1_score(o.y_true, o.y_pred, labels=EMOTION_ORDER, average=None)))

out["mcnemar"] = pd.read_csv(RESULTS_DIR / "mcnemar_tests.csv").to_dict(orient="records")
leak = pd.read_csv(RESULTS_DIR / "leakage_check.csv")
out["leak"] = {f"{r.Model}|{r.Protocol}": r.Accuracy for r in leak.itertuples()}
out["stability"] = pd.read_csv(RESULTS_DIR / "selection_stability.csv", index_col=0).iloc[:, 0].to_dict()

# stable features (chosen in >= 4 of 6 folds) and family shares
sel = {s: json.load(open(RESULTS_DIR / f"{s}_selected.json"))["selected_per_fold"] for s in ["s1", "s2", "s3"]}
stable = {}
for s, folds in sel.items():
    cnt = pd.Series([f for fold in folds for f in fold]).value_counts()
    stable[s] = sorted(cnt[cnt >= 4].index)
out["stable"] = stable
out["stable_common"] = sorted(set(stable["s1"]) & set(stable["s2"]) & set(stable["s3"]))
fam = {}
for s, folds in sel.items():
    fam[s] = (pd.Series([feature_group(f) for fold in folds for f in fold])
              .value_counts(normalize=True).mul(100).round(1).to_dict())
df = pd.read_csv(FEATURES_CSV)
meta = ["file", "actor", "gender", "emotion_id", "emotion", "intensity", "statement", "repetition"]
feats = [c for c in df.columns if c not in meta]
fam["all"] = pd.Series([feature_group(f) for f in feats]).value_counts(normalize=True).mul(100).round(1).to_dict()
out["family_share"] = fam
out["n_features"] = len(feats)

out["data"]["pitch_fallback"] = 13
out["data"]["dur_min"], out["data"]["dur_max"] = 2.94, 5.27
out["data"]["trim_pct"] = 49

with open(RESULTS_DIR / "paper_numbers.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(json.dumps({k: {m: round(v, 3) for m, v in r.items() if isinstance(v, float)} for k, r in out["runs"].items()}, indent=1))
