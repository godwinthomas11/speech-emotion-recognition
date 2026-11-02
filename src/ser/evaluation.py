"""Speaker-independent evaluation shared by every model in the project.

Protocol (nested cross-validation):

* Outer loop: 6 folds. Each fold tests on 4 actors (2 male + 2 female) that
  the model never saw. Every actor is tested exactly once.
* Inner loop: inside the 20 training actors, 3-fold GroupKFold (again split
  by actor) chooses the number of features k and the model settings.
* The scaler and the feature selector are fitted on training data only,
  never on the test actors. This prevents selection bias.
"""
import itertools
import re
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, RESULTS_DIR

N_OUTER = 6
N_INNER = 3


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------
def actor_folds(seed=RANDOM_STATE, n_folds=N_OUTER):
    """Return a list of test-actor lists, each with equal numbers of men and women."""
    rng = np.random.default_rng(seed)
    males = rng.permutation(np.arange(1, 25, 2))
    females = rng.permutation(np.arange(2, 25, 2))
    per = len(males) // n_folds
    return [sorted(np.r_[males[i * per:(i + 1) * per], females[i * per:(i + 1) * per]].tolist())
            for i in range(n_folds)]


def outer_splits(actors, folds=None):
    """Yield (fold_no, train_idx, test_idx) with no actor in both sets."""
    actors = np.asarray(actors)
    for i, test_actors in enumerate(folds or actor_folds()):
        test = np.isin(actors, test_actors)
        tr, te = np.where(~test)[0], np.where(test)[0]
        assert not set(actors[tr]) & set(actors[te]), "actor leakage between train and test"
        yield i, tr, te


def inner_splits(actors_train):
    """Actor-wise inner folds on the training portion only."""
    return list(GroupKFold(n_splits=N_INNER).split(actors_train, groups=actors_train))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def scores(y_true, y_pred):
    """Accuracy, macro-F1 and UAR (unweighted average recall = balanced accuracy)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "uar": balanced_accuracy_score(y_true, y_pred),
    }


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test: do two models make significantly different errors?"""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


# --------------------------------------------------------------------------
# Feature rankers (each returns feature indices from best to worst)
# --------------------------------------------------------------------------
def rank_all(X, y):
    """No selection: keep the original order (used for baselines)."""
    return np.arange(X.shape[1])


def rank_mutual_info(X, y):
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, n_neighbors=3, random_state=RANDOM_STATE)
    return np.argsort(-mi, kind="stable")


def rank_rfe_rf(X, y, step=10):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    rfe = RFE(rf, n_features_to_select=step, step=step).fit(X, y)
    # ranking_ is 1 for the last surviving block, 2 for the block removed just before, ...
    # Inside a block, break ties with the forest's importances from the final round.
    imp = np.zeros(X.shape[1])
    imp[rfe.support_] = rfe.estimator_.feature_importances_
    return np.lexsort((-imp, rfe.ranking_))


# --------------------------------------------------------------------------
# Nested CV for "rank, then keep the top k" selectors (S1, S2 and baselines)
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    name: str
    folds: pd.DataFrame
    oof: pd.DataFrame
    selected: list
    inner_curve: pd.DataFrame
    outer_curve: pd.DataFrame
    extra: dict = field(default_factory=dict)

    def save(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.split(":")[0].strip().lower()).strip("_")
        self.folds.to_csv(RESULTS_DIR / f"{slug}_folds.csv", index=False)
        self.oof.to_csv(RESULTS_DIR / f"{slug}_oof.csv", index=False)
        if not self.inner_curve.empty:
            self.inner_curve.to_csv(RESULTS_DIR / f"{slug}_inner_curve.csv", index=False)
        if not self.outer_curve.empty:
            self.outer_curve.to_csv(RESULTS_DIR / f"{slug}_outer_curve.csv", index=False)
        with open(RESULTS_DIR / f"{slug}_selected.json", "w") as fh:
            json.dump({"name": self.name, "selected_per_fold": self.selected, **self.extra}, fh, indent=1)

    def summary(self):
        m = self.folds[["accuracy", "macro_f1", "uar", "n_features", "fit_seconds"]]
        return pd.DataFrame({"mean": m.mean(), "std": m.std()}).T


def _param_list(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def _eval_inner(Xtr, ytr, order, make_model, params, k):
    model = make_model(k=k, **params)
    return f1_score(ytr[1], model.fit(Xtr[0][:, order[:k]], ytr[0]).predict(Xtr[1][:, order[:k]]),
                    average="macro")


def nested_ranked_cv(name, X, y, actors, feature_names, ranker, make_model, param_grid,
                     k_grid, files=None, n_jobs=4, outer_curve=True, one_se=True, verbose=True):
    """Nested speaker-independent CV for a ranking-based feature selector.

    ranker(X, y)              -> feature indices ordered best to worst
    make_model(k=..., **p)    -> an unfitted classifier
    """
    X, y, actors = np.asarray(X, float), np.asarray(y), np.asarray(actors)
    params_list = _param_list(param_grid)
    fold_rows, oof, selected, inner_rows, outer_rows = [], [], [], [], []

    for fold, tr, te in outer_splits(actors):
        t0 = time.time()
        # ---- inner loop: choose k and model settings on training actors only
        per_split = np.zeros((N_INNER, len(params_list), len(k_grid)))
        for s_i, (itr, iva) in enumerate(inner_splits(actors[tr])):
            a, b = tr[itr], tr[iva]
            sc = StandardScaler().fit(X[a])
            Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
            order = ranker(Xa, y[a])
            jobs = [(pi, ki) for pi in range(len(params_list)) for ki in range(len(k_grid))]
            res = Parallel(n_jobs=n_jobs)(
                delayed(_eval_inner)((Xa, Xb), (y[a], y[b]), order, make_model, params_list[pi], k_grid[ki])
                for pi, ki in jobs)
            for (pi, ki), r in zip(jobs, res):
                per_split[s_i, pi, ki] = r
        inner = per_split.mean(axis=0)
        best_p, best_k = np.unravel_index(np.argmax(inner), inner.shape)
        k_best_raw = k_grid[best_k]
        if one_se and len(k_grid) > 1:
            # One-standard-error rule (Hastie et al., ESL 7.10): take the smallest k whose
            # best inner score is within one standard error of the overall best.
            se = per_split[:, best_p, best_k].std(ddof=1) / np.sqrt(N_INNER)
            best_per_k = inner.max(axis=0)
            best_k = int(np.flatnonzero(best_per_k >= inner[best_p, best_k] - se)[0])
            best_p = int(np.argmax(inner[:, best_k]))
        params, k = params_list[best_p], k_grid[best_k]
        for ki, kk in enumerate(k_grid):
            inner_rows.append({"fold": fold, "k": kk, "inner_macro_f1": inner[:, ki].max()})

        # ---- outer fit with the chosen settings
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        t_rank = time.time()
        order = ranker(Xtr, y[tr])
        rank_seconds = time.time() - t_rank
        model = make_model(k=k, **params)
        t_fit = time.time()
        model.fit(Xtr[:, order[:k]], y[tr])
        fit_seconds = time.time() - t_fit
        t_pred = time.time()
        pred = model.predict(Xte[:, order[:k]])
        pred_ms = (time.time() - t_pred) / len(te) * 1000

        s = scores(y[te], pred)
        fold_rows.append({"fold": fold, "test_actors": " ".join(map(str, sorted(set(actors[te])))),
                          **s, "n_features": k, "k_best_inner": k_best_raw, "params": json.dumps(params),
                          "select_seconds": rank_seconds, "fit_seconds": fit_seconds,
                          "predict_ms_per_clip": pred_ms, "fold_seconds": time.time() - t0})
        oof.append(pd.DataFrame({"file": None if files is None else np.asarray(files)[te],
                                 "actor": actors[te], "y_true": y[te], "y_pred": pred, "fold": fold}))
        selected.append([feature_names[i] for i in order[:k]])

        # ---- for the figure only: test score at every k with the chosen settings
        if outer_curve and len(k_grid) > 1:
            curve = Parallel(n_jobs=n_jobs)(
                delayed(lambda kk: f1_score(y[te], make_model(k=kk, **params).fit(Xtr[:, order[:kk]], y[tr])
                                            .predict(Xte[:, order[:kk]]), average="macro"))(kk)
                for kk in k_grid)
            outer_rows += [{"fold": fold, "k": kk, "test_macro_f1": v} for kk, v in zip(k_grid, curve)]

        if verbose:
            print(f"[{name}] fold {fold + 1}/{N_OUTER} actors {sorted(set(actors[te]))}: "
                  f"acc={s['accuracy']:.3f} F1={s['macro_f1']:.3f} k={k} {params} "
                  f"({time.time() - t0:.0f}s)")

    return RunResult(name, pd.DataFrame(fold_rows), pd.concat(oof, ignore_index=True), selected,
                     pd.DataFrame(inner_rows), pd.DataFrame(outer_rows))


# --------------------------------------------------------------------------
# Nested CV for the Genetic Algorithm selector (S3)
# --------------------------------------------------------------------------
def _eval_mask_params(X, y, splits, cols, make_model, params):
    f1s = []
    for a, b in splits:
        sc = StandardScaler().fit(X[a][:, cols])
        m = make_model(k=len(cols), **params).fit(sc.transform(X[a][:, cols]), y[a])
        f1s.append(f1_score(y[b], m.predict(sc.transform(X[b][:, cols])), average="macro"))
    return float(np.mean(f1s))


def nested_ga_cv(name, X, y, actors, feature_names, make_selector, make_model, param_grid,
                 files=None, n_jobs=4, verbose=True):
    """make_selector(fitness_params) -> GA selector. Before the GA starts, the classifier's
    settings are tuned on all features with the inner folds, and the GA's fitness uses them."""
    """GA selects a feature mask on the training actors; the classifier is then
    tuned on that mask (same inner folds) and tested once on the unseen actors."""
    X, y, actors = np.asarray(X, float), np.asarray(y), np.asarray(actors)
    params_list = _param_list(param_grid)
    fold_rows, oof, selected, histories = [], [], [], []

    for fold, tr, te in outer_splits(actors):
        t0 = time.time()
        splits = inner_splits(actors[tr])
        if verbose:
            print(f"[{name}] fold {fold + 1}/{N_OUTER}: running GA ...")
        all_cols = np.arange(X.shape[1])
        res0 = Parallel(n_jobs=n_jobs)(
            delayed(_eval_mask_params)(X[tr], y[tr], splits, all_cols, make_model, p) for p in params_list)
        fitness_params = params_list[int(np.argmax(res0))]
        if verbose:
            print(f"  fitness model settings (tuned on all features): {fitness_params}")
        t_sel = time.time()
        ga = make_selector(fitness_params).fit(X[tr], y[tr], splits)
        select_seconds = time.time() - t_sel
        cols = np.flatnonzero(ga.support_)
        hist = pd.DataFrame(ga.history_).assign(fold=fold)
        histories.append(hist)

        # tune the final classifier on the GA subset (training actors only)
        res = Parallel(n_jobs=n_jobs)(
            delayed(_eval_mask_params)(X[tr], y[tr], splits, cols, make_model, p) for p in params_list)
        params = params_list[int(np.argmax(res))]

        sc = StandardScaler().fit(X[tr][:, cols])
        model = make_model(k=len(cols), **params)
        t_fit = time.time()
        model.fit(sc.transform(X[tr][:, cols]), y[tr])
        fit_seconds = time.time() - t_fit
        t_pred = time.time()
        pred = model.predict(sc.transform(X[te][:, cols]))
        pred_ms = (time.time() - t_pred) / len(te) * 1000

        s = scores(y[te], pred)
        fold_rows.append({"fold": fold, "test_actors": " ".join(map(str, sorted(set(actors[te])))),
                          **s, "n_features": len(cols), "params": json.dumps(params),
                          "select_seconds": select_seconds, "fit_seconds": fit_seconds,
                          "predict_ms_per_clip": pred_ms, "fold_seconds": time.time() - t0,
                          "fitness_params": json.dumps(fitness_params),
                          "ga_generations": len(ga.history_) - 1, "ga_evaluations": ga.n_evaluations_,
                          "ga_best_fitness": ga.best_fitness_})
        oof.append(pd.DataFrame({"file": None if files is None else np.asarray(files)[te],
                                 "actor": actors[te], "y_true": y[te], "y_pred": pred, "fold": fold}))
        selected.append([feature_names[i] for i in cols])
        if verbose:
            print(f"[{name}] fold {fold + 1}/{N_OUTER} actors {sorted(set(actors[te]))}: "
                  f"acc={s['accuracy']:.3f} F1={s['macro_f1']:.3f} features={len(cols)} {params} "
                  f"({time.time() - t0:.0f}s)")

    result = RunResult(name, pd.DataFrame(fold_rows), pd.concat(oof, ignore_index=True), selected,
                       pd.DataFrame(), pd.DataFrame())
    result.extra["ga_history"] = pd.concat(histories).to_dict(orient="records")
    result.history = pd.concat(histories, ignore_index=True)
    return result
