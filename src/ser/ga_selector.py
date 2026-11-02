"""Genetic Algorithm (GA) for feature selection, written from scratch.

Idea in one paragraph
---------------------
Each candidate solution (a "chromosome") is a row of 0s and 1s, one bit per
feature: 1 = use the feature, 0 = drop it. We start with a random population
of such masks, score each one (its "fitness"), and build the next generation
by picking good parents (tournament selection), mixing their bits (uniform
crossover) and flipping a few bits at random (mutation). The best few masks
are copied unchanged (elitism) so good solutions are never lost. Over the
generations the population moves towards small feature subsets that
classify well.

Fitness = mean macro-F1 of the classifier over actor-wise inner folds
          - penalty * (fraction of features used)
The small penalty prefers smaller subsets when accuracy is about the same.
"""
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def _fitness_one(mask, X, y, splits, make_model, penalty):
    if mask.sum() == 0:
        return -1.0
    cols = np.flatnonzero(mask)
    f1s = []
    for tr, va in splits:
        sc = StandardScaler().fit(X[tr][:, cols])
        model = make_model().fit(sc.transform(X[tr][:, cols]), y[tr])
        f1s.append(f1_score(y[va], model.predict(sc.transform(X[va][:, cols])), average="macro"))
    return float(np.mean(f1s)) - penalty * mask.mean()


class GeneticFeatureSelector:
    def __init__(self, make_model, pop_size=24, generations=25, crossover_rate=0.9,
                 mutation_rate=None, tournament=3, elite=2, penalty=0.02, patience=8,
                 init_density=(0.2, 0.6), n_jobs=4, random_state=42, verbose=True):
        self.make_model = make_model
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate  # default: 1.5 / n_features bits per child
        self.tournament = tournament
        self.elite = elite
        self.penalty = penalty
        self.patience = patience
        self.init_density = init_density
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # ---- GA operators -------------------------------------------------------
    def _tournament_pick(self, fitness):
        contestants = self.rng_.choice(len(fitness), self.tournament, replace=False)
        return contestants[np.argmax(fitness[contestants])]

    def _crossover(self, a, b):
        if self.rng_.random() > self.crossover_rate:
            return a.copy()
        take_from_a = self.rng_.random(a.size) < 0.5
        return np.where(take_from_a, a, b)

    def _mutate(self, child):
        flip = self.rng_.random(child.size) < self.mut_
        child = child ^ flip
        if not child.any():  # never allow an empty subset
            child[self.rng_.integers(child.size)] = True
        return child

    # ---- main loop ----------------------------------------------------------
    def fit(self, X, y, splits):
        """splits: list of (train_idx, val_idx) actor-wise folds inside the training set."""
        X, y = np.asarray(X, float), np.asarray(y)
        n = X.shape[1]
        self.rng_ = np.random.default_rng(self.random_state)
        self.mut_ = self.mutation_rate or 1.5 / n
        cache = {}

        def evaluate(pop):
            keys = [m.tobytes() for m in pop]
            todo = [i for i, k in enumerate(keys) if k not in cache]
            # identical masks are scored once
            uniq = {keys[i]: pop[i] for i in todo}
            vals = Parallel(n_jobs=self.n_jobs)(
                delayed(_fitness_one)(m, X, y, splits, self.make_model, self.penalty)
                for m in uniq.values())
            cache.update(dict(zip(uniq.keys(), vals)))
            return np.array([cache[k] for k in keys])

        densities = self.rng_.uniform(*self.init_density, size=self.pop_size)
        pop = np.array([self.rng_.random(n) < d for d in densities])
        fit = evaluate(pop)
        self.history_ = []
        best_so_far, stale = -np.inf, 0
        t0 = time.time()
        for gen in range(self.generations + 1):
            order = np.argsort(-fit)
            pop, fit = pop[order], fit[order]
            self.history_.append({"generation": gen, "best_fitness": fit[0],
                                  "mean_fitness": fit.mean(), "best_n_features": int(pop[0].sum()),
                                  "mean_n_features": pop.sum(1).mean(), "unique_evaluated": len(cache)})
            if self.verbose:
                print(f"  gen {gen:2d}: best={fit[0]:.4f} mean={fit.mean():.4f} "
                      f"features={int(pop[0].sum())} ({time.time() - t0:.0f}s)")
            if fit[0] > best_so_far + 1e-4:
                best_so_far, stale = fit[0], 0
            else:
                stale += 1
            if gen == self.generations or stale >= self.patience:
                break
            children = [pop[i].copy() for i in range(self.elite)]
            while len(children) < self.pop_size:
                p1, p2 = self._tournament_pick(fit), self._tournament_pick(fit)
                children.append(self._mutate(self._crossover(pop[p1], pop[p2])))
            pop = np.array(children)
            fit = evaluate(pop)

        self.support_ = pop[0]
        self.best_fitness_ = fit[0]
        self.n_evaluations_ = len(cache)
        return self
