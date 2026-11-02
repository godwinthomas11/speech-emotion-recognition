"""Classifier factories. Each takes k (number of selected features) so that
settings which depend on the input size, like the SVM's gamma, scale with it."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import RANDOM_STATE


def make_svm(k, C=10.0, gamma_mult=1.0):
    # gamma = gamma_mult / k  ('scale' rule for standardised features)
    return SVC(kernel="rbf", C=C, gamma=gamma_mult / k, class_weight="balanced")


def make_rf(k, min_samples_leaf=1, max_features="sqrt"):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=min_samples_leaf,
                                  max_features=max_features, class_weight="balanced",
                                  n_jobs=1, random_state=RANDOM_STATE)


def make_mlp(k, hidden=(128,), alpha=1e-2):
    return MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=1e-3,
                         max_iter=500, early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=20, random_state=RANDOM_STATE)


SVM_GRID = {"C": [0.3, 1.0, 3.0, 10.0, 30.0], "gamma_mult": [0.25, 0.5, 1.0, 2.0]}
RF_GRID = {"min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.2]}
MLP_GRID = {"hidden": [(128,), (256, 128)], "alpha": [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0]}
K_GRID = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 190]
