"""Utilities for leakage-free EHOA feature-selection experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable
import warnings

# Some restricted/virtualized Windows environments cannot expose physical CPU
# topology to joblib. The algorithms here use deterministic single-process jobs.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - covered by the actionable error below
    SMOTE = None


Array = np.ndarray


# ---------------------------------------------------------------------------
# Chaotic maps (Table 1 family in the paper)
# ---------------------------------------------------------------------------

def chaotic_singer(x: float, mu: float = 1.07) -> float:
    return mu * (7.86 * x - 23.31 * x**2 + 28.75 * x**3 - 13.302875 * x**4)


def chaotic_sinusoidal(x: float, a: float = 2.3) -> float:
    return a * x**2 * np.sin(np.pi * x)


def chaotic_gauss(x: float) -> float:
    return 0.0 if abs(x) < 1e-15 else float(np.mod(1.0 / x, 1.0))


def chaotic_circle(x: float, a: float = 0.5, b: float = 0.2) -> float:
    return float(np.mod(x + b - (a / (2 * np.pi)) * np.sin(2 * np.pi * x), 1.0))


def chaotic_chebyshev(x: float, k: int = 4) -> float:
    # Work in [-1, 1], then map back to [0, 1].
    z = np.clip(2.0 * x - 1.0, -1.0, 1.0)
    return float((np.cos(k * np.arccos(z)) + 1.0) / 2.0)


def chaotic_iterative(x: float, a: float = 0.7) -> float:
    safe_x = x if abs(x) > 1e-12 else 1e-12
    return float(np.mod(np.sin(a * np.pi / safe_x), 1.0))


def chaotic_sine(x: float, a: float = 4.0) -> float:
    return float((a / 4.0) * np.sin(np.pi * x))


def chaotic_piecewise(x: float, p: float = 0.4) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    if x < p:
        return x / p
    if x < 0.5:
        return (x - p) / (0.5 - p)
    if x < 1.0 - p:
        return (1.0 - p - x) / (0.5 - p)
    return (1.0 - x) / p


def chaotic_logistic(x: float, r: float = 4.0) -> float:
    return r * x * (1.0 - x)


def chaotic_tent(x: float, p: float = 0.4) -> float:
    return x / p if x < p else (1.0 - x) / (1.0 - p)


CHAOTIC_MAPS: dict[str, Callable[[float], float]] = {
    "singer": chaotic_singer,
    "sinusoidal": chaotic_sinusoidal,
    "gauss": chaotic_gauss,
    "circle": chaotic_circle,
    "chebyshev": chaotic_chebyshev,
    "iterative": chaotic_iterative,
    "sine": chaotic_sine,
    "piecewise": chaotic_piecewise,
    "logistic": chaotic_logistic,
    "tent": chaotic_tent,
}


def chaotic_sequence(name: str, size: int, x0: float = 0.7) -> Array:
    """Return a finite, sanitized sequence in [0, 1]."""
    if name not in CHAOTIC_MAPS:
        raise ValueError(f"Unknown chaotic map {name!r}. Choose from {sorted(CHAOTIC_MAPS)}")
    fn = CHAOTIC_MAPS[name]
    sequence = np.empty(size, dtype=float)
    x = float(x0)
    for index in range(size):
        x = float(fn(x))
        if not np.isfinite(x):
            x = 0.5
        x = float(np.mod(x, 1.0))
        sequence[index] = np.clip(x, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return sequence


def sigmoid(values: Array) -> Array:
    values = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


# ---------------------------------------------------------------------------
# Data and evaluation helpers
# ---------------------------------------------------------------------------

def clean_dataset(X: Array, y: Array) -> tuple[Array, Array]:
    """Encode labels and remove exact duplicate (X, y) rows without reordering."""
    X = np.asarray(X, dtype=float).copy()
    y = np.asarray(y).copy()
    if y.dtype.kind not in "biufc":
        y = LabelEncoder().fit_transform(y)
    else:
        y = LabelEncoder().fit_transform(y)

    frame = pd.DataFrame(X)
    frame["__target__"] = y
    keep = ~frame.duplicated(keep="first")
    return X[keep.to_numpy()], y[keep.to_numpy()].astype(int)


@dataclass
class PreparedFold:
    X_train: Array
    y_train: Array
    X_valid: Array
    y_valid: Array


def _resample_training(X: Array, y: Array, random_state: int) -> tuple[Array, Array]:
    """Apply SMOTE to training data only, adapting k for small minority classes."""
    counts = np.bincount(y.astype(int))
    positive_counts = counts[counts > 0]
    if len(positive_counts) < 2 or positive_counts.min() < 2:
        return X, y
    if SMOTE is None:
        raise ImportError("imbalanced-learn is required when apply_smote=True")
    k_neighbors = int(min(5, positive_counts.min() - 1))
    sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return sampler.fit_resample(X, y)


def prepare_cv_folds(
    X: Array,
    y: Array,
    n_folds: int = 10,
    random_state: int = 42,
    apply_smote: bool = True,
) -> list[PreparedFold]:
    """Fit imputation/scaling/SMOTE exclusively on each fold's training split."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    min_class = int(np.bincount(y).min())
    n_splits = min(n_folds, min_class)
    if n_splits < 2:
        raise ValueError("Each class needs at least two samples for stratified CV")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[PreparedFold] = []
    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        X_train = imputer.fit_transform(X[train_idx])
        X_valid = imputer.transform(X[valid_idx])
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        y_train = y[train_idx]
        if apply_smote:
            X_train, y_train = _resample_training(
                X_train, y_train, random_state + fold_index
            )
        folds.append(PreparedFold(X_train, y_train, X_valid, y[valid_idx]))
    return folds


def evaluate_mask_cv(
    mask: Array,
    folds: list[PreparedFold],
    alpha: float = 0.99,
) -> tuple[float, float, int]:
    """Evaluate Eq. 12 with deterministic 5-NN predictions over prepared folds."""
    mask = np.asarray(mask, dtype=bool)
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return 1.0, 0.0, 0
    accuracies = []
    for fold in folds:
        model = KNeighborsClassifier(n_neighbors=min(5, len(fold.y_train)))
        model.fit(fold.X_train[:, selected], fold.y_train)
        prediction = model.predict(fold.X_valid[:, selected])
        accuracies.append(accuracy_score(fold.y_valid, prediction))
    accuracy = float(np.mean(accuracies))
    fitness = alpha * (1.0 - accuracy) + (1.0 - alpha) * (selected.size / mask.size)
    return float(fitness), accuracy, int(selected.size)


def preprocess_train_test(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    *,
    apply_smote: bool,
    random_state: int,
) -> tuple[Array, Array, Array, SimpleImputer, StandardScaler]:
    """Fit all transforms on train and transform test without changing row order."""
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_p = scaler.transform(imputer.transform(X_test))
    y_train_p = np.asarray(y_train, dtype=int)
    if apply_smote:
        X_train_p, y_train_p = _resample_training(X_train_p, y_train_p, random_state)
    return X_train_p, y_train_p, X_test_p, imputer, scaler


def classifier_factories(random_state: int = 42) -> dict[str, object]:
    """The four classifiers and settings reported in Section 7.2."""
    return {
        "knn": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "logistic_regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "svm": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=random_state, n_jobs=1
        ),
    }


def evaluate_classifiers(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    selected_features: Array,
    *,
    apply_smote: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object], tuple[SimpleImputer, StandardScaler]]:
    selected = np.asarray(selected_features, dtype=int)
    X_train_p, y_train_p, X_test_p, imputer, scaler = preprocess_train_test(
        X_train, y_train, X_test,
        apply_smote=apply_smote,
        random_state=random_state,
    )
    fitted: dict[str, object] = {}
    rows = []
    for name, template in classifier_factories(random_state).items():
        model = clone(template)
        model.fit(X_train_p[:, selected], y_train_p)
        prediction = model.predict(X_test_p[:, selected])
        rows.append({"classifier": name, **calculate_metrics(y_test, prediction)})
        fitted[name] = model
    return pd.DataFrame(rows), fitted, (imputer, scaler)


def calculate_metrics(y_true: Array, y_pred: Array) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    average = "binary" if len(labels) == 2 else "macro"
    specificity_values: list[float] = []
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()
    for index in range(len(labels)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        tn = total - tp - fp - fn
        specificity_values.append(tn / (tn + fp) if tn + fp else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        specificity = specificity_values[1] if len(labels) == 2 else float(np.mean(specificity_values))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "sensitivity": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "specificity": float(specificity),
            "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }


# Backward-compatible name, now safe and intentionally training-only.
def preprocess_data(X: Array, y: Array, apply_smote: bool = True) -> tuple[Array, Array]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(imputer.fit_transform(np.asarray(X, dtype=float)))
    y_processed = np.asarray(y, dtype=int)
    if apply_smote:
        X_processed, y_processed = _resample_training(X_processed, y_processed, 42)
    return X_processed, y_processed
