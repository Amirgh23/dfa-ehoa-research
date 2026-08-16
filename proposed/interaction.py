"""Target-relevant complementarity minus redundancy guidance."""
from __future__ import annotations
import time
import numpy as np
from sklearn.feature_selection import mutual_info_classif


class InteractionModel:
    """Memory-aware pairwise redundancy with target relevance.

    Absolute Pearson association is used only as the redundancy term; target
    mutual information supplies relevance. The complementarity term rewards a
    target-relevant candidate not already represented by the current subset.
    """
    def __init__(self, top_k: int = 50, random_state: int = 42):
        self.top_k, self.random_state = top_k, random_state
        self.computation_seconds = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "InteractionModel":
        started = time.perf_counter()
        X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
        self.relevance_ = mutual_info_classif(X, y, random_state=self.random_state)
        maximum = self.relevance_.max(initial=0)
        self.relevance_ = self.relevance_ / maximum if maximum > 0 else self.relevance_
        chosen = np.argsort(self.relevance_)[-min(self.top_k, X.shape[1]):]
        self.neighborhood_ = np.zeros(X.shape[1], dtype=bool); self.neighborhood_[chosen] = True
        self.redundancy_ = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
        if len(chosen) > 1:
            corr = np.nan_to_num(np.abs(np.corrcoef(X[:, chosen], rowvar=False)))
            self.redundancy_[np.ix_(chosen, chosen)] = corr
            np.fill_diagonal(self.redundancy_, 0)
        self.computation_seconds = time.perf_counter() - started
        return self

    def score(self, mask: np.ndarray) -> np.ndarray:
        selected = np.flatnonzero(mask)
        redundancy = self.redundancy_[:, selected].mean(axis=1) if len(selected) else 0.0
        score = self.relevance_ * (1.0 - redundancy) - redundancy
        score = np.where(self.neighborhood_, score, 0.0)
        scale = np.max(np.abs(score), initial=0)
        return score / scale if scale > 0 else score


def subset_redundancy(X: np.ndarray, mask: np.ndarray) -> float:
    selected = np.flatnonzero(mask)
    if len(selected) < 2: return 0.0
    corr = np.abs(np.corrcoef(np.nan_to_num(X[:, selected]), rowvar=False))
    return float(corr[np.triu_indices(len(selected), 1)].mean())


def subset_interaction_quality(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                               random_state: int = 42) -> float:
    """Mean relevance/complementarity score of selected features."""
    selected=np.flatnonzero(mask)
    if not len(selected): return 0.0
    model=InteractionModel(top_k=X.shape[1],random_state=random_state).fit(X,y)
    return float(np.mean(model.score(mask)[selected]))
