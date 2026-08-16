"""Binary-population stability, reliability, diversity and stagnation tools."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def _masks(masks: np.ndarray) -> np.ndarray:
    value = np.asarray(masks, dtype=bool)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("masks must be a non-empty 2-D array")
    return value


def compute_feature_reliability(masks: np.ndarray, previous: np.ndarray | None = None,
                                rho: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed selection frequencies q and reliability R=2q-1."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    observed = _masks(masks).mean(axis=0)
    q = observed if previous is None else rho * np.asarray(previous) + (1-rho) * observed
    q = np.clip(q, 0.0, 1.0)
    return q, 2.0 * q - 1.0


def compute_jaccard_stability(masks: np.ndarray) -> float:
    masks = _masks(masks)
    if len(masks) < 2:
        return 1.0
    scores = []
    for i in range(len(masks)-1):
        for j in range(i+1, len(masks)):
            union = np.logical_or(masks[i], masks[j]).sum()
            scores.append(1.0 if union == 0 else np.logical_and(masks[i], masks[j]).sum()/union)
    return float(np.mean(scores))


def compute_nogueira_stability(masks: np.ndarray) -> float:
    """Nogueira et al. chance-corrected stability, bounded for degenerate cases."""
    masks = _masks(masks)
    if len(masks) < 2:
        return 1.0
    q = masks.mean(axis=0)
    kbar = masks.sum(axis=1).mean()
    denom = (kbar / masks.shape[1]) * (1.0 - kbar / masks.shape[1])
    if denom <= np.finfo(float).eps:
        return 1.0 if np.all(masks == masks[0]) else 0.0
    value = 1.0 - (masks.shape[0] / (masks.shape[0]-1)) * np.mean(q*(1-q)) / denom
    return float(np.clip(value, -1.0, 1.0))


def compute_population_diversity(masks: np.ndarray) -> float:
    masks = _masks(masks)
    if len(masks) < 2:
        return 0.0
    distances = [np.not_equal(masks[i], masks[j]).mean()
                 for i in range(len(masks)-1) for j in range(i+1, len(masks))]
    return float(np.mean(distances))


def resampled_feature_solutions(X: np.ndarray, y: np.ndarray, n_resamples: int,
                                subset_size: int, rng: np.random.Generator,
                                fraction: float = 0.8) -> np.ndarray:
    """Build deterministic target-relevant solutions on stratified train resamples.

    Each resample independently ranks features by mutual information and returns
    a top-k binary solution. This is a leakage-safe stability proxy: it never sees
    outer validation/test data and avoids recursively running DFA-EHOA.
    """
    X=np.nan_to_num(np.asarray(X,float),nan=0.0,posinf=0.0,neginf=0.0); y=np.asarray(y)
    if n_resamples < 1: raise ValueError("n_resamples must be positive")
    k=int(np.clip(subset_size,1,X.shape[1])); masks=np.zeros((n_resamples,X.shape[1]),bool)
    classes=np.unique(y)
    for b in range(n_resamples):
        pieces=[]
        for label in classes:
            pool=np.flatnonzero(y==label); size=max(2,int(np.ceil(fraction*len(pool))))
            pieces.append(rng.choice(pool,size=size,replace=True))
        indices=np.concatenate(pieces); relevance=mutual_info_classif(X[indices],y[indices],random_state=int(rng.integers(0,2**31-1)))
        masks[b,np.argsort(relevance,kind="stable")[-k:]]=True
    return masks


@dataclass
class StagnationDetector:
    patience: int = 5
    epsilon: float = 1e-6
    best: float = float("inf")
    counter: int = 0

    def update(self, value: float) -> bool:
        if value < self.best - self.epsilon:
            self.best, self.counter = float(value), 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    @property
    def signal(self) -> float:
        return float(np.clip(self.counter / max(1, self.patience), 0, 1))
