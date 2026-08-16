"""Configurable reliability/interaction-guided binary transition."""
import numpy as np
from utils import sigmoid

def scheduled_weight(value: float, iteration: int, maximum: int, strategy: str,
                     signal: float = 1.0) -> float:
    if strategy == "constant": return value
    if strategy == "linear": return value * iteration / max(1, maximum)
    if strategy == "adaptive": return value * np.clip(signal, 0, 1)
    raise ValueError("strategy must be constant, linear, or adaptive")

def transition_probabilities(position, reliability, interaction, *, mode="dual",
                             lambda_stability=0.5, lambda_interaction=0.5):
    z = np.asarray(position, float).copy()
    if mode in {"stability", "dual"}: z += lambda_stability*np.clip(reliability, -1, 1)
    if mode in {"interaction", "dual"}: z += lambda_interaction*np.clip(interaction, -1, 1)
    if mode not in {"baseline", "stability", "interaction", "dual"}: raise ValueError("invalid transition mode")
    return sigmoid(z)

def sample_mask(probabilities, rng):
    p = np.clip(np.asarray(probabilities, float), 0, 1)
    mask = rng.random(p.shape) < p
    if not mask.any(): mask[int(np.argmax(p))] = True
    return mask
