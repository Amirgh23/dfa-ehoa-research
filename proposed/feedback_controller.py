"""Bounded closed-loop control of EHOA sweep factor and inertia."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FeedbackController:
    eta_stability: float = 0.35
    eta_diversity: float = 0.20
    eta_stagnation: float = 0.30
    smoothing: float = 0.5

    def update(self, sf_base, inertia_base, stability, diversity, stagnation,
               sf_bounds, inertia_bounds, previous=None):
        uncertainty = 1 - np.clip(stability, 0, 1)
        low_diversity = 1 - np.clip(diversity / 0.5, 0, 1)
        pressure = self.eta_stability*uncertainty + self.eta_diversity*low_diversity + self.eta_stagnation*stagnation
        sf = np.clip(sf_base*(1+pressure), *sf_bounds)
        inertia = np.clip(inertia_base + 0.25*pressure, *inertia_bounds)
        if previous is not None:
            sf = self.smoothing*previous[0] + (1-self.smoothing)*sf
            inertia = self.smoothing*previous[1] + (1-self.smoothing)*inertia
        return float(sf), float(inertia)
