"""Bounded, regime-aware closed-loop control of EHOA search dynamics."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FeedbackController:
    """Convert population evidence into bounded exploration/consolidation.

    Instability is deliberately *not* an exploration trigger by itself.  Early
    collapse or stagnation increases exploration, whereas late instability with
    adequate diversity promotes consolidation.  This prevents the positive
    feedback loop in the legacy controller (unstable -> more exploration ->
    still less stable).
    """
    eta_exploration: float = 0.45
    eta_consolidation: float = 0.30
    smoothing: float = 0.5

    def update(self, sf_base, inertia_base, stability, diversity, stagnation,
               sf_bounds, inertia_bounds, previous=None, progress=0.5):
        uncertainty = 1 - np.clip(stability, 0, 1)
        diversity_ratio = np.clip(diversity / 0.5, 0, 1)
        collapse = 1 - diversity_ratio
        stagnation = np.clip(stagnation, 0, 1)
        progress = np.clip(progress, 0, 1)

        # Collapse and stagnation must corroborate an exploration response.
        exploration = collapse * (0.5 * (1 - progress) + 0.5 * stagnation)
        # Late uncertainty with a non-collapsed population calls for consensus.
        consolidation = uncertainty * diversity_ratio * progress
        adjustment = (
            self.eta_exploration * exploration
            - self.eta_consolidation * consolidation
        )
        sf = np.clip(sf_base * (1 + adjustment), *sf_bounds)
        inertia = np.clip(inertia_base + 0.25 * adjustment, *inertia_bounds)
        if previous is not None:
            sf = self.smoothing*previous[0] + (1-self.smoothing)*sf
            inertia = self.smoothing*previous[1] + (1-self.smoothing)*inertia
        return float(sf), float(inertia)
