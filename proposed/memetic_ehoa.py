"""Budget-auditable memetic refinement for EHOA."""
from __future__ import annotations

import time
import numpy as np

from ehoa import EHOA, EHOAResult


class MemeticEHOA(EHOA):
    """Refine the EHOA incumbent with deterministic best-improvement flips.

    Only inner-CV fitness is queried. Each round evaluates every one-feature
    add/remove neighbour and accepts the best strict improvement, making the
    extra evaluation cost explicit and reproducible.
    """

    def __init__(self, *args, local_search_rounds=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_search_rounds=int(local_search_rounds)
        if self.local_search_rounds < 1:
            raise ValueError("local_search_rounds must be >= 1")
        self.local_improvements_=0

    def fit(self, X, y):
        mask,_,_=super().fit(X,y)
        started=time.perf_counter(); self.local_improvements_=0
        for _ in range(self.local_search_rounds):
            candidate_best=None; candidate_score=(self.best_fitness,len(self.best_features))
            for feature in range(len(mask)):
                candidate=mask.astype(bool).copy(); candidate[feature]=~candidate[feature]
                if not candidate.any():
                    continue
                fitness,accuracy,count=self._evaluate(candidate)
                score=(fitness,count)
                if score < candidate_score:
                    candidate_best=(candidate,fitness,accuracy); candidate_score=score
            if candidate_best is None:
                break
            mask,fitness,accuracy=candidate_best
            position=np.where(mask,self.position_max,self.position_min)
            self._consider_global(position,mask,fitness,accuracy)
            self.local_improvements_+=1
        self.runtime_seconds_ += time.perf_counter()-started
        self.result_=EHOAResult(self.best_solution.copy(),self.best_features.copy(),
            self.best_accuracy,self.best_fitness,self.runtime_seconds_,self.evaluations_)
        return self.best_solution.copy(),self.best_accuracy,self.best_features.copy()

    def history_frame(self):
        frame=super().history_frame()
        frame["local_improvements"]=self.local_improvements_
        return frame
