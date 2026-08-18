"""Metric-aligned EHOA for imbalanced feature-selection studies."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from ehoa import EHOA


class AlignedEHOA(EHOA):
    """Optimize the same balanced-accuracy target used in outer evaluation."""

    def _evaluate(self, mask):
        mask=np.asarray(mask,bool)
        key=np.packbits(mask.astype(np.uint8)).tobytes()+len(mask).to_bytes(4,"little")
        if key not in self._fitness_cache:
            selected=np.flatnonzero(mask)
            if not len(selected):
                result=(1.0,0.0,0)
            else:
                scores=[]
                for fold in self._folds:
                    model=KNeighborsClassifier(n_neighbors=min(5,len(fold.y_train)))
                    model.fit(fold.X_train[:,selected],fold.y_train)
                    scores.append(balanced_accuracy_score(fold.y_valid,model.predict(fold.X_valid[:,selected])))
                score=float(np.mean(scores))
                result=(float(self.alpha*(1-score)+(1-self.alpha)*len(selected)/len(mask)),score,len(selected))
            self._fitness_cache[key]=result; self.evaluations_+=1
        return self._fitness_cache[key]
