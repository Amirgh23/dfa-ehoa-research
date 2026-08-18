"""Risk-controlled abstention guard for EHOA feature selection."""
from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from ehoa import EHOA, EHOAResult


class GuardedEHOA(EHOA):
    """Keep a subset only when training CV supports a material benefit.

    Feature selection is allowed to abstain and return all features. This avoids
    forcing dimensionality reduction when its estimated predictive risk is
    worse than the no-selection reference.
    """

    def __init__(self,*args,selection_margin=.005,**kwargs):
        super().__init__(*args,**kwargs)
        self.selection_margin=float(selection_margin)
        self.abstained_=False

    def _balanced_cv(self,mask):
        selected=np.flatnonzero(mask); scores=[]
        for fold in self._folds:
            model=KNeighborsClassifier(n_neighbors=min(5,len(fold.y_train)))
            model.fit(fold.X_train[:,selected],fold.y_train)
            scores.append(balanced_accuracy_score(fold.y_valid,model.predict(fold.X_valid[:,selected])))
        return float(np.mean(scores))

    def fit(self,X,y):
        mask,_,_=super().fit(X,y); started=time.perf_counter()
        full=np.ones_like(mask,dtype=np.uint8)
        subset_score=self._balanced_cv(mask); full_score=self._balanced_cv(full)
        self.guard_subset_score_=subset_score; self.guard_full_score_=full_score
        self.abstained_=subset_score < full_score+self.selection_margin
        if self.abstained_:
            self.best_solution=full
            self.best_features=np.flatnonzero(full)
            self.best_accuracy=full_score
            self.best_fitness=self.alpha*(1-full_score)+(1-self.alpha)
            self.best_position=np.full(len(full),self.position_max)
        self.runtime_seconds_+=time.perf_counter()-started
        self.result_=EHOAResult(self.best_solution.copy(),self.best_features.copy(),
            self.best_accuracy,self.best_fitness,self.runtime_seconds_,self.evaluations_)
        return self.best_solution.copy(),self.best_accuracy,self.best_features.copy()

    def history_frame(self):
        frame=super().history_frame()
        frame["guard_abstained"]=self.abstained_
        frame["guard_subset_balanced_accuracy"]=self.guard_subset_score_
        frame["guard_full_balanced_accuracy"]=self.guard_full_score_
        return frame
