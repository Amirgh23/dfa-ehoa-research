"""Safe dimension-aware portfolio selected after the innovation tournament."""
from __future__ import annotations

from ehoa import EHOA
from .memetic_ehoa import MemeticEHOA


class SafePortfolioEHOA:
    """Use M-EHOA below the threshold and preserve baseline otherwise."""

    def __init__(self,*args,dimension_threshold=20,local_search_rounds=2,**kwargs):
        self._args=args; self._kwargs=kwargs
        self.dimension_threshold=int(dimension_threshold)
        self.local_search_rounds=int(local_search_rounds)

    def fit(self,X,y):
        if X.shape[1] >= self.dimension_threshold:
            self.selected_strategy_="baseline"
            self.delegate_=EHOA(*self._args,**self._kwargs)
        else:
            self.selected_strategy_="memetic"
            self.delegate_=MemeticEHOA(*self._args,local_search_rounds=self.local_search_rounds,**self._kwargs)
        result=self.delegate_.fit(X,y)
        for name in ("best_solution","best_features","best_accuracy","best_fitness",
                     "evaluations_","runtime_seconds_","result_"):
            setattr(self,name,getattr(self.delegate_,name))
        return result

    def history_frame(self):
        frame=self.delegate_.history_frame().copy(); frame["portfolio_strategy"]=self.selected_strategy_
        return frame
