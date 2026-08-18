"""Dimension-aware portfolio of the tournament-winning EHOA innovations."""
from __future__ import annotations

from .dfa_ehoa import DFAEHOA
from .memetic_ehoa import MemeticEHOA


class PortfolioEHOA:
    """Choose the pre-registered winner for low/high-dimensional regimes."""

    def __init__(self,*args,dimension_threshold=20,local_search_rounds=2,
                 interaction_top_k=30,lambda_interaction=.5,
                 weight_schedule="adaptive",**kwargs):
        self._args=args; self._kwargs=kwargs
        self.dimension_threshold=int(dimension_threshold)
        self.local_search_rounds=int(local_search_rounds)
        self.interaction_top_k=int(interaction_top_k)
        self.lambda_interaction=float(lambda_interaction)
        self.weight_schedule=weight_schedule
        self.selected_strategy_=None; self.delegate_=None

    def fit(self,X,y):
        if X.shape[1] >= self.dimension_threshold:
            self.selected_strategy_="interaction"
            self.delegate_=DFAEHOA(*self._args,feedback_enabled=False,
                transition_mode="interaction",interaction_top_k=self.interaction_top_k,
                lambda_interaction=self.lambda_interaction,
                weight_schedule=self.weight_schedule,**self._kwargs)
        else:
            self.selected_strategy_="memetic"
            self.delegate_=MemeticEHOA(*self._args,
                local_search_rounds=self.local_search_rounds,**self._kwargs)
        result=self.delegate_.fit(X,y)
        for name in ("best_solution","best_features","best_accuracy","best_fitness",
                     "evaluations_","runtime_seconds_","result_"):
            setattr(self,name,getattr(self.delegate_,name))
        return result

    def history_frame(self):
        frame=self.delegate_.history_frame().copy()
        frame["portfolio_strategy"]=self.selected_strategy_
        return frame
