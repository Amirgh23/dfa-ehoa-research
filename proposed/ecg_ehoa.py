"""Evidence-Consensus Gated EHOA (ECG-EHOA)."""
from __future__ import annotations

import numpy as np

from .dfa_ehoa import DFAEHOA
from .transition import sigmoid, transition_probabilities


class ECGEHOA(DFAEHOA):
    """Apply auxiliary guidance only when independent evidence agrees.

    The unguided EHOA probability is a safe fallback.  Reliability and
    interaction guidance must agree in direction and pass confidence/magnitude
    thresholds.  A training-only trust credit grows after inner-fitness
    improvement and decays otherwise; held-out labels never affect it.
    """

    def __init__(self, *args, consensus_confidence=.20, evidence_floor=.05,
                 trust_decay=.85, trust_reward=.20, minimum_trust=.10, **kwargs):
        kwargs.setdefault("feedback_enabled", False)
        kwargs.setdefault("transition_mode", "dual")
        super().__init__(*args, **kwargs)
        self.consensus_confidence=float(consensus_confidence)
        self.evidence_floor=float(evidence_floor)
        self.trust_decay=float(trust_decay)
        self.trust_reward=float(trust_reward)
        self.minimum_trust=float(minimum_trust)
        self.guidance_credit_=1.0
        self.active_guidance_fraction_=0.0

    def fit(self, X, y):
        self.guidance_credit_=1.0
        self.active_guidance_fraction_=0.0
        return super().fit(X, y)

    def _transition(self, position, reliability, interaction, confidence, ls, li):
        baseline=sigmoid(np.asarray(position,float))
        guided=transition_probabilities(position,reliability,interaction,mode="dual",
            lambda_stability=ls,lambda_interaction=li,reliability_confidence=confidence)
        reliable=np.abs(reliability)>=self.evidence_floor
        complementary=np.abs(interaction)>=self.evidence_floor
        agreement=np.sign(reliability)==np.sign(interaction)
        active=reliable & complementary & agreement & (confidence>=self.consensus_confidence)
        trust=np.where(active,confidence*self.guidance_credit_,0.0)
        self.active_guidance_fraction_=float(np.mean(active))
        return np.clip(baseline+trust*(guided-baseline),1e-9,1-1e-9)

    def _record_iteration_outcome(self, improved):
        if improved:
            self.guidance_credit_=min(1.0,self.guidance_credit_+self.trust_reward*(1-self.guidance_credit_))
        else:
            self.guidance_credit_=max(self.minimum_trust,self.guidance_credit_*self.trust_decay)

    def _extra_telemetry(self):
        return {"guidance_credit":self.guidance_credit_,
                "active_guidance_fraction":self.active_guidance_fraction_}
