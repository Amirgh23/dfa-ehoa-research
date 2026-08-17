"""Dual-Feedback Adaptive EHOA; baseline EHOA remains unchanged in ehoa.py."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from ehoa import EHOA, EHOAResult
from .feedback_controller import FeedbackController
from .interaction import InteractionModel
from .stability import (StagnationDetector, compute_feature_reliability,
    compute_jaccard_stability, compute_nogueira_stability, compute_population_diversity,
    resampled_feature_solutions)
from .transition import sample_mask, scheduled_weight, transition_probabilities


class DFAEHOA(EHOA):
    """2x2-ablatable DFA-EHOA.

    `feedback_enabled` controls SF/inertia; transition_mode independently controls
    reliability and interaction injection. All estimates use only X passed to fit.
    """
    def __init__(self, *args, feedback_enabled=True, transition_mode="dual",
                 stability_interval=2, stability_strategy="periodic", warmup=2,
                 reliability_rho=.6, bootstrap_count=5, interaction_top_k=50,
                 resample_fraction=.8, reliability_source="resampling",
                 lambda_stability=.5, lambda_interaction=.5, weight_schedule="adaptive",
                 stagnation_patience=5, stagnation_epsilon=1e-6, controller=None, **kwargs):
        eta_exploration=kwargs.pop("eta_exploration",.45); eta_consolidation=kwargs.pop("eta_consolidation",.30)
        super().__init__(*args, **kwargs)
        self.feedback_enabled, self.transition_mode = feedback_enabled, transition_mode
        self.stability_interval, self.stability_strategy, self.warmup = stability_interval, stability_strategy, warmup
        self.reliability_rho, self.bootstrap_count = reliability_rho, bootstrap_count
        self.resample_fraction, self.reliability_source = resample_fraction, reliability_source
        self.interaction_top_k = interaction_top_k
        self.lambda_stability, self.lambda_interaction = lambda_stability, lambda_interaction
        self.weight_schedule = weight_schedule
        self.stagnation_patience, self.stagnation_epsilon = stagnation_patience, stagnation_epsilon
        self.controller = controller or FeedbackController(eta_exploration,eta_consolidation)

    def _should_update(self, iteration, improved, detector, diversity, previous_diversity):
        if iteration <= self.warmup: return False
        if self.stability_strategy == "every": return True
        if self.stability_strategy == "periodic": return iteration % max(1, self.stability_interval) == 0
        if self.stability_strategy == "event": return improved or detector.counter > 0 or abs(diversity-previous_diversity) > .1
        raise ValueError("stability_strategy must be every, periodic, or event")

    def fit(self, X, y):
        started=time.perf_counter(); self._reset_state(); self._rng=np.random.default_rng(self.random_state)
        X=np.asarray(X,float); y=np.asarray(y,int)
        self._folds = __import__('utils').prepare_cv_folds(X,y,self.n_folds,self.random_state,self.apply_smote)
        n_features=X.shape[1]; population,velocities=self._initialize_population(n_features)
        personal_positions=population.copy(); personal_fitness=np.full(self.n_hikers,np.inf)
        interaction=np.zeros(n_features); self.interaction_seconds_=0.0
        if self.transition_mode in {"interaction","dual"}:
            model=InteractionModel(self.interaction_top_k,self.random_state).fit(X,y)
            interaction=model.score(np.zeros(n_features,bool)); self._interaction_model=model; self.interaction_seconds_=model.computation_seconds
        reliability=np.zeros(n_features); q=None; masks=[]
        for i in range(self.n_hikers):
            mask=sample_mask(transition_probabilities(population[i],reliability,interaction,mode=self.transition_mode),self._rng)
            masks.append(mask); f,a,_=self._evaluate(mask); personal_fitness[i]=f; self._consider_global(population[i],mask,f,a)
        detector=StagnationDetector(self.stagnation_patience,self.stagnation_epsilon)
        self.telemetry_=[]; previous_control=None; previous_diversity=compute_population_diversity(np.array(masks))
        stability_seconds=0.; velocity_limit=self.position_max-self.position_min
        for iteration in range(1,self.max_iter+1):
            current_masks=np.asarray(masks,bool); diversity=compute_population_diversity(current_masks)
            jaccard=compute_jaccard_stability(current_masks); nogueira=compute_nogueira_stability(current_masks)
            sf_base=self._adaptive_sweep_factor(iteration); w_base=self._inertia_weight(iteration)
            stagnation=detector.signal
            if self.feedback_enabled and iteration>self.warmup:
                sf,inertia=self.controller.update(sf_base,w_base,max(0,nogueira),diversity,stagnation,
                    (self.sf_min,self.sf_max),(self.w_min,self.w_max),previous_control,
                    progress=iteration/self.max_iter); previous_control=(sf,inertia)
            else: sf,inertia=sf_base,w_base
            improved=False; new_masks=[]
            for i in range(self.n_hikers):
                r1=self._rng.random(n_features); r2=self._rng.random(n_features)
                velocities[i]=np.clip(inertia*velocities[i]+sf*(self.c1*r1*(personal_positions[i]-population[i])+self.c2*r2*(self.best_position-population[i])),-velocity_limit,velocity_limit)
                population[i]=np.clip(population[i]+velocities[i],self.position_min,self.position_max)
                ls=scheduled_weight(self.lambda_stability,iteration,self.max_iter,self.weight_schedule,1-max(0,nogueira))
                li=scheduled_weight(self.lambda_interaction,iteration,self.max_iter,self.weight_schedule,diversity)
                local_interaction=self._interaction_model.score(self.best_solution) if self.transition_mode in {"interaction","dual"} else interaction
                if q is None:
                    confidence=np.zeros(n_features)
                else:
                    q_safe=np.clip(q,1e-12,1-1e-12)
                    entropy=-(q_safe*np.log2(q_safe)+(1-q_safe)*np.log2(1-q_safe))
                    confidence=1-entropy
                mask=sample_mask(transition_probabilities(population[i],reliability,local_interaction,mode=self.transition_mode,lambda_stability=ls,lambda_interaction=li,reliability_confidence=confidence),self._rng)
                new_masks.append(mask); f,a,_=self._evaluate(mask)
                if f<personal_fitness[i]: personal_fitness[i]=f; personal_positions[i]=population[i].copy()
                improved |= self._consider_global(population[i],mask,f,a)
            masks=new_masks; detector.update(self.best_fitness)
            if self._should_update(iteration,improved,detector,diversity,previous_diversity):
                tick=time.perf_counter()
                if self.reliability_source == "resampling":
                    evidence=resampled_feature_solutions(X,y,self.bootstrap_count,len(self.best_features),self._rng,self.resample_fraction)
                elif self.reliability_source == "population":
                    evidence=np.asarray(masks,bool)
                    if self.bootstrap_count>len(evidence): evidence=evidence[self._rng.integers(0,len(evidence),self.bootstrap_count)]
                else: raise ValueError("reliability_source must be resampling or population")
                q,reliability=compute_feature_reliability(evidence,q,self.reliability_rho); stability_seconds+=time.perf_counter()-tick
            previous_diversity=diversity
            self.convergence_history.append(self.best_fitness); self.accuracy_history.append(self.best_accuracy)
            self.feature_count_history.append(len(self.best_features)); self.diversity_history.append(diversity)
            finite=personal_fitness[np.isfinite(personal_fitness)]
            delta=sf-sf_base
            controller_state="explore" if delta>1e-9 else ("consolidate" if delta < -1e-9 else "baseline")
            self.telemetry_.append(dict(iteration=iteration,fitness=self.best_fitness,
                mean_population_fitness=float(finite.mean()) if len(finite) else np.nan,
                cv_accuracy=self.best_accuracy,
                selected_features=len(self.best_features),population_diversity=diversity,jaccard_stability=jaccard,
                nogueira_stability=nogueira,sweep_factor=sf,inertia=inertia,stagnation_counter=detector.counter))
            self.telemetry_[-1]["controller_state"]=controller_state
        self.stability_seconds_=stability_seconds; self.runtime_seconds_=time.perf_counter()-started
        self.result_=EHOAResult(self.best_solution.copy(),self.best_features.copy(),self.best_accuracy,self.best_fitness,self.runtime_seconds_,self.evaluations_)
        return self.best_solution.copy(),self.best_accuracy,self.best_features.copy()

    def history_frame(self): return pd.DataFrame(self.telemetry_)
