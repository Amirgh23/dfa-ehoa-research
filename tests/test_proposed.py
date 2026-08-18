import numpy as np
from sklearn.datasets import make_classification
from proposed.stability import *
from proposed.feedback_controller import FeedbackController
from proposed.transition import transition_probabilities, sample_mask
from proposed.interaction import InteractionModel, subset_interaction_quality
from proposed import DFAEHOA, ECGEHOA, MemeticEHOA, AlignedEHOA, GuardedEHOA, PortfolioEHOA, SafePortfolioEHOA
from run_experiments import evaluation_splits, statistics

def test_stability_metrics_known_cases():
    same=np.array([[1,0,1],[1,0,1]],bool); disjoint=np.array([[1,0],[0,1]],bool)
    assert compute_jaccard_stability(same)==1 and compute_nogueira_stability(same)==1
    assert compute_jaccard_stability(disjoint)==0

def test_reliability_and_diversity_bounds():
    masks=np.array([[1,0],[1,1],[0,1]],bool); q,r=compute_feature_reliability(masks)
    assert np.all((q>=0)&(q<=1)) and np.all((r>=-1)&(r<=1))
    assert 0<=compute_population_diversity(masks)<=1

def test_controller_and_transition_bounds():
    sf,w=FeedbackController().update(2,.7,.1,.05,1,(1,3),(.4,.9))
    assert 1<=sf<=3 and .4<=w<=.9
    p=transition_probabilities(np.array([-100,100.]),np.ones(2),np.ones(2)); assert np.all((p>=0)&(p<=1))
    assert sample_mask(np.zeros(4),np.random.default_rng(1)).any()

def test_regime_controller_avoids_instability_positive_feedback():
    controller=FeedbackController(smoothing=0)
    # Late instability with healthy diversity consolidates (below baseline).
    sf_late,_=controller.update(2,.7,.1,.5,0,(1,3),(.4,.9),progress=.9)
    # Early population collapse drives exploration (above baseline).
    sf_early,_=controller.update(2,.7,.9,.01,1,(1,3),(.4,.9),progress=.1)
    assert sf_late < 2 < sf_early

def test_entropy_confidence_gates_ambiguous_reliability():
    z=np.zeros(2); r=np.array([1.,-1.]); interaction=np.zeros(2)
    neutral=transition_probabilities(z,r,interaction,mode="stability",reliability_confidence=np.zeros(2))
    certain=transition_probabilities(z,r,interaction,mode="stability",reliability_confidence=np.ones(2))
    np.testing.assert_allclose(neutral,.5); assert certain[0]>.5 and certain[1]<.5

def test_ecg_falls_back_without_consensus_and_uses_agreed_evidence():
    model=ECGEHOA(n_hikers=2,max_iter=1,n_folds=2,apply_smote=False,random_state=3)
    position=np.zeros(3); confidence=np.ones(3)
    fallback=model._transition(position,np.array([1.,1.,0.]),np.array([-1.,0.,1.]),confidence,.5,.5)
    np.testing.assert_allclose(fallback,.5)
    guided=model._transition(position,np.array([1.,-1.,1.]),np.array([1.,-1.,1.]),confidence,.5,.5)
    assert guided[0]>.5 and guided[1]<.5

def test_ecg_credit_is_bounded_and_integration_is_deterministic():
    X,y=make_classification(n_samples=50,n_features=7,n_informative=3,random_state=14)
    kwargs=dict(n_hikers=3,max_iter=2,n_folds=2,apply_smote=False,random_state=11,verbose=False,warmup=0)
    a=ECGEHOA(**kwargs); b=ECGEHOA(**kwargs)
    ma,_,_=a.fit(X,y); mb,_,_=b.fit(X,y)
    np.testing.assert_array_equal(ma,mb)
    assert a.minimum_trust<=a.guidance_credit_<=1
    assert {"guidance_credit","active_guidance_fraction"}<=set(a.history_frame())

def test_memetic_refinement_never_worsens_inner_objective_and_is_deterministic():
    X,y=make_classification(n_samples=60,n_features=8,n_informative=3,random_state=19)
    kwargs=dict(n_hikers=3,max_iter=2,n_folds=2,apply_smote=False,random_state=17,verbose=False)
    baseline=DFAEHOA(**kwargs,feedback_enabled=False,transition_mode="baseline")
    baseline.fit(X,y)
    a=MemeticEHOA(**kwargs,local_search_rounds=1); b=MemeticEHOA(**kwargs,local_search_rounds=1)
    ma,_,_=a.fit(X,y); mb,_,_=b.fit(X,y)
    np.testing.assert_array_equal(ma,mb)
    assert a.best_fitness<=baseline.best_fitness+1e-12
    assert a.evaluations_>=baseline.evaluations_

def test_metric_aligned_ehoa_is_deterministic_and_budget_matched():
    X,y=make_classification(n_samples=70,n_features=8,n_informative=3,weights=[.75,.25],random_state=23)
    kwargs=dict(n_hikers=3,max_iter=2,n_folds=2,apply_smote=False,random_state=29,verbose=False)
    a=AlignedEHOA(**kwargs); b=AlignedEHOA(**kwargs)
    ma,_,_=a.fit(X,y); mb,_,_=b.fit(X,y)
    np.testing.assert_array_equal(ma,mb)
    assert a.evaluations_==b.evaluations_ and 0<=a.best_accuracy<=1

def test_guarded_ehoa_returns_valid_deterministic_decision():
    X,y=make_classification(n_samples=70,n_features=8,n_informative=3,random_state=31)
    kwargs=dict(n_hikers=3,max_iter=2,n_folds=2,apply_smote=False,random_state=37,verbose=False)
    a=GuardedEHOA(**kwargs); b=GuardedEHOA(**kwargs)
    ma,_,_=a.fit(X,y); mb,_,_=b.fit(X,y)
    np.testing.assert_array_equal(ma,mb)
    assert isinstance(a.abstained_,(bool,np.bool_)) and ma.any()
    if a.abstained_: assert ma.all()

def test_portfolio_uses_preregistered_dimension_rule():
    common=dict(n_hikers=2,max_iter=1,n_folds=2,apply_smote=False,random_state=41,verbose=False)
    low_X,low_y=make_classification(n_samples=50,n_features=8,n_informative=3,random_state=42)
    high_X,high_y=make_classification(n_samples=50,n_features=20,n_informative=3,random_state=43)
    low=PortfolioEHOA(**common); high=PortfolioEHOA(**common)
    low.fit(low_X,low_y); high.fit(high_X,high_y)
    assert low.selected_strategy_=="memetic"
    assert high.selected_strategy_=="interaction"

def test_safe_portfolio_preserves_baseline_in_high_dimension():
    X,y=make_classification(n_samples=50,n_features=20,n_informative=3,random_state=44)
    common=dict(n_hikers=2,max_iter=1,n_folds=2,apply_smote=False,random_state=45,verbose=False)
    safe=SafePortfolioEHOA(**common); reference=__import__("ehoa").EHOA(**common)
    safe_mask,_,_=safe.fit(X,y); reference_mask,_,_=reference.fit(X,y)
    assert safe.selected_strategy_=="baseline"
    np.testing.assert_array_equal(safe_mask,reference_mask)

def test_stagnation():
    d=StagnationDetector(2); assert not d.update(1); assert not d.update(1); assert d.update(1)

def test_interaction_deterministic_and_finite():
    X,y=make_classification(n_samples=60,n_features=8,random_state=2)
    a=InteractionModel(5,3).fit(X,y).score(np.zeros(8,bool)); b=InteractionModel(5,3).fit(X,y).score(np.zeros(8,bool))
    np.testing.assert_allclose(a,b); assert np.isfinite(a).all()
    mask=np.array([1,1,0,0,0,0,0,0],bool)
    assert np.isfinite(subset_interaction_quality(X,y,mask,3))

def test_four_variants_integration_and_determinism():
    X,y=make_classification(n_samples=60,n_features=8,n_informative=3,random_state=4)
    variants=[(False,"baseline"),(True,"stability"),(False,"interaction"),(True,"dual")]
    for feedback,mode in variants:
        kwargs=dict(n_hikers=3,max_iter=2,n_folds=2,apply_smote=False,random_state=9,verbose=False,feedback_enabled=feedback,transition_mode=mode,warmup=0)
        a=DFAEHOA(**kwargs); b=DFAEHOA(**kwargs); ma,_,_=a.fit(X,y); mb,_,_=b.fit(X,y)
        np.testing.assert_array_equal(ma,mb); assert ma.any(); assert len(a.history_frame())==2

def test_selected_indices_reconstruct_equal_width_masks():
    indices=[[0,2],[1],[0,1,2]]; masks=np.zeros((3,3),bool)
    for i, selected in enumerate(indices): masks[i,selected]=True
    assert masks.shape==(3,3) and np.isfinite(compute_nogueira_stability(masks))

def test_training_resampling_is_deterministic_stratified_and_nonempty():
    X,y=make_classification(n_samples=50,n_features=7,n_informative=3,weights=[.8,.2],random_state=8)
    a=resampled_feature_solutions(X,y,4,3,np.random.default_rng(10))
    b=resampled_feature_solutions(X,y,4,3,np.random.default_rng(10))
    np.testing.assert_array_equal(a,b); assert a.shape==(4,7); assert np.all(a.sum(axis=1)==3)

def test_nested_outer_splits_are_deterministic_disjoint_and_complete():
    X,y=make_classification(n_samples=60,n_features=6,random_state=12)
    cfg={"evaluation_protocol":"repeated_nested_cv","seeds":[7],"outer_folds":3}
    first=list(evaluation_splits(X,y,cfg)); second=list(evaluation_splits(X,y,cfg))
    assert [(a,b,c) for a,b,c,_,_ in first]==[(a,b,c) for a,b,c,_,_ in second]
    tests=[]
    for _,_,_,train,test in first:
        assert not set(train)&set(test); tests.extend(test.tolist())
    assert sorted(tests)==list(range(len(y)))

def test_statistics_handle_all_tied_methods_without_nan():
    rows=[]
    for seed in [1,2,3]:
        for method in ["EHOA","SF-EHOA","IG-EHOA","DFA-EHOA"]:
            rows.append({"dataset":"tiny","method":method,"seed":seed,"fold":"outer_1","balanced_accuracy":.5})
    frame=statistics(__import__("pandas").DataFrame(rows))
    assert np.isfinite(frame[["statistic","p_value"]]).all().all()
    assert frame.loc[frame.test=="friedman","p_value"].iloc[0]==1.0
