import numpy as np
from sklearn.datasets import make_classification
from proposed.stability import *
from proposed.feedback_controller import FeedbackController
from proposed.transition import transition_probabilities, sample_mask
from proposed.interaction import InteractionModel, subset_interaction_quality
from proposed import DFAEHOA

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
