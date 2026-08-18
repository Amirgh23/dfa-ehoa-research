import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ehoa import EHOA
from run_experiments import load_data
from utils import (
    CHAOTIC_MAPS,
    chaotic_sequence,
    clean_dataset,
    calculate_metrics,
    evaluate_classifiers,
    prepare_cv_folds,
)


def test_all_chaotic_maps_are_finite_and_bounded():
    for name in CHAOTIC_MAPS:
        values = chaotic_sequence(name, 100)
        assert np.isfinite(values).all()
        assert ((values > 0) & (values < 1)).all()


def test_clean_dataset_preserves_first_occurrence_order():
    X = np.array([[9.0], [1.0], [9.0], [5.0]])
    y = np.array([1, 0, 1, 0])
    cleaned_X, cleaned_y = clean_dataset(X, y)
    np.testing.assert_array_equal(cleaned_X.ravel(), [9.0, 1.0, 5.0])
    np.testing.assert_array_equal(cleaned_y, [1, 0, 0])


def test_validation_fold_is_not_smote_resampled():
    X, y = make_classification(
        n_samples=80,
        n_features=8,
        weights=[0.8, 0.2],
        random_state=3,
    )
    folds = prepare_cv_folds(X, y, n_folds=4, apply_smote=True)
    assert sum(len(fold.y_valid) for fold in folds) == len(y)
    assert all(len(fold.y_train) >= 60 for fold in folds)


def test_ehoa_is_reproducible_and_returns_nonempty_mask():
    X, y = make_classification(
        n_samples=70,
        n_features=10,
        n_informative=4,
        random_state=7,
    )
    kwargs = dict(
        n_hikers=4,
        max_iter=2,
        n_folds=3,
        random_state=11,
        verbose=False,
    )
    first = EHOA(**kwargs)
    second = EHOA(**kwargs)
    mask_a, accuracy_a, features_a = first.fit(X, y)
    mask_b, accuracy_b, features_b = second.fit(X, y)
    np.testing.assert_array_equal(mask_a, mask_b)
    np.testing.assert_array_equal(features_a, features_b)
    assert accuracy_a == accuracy_b
    assert len(features_a) > 0
    assert first.best_fitness == min(first.convergence_history)


def test_held_out_evaluation_keeps_test_alignment():
    X, y = make_classification(n_samples=100, n_features=8, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5, stratify=y
    )
    metrics, models, _ = evaluate_classifiers(
        X_train, y_train, X_test, y_test, np.arange(X.shape[1]), random_state=5
    )
    assert len(metrics) == 4
    assert set(models) == {"knn", "logistic_regression", "svm", "random_forest"}
    assert metrics["accuracy"].between(0, 1).all()
    assert metrics["roc_auc"].between(0, 1).all()
    assert metrics["pr_auc"].between(0, 1).all()


def test_binary_specificity_is_true_negative_rate():
    # tn=2, fp=1, fn=1, tp=2 => specificity=2/3
    metrics = calculate_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
    assert metrics["specificity"] == 2 / 3


def test_committed_dataset_snapshots_are_complete():
    breast_X, breast_y = load_data("breast_cancer")
    wine_X, wine_y = load_data("wine")
    assert breast_X.shape == (569, 30) and breast_y.shape == (569,)
    assert wine_X.shape == (178, 13) and wine_y.shape == (178,)
    assert set(np.unique(breast_y)) == {0, 1}
    assert set(np.unique(wine_y)) == {0, 1, 2}
