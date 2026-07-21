# Article-to-code traceability

This table makes the implementation auditable during review.

| Paper component | Equation / section | Implementation |
|---|---|---|
| Ten chaotic maps | Table 1 | `utils.py`: `CHAOTIC_MAPS` and map functions |
| Chaotic population initialization | Eq. 6 | `utils.py`: `chaotic_sequence`; `ehoa.py`: `_initialize_population` |
| Adaptive sweep factor | Eq. 7 | `ehoa.py`: `_adaptive_sweep_factor` |
| Linearly decreasing inertia | Section 4.3 | `ehoa.py`: `_inertia_weight` |
| Enhanced velocity | Eq. 9 | `ehoa.py`: main update loop in `fit` |
| S-shaped transfer | Eq. 10 | `utils.py`: `sigmoid` |
| Stochastic binary decision | Eq. 11 | `ehoa.py`: `_to_mask` |
| Accuracy/feature-count fitness | Eq. 12 | `utils.py`: `evaluate_mask_cv` |
| 5-NN wrapper | Section 5.2 | `utils.py`: `evaluate_mask_cv` |
| 10-fold stratified CV | Sections 5.2/6.2 | `utils.py`: `prepare_cv_folds` |
| Mean imputation | Section 6.2 | `prepare_cv_folds`, `preprocess_train_test` |
| Duplicate removal | Section 6.2 | `utils.py`: `clean_dataset` |
| SMOTE | Section 6.2 | `utils.py`: `_resample_training`, applied to train only |
| Four classifiers | Section 7.2 | `utils.py`: `classifier_factories` |
| Imbalance-aware metrics | Eq. 19 | `utils.py`: `calculate_metrics` |
| Global feature explanation | Section 5.4 | `main.py`: `explain_features` |
| Independent trials | Section 6.4 | `main.py`: `--runs`; paper profile uses 20 |

## Deliberate engineering clarifications

- Continuous positions and binary masks are stored separately so Eq. 9 remains
  dimensionally consistent.
- The continuous search bounds default to `[-6, 6]`, allowing the sigmoid to
  represent both low and high selection probabilities.
- Imputation, scaling, and SMOTE are fitted inside each training fold to prevent
  leakage. This is stricter than applying preprocessing once before CV.
- The held-out test split is not used by EHOA and is evaluated only after the
  feature subset has been finalized.

