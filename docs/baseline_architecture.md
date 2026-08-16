# Baseline EHOA architecture

## Execution flow

`main.py` loads a built-in dataset, removes exact duplicates, creates a stratified hold-out split, and calls `EHOA.fit` **only on training data**. `utils.prepare_cv_folds` fits mean imputation and standardization on each inner-training fold; SMOTE is applied only after those transforms and only to that training fold. The untouched hold-out is evaluated after selection.

## Paper-to-code map

| Component | Implementation | Status |
|---|---|---|
| Ten chaotic maps / chaotic initialization | `utils.py:47-124`, `ehoa.py:_initialize_population` | paper component |
| Adaptive sweep factor | `ehoa.py:_adaptive_sweep_factor` | linear Eq. 7 baseline |
| Inertia weight | `ehoa.py:_inertia_weight` | linear paper enhancement |
| Velocity update, personal/global best | `ehoa.py:168-209` | PSO-like Eq. 9 |
| S-shaped transfer / binary mask | `utils.sigmoid`, `EHOA._to_mask` | Eqs. 10-11; non-empty repair is engineering |
| Fitness | `utils.evaluate_mask_cv` | `alpha*(1-accuracy)+(1-alpha)*feature_ratio`, Eq. 12 |
| Classification | deterministic 5-NN inside stratified CV | paper component |
| Explainability | `main.explain_features` | repository engineering; permutation or optional SHAP |
| Imbalance | `utils._resample_training` | fold-local SMOTE; leakage-safe engineering |
| Repeated seeds | `main.run_dataset` | seed + run index |

The baseline stores continuous position separately from the sampled mask, caches identical masks, clips positions/velocities, repairs empty masks, and offers early stopping. These are repository engineering clarifications. It has no stability feedback, per-feature reliability, interaction guidance, nested outer CV, or statistical multi-dataset runner.

## Optimization flow

Chaotic positions → sigmoid masks → inner-CV fitness → personal/global best → linearly scheduled SF/inertia → velocity/position update → stochastic mask → cached fitness → convergence history. `EHOA` remains unchanged; DFA-EHOA lives under `proposed/`.
