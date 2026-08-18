# Final Research Report: DFA/SFIG-EHOA

**Author:** Amirgh23

**Date:** 17 August 2026

**Status:** implementation and paired small-benchmark evaluation complete

## Abstract

This project extends Enhanced Hiking Optimization Algorithm (EHOA) feature
selection with a regime-aware stability-feedback controller and a
confidence-gated reliability/interaction transition. The untouched EHOA
fitness remains the optimization objective. Four ablated variants were tested
with identical seeds and nominal search budgets on Breast Cancer Wisconsin and
Wine using ten repeated stratified hold-outs. The full DFA/SFIG-EHOA did not
significantly improve balanced accuracy: its mean change versus EHOA was
-0.0056 on Breast Cancer and 0.0000 on Wine. It did, however, improve
chance-corrected Nogueira stability by 0.0726 and 0.0365, respectively, while
selecting 1.4 fewer features on Breast Cancer. This is evidence of a stability-
reduction trade-off, not universal predictive superiority. Runtime was about
3.0--3.2 times the baseline. No Holm-corrected comparison was significant.

## Problem and research gap

Published EHOA uses chaotic initialization, linear sweep adaptation,
velocity-inspired updates, and an S-shaped transfer. It does not use
cross-resampling feature-selection reliability to close the loop around search
dynamics. Recent adaptive optimizers and interaction-aware feature selectors
exist, so the defensible novelty is narrower: the specific combination of
corroborated stability control and entropy-gated reliability/interaction
guidance inside a modular EHOA binary transition.

## Proposed method

Let `q_j` be the exponentially smoothed selection frequency estimated from
class-stratified training resamples, `R_j = 2q_j-1`, and
`c_j = 1-H_2(q_j)` the confidence gate. The transition probability is

`p_ij = sigmoid(x_ij + lambda_s c_j R_j + lambda_i I_j(S))`,

where `I_j(S)` combines target mutual information with cached absolute
correlation redundancy. Correlation is treated as redundancy, not claimed as a
higher-order interaction.

The revised controller uses two drives. Early diversity collapse or stagnation
supports exploration; late instability with adequate diversity supports
consolidation. Instability alone cannot increase exploration. Sweep and inertia
are clipped and smoothed. This fixes the positive-feedback weakness found in
the first prototype.

## Experimental protocol

- Datasets: scikit-learn Breast Cancer Wisconsin (30 features, binary) and Wine
  Recognition (13 features, three classes).
- Independent paired seeds: 10 fixed seeds listed in
  `configs/paper_revised.yaml`.
- Split: repeated 80/20 stratified hold-out; inner training-only 5-fold CV.
- Search: population 8, iterations 10, identical nominal settings for all four
  methods.
- Fitness: unchanged EHOA 5-NN error/feature-ratio objective (`alpha=0.99`).
- Leakage prevention: imputation, scaling, and SMOTE fit only within training;
  reliability and interaction evidence never see the held-out test set.
- Primary held-out metric: balanced accuracy. Secondary metrics: F1, MCC,
  ROC-AUC, PR-AUC, sensitivity, specificity, feature count, redundancy,
  interaction quality, Jaccard and Nogueira stability, runtime and peak memory.
- Statistics: per-dataset Friedman; paired Wilcoxon versus EHOA; Holm correction;
  paired rank-biserial and Cliff's delta; bootstrap CI of median paired change.

Repeated nested CV was not used because the wrapper already performs inner CV
for every candidate and the available experiment budget was bounded. Repeated
stratified hold-out is therefore an explicitly documented, weaker alternative;
results must not be generalized beyond these datasets.

## Main results

| Dataset | Method | Balanced accuracy mean ± SD | Features | Nogueira | Jaccard | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | EHOA | 0.9554 ± 0.0198 | 15.3 | 0.0322 | 0.3589 | 2.99 |
| Breast Cancer | SF-EHOA | **0.9579 ± 0.0185** | **13.2** | 0.0891 | 0.3293 | 9.12 |
| Breast Cancer | IG-EHOA | 0.9574 ± 0.0185 | 14.1 | 0.0499 | 0.3351 | 9.49 |
| Breast Cancer | DFA-EHOA | 0.9498 ± 0.0215 | 13.9 | **0.1048** | 0.3529 | 9.50 |
| Wine | EHOA | 0.9558 ± 0.0321 | **6.8** | 0.2188 | 0.4731 | 2.02 |
| Wine | SF-EHOA | 0.9425 ± 0.0426 | 6.9 | 0.2073 | 0.4697 | 5.75 |
| Wine | IG-EHOA | **0.9563 ± 0.0449** | 7.2 | **0.2667** | **0.5180** | 5.71 |
| Wine | DFA-EHOA | 0.9558 ± 0.0321 | 6.9 | 0.2553 | 0.4957 | 6.08 |

ROC-AUC/PR-AUC for DFA-EHOA were 0.9836/0.9832 on Breast Cancer and
0.9967/0.9921 on Wine. Complete means, standard deviations, medians, IQRs and
95% intervals are stored in `results/aggregated/paper_revised.csv`.

## Ablation interpretation

- **SF-EHOA** gave the best Breast Cancer mean balanced accuracy and smallest
  subset, but lost accuracy on Wine. The feedback/reliability component is
  promising but dataset-dependent.
- **IG-EHOA** gave the best mean Wine balanced accuracy and stability, and the
  best Breast Cancer ROC-AUC/PR-AUC. It is the most promising component for
  wider validation.
- **Full DFA-EHOA** improved Nogueira stability on both datasets. On Breast
  Cancer this came with 1.4 fewer features and a 0.0056 balanced-accuracy loss;
  on Wine balanced accuracy was exactly tied in mean and features were similar.
- Jaccard and Nogueira need not rank methods identically because Nogueira
  corrects for chance and variable subset size. Both are retained.

## Statistical findings

Friedman p-values were 0.3039 (Breast Cancer) and 0.0633 (Wine). For full
DFA-EHOA versus EHOA, paired Wilcoxon p-values were 0.2637 and 1.0000;
Holm-adjusted p-values were 1.0. The paired median balanced-accuracy changes
were -0.0060 (95% bootstrap CI -0.0169 to 0.0069) and 0.0000 (CI 0 to 0).
The Breast Cancer win/tie/loss count was 3/0/7 and Wine was 1/8/1. Therefore no
predictive-superiority claim is accepted.

## Cost and complexity

The wrapper dominates at approximately `O(T P CV C_classifier)`. Stability
adds population-mask comparisons and periodic resampling; interaction setup
adds target mutual information and a top-k correlation cache. Measured full
DFA-EHOA runtime was 3.18x EHOA on Breast Cancer and 3.01x on Wine. Peak-memory,
classifier-time and evaluation-count columns are retained in raw/aggregate
files. Nominal iteration/population budgets are matched; evaluation counts are
also exposed because stochastic cache hits make effective counts differ.

## Claim audit

| Claim | Decision | Evidence |
|---|---|---|
| Novel controller/transition integration is implemented | Accepted | Code, equations, ablation and tests |
| Full method improves chance-corrected stability | Accepted for these runs | Nogueira +0.0726 and +0.0365 |
| Full method improves predictive accuracy | Rejected | negative/tied means; non-significant tests |
| Full method is computationally efficient | Rejected | about 3x runtime |
| Full method is universally superior/state of the art | Rejected | two datasets, no broad competitor suite |
| A meaningful trade-off exists on at least one benchmark | Accepted, scoped | Breast Cancer: fewer features and higher Nogueira with small BA loss |
| Full DFA-EHOA is Pareto non-dominated in BA/stability/feature count | Accepted for both datasets | No variant is no-worse on all three objectives; full method dominates SF-EHOA on Wine |

## Acceptance-oriented positioning

The paper should be defended as a **mechanism and trade-off contribution**, not
as another accuracy-only optimizer. Its strongest point is the separation of
two feedback channels: corroborated population-level evidence changes
continuous search dynamics, while entropy-gated feature-level evidence changes
the discrete transition. The three falsifiable hypotheses are stability gain,
interaction-guided trade-off improvement, and Pareto non-dominance. Current
evidence supports the first descriptively and the third on both datasets; the
second remains dataset-dependent. This framing is stronger and more honest than
claiming state-of-the-art accuracy from a two-dataset study.

## Limitations and threats to validity

Only two low-dimensional public datasets were available in the supplied
repository. Ten repeats are adequate for a course-scale paired study but not a
state-of-the-art benchmark. Repeated hold-out reuses samples, and per-dataset
Friedman tests have limited power. Hyperparameters were fixed before the final
run; no test-result tuning or selective deletion was performed. The
relevance/redundancy model is pairwise and cannot represent arbitrary synergistic
higher-order interactions. Runtime results are platform-specific.

## Reproducibility and deliverables

The repository contains the immutable baseline, proposed package, 16 passing
tests, YAML configuration, all 80 raw runs, 80 convergence traces, aggregate and
statistical CSV files, figures, LaTeX tables, bilingual README, methodology,
novelty comparison, environment record, English LaTeX manuscript and rendered
PDF. `resume: true` makes the experiment interruption-safe. Every conclusion in
this report is recoverable from committed raw data.

## Conclusion

DFA/SFIG-EHOA is a real and testable methodological extension, but the present
evidence supports a stability/reduction trade-off rather than predictive
superiority. The scientifically strongest next step is preregistered validation
of the full method and especially IG-EHOA on many high-dimensional datasets,
with repeated nested CV and evaluation-count-matched stopping.
