# DFA-EHOA research report

## 1. Repository Analysis
See `docs/baseline_architecture.md`. The unchanged baseline is leakage-aware for its inner CV and hold-out protocol.

## 2. Baseline Reproduction
See `results/baseline_reproduction.md`. Committed quick results are not treated as newly executed evidence.

## 3–5. Implemented Novelty, Mathematics, Protocol
See `docs/dfa_ehoa_methodology.md` and the YAML configs. The implementation supports the required 2x2 design, deterministic seeds, resume, raw traces, and evaluation counts.

## 6–13. Final Available Results, Stability, Interaction, Runtime, Statistics, Sensitivity, Failures

The final available 2-dataset × 4-method × 5-seed ablation used population 8, 8 iterations and 5 inner folds. Breast-cancer mean balanced accuracy was EHOA 0.9597, SF 0.9544, IG 0.9492 and DFA 0.9643. Wine values were 0.9343, 0.9438, 0.9287 and 0.9486. DFA selected more features than EHOA on both datasets (14.2 vs 13.0; 7.4 vs 6.6).

Selection stability did not improve: DFA Nogueira was 0.0774 vs EHOA 0.1584 on breast cancer and 0.0903 vs 0.1998 on wine. IG reduced mean redundancy on both datasets (0.4068 vs 0.4408; 0.3042 vs 0.3357), but interaction quality improved only on breast cancer (-0.2864 vs -0.3284; higher is better) and worsened on wine (0.3949 vs 0.4643). Thus Claim 1 is contradicted, while Claim 2 has partial metric-dependent support.

Friedman p-values were 0.2063 and 0.9288. DFA-vs-EHOA Wilcoxon p-values were 0.75 and 0.625 with Cliff's delta 0.12 and 0.16; no Holm-adjusted comparison was significant. Same-iteration runtime for DFA was roughly twice EHOA on breast cancer. In the cost-oriented comparison, 16-iteration EHOA reached 0.9663 vs 8-iteration DFA 0.9643 on breast cancer and tied DFA at 0.9486 on wine; the apparent same-iteration predictive advantage therefore disappeared.

A 90-run one-factor-at-a-time sensitivity experiment covered the requested controller coefficients, transition weights, bootstrap count, patience, interaction top-k, reliability smoothing and every/periodic/event strategies. Results were highly tied under the deliberately small 4-iteration sensitivity budget; `interaction_top_k=10` was descriptively best at 0.9507. The earlier 8-iteration sensitivity showed bootstrap-count sensitivity (5 outperformed 3 and 8), so robustness is not established.

Raw data, traces, aggregates, statistics, figures and CSV/Markdown/LaTeX tables are stored under `results/`.

## 14. Threats to Validity
The current interaction proxy combines mutual-information relevance with correlation-based redundancy; it is not exact conditional mutual information. Reliability uses independent stratified bootstrap MI solutions but does not recursively rerun the full wrapper selector. Hold-out experiments are not repeated nested CV. Normal-approximation CIs are weak for five runs. Stability is one across-run estimate repeated in raw rows, so it must not be interpreted as five independent stability observations. The original benchmark datasets are absent.

## 15–17. Novelty Assessment and Claims
Code-level novelty is measurable: stability changes SF/inertia and reliability plus interaction changes transition probabilities. The available evidence does **not** validate the proposed full DFA-EHOA as a superior trade-off: stability decreased, feature count increased, differences were not significant, and a stronger compute-matched EHOA removed the predictive advantage. Interaction guidance alone shows partial redundancy evidence and is the only component worth carrying forward for redesign.

## 18–19. Recommended Contributions and Future Work
Position the contribution conservatively as a closed-loop, stability-informed EHOA with reliability/interaction-guided binary transition. Next priorities are true training-data bootstrap refits, repeated nested CV, more datasets, evaluation-budget matching, exact/sparse conditional interaction estimators, and preregistered sensitivity ranges.
