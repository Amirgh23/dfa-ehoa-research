# Conservative novelty comparison (search updated 2026-08-17)

This is a scoped comparison, not proof that no similar method exists. Searches
covered EHOA/HOA feature selection, adaptive metaheuristics, stability feedback,
feature reliability, and interaction-guided binary transitions. The defensible
claim is the *specific integration and controller logic*, not ownership of
adaptive control, feature interaction, or stability estimation individually.

| Method | Adaptive search state | Feature interaction/redundancy | Resampling stability controls dynamics | Confidence-gated reliability transition | EHOA-based |
|---|---:|---:|---:|---:|---:|
| Hegazy et al. EHOA (2026), DOI 10.1007/s10586-026-05946-9 | Linear dynamic sweep | No search guidance | No | No | Yes |
| Abdel-salam et al. AEDHOA (2025), DOI 10.1016/j.knosys.2025.113286 | Four adaptive/diversification strategies | Not cross-resampling reliability | No | No | HOA |
| Zhang et al. TQFEO (2025), DOI 10.1016/j.knosys.2025.113323 | Twin Q-learning and fitness-variance status | MIC/ReliefF and Manhattan guidance | No | No | No |
| Lian et al. TAM (2025), DOI 10.1016/j.asoc.2025.113505 | Historical trend/covariance | No FS-specific reliability transition | No | No | No |
| **DFA/SFIG-EHOA (this work)** | Regime-aware exploration/consolidation | Training-only relevance/redundancy transition | Yes | Yes | Yes |

## Defensible contribution statement

The contribution is a modular EHOA extension in which chance-corrected
cross-resampling selection evidence is used twice but with separate safeguards:
(1) a regime-aware controller changes sweep and inertia only under corroborated
population states; and (2) entropy-gated per-feature reliability, together with a
target-relevance/redundancy signal, biases the binary transition. A full 2x2
ablation isolates the two mechanisms while leaving the baseline wrapper fitness
unchanged.

## Claims intentionally avoided

- “first adaptive metaheuristic for feature selection”;
- “first interaction-aware feature selector”;
- “universally superior” or “state of the art”;
- causal claims from two datasets;
- treating Pearson redundancy as a higher-order statistical interaction.
