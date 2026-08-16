# Final ablation report

```csv
dataset,method,balanced_accuracy_mean,f1_score_mean,mcc_mean,selected_features_mean,jaccard_stability_mean,redundancy_mean,runtime_seconds_mean
breast_cancer,DFA-EHOA,0.9643,0.9721,0.9254,14.2,0.3508,0.4031,2.6524
breast_cancer,EHOA,0.9597,0.9661,0.9118,13.0,0.3645,0.4408,1.3636
breast_cancer,IG-EHOA,0.9492,0.9607,0.8959,14.2,0.3604,0.4068,2.7083
breast_cancer,SF-EHOA,0.9544,0.9652,0.9068,14.2,0.3747,0.4141,2.5505
wine,DFA-EHOA,0.9486,0.9463,0.9193,7.4,0.4555,0.3191,1.411
wine,EHOA,0.9343,0.9295,0.8957,6.6,0.4468,0.3357,0.7008
wine,IG-EHOA,0.9287,0.9225,0.8885,8.0,0.4987,0.3042,1.3424
wine,SF-EHOA,0.9438,0.9406,0.9111,6.6,0.4439,0.3354,1.4183
```

Statistical results are exploratory unless run counts and datasets meet the registered protocol.

## Component effects

- **Stability feedback:** did not improve Nogueira stability and did not yield a consistent predictive gain. The controller/reliability mechanism requires redesign.
- **Interaction guidance:** reduced mean correlation redundancy on both datasets, but its independent interaction-quality score improved only on breast cancer. Evidence is partial.
- **Combination:** DFA had the highest descriptive balanced accuracy in the same-iteration experiment, but selected more features, cost more, and was not statistically significant.
- **Cost justification:** not established. Doubling baseline EHOA iterations removed DFA's descriptive predictive advantage.

## Claim decisions

| Claim | Decision | Evidence |
|---|---|---|
| Stability feedback improves subset stability | Rejected for current implementation | Nogueira decreased on both datasets |
| Interaction guidance reduces redundancy | Partially supported | Redundancy decreased on both; interaction quality mixed |
| Dual feedback gives the best trade-off | Not supported | More features, higher runtime, no significance, cost-matched baseline competitive |
