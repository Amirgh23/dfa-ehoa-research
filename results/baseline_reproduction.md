# Baseline reproduction

The repository claims quick-profile hold-out results of 0.9276 balanced accuracy (breast cancer) and 0.9762 (wine), seed 42. These are repository demo results, not the paper's 33-dataset reproduction.

Fresh execution (`python main.py --profile quick --datasets breast_cancer,wine --output results/fresh_baseline --explain none`) reproduced both values exactly: breast cancer selected 15/30 features with 0.9276 balanced accuracy; wine selected 6/13 with 0.9762. The observed difference from the README is zero at the reported precision.

This validates the repository quick baseline only. Full paper reproduction remains unavailable without the original 33-dataset protocol and substantial compute.
