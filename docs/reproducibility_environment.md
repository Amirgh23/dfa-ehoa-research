# Reproducibility environment

The `paper_revised` artifact was generated on 2026-08-17 with:

- Windows 11 (`10.0.26200`), Intel64 Family 6 Model 165;
- Python 3.13.1;
- NumPy 2.2.5, pandas 3.0.1, SciPy 1.17.1;
- scikit-learn 1.8.0, imbalanced-learn 0.14.2;
- Matplotlib 3.10.7, PyYAML 6.0.3.

The repository pins a clean compatible environment in `requirements.txt`.
Algorithmic seeds are specified in YAML, every split is stratified, estimator
parallelism is disabled, and completed run keys are checkpointed. Small runtime
differences are expected across processors; masks and metrics should be stable
under the pinned dependency set.

The study uses the versioned scikit-learn copies of Breast Cancer Wisconsin
Diagnostic (569 samples, 30 numeric features, 2 classes before duplicate
removal) and Wine Recognition (178 samples, 13 numeric features, 3 classes).
No external patient files or untracked private datasets are required.
