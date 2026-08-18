# Versioned datasets

This directory makes the reported experiment fully self-contained. The CSV
snapshots are exported from the scikit-learn 1.8 dataset loaders without row
sampling or feature transformation.

| File | Rows | Features | Target | Source DOI | License |
|---|---:|---:|---|---|---|
| `breast_cancer.csv` | 569 | 30 | `target` (0/1) | 10.24432/C5DW2B | CC BY 4.0 |
| `wine.csv` | 178 | 13 | `target` (0/1/2) | 10.24432/C5PC7J | CC BY 4.0 |

The runner automatically prefers these committed snapshots. The scikit-learn
loaders are retained only as a compatibility fallback. SHA-256 digests are in
`SHA256SUMS.txt`.

Dataset licensing and attribution are separate from the repository's MIT code
license; see `DATA_LICENSE.md`.
