# Conservative novelty comparison

This table is an implementation-positioning checklist, not a systematic-review claim. A current literature search is required before using “to the best of our knowledge” in a paper.

| Method | Stability-aware | Interaction-aware | Feedback search | Stability controls dynamics | Interaction controls transition | EHOA-based | Dual feedback | Difference |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Repository EHOA | No | No | No | No | No | Yes | No | Linear schedules and sigmoid only |
| Stability-regularized FS (general) | Often | Varies | Usually no | Usually no | No | Varies | No | Stability commonly enters objective/reporting |
| mRMR-style selector | No | Redundancy-aware | No | No | No | No | No | Filter ranking rather than binary dynamics |
| DFA-EHOA | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Closed-loop stability control plus guided transition |
