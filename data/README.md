# Local datasets

Dataset files are intentionally ignored by Git. Copy CSV files into this folder
or keep them anywhere on the local machine, then copy
`configs/home_datasets.example.yaml` and edit each absolute `path` and `target`
column. Feature columns must be numeric; labels may be strings or numbers.

Run with:

```powershell
python run_experiments.py --config configs/home_datasets.yaml
```

Completed dataset/method/seed tuples are checkpointed under `results/raw`, so an
interrupted experiment can be resumed with the same command.
