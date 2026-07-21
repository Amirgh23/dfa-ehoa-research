"""Reproducible command-line experiment for EHOA feature selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from ehoa import EHOA
from utils import clean_dataset, evaluate_classifiers, preprocess_train_test


PAPER_URL = "https://doi.org/10.1007/s10586-026-05946-9"


def load_builtin_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if name == "breast_cancer":
        bunch = load_breast_cancer()
        return bunch.data, bunch.target, list(bunch.feature_names)
    if name == "wine":
        bunch = load_wine()
        return bunch.data, bunch.target, list(bunch.feature_names)
    if name == "high_dimensional":
        X, y = make_classification(
            n_samples=300,
            n_features=500,
            n_informative=20,
            n_redundant=30,
            weights=[0.7, 0.3],
            random_state=42,
        )
        return X, y, [f"feature_{i}" for i in range(X.shape[1])]
    raise ValueError(f"Unknown dataset {name!r}")


def plot_convergence(selector: EHOA, output: Path, title: str) -> None:
    history = selector.history_frame()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history["iteration"], history["fitness"], color="#2563eb")
    axes[0, 0].set(title="Fitness convergence", xlabel="Iteration", ylabel="Fitness")
    axes[0, 1].plot(history["iteration"], history["cv_accuracy"], color="#16a34a")
    axes[0, 1].set(title="Cross-validation accuracy", xlabel="Iteration", ylabel="Accuracy")
    axes[1, 0].plot(history["iteration"], history["selected_features"], color="#dc2626")
    axes[1, 0].set(title="Selected feature count", xlabel="Iteration", ylabel="Features")
    axes[1, 1].plot(history["iteration"], history["population_diversity"], color="#9333ea")
    axes[1, 1].set(title="Population diversity", xlabel="Iteration", ylabel="Mean std")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(model, X_test: np.ndarray, y_test: np.ndarray, output: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, model.predict(X_test), cmap="Blues", colorbar=False, ax=axis
    )
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_comparison(comparison: pd.DataFrame, output: Path, title: str) -> None:
    """Plot held-out balanced accuracy before and after feature selection."""
    plot_data = comparison.copy()
    plot_data["feature_set"] = plot_data["feature_set"].map(
        {"ehoa": "EHOA-selected", "all_features": "All features"}
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=plot_data,
        x="classifier",
        y="balanced_accuracy",
        hue="feature_set",
        palette=["#2563eb", "#94a3b8"],
        ax=axis,
    )
    axis.set(
        title=title,
        xlabel="Classifier",
        ylabel="Held-out balanced accuracy",
        ylim=(0.0, 1.05),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Feature set", loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def explain_features(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output: Path,
    method: str,
    random_state: int,
) -> pd.DataFrame:
    """Produce honest global importances; SHAP is optional, permutation is default."""
    importances: np.ndarray
    actual_method = method
    if method == "shap":
        try:
            import shap

            background = shap.sample(X_train, min(50, len(X_train)), random_state=random_state)
            sample = X_test[: min(80, len(X_test))]
            explainer = shap.Explainer(model.predict_proba, background)
            values = np.asarray(explainer(sample).values)
            if values.ndim == 3:
                importances = np.mean(np.abs(values), axis=(0, 2))
            else:
                importances = np.mean(np.abs(values), axis=0)
        except Exception as error:
            print(f"warning: SHAP unavailable ({error}); using permutation importance", file=sys.stderr)
            actual_method = "permutation"
    if actual_method == "permutation":
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="balanced_accuracy",
            n_repeats=20,
            random_state=random_state,
            n_jobs=1,
        )
        importances = result.importances_mean

    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    shown = frame.head(20).sort_values("importance")
    fig, axis = plt.subplots(figsize=(9, max(4, len(shown) * 0.35)))
    axis.barh(shown["feature"], shown["importance"], color="#2563eb")
    axis.set_xlabel("Mean |SHAP value|" if actual_method == "shap" else "Permutation importance")
    axis.set_title(f"Global feature importance ({actual_method})")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    frame["method"] = actual_method
    return frame


def run_dataset(name: str, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    X, y, feature_names = load_builtin_dataset(name)
    X, y = clean_dataset(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"\n[{name}] samples={len(X)}, features={X.shape[1]}, classes={len(np.unique(y))}")

    selectors: list[EHOA] = []
    run_rows = []
    for run in range(args.runs):
        selector = EHOA(
            n_hikers=args.hikers,
            max_iter=args.iterations,
            n_folds=args.folds,
            chaotic_map=args.chaotic_map,
            alpha=args.alpha,
            apply_smote=not args.no_smote,
            random_state=args.seed + run,
            patience=args.patience,
            verbose=args.verbose,
        )
        _, cv_accuracy, features = selector.fit(X_train, y_train)
        selectors.append(selector)
        run_rows.append(
            {
                "run": run + 1,
                "seed": args.seed + run,
                "fitness": selector.best_fitness,
                "cv_accuracy": cv_accuracy,
                "selected_features": len(features),
                "reduction_percent": 100 * (1 - len(features) / X.shape[1]),
                "evaluations": selector.evaluations_,
                "runtime_seconds": selector.runtime_seconds_,
            }
        )

    best = min(selectors, key=lambda item: item.best_fitness)
    selected = best.best_features
    selected_metrics, models, _ = evaluate_classifiers(
        X_train,
        y_train,
        X_test,
        y_test,
        selected,
        apply_smote=not args.no_smote,
        random_state=args.seed,
    )
    baseline_metrics, _, _ = evaluate_classifiers(
        X_train,
        y_train,
        X_test,
        y_test,
        np.arange(X.shape[1]),
        apply_smote=not args.no_smote,
        random_state=args.seed,
    )
    selected_metrics["feature_set"] = "ehoa"
    baseline_metrics["feature_set"] = "all_features"
    comparison = pd.concat([selected_metrics, baseline_metrics], ignore_index=True)

    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(dataset_dir / "runs.csv", index=False)
    comparison.to_csv(dataset_dir / "classifier_metrics.csv", index=False)
    best.history_frame().to_csv(dataset_dir / "convergence.csv", index=False)
    pd.DataFrame(
        {
            "feature_index": selected,
            "feature_name": [feature_names[index] for index in selected],
        }
    ).to_csv(dataset_dir / "selected_features.csv", index=False)
    plot_convergence(best, dataset_dir / "convergence.png", f"EHOA — {name}")
    plot_classifier_comparison(
        comparison,
        dataset_dir / "classifier_comparison.png",
        f"EHOA vs. all features — {name}",
    )

    X_train_p, y_train_p, X_test_p, _, _ = preprocess_train_test(
        X_train,
        y_train,
        X_test,
        apply_smote=not args.no_smote,
        random_state=args.seed,
    )
    X_train_selected = X_train_p[:, selected]
    X_test_selected = X_test_p[:, selected]
    plot_confusion(
        models["knn"],
        X_test_selected,
        y_test,
        dataset_dir / "confusion_matrix_knn.png",
        f"Held-out test confusion matrix — {name}",
    )
    if args.explain != "none":
        importance = explain_features(
            models["knn"],
            X_train_selected,
            X_test_selected,
            y_test,
            [feature_names[index] for index in selected],
            dataset_dir / "feature_importance.png",
            args.explain,
            args.seed,
        )
        importance.to_csv(dataset_dir / "feature_importance.csv", index=False)

    knn_test = selected_metrics.loc[selected_metrics["classifier"] == "knn"].iloc[0]
    print(
        f"best: cv_accuracy={best.best_accuracy:.4f}, test_accuracy={knn_test['accuracy']:.4f}, "
        f"features={len(selected)}/{X.shape[1]}, reduction={100 * (1-len(selected)/X.shape[1]):.1f}%"
    )
    return {
        "dataset": name,
        "samples": int(len(X)),
        "total_features": int(X.shape[1]),
        "selected_features": int(len(selected)),
        "reduction_percent": float(100 * (1 - len(selected) / X.shape[1])),
        "best_cv_accuracy": float(best.best_accuracy),
        "best_fitness": float(best.best_fitness),
        "knn_test_accuracy": float(knn_test["accuracy"]),
        "knn_test_balanced_accuracy": float(knn_test["balanced_accuracy"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "paper"], default="quick")
    parser.add_argument(
        "--datasets",
        default="breast_cancer,wine",
        help="Comma-separated: breast_cancer,wine,high_dimensional",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--chaotic-map", default="tent")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hikers", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--explain", choices=["permutation", "shap", "none"], default="permutation")
    parser.add_argument("--verbose", action="store_true")
    return parser


def write_experiment_report(
    summaries: list[dict[str, object]],
    args: argparse.Namespace,
    output: Path,
) -> None:
    """Create a compact, professor-ready report from the exact run artifacts."""
    rows = []
    for item in summaries:
        rows.append(
            "| {dataset} | {selected_features}/{total_features} | {reduction_percent:.2f}% "
            "| {best_cv_accuracy:.4f} | {knn_test_accuracy:.4f} "
            "| {knn_test_balanced_accuracy:.4f} |".format(**item)
        )
    report = f"""# گزارش اجرای EHOA

این گزارش به‌صورت خودکار از artifactهای همین اجرا تولید شده است؛ بنابراین اعداد آن
با فایل‌های CSV پوشه‌ی نتایج یکسان‌اند.

## تنظیمات

- Profile: `{args.profile}`
- Chaotic map: `{args.chaotic_map}`
- Population / iterations: `{args.hikers}` / `{args.iterations}`
- CV folds / independent runs: `{args.folds}` / `{args.runs}`
- Fitness alpha: `{args.alpha}`
- Random seed: `{args.seed}`
- SMOTE: `{'disabled' if args.no_smote else 'training folds only'}`

## خلاصه نتایج

| Dataset | Selected/total | Reduction | CV accuracy | Test accuracy | Test balanced accuracy |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## تفسیر صحیح

`CV accuracy` فقط روی داده‌ی آموزش و برای بهینه‌سازی subset محاسبه شده است.
`Test accuracy` و `Test balanced accuracy` روی hold-out دست‌نخورده محاسبه شده‌اند
و معیار اصلی تعمیم هستند. پروفایل quick برای دموی کد است؛ ادعای بازتولید مقاله
به پروفایل paper، ۲۰ اجرای مستقل و مجموعه‌داده‌های اصلی نیاز دارد.

برای جزئیات هر classifier، ویژگی‌های منتخب و نمودارها به زیرپوشه‌ی هر dataset
مراجعه کنید.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> list[dict[str, object]]:
    args = build_parser().parse_args(argv)
    defaults = {
        "quick": {"hikers": 8, "iterations": 8, "folds": 5, "runs": 1, "patience": None},
        "paper": {"hikers": 30, "iterations": 50, "folds": 10, "runs": 20, "patience": None},
    }[args.profile]
    for field, value in defaults.items():
        if getattr(args, field) is None:
            setattr(args, field, value)

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "paper": PAPER_URL,
        "profile": args.profile,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "parameters": vars(args) | {"output": str(args.output)},
    }
    (args.output / "experiment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summaries = [
        run_dataset(name.strip(), args, args.output)
        for name in args.datasets.split(",")
        if name.strip()
    ]
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output / "summary.csv", index=False)
    write_experiment_report(summaries, args, args.output)
    print("\n" + summary_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return summaries


if __name__ == "__main__":
    main()
