"""Run the full data-size and minority-ratio experiment."""

import argparse
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from data import load_data, split_data
from models import build_model


TRAIN_SIZES = [500, 1_000, 2_000, 5_000]
MINORITY_RATIOS = [0.50, 0.30, 0.10, 0.05, 0.01]
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["logistic_regression", "random_forest"]


def make_subset(X, y, n_total, minority_ratio, seed):
    """Sample an exact yes/no ratio from the training partition without replacement."""
    n_yes = round(n_total * minority_ratio)
    n_no = n_total - n_yes
    available_yes = int((y == 1).sum())
    available_no = int((y == 0).sum())

    if n_yes > available_yes or n_no > available_no:
        raise ValueError(
            f"Cannot create n={n_total}, yes_ratio={minority_ratio:.0%}: "
            f"requested yes/no={n_yes}/{n_no}, available={available_yes}/{available_no}"
        )

    yes_indices = y[y == 1].sample(n=n_yes, random_state=seed).index
    no_indices = y[y == 0].sample(n=n_no, random_state=seed).index
    subset_indices = yes_indices.append(no_indices)
    shuffled_indices = X.loc[subset_indices].sample(frac=1, random_state=seed).index

    return X.loc[shuffled_indices].copy(), y.loc[shuffled_indices].copy()


def evaluate(model, X_test, y_test):
    """Return the fixed test-set metrics for the positive class yes=1."""
    predictions = model.predict(X_test)
    yes_probabilities = model.predict_proba(X_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[1],
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision_score(y_test, yes_probabilities),
    }


def _unique_output_path(output_dir, stem):
    """Choose a dated result path without overwriting an earlier run."""
    date_stamp = datetime.now().astimezone().strftime("%Y-%m-%d")
    output_path = output_dir / f"{stem}_{date_stamp}.csv"
    if output_path.exists():
        time_stamp = datetime.now().astimezone().strftime("%H%M%S")
        output_path = output_dir / f"{stem}_{date_stamp}_{time_stamp}.csv"
    return output_path


def write_run_log(output_path, results, skipped, X_train, y_train, X_test, y_test, quick):
    """Append run settings, software versions, and anomaly checks to the log."""
    import sklearn
    zero_recall_rows = int((results["recall"] == 0).sum())
    group_columns = ["model", "n_train", "minority_ratio"]
    all_seed_zero_groups = results.groupby(group_columns)["recall"].max()
    all_seed_zero_groups = all_seed_zero_groups[all_seed_zero_groups == 0]

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"### {'Quick run' if quick else 'Full experiment'} — {now}",
        "",
        f"- Results file: `../results/{output_path.name}`",
        f"- Result rows: {len(results)}",
        f"- Skipped conditions: {len(skipped)}",
        f"- Fixed split: train {len(X_train)} rows ({int(y_train.sum())} yes), "
        f"test {len(X_test)} rows ({int(y_test.sum())} yes); 70/30, stratified, seed 42",
        f"- Models: {', '.join(MODELS)}",
        f"- Training sizes: {TRAIN_SIZES if not quick else [500]}",
        f"- Requested yes ratios: {MINORITY_RATIOS if not quick else [0.10]}",
        f"- Sampling seeds: {SEEDS if not quick else [42]}",
        "- Metrics: accuracy, yes precision, yes recall, yes F1 at threshold 0.5, "
        "and PR-AUC (Average Precision)",
        f"- Recall-zero rows: {zero_recall_rows}/{len(results)}",
        f"- Groups with recall zero for every seed: {list(all_seed_zero_groups.index)}",
        f"- Versions: pandas {pd.__version__}, scikit-learn {sklearn.__version__}, "
        f"ucimlrepo {version('ucimlrepo')}",
        "",
    ]

    log_path = Path(__file__).resolve().parents[1] / "docs" / "experiment_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Experiment log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def run_experiments(quick=False):
    """Run the approved conditions, or a two-model sample with --quick."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    rows = []
    skipped = []
    train_sizes = [500] if quick else TRAIN_SIZES
    minority_ratios = [0.10] if quick else MINORITY_RATIOS
    seeds = [42] if quick else SEEDS
    conditions = [
        (n_total, ratio, seed)
        for n_total in train_sizes
        for ratio in minority_ratios
        for seed in seeds
    ]

    for condition_number, (n_total, ratio, seed) in enumerate(conditions, start=1):
        try:
            X_subset, y_subset = make_subset(
                X_train,
                y_train,
                n_total=n_total,
                minority_ratio=ratio,
                seed=seed,
            )
        except ValueError as error:
            skipped.append(
                {
                    "n_train": n_total,
                    "minority_ratio": ratio,
                    "seed": seed,
                    "reason": str(error),
                }
            )
            print(f"Skipping condition {condition_number}/{len(conditions)}: {error}")
            continue

        actual_ratio = float(y_subset.mean())
        print(
            f"Condition {condition_number}/{len(conditions)}: "
            f"n={n_total}, yes_ratio={actual_ratio:.0%}, seed={seed}"
        )

        for model_name in MODELS:
            model = build_model(model_name, X_subset, random_state=seed)
            model.fit(X_subset, y_subset)
            metrics = evaluate(model, X_test, y_test)
            rows.append(
                {
                    "model": model_name,
                    "n_train": n_total,
                    "minority_ratio": ratio,
                    "actual_minority_ratio": actual_ratio,
                    "n_yes": int(y_subset.sum()),
                    "n_no": int((y_subset == 0).sum()),
                    "seed": seed,
                    **metrics,
                }
            )

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _unique_output_path(
        output_dir,
        "quick_results" if quick else "results",
    )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(rows)} experiment rows to: {output_path}")

    if skipped:
        skipped_path = output_dir / "skipped_conditions.csv"
        pd.DataFrame(skipped).to_csv(skipped_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(skipped)} skipped conditions to: {skipped_path}")

    write_run_log(
        output_path,
        pd.DataFrame(rows),
        skipped,
        X_train,
        y_train,
        X_test,
        y_test,
        quick,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one small condition with both models for a fast demonstration.",
    )
    arguments = parser.parse_args()
    run_experiments(quick=arguments.quick)
