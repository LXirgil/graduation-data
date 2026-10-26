"""Fit and evaluate one baseline run for each approved model."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
)

from data import load_data, split_data
from models import build_model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train rows: {len(X_train)} / test rows: {len(X_test)}")
    print(f"Minority class yes: train {int(y_train.sum())} / test {int(y_test.sum())}")

    rows = []
    for model_name, display_name in [
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
    ]:
        model = build_model(model_name, X_train, random_state=42)
        model.fit(X_train, y_train)
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
        accuracy = accuracy_score(y_test, predictions)
        pr_auc = average_precision_score(y_test, yes_probabilities)

        print(f"\n--- {display_name} ---")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["no", "yes"],
                zero_division=0,
            )
        )
        print(f"Accuracy: {accuracy:.4f}")
        print(f"yes PR-AUC (Average Precision): {pr_auc:.4f}")

        rows.append(
            {
                "model": display_name,
                "accuracy": accuracy,
                "yes_precision": precision,
                "yes_recall": recall,
                "yes_f1": f1,
                "yes_pr_auc_average_precision": pr_auc,
            }
        )

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved baseline metrics to: {output_path}")


if __name__ == "__main__":
    main()
