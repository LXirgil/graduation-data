"""Summarize experiment runs and create figures for analysis and presentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
METRICS = ["accuracy", "precision", "recall", "f1", "pr_auc"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def build_summary(results):
    """Calculate mean and sample standard deviation across sampling seeds."""
    group_columns = ["model", "n_train", "minority_ratio"]
    aggregations = {"n_seeds": ("seed", "nunique")}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return (
        results.groupby(group_columns, as_index=False)
        .agg(**aggregations)
        .sort_values(group_columns)
    )


def create_figures(summary):
    """Write line plots and heatmaps as reusable PNG files."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    ratio_order = [0.01, 0.05, 0.10, 0.30, 0.50]
    size_order = [500, 1_000, 2_000, 5_000]

    # Line plots: class ratio on x, metric on y, separate line per train size.
    for model_name, model_label in MODEL_LABELS.items():
        model_data = summary[summary["model"] == model_name]
        for metric in ["recall", "f1"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for n_train in size_order:
                line = model_data[model_data["n_train"] == n_train].sort_values(
                    "minority_ratio"
                )
                ax.errorbar(
                    line["minority_ratio"] * 100,
                    line[f"{metric}_mean"],
                    yerr=line[f"{metric}_std"],
                    marker="o",
                    capsize=3,
                    label=f"n={n_train:,}",
                )
            ax.set_xticks([1, 5, 10, 30, 50])
            ax.set_xlabel("Minority class (yes) in training data (%)")
            ax.set_ylabel(f"{metric.upper()} on fixed test set")
            ax.set_ylim(0, 1)
            ax.set_title(f"{model_label}: {metric.upper()} by training ratio and size")
            ax.legend(title="Training rows")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"line_{metric}_{model_name}.png", dpi=200)
            plt.close(fig)

        # F1 heatmap, one per model.
        f1_table = (
            model_data.pivot(index="n_train", columns="minority_ratio", values="f1_mean")
            .reindex(index=size_order, columns=ratio_order)
        )
        f1_table.columns = [f"{ratio:.0%}" for ratio in f1_table.columns]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            f1_table,
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            cmap="viridis",
            ax=ax,
        )
        ax.set_xlabel("Minority class (yes) in training data")
        ax.set_ylabel("Training rows")
        ax.set_title(f"{model_label}: mean F1")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"heatmap_f1_{model_name}.png", dpi=200)
        plt.close(fig)

        # Paired Accuracy and Recall heatmaps make majority-class bias visible.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, metric in zip(axes, ["accuracy", "recall"]):
            table = (
                model_data.pivot(
                    index="n_train",
                    columns="minority_ratio",
                    values=f"{metric}_mean",
                )
                .reindex(index=size_order, columns=ratio_order)
            )
            table.columns = [f"{ratio:.0%}" for ratio in table.columns]
            sns.heatmap(
                table,
                annot=True,
                fmt=".2f",
                vmin=0,
                vmax=1,
                cmap="viridis",
                ax=ax,
                cbar=ax is axes[-1],
            )
            ax.set_xlabel("Minority class (yes) in training data")
            ax.set_ylabel("Training rows" if ax is axes[0] else "")
            ax.set_title(metric.capitalize())
        fig.suptitle(f"{model_label}: Accuracy and minority-class Recall")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"heatmap_accuracy_recall_{model_name}.png", dpi=200)
        plt.close(fig)


def create_analysis_notes(results, summary):
    """Write concise numeric observations without imposing a success threshold."""
    lines = [
        "# Analysis notes",
        "",
        "These are descriptive results from the fixed test set. No statistical "
        "significance test or minimum acceptable performance threshold was set.",
        "",
        "## Mean Recall and F1 across class-ratio settings",
        "",
    ]

    by_size = summary.groupby(["model", "n_train"])[["recall_mean", "f1_mean"]].mean()
    for model_name, model_label in MODEL_LABELS.items():
        lines.append(f"### {model_label}")
        lines.append("")
        for n_train in [500, 1_000, 2_000, 5_000]:
            row = by_size.loc[(model_name, n_train)]
            lines.append(
                f"- n={n_train:,}: Recall {row['recall_mean']:.3f}, "
                f"F1 {row['f1_mean']:.3f} (averaged equally across the five ratios)."
            )
        lines.append("")

    lines.extend(
        [
            "## Mean Accuracy and Recall across training sizes",
            "",
        ]
    )
    by_ratio = summary.groupby(["model", "minority_ratio"])[
        ["accuracy_mean", "recall_mean", "f1_mean", "pr_auc_mean"]
    ].mean()
    for model_name, model_label in MODEL_LABELS.items():
        lines.append(f"### {model_label}")
        lines.append("")
        for ratio in [0.50, 0.30, 0.10, 0.05, 0.01]:
            row = by_ratio.loc[(model_name, ratio)]
            lines.append(
                f"- yes={ratio:.0%}: Accuracy {row['accuracy_mean']:.3f}, "
                f"Recall {row['recall_mean']:.3f}, F1 {row['f1_mean']:.3f}, "
                f"PR-AUC {row['pr_auc_mean']:.3f} "
                "(averaged equally across the four training sizes)."
            )
        lines.append("")

    lines.extend(
        [
            "## Relative descriptive effects on Recall",
            "",
            "The comparisons below use equal-weight averages over the other conditions; "
            "they are descriptive, not statistical significance tests.",
            "",
        ]
    )
    for model_name, model_label in MODEL_LABELS.items():
        size_effect = (
            by_size.loc[(model_name, 5_000), "recall_mean"]
            - by_size.loc[(model_name, 500), "recall_mean"]
        )
        ratio_effect = (
            by_ratio.loc[(model_name, 0.50), "recall_mean"]
            - by_ratio.loc[(model_name, 0.01), "recall_mean"]
        )
        lines.append(
            f"- {model_label}: increasing training size from 500 to 5,000 changed "
            f"mean Recall by {size_effect:+.3f}; increasing yes ratio from 1% to 50% "
            f"changed mean Recall by {ratio_effect:+.3f}."
        )
    lines.append("")

    zero_recall_rows = int((results["recall"] == 0).sum())
    lines.extend(
        [
            "## Recall-zero runs",
            "",
            f"Recall was zero in {zero_recall_rows} of {len(results)} individual model runs. "
            "The only model/size/ratio group with zero Recall across all five seeds was "
            "Random Forest, n=500, yes=1%. This is recorded as an observed result; "
            "the run itself had no skipped conditions.",
            "",
            "## Interpretation note",
            "",
            "Accuracy should be read together with yes Recall, F1, and PR-AUC. "
            "A high Accuracy alone does not show whether the model found the minority class. "
            "The project has not defined a minimum performance threshold, so choosing a "
            "minimum data size or ratio requires a decision with the supervisor.",
            "",
        ]
    )
    (PROJECT_DIR / "docs" / "analysis_notes.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(source_path=None):
    if source_path is None:
        candidates = list(RESULTS_DIR.glob("results_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No full experiment result CSV found in {RESULTS_DIR}"
            )
        source_path = max(candidates, key=lambda path: path.stat().st_mtime)
    else:
        source_path = Path(source_path)

    results = pd.read_csv(source_path)
    summary = build_summary(results)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    create_figures(summary)
    create_analysis_notes(results, summary)
    print(f"Saved summary: {RESULTS_DIR / 'summary.csv'}")
    print(f"Source results: {source_path}")
    print(f"Saved figures: {FIGURES_DIR}")
    print(f"Saved analysis notes: {PROJECT_DIR / 'docs' / 'analysis_notes.md'}")


if __name__ == "__main__":
    main()
