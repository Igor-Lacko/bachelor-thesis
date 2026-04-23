"""
eval_classification_ensemble.py
Evaluates classification results of the full ensemble and saves tables/figures.
Author: Igor Lacko
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate classification results of the full ensemble."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the csv file containing the classification results.",
    )
    parser.add_argument(
        "-t",
        "--table-dir",
        type=Path,
        required=True,
        help="Directory to save the tables.",
    )
    parser.add_argument(
        "-f",
        "--figure-dir",
        type=Path,
        required=True,
        help="Directory to save the figures.",
    )
    return parser.parse_args()


def full_ensemble_table(df_table: pd.DataFrame, output_dir: Path):
    """Creates a evaluation table from the full ensemble classification results.

    Args:
        df_table (pd.DataFrame): DataFrame containing the classification results.
        output_dir (Path): Directory to save the evaluation table.
    """
    table_path = output_dir / "classification_full_ensemble_table.tex"

    # model, precision, recall, f1-score, variant, without
    df_table = df_table[df_table["Without"] == "full"]

    """
    best to index by variant
    """

    df_table = df_table.set_index(["Variant", "Model"])
    df_table = df_table.sort_index().drop(columns=["Without"])

    # Sort by F1-Score
    df_table = df_table.sort_values(by="F1-Score", ascending=False)
    df_table.to_latex(table_path, float_format="%.3f", multirow=True)


def partial_ensemble_figure(df_ensemble: pd.DataFrame, output_dir: Path):
    """Creates 2 figures for each dataset variant, comparing the ensemble model
        when full and when one feature is removed.
        Done as a scatterplot where all metrics are shown (i have barplot fatigue)
    Args:
        df_ensemble (pd.DataFrame): DataFrame containing the ensemble classification results.
        output_dir (Path): Directory to save the figure.
    """
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.5):
        df_chosen_model = df_ensemble[df_ensemble["Model"] == "RF"].copy()

        # Melt metrics into one column
        df_melted = df_chosen_model.melt(
            id_vars=["Variant", "Without"],
            value_vars=["Precision", "Recall", "F1-Score"],
            var_name="Metric",
            value_name="Value",
        )

        feature_name_map = {
            "full": "None",
            "tfidf": "TF-IDF",
            "statistical": "Statistical",
            "transformer": "Transformer",
        }

        df_melted["Without"] = df_melted["Without"].map(feature_name_map)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, sharex=True)
        for i, variant in enumerate(df_chosen_model["Variant"].unique()):
            df_filtered = df_melted[df_melted["Variant"] == variant]
            g = sns.stripplot(
                data=df_filtered,
                x="Without",
                y="Value",
                hue="Metric",
                ax=axes[i],
                palette="Set2",
                jitter=True,
                size=10,
            )

            g.get_xaxis().labelpad = 10
            g.get_yaxis().labelpad = 10
            g.set_title(f"{variant.title()} dataset variant", pad=20)
            g.set_xlabel("Removed Feature")
            g.set_yticks(np.arange(0, 1.1, 0.1))
            g.legend(loc="lower left", title="Metric")
            g.set_ylabel("F1-Score")
            g.set(ylim=(0, 1))

        fig.tight_layout()
        figure_path = output_dir / "classification_partial_ensemble.svg"
        fig.savefig(figure_path, dpi=300)
        fig.clf()


def rename_df(df: pd.DataFrame) -> pd.DataFrame:
    """Renames the columns of the DataFrame for better presentation.

    Args:
        df (pd.DataFrame): DataFrame to rename.

    Returns:
        pd.DataFrame: Renamed DataFrame.
    """
    df = df.rename(
        columns={
            "model": "Model",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "variant": "Variant",
            "without": "Without",
            "features": "Features",
        }
    )

    model_mapping = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "knn": "KNN",
        "svm": "SVM",
        "mlp": "MLP",
    }

    def model_map_func(model_name: str) -> str:
        return model_mapping.get(model_name, model_name)

    df["Model"] = df["Model"].apply(model_map_func)
    return df


def main():
    args = parse_args()

    input_file = args.input
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    table_dir, figure_dir = args.table_dir, args.figure_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    df = rename_df(df)
    full_ensemble_table(df, table_dir)
    partial_ensemble_figure(df, figure_dir)


if __name__ == "__main__":
    main()
