"""
eval_experiments.py
Evaluates the results of experiments.
Author: Igor Lacko
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parses csv files and output folders

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument(
        "--slovakmistral-csv",
        type=Path,
        required=False,
        help="Path to the SlovakMistral results CSV file",
    )
    parser.add_argument(
        "--without-punctuation-csv",
        type=Path,
        required=False,
        help="Path to the CSV file for results without punctuation",
    )
    parser.add_argument(
        "--synonymized-csv",
        type=Path,
        required=False,
        help="Path to the CSV file for results with synonymized data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the output folder where the evaluation results will be saved",
    )

    return parser.parse_args()


def model_stripplot(df: pd.DataFrame, output: Path, **kwargs):
    """Creates 2 stripplots for each dataset variant. Compares the full model
        ensemble with individual models, and shows f1/pre/rec for each one.

    Args:
        df (pd.DataFrame): DataFrame containing the results to be plotted
        output (Path): Path to the output file where the plot will be saved
    """
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.5):
        column_mapping = {
            "features": "Features",
            "variant": "Variant",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-Score",
        }

        df = df.rename(columns=column_mapping)

        df_melted = df.melt(
            id_vars=["Features", "Variant"],
            value_vars=["Precision", "Recall", "F1-Score", "Accuracy"],
            var_name="Metric",
            value_name="Score",
        )

        feature_map = {
            "statistical": "Statistical",
            "tfidf": "TF-IDF",
            "meta": "Full Ensemble",
            "transformer": "Transformer",
        }

        df_melted["Features"] = df_melted["Features"].map(feature_map)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True, sharex=True)
        for variant, ax in zip(df_melted["Variant"].unique(), axes):
            df_filtered = df_melted[df_melted["Variant"] == variant]
            g = sns.stripplot(
                data=df_filtered,
                x="Features",
                y="Score",
                hue="Metric",
                ax=ax,
                palette="Set2",
                jitter=True,
                size=10,
            )

            (xaxis := g.get_xaxis()).labelpad = 10
            xaxis.label.set_visible(False)
            g.get_yaxis().labelpad = 10
            g.set_yticks(np.arange(0, 1.1, 0.1))
            ax.set_title(f"Variant: {variant.capitalize()}")
            g.set(ylim=(0, 1))
            g.legend(loc=kwargs.get("legend_loc"), title="Metric")

        fig.suptitle(kwargs.get("title", "Model Performance Comparison"), y=1.02)
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        fig.clf()


def main():
    args = parse_args()
    output_folder = args.output

    if (slovakmistral := args.slovakmistral_csv) is not None:
        df_slovakmistral = pd.read_csv(slovakmistral)
        model_stripplot(
            df_slovakmistral,
            output_folder / "slovakmistral_evaluation.svg",
            title="Classifier performance on SlovakMistral reviews",
            legend_loc="lower left",
        )

    if (without_punctuation := args.without_punctuation_csv) is not None:
        df_without_punctuation = pd.read_csv(without_punctuation)
        model_stripplot(
            df_without_punctuation,
            output_folder / "without_punctuation_evaluation.svg",
            title="Classifier performance on reviews without punctuation",
            legend_loc="lower right",
        )

    if (synonymized := args.synonymized_csv) is not None:
        raise NotImplementedError("Synonymized data evaluation is not implemented yet.")


if __name__ == "__main__":
    main()
