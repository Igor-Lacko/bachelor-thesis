"""
eval_individual_models.py
Evaluates classification results of individual models and saves tables/figures.
Author: Igor Lacko
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate classification results of individual models."
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


def create_table(df_table: pd.DataFrame, output_dir: Path):
    """
    Create an evaluation table from the classification results.
    Args:
        df_table (pd.DataFrame): DataFrame containing the classification results.
        output_dir (Path): Directory to save the evaluation table.
    """
    table_path = output_dir / "classification_table.tex"

    """
    df looks like model, precision, recall, f1-score, variant, features
    multiindex on features first and then model is probably best for presentation
    """
    df_table = df_table.set_index(["Features", "Model", "Variant"])
    df_table = df_table.sort_index()

    print(df_table)
    df_table.to_latex(
        table_path,
        float_format="%.2f",
        multirow=True,
        sparsify=True,
        label="tab:classification_results",
    )


def f1_score_comparison(df: pd.DataFrame, output_dir: Path):
    """Creates 3 figures for each feature. Compares the F1 score for each model across both variants.

    Args:
        df (pd.DataFrame): DataFrame containing the classification results.
        output_dir (Path): Directory to save the figures.
    """
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.5):
        for feature in df["Features"].unique():
            df_feature = df[df["Features"] == feature]

            fig = plt.figure(figsize=(10, 6))

            # For hlines later
            max_separate = df_feature[df_feature["Variant"] == "Separate"][
                "F1-Score"
            ].max()
            max_unified = df_feature[df_feature["Variant"] == "Unified"][
                "F1-Score"
            ].max()

            g = sns.barplot(
                data=df_feature,
                x="Model",
                y="F1-Score",
                hue="Variant",
                palette=(palette := sns.color_palette("Paired")),
            )

            g.axhline(max_separate, color=palette[0], linestyle="--")
            g.axhline(max_unified, color=palette[1], linestyle="--")

            ticks = [0, 0.5, max_unified, max_separate, 1]

            g.set_yticks(sorted(set(ticks)))
            g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

            # Different things look good in tables and graphs... :((
            feature_map = {
                "TF-IDF": "TF-IDF",
                "Embeddings": "embedding",
                "Statistical": "statistical",
            }

            g.set_title(
                f"F1-Score Comparison for models using {feature_map.get(feature, feature)} features"
            )
            g.set_ylabel("F1-Score", labelpad=10)
            g.set(ylim=(0, 1))
            g.legend(title="Variant", loc="best")
            g.xaxis.label.set_visible(False)

            fig.tight_layout()
            figure_path = output_dir / f"f1_score_comparision_{feature.lower()}.svg"
            fig.savefig(figure_path)
            fig.clf()


def precision_recall_comparison(df: pd.DataFrame, output_dir: Path):
    """Creates a 3 X 2 grid of figures for each feature and variant. Compares the Precision and Recall for each model.

    Args:
        df (pd.DataFrame): DataFrame containing the classification results.
        output_dir (Path): Directory to save the figures.
    """
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=2):
        fig, axes = plt.subplots(3, 2, figsize=(15, 18), sharey=True)
        for i, feature in enumerate(df["Features"].unique()):
            for j, variant in enumerate(df["Variant"].unique()):
                df_subset = df[(df["Features"] == feature) & (df["Variant"] == variant)]

                df_melted = df_subset.melt(
                    id_vars=["Model"],
                    value_vars=["Precision", "Recall"],
                    var_name="Metric",
                    value_name="Score",
                )

                sns.barplot(
                    data=df_melted,
                    x="Model",
                    y="Score",
                    hue="Metric",
                    palette="Paired",
                    ax=axes[i, j],
                )

                axes[i, j].set_title(f"{variant} - {feature}")
                axes[i, j].set_ylabel("Score", labelpad=10)
                axes[i, j].set(ylim=(0, 1))
                axes[i, j].legend_.remove()
                axes[i, j].xaxis.label.set_visible(False)
                axes[i, j].yaxis.label.set_visible(False)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        handles, _ = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            ["Precision", "Recall"],
            title="Metric",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.07),
            ncol=1,
        )
        figure_path = output_dir / "precision_recall_comparison.svg"
        fig.savefig(figure_path, bbox_inches="tight")
        fig.clf()


def rename_df(df: pd.DataFrame) -> pd.DataFrame:
    """Renames cols/values in the DataFrame for better presentation in tables/figures.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with renamed columns and values.
    """
    # Rename cols first
    col_map = {
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1-Score",
        "variant": "Variant",
        "features": "Features",
        "model": "Model",
    }

    df = df.rename(columns=col_map)

    model_name_map = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "knn": "KNN",
        "multinomial_bayes": "NB",
        "mlp": "MLP",
        "svm": "SVM",
        "slovakbert": "SlovakBERT",
    }

    feature_name_map = {
        "tfidf": "TF-IDF",
        "transformer": "Embeddings",
        "statistical": "Statistical",
    }

    def model_map_func(model_name: str) -> str:
        if model_name in model_name_map.values():
            return model_name
        return model_name_map.get(model_name, model_name)

    def feature_map_func(feature_name: str) -> str:
        if feature_name in feature_name_map.values():
            return feature_name
        return feature_name_map.get(feature_name, feature_name)

    df["Model"] = df["Model"].map(model_map_func)
    df["Variant"] = df["Variant"].str.title()
    df["Features"] = df["Features"].map(feature_map_func)

    return df


def main():
    args = parse_args()

    input_file = args.input
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    table_dir, figure_dir = args.table_dir, args.figure_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.read_csv(input_file)
    df_results = rename_df(df_results)

    # 1. Tables!
    create_table(df_results, table_dir)

    # 2. Graphs!
    f1_score_comparison(df_results, figure_dir)
    precision_recall_comparison(df_results, figure_dir)


if __name__ == "__main__":
    main()
