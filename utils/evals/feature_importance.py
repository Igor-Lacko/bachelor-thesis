"""
feature_importance.py
Compares the feature importance of individual models and saves tables/figures.
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
    parser = argparse.ArgumentParser(description="View feature importance of models")
    parser.add_argument(
        "-b",
        "--base_path",
        type=Path,
        required=True,
        help="Path to the folder containint meta.csv, statistical.csv and tfidf.csv files",
    )
    parser.add_argument(
        "-t",
        "--table-dir",
        type=Path,
        required=True,
        help="Path to the directory where tables will be saved",
    )
    parser.add_argument(
        "-f",
        "--figure-dir",
        type=Path,
        required=True,
        help="Path to the directory where figures will be saved",
    )
    return parser.parse_args()


def tfidf_table(df_tfidf: pd.DataFrame, output_path: Path):
    """Creates a table of the top 10 most important TF-IDF features and saves it as a LaTeX file.

    Args:
        df_tfidf (pd.DataFrame): DataFrame containing TF-IDF features and their importance.
        output_path (Path): Path to the output LaTeX file where the table will be saved.
    """

    for variant in ["separate", "unified"]:
        df_variant = df_tfidf[df_tfidf["variant"] == variant]
        df_variant = df_variant[["feature", "importance"]].sort_values(
            by="importance", ascending=False
        )
        df_variant.to_latex(
            output_path.with_name(output_path.stem + f"_{variant}.tex"),
            index=False,
            label=f"tab:feature_importance_tfidf_{variant}",
            caption=f"Top 10 most important TF-IDF features for the {variant} variant.",
        )


def meta_barchart(df_meta: pd.DataFrame, output_path: Path):
    """Creates a bar chart of the 3 meta features side by side for each variant.

    Args:
        df_meta (pd.DataFrame): DataFrame containing meta features and their importance.
        output_path (Path): Path to the output file where the figure will be saved.
    """
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        for ax, variant in zip(axes, ["separate", "unified"]):
            df_variant = df_meta[df_meta["variant"] == variant]
            g = sns.barplot(data=df_variant, y="importance", x="feature", ax=ax)

            g.set_title(f"{variant.title()} Variant", pad=10)
            g.set_ylabel("Importance")
            g.set(ylim=(0, 1))
            g.set_yticks(np.arange(0, 1.1, 0.1))
            xaxis = g.get_xaxis()
            xaxis.label.set_visible(False)
            yaxis = g.get_yaxis()
            yaxis.label.set_visible(False)

        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        fig.clf()


def statistical_barchart(df_statistical: pd.DataFrame, output_path: Path):
    """Creates a bar chart of the top 10 most important statistical features side by side for each variant.

    Args:
        df_statistical (pd.DataFrame): DataFrame containing statistical features and their importance.
        output_path (Path): Path to the output file where the figure will be saved.
    """
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharey=True)
        for ax, variant in zip(axes, ["separate", "unified"]):
            df_variant = df_statistical[df_statistical["variant"] == variant]
            df_variant = (
                df_variant[["feature", "importance"]]
                .sort_values(by="importance", ascending=False)
                .head(10)
            )
            g = sns.barplot(data=df_variant, y="importance", x="feature", ax=ax)

            g.set_title(f"{variant.capitalize()} Variant", pad=20)
            g.set_ylabel("Importance")
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
            xaxis = g.get_xaxis()
            xaxis.label.set_visible(False)
            yaxis = g.get_yaxis()
            yaxis.label.set_visible(False)

        fig.subplots_adjust(hspace=0.2)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        fig.clf()


def main():
    """Script entry point"""
    args = parse_args()
    base_path = args.base_path
    table_dir = args.table_dir
    figure_dir = args.figure_dir

    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    if not (base_path / "meta.csv").is_file():
        raise FileNotFoundError(
            f"Meta features file not found: {base_path / 'meta.csv'}"
        )
    df_meta = pd.read_csv(base_path / "meta.csv")

    if not (base_path / "statistical.csv").is_file():
        raise FileNotFoundError(
            f"Statistical features file not found: {base_path / 'statistical.csv'}"
        )
    df_statistical = pd.read_csv(base_path / "statistical.csv")

    if not (base_path / "tfidf.csv").is_file():
        raise FileNotFoundError(
            f"TF-IDF features file not found: {base_path / 'tfidf.csv'}"
        )
    df_tfidf = pd.read_csv(base_path / "tfidf.csv")

    if not table_dir.exists():
        table_dir.mkdir(parents=True)

    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)

    tfidf_table(df_tfidf, table_dir / "tfidf_importance.tex")
    statistical_barchart(df_statistical, figure_dir / "statistical_importance.svg")
    meta_barchart(df_meta, figure_dir / "meta_importance.svg")


if __name__ == "__main__":
    main()
