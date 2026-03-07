"""
compare_datasets.py
Compares two datasets by things like review length distribution and generates graphs, etc.
Author: Igor Lacko
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(
    description="Compare two datasets by various metrics and generate visualizations."
)

parser.add_argument(
    "-d",
    "--datasets",
    nargs=2,
    required=True,
    help="Paths to the two dataset CSV files to compare.",
)

parser.add_argument(
    "-n",
    "--names",
    nargs=2,
    required=False,
    help="Dataset names used in the graphs. If not provided, will use 'Dataset A' and 'Dataset B'.",
)

parser.add_argument(
    "-o", "--output", type=str, required=True, help="Output folder for graphs."
)

args = parser.parse_args()

OUTPUT = args.output
DATASET_A, DATASET_B = args.datasets

(NAME_A, NAME_B) = args.names if args.names else ("Dataset A", "Dataset B")
COLOR_A = "#1f77b4"
COLOR_B = "#d95f02"


def plot_metric_comparison(
    values_a: pd.Series,
    values_b: pd.Series,
    x_label: str,
    output_name: str,
):
    """Plot one metric on side-by-side axes with comparable normalization.

    Args:
        values_a (pd.Series): Metric values from the first dataset.
        values_b (pd.Series): Metric values from the second dataset.
        title (str): Figure title.
        x_label (str): X-axis label.
        output_name (str): Output SVG filename.
    """
    values_a = values_a.dropna()
    values_b = values_b.dropna()
    combined_values = pd.concat([values_a, values_b], ignore_index=True)

    if combined_values.empty:
        return

    min_value = combined_values.min()
    max_value = combined_values.max()
    if min_value == max_value:
        bins = 10
    else:
        bins = np.linspace(min_value, max_value, 31)

    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

        plot_config = [
            (axes[0], values_a, NAME_A, COLOR_A),
            (axes[1], values_b, NAME_B, COLOR_B),
        ]

        for axis, values, name, color in plot_config:
            sns.histplot(
                x=values,
                bins=bins,
                stat="density",
                kde=True,
                color=color,
                alpha=0.45,
                edgecolor="white",
                linewidth=0.8,
                ax=axis,
            )
            axis.axvline(values.median(), color=color, linestyle="--", linewidth=2)
            axis.set_title(f"{name}\nn={len(values)}, median={values.median():.1f}")
            axis.set_xlabel(x_label)
            axis.set_ylabel("Density")

        fig.tight_layout()
        fig.savefig(
            os.path.join(OUTPUT, output_name), format="svg", bbox_inches="tight"
        )
        plt.close(fig)


def compare_length_chars(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Compare review character lengths and save the distribution plot.

    Args:
        df_a (pd.DataFrame): First dataset with a ``content`` column.
        df_b (pd.DataFrame): Second dataset with a ``content`` column.
    """
    plot_metric_comparison(
        values_a=df_a["content"].str.len(),
        values_b=df_b["content"].str.len(),
        x_label="Review Length (characters)",
        output_name="length_distribution.svg",
    )

    mean_a = df_a["content"].str.len().mean()
    mean_b = df_b["content"].str.len().mean()
    print(f"{NAME_A} mean review length (chars): {mean_a:.1f}")
    print(f"{NAME_B} mean review length (chars): {mean_b:.1f}")

    median_a = df_a["content"].str.len().median()
    median_b = df_b["content"].str.len().median()
    print(f"{NAME_A} median review length (chars): {median_a:.1f}")
    print(f"{NAME_B} median review length (chars): {median_b:.1f}")


def compare_length_words(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Compare review word counts and save the distribution plot.

    Args:
        df_a (pd.DataFrame): First dataset with a ``content`` column.
        df_b (pd.DataFrame): Second dataset with a ``content`` column.
    """
    plot_metric_comparison(
        values_a=df_a["content"].str.split().str.len(),
        values_b=df_b["content"].str.split().str.len(),
        x_label="Review Length (words)",
        output_name="word_count_distribution.svg",
    )

    mean_a = df_a["content"].str.split().str.len().mean()
    mean_b = df_b["content"].str.split().str.len().mean()
    print(f"{NAME_A} mean review length (words): {mean_a:.1f}")
    print(f"{NAME_B} mean review length (words): {mean_b:.1f}")


def compare_ttr(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Compare review type-token ratios and save the distribution plot.

    Args:
        df_a (pd.DataFrame): First dataset with a ``content`` column.
        df_b (pd.DataFrame): Second dataset with a ``content`` column.

    Note: Might be useless, humans often write short reviews.
    """
    longer_than_one = lambda df: df[df["content"].str.split().str.len() > 1]
    ttr = lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
    plot_metric_comparison(
        values_a=longer_than_one(df_a)["content"].apply(ttr),
        values_b=longer_than_one(df_b)["content"].apply(ttr),
        x_label="Type-Token Ratio",
        output_name="ttr_distribution.svg",
    )


def compare_exclamation_mark_usage(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Compare exclamation mark usage.

    Args:
        df_a (pd.DataFrame): First dataset with a ``content`` column.
        df_b (pd.DataFrame): Second dataset with a ``content`` column.
    """
    mean_a = df_a["content"].str.count("!").mean()
    mean_b = df_b["content"].str.count("!").mean()
    print(f"{NAME_A} mean exclamation marks per review: {mean_a:.2f}")
    print(f"{NAME_B} mean exclamation marks per review: {mean_b:.2f}")

    total_a = df_a["content"].str.count("!").sum()
    total_b = df_b["content"].str.count("!").sum()
    print(f"{NAME_A} total exclamation marks: {total_a}")
    print(f"{NAME_B} total exclamation marks: {total_b}")

    median_a = df_a["content"].str.count("!").median()
    median_b = df_b["content"].str.count("!").median()
    print(f"{NAME_A} median exclamation marks per review: {median_a}")
    print(f"{NAME_B} median exclamation marks per review: {median_b}")


def main():
    """Load both datasets and generate comparison plots."""
    os.makedirs(OUTPUT, exist_ok=True)
    df_a = pd.read_csv(DATASET_A)
    df_b = pd.read_csv(DATASET_B)

    compare_length_chars(df_a, df_b)
    compare_length_words(df_a, df_b)
    compare_ttr(df_a, df_b)
    compare_exclamation_mark_usage(df_a, df_b)


if __name__ == "__main__":
    main()
