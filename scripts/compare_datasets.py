"""
compare_datasets.py
Compares two datasets by things like review length distribution and generates graphs, etc.
Author: Igor Lacko
"""

import argparse
import os
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


def compare_length_chars(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Compare review character lengths and save the distribution plot.

    Args:
        df_a (pd.DataFrame): First dataset with a ``content`` column.
        df_b (pd.DataFrame): Second dataset with a ``content`` column.
    """
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            df_a["content"].str.len().to_numpy(),
            color="blue",
            label=NAME_A,
            kde=True,
            stat="density",
            bins=30,
        )
        sns.histplot(
            df_b["content"].str.len().to_numpy(),
            color="orange",
            label=NAME_B,
            kde=True,
            stat="density",
            bins=30,
        )
        plt.title("Distribution of Review Lengths")
        plt.xlabel("Review Length (characters)")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT, "length_distribution.svg"))
        plt.close()


def compare_length_words(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Compare review word counts and save the distribution plot.

    Args:
        df_a (pd.DataFrame): First dataset with a ``content`` column.
        df_b (pd.DataFrame): Second dataset with a ``content`` column.
    """
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            df_a["content"].str.split().str.len().to_numpy(),
            color="blue",
            label=NAME_A,
            kde=True,
            stat="density",
            bins=30,
        )
        sns.histplot(
            df_b["content"].str.split().str.len().to_numpy(),
            color="orange",
            label=NAME_B,
            kde=True,
            stat="density",
            bins=30,
        )
        plt.title("Distribution of Review Word Counts")
        plt.xlabel("Review Length (words)")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT, "word_count_distribution.svg"))
        plt.close()


def main():
    """Load both datasets and generate comparison plots."""
    os.makedirs(OUTPUT, exist_ok=True)
    df_a = pd.read_csv(DATASET_A)
    df_b = pd.read_csv(DATASET_B)

    compare_length_chars(df_a, df_b)
    compare_length_words(df_a, df_b)


if __name__ == "__main__":
    main()