"""
remove_outliers.py
Removes outlier reviews (regarding length) from the dataset and prints the remaining statistics.
Author: Igor Lacko
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description="Remove outlier reviews (regarding length) from the dataset and print the remaining statistics."
)

parser.add_argument(
    "-d", "--dataset", type=str, required=True, help="Path to the dataset CSV file."
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to save the cut dataset CSV file.",
)

args = parser.parse_args()

DATASET = args.dataset
OUTPUT = args.output


def main():
    df = pd.read_csv(DATASET)
    print(f"{"*" * 10} Original Dataset Statistics {"*" * 10}")
    print(df["content"].str.len().describe())

    df["length"] = df["content"].apply(len)
    Q1 = df["length"].quantile(0.25)
    Q3 = df["length"].quantile(0.75)

    print(f"Q1 (25th percentile): {Q1}")
    print(f"Q3 (75th percentile): {Q3}")

    # Remaining reviews in {Q1, Q3 + 1.5 * IQR}
    IQR = Q3 - Q1
    print(f"IQR (Interquartile Range): {IQR}")

    lower_bound = Q1
    upper_bound = min(1000, Q3 + 1.5 * IQR)
    df_cut = df[(df["length"] >= lower_bound) & (df["length"] <= upper_bound)].drop(
        columns=["length"]
    )

    print(f"{"*" * 10} Cut Dataset Statistics {"*" * 10}")
    print(df_cut["content"].str.len().describe())
    if "content_type" in df_cut.columns:
        print(df_cut["content_type"].value_counts())
    df_cut.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
