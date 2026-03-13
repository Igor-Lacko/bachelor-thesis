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
parser.add_argument(
    "--use-words",
    action="store_true",
    help="Use word count instead of character count.",
)

args = parser.parse_args()

DATASET = args.dataset
OUTPUT = args.output


def main():
    df = pd.read_csv(DATASET)
    len_func = lambda x: len(x.split()) if args.use_words else len(x)
    df["length"] = df["content"].apply(len_func)

    print(f"{"*" * 10} Original Dataset Statistics {"*" * 10}")
    print(df["length"].describe())

    Q1 = df["length"].quantile(0.25)
    Q3 = df["length"].quantile(0.75)

    print(f"Q1 (25th percentile): {Q1}")
    print(f"Q3 (75th percentile): {Q3}")

    # Remaining reviews in {Q1, Q3 + 1.5 * IQR}
    IQR = Q3 - Q1
    print(f"IQR (Interquartile Range): {IQR}")

    lower_bound = Q1
    upper_bound = min(1000, Q3 + 1.5 * IQR)
    df_cut = df[(df["length"] >= lower_bound) & (df["length"] <= upper_bound)]

    print(f"{"*" * 10} Cut Dataset Statistics {"*" * 10}")
    print(df_cut["length"].describe())
    if "content_type" in df_cut.columns:
        print(df_cut["content_type"].value_counts())
    df_cut = df_cut.drop(columns=["length"])
    df_cut.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
