"""
to_separate.py
Converts the default dataset csv into one big column with content_good and content_bad used as individual rows.
Author: Igor Lacko
"""

import pandas as pd
import argparse
import os

# Dataset arg
parser = argparse.ArgumentParser(
    description="Convert dataset CSV to separate content rows."
)
parser.add_argument(
    "-d", "--dataset", type=str, required=True, help="Path to the dataset CSV file."
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to the output CSV file with separated content.",
)

args = parser.parse_args()


def to_separate(input_csv: pd.DataFrame):
    """Converts the default dataset csv to the output and saves it.

    Args:
        input_csv (pd.DataFrame): The input dataframe.
    """
    df = input_csv.melt(
        id_vars=["rating"],
        value_vars=["content_good", "content_bad"],
        var_name="content_type",
        value_name="content",
    )

    # Drop NaN values if any
    df = df.dropna(subset=["content"])

    # And duplicates ofc
    df = df.drop_duplicates(subset=["content"])

    df.to_csv(args.output, index=False)


def main():
    """Script main function."""
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file {args.dataset} not found.")

    input_csv = pd.read_csv(args.dataset)
    to_separate(input_csv)


if __name__ == "__main__":
    main()
