"""
make_splits.py
Creates training and testing set via a 90-10 split.
Author: Igor Lacko
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description="Create training and testing splits from a DataFrame."
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="Path to the input dataset CSV file.",
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help="Output folder to save the splits."
)
parser.add_argument(
    "-t",
    "--test",
    type=float,
    default=0.1,
    help="Proportion of the dataset to include in the test split.",
)


def make_splits(
    df: pd.DataFrame, out_folder: str, test_size: float = 0.1, random_state: int = 67
):
    """Splits the DataFrame into training and testing sets.
        Training is used in k-fold cross-validation, test to evaluate the whole ensemble.

    Args:
        df (pd.DataFrame): Input DataFrame to be split. Has to contan 'stratify_on' column.
        out_folder (str): Output folder where the splits will be saved.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.1.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 67.
    """
    if not os.path.exists(out_folder):
        raise FileNotFoundError(f"Output folder {out_folder} does not exist.")

    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["stratify_on"]
    )

    train.to_csv(os.path.join(out_folder, "train.csv"), index=False)
    test.to_csv(os.path.join(out_folder, "test.csv"), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        df = pd.read_csv(args.dataset)
        make_splits(df, args.output, test_size=args.test)

    except Exception as e:
        print(f"An error occurred: {e}")
