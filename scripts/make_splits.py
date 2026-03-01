"""
make_splits.py
Creates training and testing set via a 90-10 split.
Can also create 80-10-10 if a validation set output is provided.
Author: Igor Lacko
"""

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
    "-tr",
    "--train-output",
    type=str,
    required=True,
    help="Output path for the train split CSV.",
)
parser.add_argument(
    "-ts",
    "--test-output",
    type=str,
    required=True,
    help="Output path for the test split CSV.",
)
parser.add_argument(
    "-v",
    "--validation-output",
    type=str,
    help="Optional output path for the validation split CSV. If provided, 80-10-10 split is used.",
)


def make_splits(
    df: pd.DataFrame,
    train_output: str,
    test_output: str,
    validation_output: str | None = None,
):
    """Splits the DataFrame into training and testing sets.
        Training is used in k-fold cross-validation, test to evaluate the whole ensemble.

    Args:
        df (pd.DataFrame): Input DataFrame to be split. Has to contan 'stratify_on' column.
        train_output (str): Output file path for train split.
        test_output (str): Output file/folder path for test split.
        validation_output (str | None): Output file path for validation split.
    """

    if validation_output is None:
        train, test = train_test_split(
            df, test_size=0.1, stratify=df["stratify_on"]
        )
        train.to_csv(train_output, index=False)
        test.to_csv(test_output, index=False)
        return

    train_val, test = train_test_split(
        df, test_size=0.1, stratify=df["stratify_on"]
    )
    train, val = train_test_split(
        train_val,
        test_size=(1 / 9),
        stratify=train_val["stratify_on"],
    )

    train.to_csv(train_output, index=False)
    test.to_csv(test_output, index=False)
    val.to_csv(validation_output, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        df = pd.read_csv(args.dataset)
        make_splits(
            df,
            train_output=args.train_output,
            test_output=args.test_output,
            validation_output=args.validation_output,
        )

    except Exception as e:
        print(f"An error occurred: {e}")
