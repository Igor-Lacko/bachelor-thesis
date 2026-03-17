"""
make_splits.py
Creates training and testing sets via multiple split strategies.
Generates 80-10-10 split, 90-10 split, and stratified 5-fold splits.
Author: Igor Lacko
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

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
    "-o",
    "--output",
    type=str,
    required=True,
    help="Output folder where 80_20, 90_10, and kfolds folders will be created.",
)


def make_splits(df: pd.DataFrame, output_folder: str):
    """Splits the DataFrame using three different strategies.
        - 80_10_10: 80% train, 10% validation, 10% test split
        - 90_10: 90% train, 10% test split
        - kfolds: Stratified 5-fold cross-validation

    Args:
        df (pd.DataFrame): Input DataFrame to be split. Has to contain 'stratify_on' column.
        output_folder (str): Output folder where splits will be stored.
    """

    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    folder_80_10_10 = os.path.join(output_folder, "80_10_10")
    folder_90_10 = os.path.join(output_folder, "90_10")
    folder_kfolds = os.path.join(output_folder, "kfolds")

    os.makedirs(folder_80_10_10, exist_ok=True)
    os.makedirs(folder_90_10, exist_ok=True)
    os.makedirs(folder_kfolds, exist_ok=True)

    # 80-10-10 split
    train_80, temp_test = train_test_split(
        df, test_size=0.2, stratify=df["stratify_on"], random_state=42
    )
    val_10, test_10 = train_test_split(
        temp_test, test_size=0.5, stratify=temp_test["stratify_on"], random_state=42
    )

    train_80.to_csv(os.path.join(folder_80_10_10, "train.csv"), index=False)
    val_10.to_csv(os.path.join(folder_80_10_10, "val.csv"), index=False)
    test_10.to_csv(os.path.join(folder_80_10_10, "test.csv"), index=False)

    # 90-10 split
    train_90, test_10 = train_test_split(
        df, test_size=0.1, stratify=df["stratify_on"], random_state=42
    )

    train_90.to_csv(os.path.join(folder_90_10, "train.csv"), index=False)
    test_10.to_csv(os.path.join(folder_90_10, "test.csv"), index=False)

    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_num = 1
    for _, test_index in skf.split(df, df["stratify_on"]):
        fold_data = df.iloc[test_index].reset_index(drop=True)
        fold_data.to_csv(
            os.path.join(folder_kfolds, f"fold_{fold_num}.csv"), index=False
        )
        fold_num += 1


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        df = pd.read_csv(args.dataset)
        make_splits(df, output_folder=args.output)
        print(f"Splits created successfully in {args.output}")

    except Exception as e:
        print(f"An error occurred: {e}")
