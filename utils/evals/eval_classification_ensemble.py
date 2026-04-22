"""
eval_classification_ensemble.py
Evaluates classification results of the full ensemble and saves tables/figures.
Author: Igor Lacko
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate classification results of the full ensemble."
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


def full_ensemble_table(df_table: pd.DataFrame, output_dir: Path):
    """Creates a evaluation table from the full ensemble classification results.

    Args:
        df_table (pd.DataFrame): DataFrame containing the classification results.
        output_dir (Path): Directory to save the evaluation table.
    """
    table_path = output_dir / "classification_full_ensemble_table.tex"

    # model, precision, recall, f1-score, variant, without
    df_table = df_table[df_table["Without"] == "full"]

    """
    best to index by variant
    """

    df_table = df_table.set_index(["Variant", "Model"])
    df_table = df_table.sort_index().drop(columns=["Without"])

    # Sort by F1-Score
    df_table = df_table.sort_values(by="F1-Score", ascending=False)
    df_table.to_latex(table_path, float_format="%.3f", multirow=True)


def rename_df(df: pd.DataFrame) -> pd.DataFrame:
    """Renames the columns of the DataFrame for better presentation.

    Args:
        df (pd.DataFrame): DataFrame to rename.

    Returns:
        pd.DataFrame: Renamed DataFrame.
    """
    df = df.rename(
        columns={
            "model": "Model",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "variant": "Variant",
            "without": "Without",
        }
    )

    model_mapping = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "knn": "KNN",
        "svm": "SVM",
        "mlp": "MLP",
    }

    def model_map_func(model_name: str) -> str:
        return model_mapping.get(model_name, model_name)

    df["Model"] = df["Model"].apply(model_map_func)
    return df


def main():
    args = parse_args()

    input_file = args.input
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    table_dir, figure_dir = args.table_dir, args.figure_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    df = rename_df(df)
    full_ensemble_table(df, table_dir)


if __name__ == "__main__":
    main()
