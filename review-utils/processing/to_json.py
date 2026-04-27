"""
to_json.py
Creates a JSON from cross validation results
Author: Igor Lacko
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parses CLI args.

    Returns:
        argparse.Namespace: Parsed CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Converts cross validation results from CSV to JSON format."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input CSV file containing cross validation results.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to save the output JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df.drop(columns=["f1_score"])
    df["parameters"] = df["parameters"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    results = {}

    for model, model_results in df.groupby("model"):
        results[model] = {}
        for variant, variant_results in model_results.groupby("variant"):
            results[model][variant] = {}
            for _, row in variant_results.iterrows():
                results[model][variant][row["features"]] = row["parameters"]

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
