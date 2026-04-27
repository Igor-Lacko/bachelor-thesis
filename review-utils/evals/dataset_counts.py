"""
dataset_counts.py
Creates a LaTeX table with dataset counts for both variants and classes (so 4 rows).
Author: Igor Lacko
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parses the inputs/output

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Create a LaTeX table with dataset counts"
    )
    parser.add_argument(
        "-u",
        "--unified",
        type=Path,
        required=True,
        help="Path to the unified variant of the dataset",
    )
    parser.add_argument(
        "-s",
        "--separate",
        type=Path,
        required=True,
        help="Path to the separate variant of the dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Folder to store the LaTeX table with dataset counts",
    )

    return parser.parse_args()


def main():
    """Script entry point."""
    args = parse_args()

    df_separate = pd.read_csv(args.separate)
    df_unified = pd.read_csv(args.unified)

    counts = {
        "separate": {
            "fake": (fake_separate := len(df_separate[df_separate["label"] == 1])),
            "real": (real_separate := len(df_separate[df_separate["label"] == 0])),
            "total": fake_separate + real_separate,
        },
        "unified": {
            "fake": (fake_unified := len(df_unified[df_unified["label"] == 1])),
            "real": (real_unified := len(df_unified[df_unified["label"] == 0])),
            "total": fake_unified + real_unified,
        },
    }

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    df_counts = pd.DataFrame(counts).T
    df_counts.to_latex(
        output / "dataset_counts.tex",
        caption="Dataset counts for separate and unified variants",
        label="tab:dataset_counts",
    )


if __name__ == "__main__":
    main()
