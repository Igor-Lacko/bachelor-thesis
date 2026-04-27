"""
utils.py
Utilities for combining datasets.
Author: Igor Lacko
"""

import argparse
from rich.console import Console
import os
import pandas as pd

# Global console instance
console = Console()

# Argument parser
__parser = argparse.ArgumentParser(description="Utilities for combining datasets.")
__parser.add_argument(
    "-r",
    "--real_dataset",
    type=str,
    help="Path to the real dataset file.",
)
__parser.add_argument(
    "-s",
    "--synthetic-dataset",
    type=str,
    help="Path to the synthetic dataset file.",
)
__parser.add_argument(
    "-sf",
    "--synthetic-folder",
    type=str,
    help="Path to the folder containing multiple synthetic dataset CSV files to combine.",
)
__parser.add_argument(
    "--stratify-on",
    type=str,
    help="Column name to stratify on when splitting datasets.",
)
__parser.add_argument(
    "--to-synthetic",
    action="store_true",
    help="Combine various synthetic dataset CSVs into one CSV file. --synthetic-folder must be provided.",
)
__parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Output file path for the resulting dataset.",
)

__args = __parser.parse_args()

DATASET_REAL = __args.real_dataset
DATASET_SYNTHETIC = __args.synthetic_dataset
SYNTHETIC_FOLDER = __args.synthetic_folder
OUTPUT_FILE = __args.output
TO_SYNTHETIC = __args.to_synthetic
STRATIFY_ON = __args.stratify_on


def validate_args():
    """Validate the provided arguments."""
    if TO_SYNTHETIC:
        if not SYNTHETIC_FOLDER:
            console.print(
                "[red]Error:[/red] --synthetic-folder must be provided when --to-synthetic is set."
            )
            exit(1)
        if not os.path.isdir(SYNTHETIC_FOLDER):
            console.print(
                f"[red]Error:[/red] The provided synthetic folder path '{SYNTHETIC_FOLDER}' is not a valid directory."
            )
            exit(1)
    else:
        if not DATASET_REAL:
            console.print("[red]Error:[/red] --real-dataset must be provided.")
            exit(1)
        if not os.path.isfile(DATASET_REAL):
            console.print(
                f"[red]Error:[/red] The provided real dataset path '{DATASET_REAL}' is not a valid file."
            )
            exit(1)
        if DATASET_SYNTHETIC and SYNTHETIC_FOLDER:
            console.print(
                "[red]Error:[/red] Provide either --synthetic-dataset or --synthetic-folder, not both."
            )
            exit(1)
        if not DATASET_SYNTHETIC and not SYNTHETIC_FOLDER:
            console.print(
                "[red]Error:[/red] Either --synthetic-dataset or --synthetic-folder must be provided."
            )
            exit(1)

        if DATASET_SYNTHETIC and not os.path.isfile(DATASET_SYNTHETIC):
            console.print(
                f"[red]Error:[/red] The provided synthetic dataset path '{DATASET_SYNTHETIC}' is not a valid file."
            )
            exit(1)

        if SYNTHETIC_FOLDER and not os.path.isdir(SYNTHETIC_FOLDER):
            console.print(
                f"[red]Error:[/red] The provided synthetic folder path '{SYNTHETIC_FOLDER}' is not a valid directory."
            )
            exit(1)


validate_args()
