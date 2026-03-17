"""
combine.py
Module to combine synthetic datasets, or synthetic + real datasets.
Author: Igor Lacko
"""

import os
import pandas as pd
import numpy as np
from utils import *


def __synthetic_to_one() -> pd.DataFrame:
    """Returns the loaded synthetic dataset if provided, else combines all CSVs in the synthetic folder.

    Returns:
        pd.DataFrame: Combined synthetic dataset.
    """
    if DATASET_SYNTHETIC:
        console.print(
            f"[green]Loading synthetic dataset from {DATASET_SYNTHETIC}[/green]"
        )
        synthetic_df = pd.read_csv(DATASET_SYNTHETIC)
        return synthetic_df

    else:
        console.print(
            f"[green]Combining synthetic datasets from folder {SYNTHETIC_FOLDER}[/green]"
        )
        combined_df = pd.DataFrame()
        for file_name in os.listdir(SYNTHETIC_FOLDER):
            if file_name.endswith(".csv"):
                file_path = os.path.join(SYNTHETIC_FOLDER, file_name)
                console.print(f"[green]Loading {file_path}[/green]")
                df = pd.read_csv(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        return combined_df


def combine():
    """Combines first the synthetic datasets (if needed), then combines with the real dataset (also if needed)."""
    synthetic = __synthetic_to_one()

    # Annotate data
    # 0 == real, 1 == synthetic
    synthetic["label"] = 1

    if TO_SYNTHETIC:
        console.print(
            f"[green]Saving combined synthetic dataset to {OUTPUT_FILE}[/green]"
        )
        synthetic.to_csv(OUTPUT_FILE, index=False)
    else:
        console.print(f"[green]Loading real dataset from {DATASET_REAL}[/green]")
        real = pd.read_csv(DATASET_REAL)
        real["label"] = 0

        # "unified" --> "content" if using the unified dataset so that the columns match
        if "unified" in real.columns and "content" in synthetic.columns:
            real = real.rename(columns={"unified": "content"})

        # Use only columns present in both datasets (so only in synthetic)
        common_columns = synthetic.columns.intersection(real.columns)
        combined = pd.concat(
            [synthetic[common_columns], real[common_columns]], ignore_index=True
        )

        """
            Trick for multi column stratification:
                Combined columns into a single column by converting to string.
                If content type it's easy, if rating round it down and append to label.
        """
        if STRATIFY_ON is not None and STRATIFY_ON == "content_type":
            combined["stratify_on"] = (
                combined["label"].astype(str)
                + "_"
                + combined["content_type"].astype(str)
            )
        elif STRATIFY_ON is not None and STRATIFY_ON == "rating":
            combined["stratify_on"] = (
                combined["label"].astype(str)
                + "_"
                # Round to nearest integer
                + combined["rating"].round(0).astype(int).astype(str)
            )
        console.print(f"[green]Saving combined dataset to {OUTPUT_FILE}[/green]")

        # Shuffle to not have first half s second r
        combined = combined.reindex(np.random.permutation(combined.index)).reset_index(
            drop=True
        )
        combined.to_csv(OUTPUT_FILE, index=False)
