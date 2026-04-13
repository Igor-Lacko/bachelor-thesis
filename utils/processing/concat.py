"""
concat.py
Concatenates all dataframes in the given folder
Author: Igor Lacko
"""

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(
    description="Concatenate all dataframes in the given folder"
)
parser.add_argument(
    "folder", type=str, help="Path to the folder containing the dataframes"
)
parser.add_argument("output", type=str, help="Path to the output file (CSV)")
args = parser.parse_args()

FOLDER = args.folder
OUTPUT = args.output


def main():
    """Main function to concatenate dataframes."""
    df = pd.DataFrame()
    for file in os.listdir(FOLDER):
        if file.endswith(".csv"):
            df = pd.concat(
                [df, pd.read_csv(os.path.join(FOLDER, file))], ignore_index=True
            )
    df.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
