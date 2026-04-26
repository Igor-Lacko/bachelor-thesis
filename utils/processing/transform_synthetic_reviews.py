"""
transform_synthetic_reviews.py
A) Synonymizes and
B) Removes punctuation
from synthetic reviews in a testing set and saves it as two separate csv files.
Author: Igor Lacko
"""

import argparse
import random
from pathlib import Path

import pandas as pd

# Is constant
PUNCTUATION = set(".!?,;:-")


def parse_args() -> argparse.Namespace:
    """Parses CLI args.

    Returns:
        argparse.Namespace: Parsed CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Transforms synthetic reviews by synonymizing and removing punctuation."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input CSV file containing synthetic reviews.",
    )
    parser.add_argument(
        "-w",
        "--wordnet",
        type=Path,
        required=True,
        help="Path to the WordNet database for synonymization.",
    )
    parser.add_argument(
        "-os",
        "--output-synonymized",
        type=Path,
        required=True,
        help="Folder to save the synonymized reviews CSV file.",
    )
    parser.add_argument(
        "-on",
        "--output-no-punctuation",
        type=Path,
        required=True,
        help="Folder to save the no punctuation reviews CSV file.",
    )
    parser.add_argument(
        "-wp",
        "--word-probability",
        type=float,
        default=0.1,
        help="Probability of synonymizing each word in the review (default: 0.1).",
    )
    return parser.parse_args()


def get_synset(line: str) -> set[str]:
    """Gets the set of synonyms from a line in the WordNet file.

    Args:
        line (str): Line from the WordNet file.
            - Structured like: ID\tPOS tag\tsyn1;syn2;...;synN\tFOR RECORD SEPARATOR english wn record
            - Where we only want the synonyms

    Returns:
        set[str]: Set of synonyms.
    """
    # Remove the english record at the end of the line
    parts = line.split("\t")
    return set(parts[2].split(";"))


def get_synsets(wordnet_path: Path) -> list[set[str]]:
    """Parses the WordNet file and returns a list of all synsets.

    Args:
        wordnet_path (Path): Path to the WordNet file.

    Returns:
        list[set[str]]: List of synsets, where each synset is a set of synonyms.
    """
    with open(wordnet_path, "r", encoding="utf-8") as f:
        return list(map(get_synset, f.readlines()))


def create_synonym_dict(synsets: list[set[str]]) -> dict[str, set[str]]:
    """Creates a dictionary mapping each word to its set of synonyms.

    Args:
        synsets (list[set[str]]): List of synsets, where each synset is a set of synonyms.

    Returns:
        dict[str, set[str]]: Dictionary mapping each word to its set of synonyms.
    """
    synonyms: dict[str, set[str]] = {}
    for synset in synsets:
        for word in synset:
            synonyms.setdefault(word, set()).update(synset - {word})
    return synonyms


def get_mask(df: pd.DataFrame) -> pd.Series:
    """Returns a mask for synthetic reviews in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to get the mask for.

    Returns:
        pd.Series: Mask for synthetic reviews.
    """
    return df["label"] == 1


def synonymize_full(
    df: pd.DataFrame,
    wordnet_path: Path,
    word_prob: float = 0.1,
) -> pd.DataFrame:
    """Synonymizes the reviews in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to synonymize.
        wordnet_path (Path): Path to the WordNet database.
        word_prob (float): Probability of synonymizing each word in the review.

    Returns:
        pd.DataFrame: DataFrame with synonymized synthetic reviews.
    """
    df = df.copy()
    mask = get_mask(df)
    synonyms = create_synonym_dict(get_synsets(wordnet_path))

    def synonymize_word(word: str) -> str:
        word = word.strip()
        word_lower = word.lower()
        word_upper = word.upper()
        word_title = word.title()
        if random.random() > word_prob:
            return word

        punctuation = None
        if word and word[-1] in PUNCTUATION:
            punctuation = word[-1]
            word = word[:-1]

        for variant in [word, word_lower, word_upper, word_title]:
            if variant in synonyms and synonyms[variant]:
                synonym = random.choice(list(synonyms[variant]))
                return synonym + (punctuation if punctuation else "")

        return word

    def synonymize_one(review: str) -> str:
        return " ".join(map(synonymize_word, review.split()))

    df.loc[mask, "content"] = df.loc[mask, "content"].apply(synonymize_one)
    return df


def remove_punctuation_full(df: pd.DataFrame) -> pd.DataFrame:
    """Removes punctuation from the reviews in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to remove punctuation from.

    Returns:
        pd.DataFrame: DataFrame with no punctuation in synthetic reviews.
    """
    df = df.copy()
    mask = get_mask(df)
    df.loc[mask, "content"] = df.loc[mask, "content"].apply(remove_punctuation_one)
    return df


def remove_punctuation_one(review: str) -> str:
    """Removes punctuation from a single review.

    Args:
        review (str): Review to remove punctuation from.

    Returns:
        str: Review with no punctuation.
    """
    for p in PUNCTUATION:
        review = review.replace(p, "")
    return review


def check_folders(withhout_punctuation_path: Path, synonymized_path: Path):
    """Checks if the output folders exist, and creates them if they don't.

    Args:
        withhout_punctuation_path (Path): Path to the folder for no punctuation reviews.
        synonymized_path (Path): Path to the folder for synonymized reviews.
    """
    if not withhout_punctuation_path.exists():
        withhout_punctuation_path.mkdir(parents=True)
    if not synonymized_path.exists():
        synonymized_path.mkdir(parents=True)


def main():
    """Script main function"""
    args = parse_args()
    df = pd.read_csv(args.input)

    check_folders(
        path_np := args.output_no_punctuation,
        path_syn := args.output_synonymized,
    )

    try:
        df_synonymized = synonymize_full(
            df,
            args.wordnet,
            args.word_probability,
        )
        df_synonymized.to_csv(path_syn := (path_syn / "test.csv"), index=False)
        print(f"Synonymized reviews saved to: {path_syn}")
    except NotImplementedError as e:
        print(f"Synonymization not implemented: {e}")

    try:
        df_no_punctuation = remove_punctuation_full(df)
        df_no_punctuation.to_csv(path_np := (path_np / "test.csv"), index=False)
        print(f"No punctuation reviews saved to: {path_np}")
    except NotImplementedError as e:
        print(f"Removing punctuation not implemented: {e}")


if __name__ == "__main__":
    main()
