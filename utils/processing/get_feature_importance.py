"""
feature_importance.py
Compares the feature importance of models trained on statistical, TFIDF and meta features.
Author: Igor Lacko
"""

import argparse
import pickle
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from unidecode import unidecode

TFeature = Literal["statistical", "tfidf", "meta"]
TVariant = Literal["unified", "separate"]

parser = argparse.ArgumentParser(description="View feature importance of models")
parser.add_argument(
    "-b",
    "--base_path",
    type=Path,
    required=True,
    help="Path to the base directory of the project",
)
parser.add_argument(
    "-ds",
    "--dataset-separate",
    type=Path,
    required=True,
    help="Path to the separate dataset. Needed to fit a tfidfvectorizer in the same way as during classification to get feature names.",
)
parser.add_argument(
    "-du",
    "--dataset-unified",
    type=Path,
    required=True,
    help="Path to the unified dataset. Needed to fit a tfidfvectorizer in the same way as during classification to get feature names.",
)
parser.add_argument(
    "--stop-word-file",
    type=Path,
    required=True,
    help="Path to the file containing stop words. Needed to fit a tfidfvectorizer in the same way as during classification to get feature names.",
)
parser.add_argument(
    "-o",
    "--output",
    type=Path,
    required=True,
    help="Path to the output folder, creates 3 files for each feature type",
)


def get_model_path(base_path: Path, feature: TFeature, variant: TVariant) -> Path:
    """Returns path to a model. Expects it to be the only file in the folder.

    Args:
        base_path (Path): Path to the base directory of the project
        feature (TFeature): Feature type (statistical, tfidf, meta)
        variant (TVariant): Dataset variant (unified, separate)

    Returns:
        Path: Path to the model file
    """
    folder = base_path / f"{variant}/{feature}"
    return next(folder.iterdir())


def get_tfidf_feature_names(df_train: pd.DataFrame, stop_words: set) -> list[str]:
    """Returns the names of the TF-IDF features

    Args:
        df_train (pd.DataFrame): The training dataset
        stop_words (set): The set of stop words

    Returns:
        list[str]: List of TF-IDF feature names
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(stop_words),
    )
    vectorizer.fit(df_train["content"])
    return vectorizer.get_feature_names_out().tolist()


def get_statistical_feature_names() -> list[str]:
    """Returns the names of the statistical features

    Returns:
        list[str]: List of statistical feature names
    """
    return [
        "Word count",
        "Char count",
        "Mean sentence length",
        "Mean word length",
        "ARI",
        "TTR",
        "Stop word ratio",
        "Sentence count",
        "Unique word count",
        "Hapax legomena rate",
        "Dot ratio",
        "Exclamation mark ratio",
        "Question mark ratio",
        "Comma ratio",
        "Semicolon ratio",
        "Colon ratio",
        "Hyphen ratio",
        "Ratio of uppercase words",
        "Ratio of titlecase words",
        "Noun frequency",
        "Adjective frequency",
        "Verb frequency",
        "Adverb frequency",
        "Count of unknown words",
    ]


def get_meta_feature_names() -> list[str]:
    """Returns the names of the meta features

    Returns:
        list[str]: List of meta feature names
    """
    return ["Statistical", "TF-IDF", "Transformer"]


def get_random_forest_importance(
    model: RandomForestClassifier, feature: TFeature
) -> dict[str, float]:
    """Returns the feature importance for a random forest model

    Args:
        model (RandomForestClassifier): The random forest model
        feature (TFeature): Feature type (statistical, tfidf, meta)
    Returns:
        dict[str, float]: Dictionary mapping feature names to their importance
    """
    importance = model.feature_importances_
    if feature == "statistical":
        feature_names = get_statistical_feature_names()
    elif feature == "meta":
        feature_names = get_meta_feature_names()
    else:
        raise ValueError(f"Unsupported feature type: {feature}")
    return dict(zip(feature_names, importance))


def parse_stop_words(stop_word_file: Path) -> set[str]:
    """Parse stop words from a file, normalize, and return as a set.

    Args:
        stop_word_file (Path): Path to the file containing stop words.

    Returns:
        set[str]: Set of normalized stop words.

    Raises:
        FileNotFoundError: If the stop word file does not exist.
    """
    if not stop_word_file.is_file():
        raise FileNotFoundError(f"Stop word file not found: {stop_word_file}")

    with stop_word_file.open("r") as f:
        stop_words = set()
        for line in f:
            stop_words.add(unidecode(line.strip().lower()))

    return stop_words


def get_logistic_regression_importance(
    model: LogisticRegression, df_train: pd.DataFrame, stop_words: set
) -> dict[str, float]:
    """Returns the feature importance for a logistic regression model (TF-IDF)

    Args:
        model (LogisticRegression): The logistic regression model
        df_train (pd.DataFrame): The training dataset
        stop_words (set): The set of stop words
    Returns:
        dict[str, float]: Dictionary mapping feature names to their importance
    """
    importance = model.coef_[0]
    feature_names = get_tfidf_feature_names(df_train, stop_words)

    # Sort features by importance
    importance, feature_names = zip(
        *sorted(zip(importance, feature_names), key=lambda x: abs(x[0]), reverse=True)
    )

    # Top 10
    return dict(zip(feature_names[:10], importance[:10]))


def get_feature_importances(
    base_path: Path,
    feature: TFeature,
    variant: TVariant,
    df_train: pd.DataFrame,
    stop_words: set,
) -> dict[str, float]:
    """Returns the feature importances for the given model

    Args:
        base_path (Path): Path to the base directory where trained models are
        feature (TFeature): Feature type (statistical, tfidf, meta)
        variant (TVariant): Dataset variant (unified, separate)
        df_train (pd.DataFrame): The training dataset, needed to get feature names for TF-IDF
        stop_words (set): The set of stop words

    Raises:
        ValueError: If the model type is not supported

    Returns:
        dict[str, float]: Dictionary mapping feature names to their importance
    """
    model_path = get_model_path(base_path, feature, variant)

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Get feature importance
    if isinstance(model, RandomForestClassifier):
        importance = get_random_forest_importance(model, feature)
        print(importance)
        return importance
    elif isinstance(model, LogisticRegression):
        importance = get_logistic_regression_importance(model, df_train, stop_words)
        print(importance)
        return importance
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def main():
    args = parser.parse_args()
    base_path = args.base_path
    stop_words = parse_stop_words(args.stop_word_file)
    df_train_separate = pd.read_csv(args.dataset_separate)
    df_train_unified = pd.read_csv(args.dataset_unified)
    (output_path := args.output).mkdir(parents=True, exist_ok=True)

    for feature in get_args(TFeature):
        feature_path = output_path / f"{feature}.csv"
        df_output = pd.DataFrame(columns=["feature", "importance", "variant"])
        for variant in get_args(TVariant):
            importances = get_feature_importances(
                base_path,
                feature,
                variant,
                df_train_separate if variant == "separate" else df_train_unified,
                stop_words,
            )

            df_output = pd.concat(
                [
                    df_output,
                    pd.DataFrame(
                        {
                            "feature": list(importances.keys()),
                            "importance": list(importances.values()),
                            "variant": variant,
                        }
                    ),
                ],
                ignore_index=True,
            )

        df_output.to_csv(feature_path, index=False)


if __name__ == "__main__":
    main()
