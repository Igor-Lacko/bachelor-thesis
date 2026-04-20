"""
eval_cross_validation.py
Evaluates the performance of several models from cross validation
results and creates latex tables with the results.
Author: Igor Lacko
"""

import argparse
from itertools import product
from json import loads
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from rich.console import Console

console = Console()


TFeatures = Literal[
    "transformer",
    "tfidf",
    "statistical",
]
TVariant = Literal["unified", "separate"]


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments for evaluating models.

    Returns:
        argparse.Namespace: Object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate models and create latex tables/figures."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the csv file containing classification results. Has to contain"
        " columns: model, features, variant, f1_score, parameters",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Directory to save the tables.",
        required=False,
    )

    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        help="Number of top models to include in the tables and figures. Default: 3, -1 (or any negative number) for all models.",
        default=3,
    )

    parser.add_argument(
        "--show-best-params",
        action="store_true",
        help="If true, shows the best parameters for each model for each variant and features. "
        "Ignores all other parameters except -i",
    )

    return parser.parse_args()


def validate_df(df: pd.DataFrame) -> bool:
    """Validates the input dataframe for evaluating models.

    Args:
        df (pd.DataFrame): Dataframe to validate. Has to contain columns: model, features, variant, f1_score, parameters

    Returns:
        bool: True if the dataframe is valid, False otherwise.
    """
    required_columns = {"model", "features", "variant", "f1_score", "parameters"}
    if not required_columns.issubset(df.columns):
        console.print(
            f"Error: Input dataframe is missing required columns. Required columns are: {required_columns}"
        )
        return False
    return True


def sort_and_filter(df: pd.DataFrame, mask: pd.Series, top_n: int) -> pd.DataFrame:
    """Utility function to sort the dataframe by f1_score and filter it by a boolean mask, then return the top n models.

    Args:
        df (pd.DataFrame): Dataframe to sort and filter.
        mask (pd.Series): Boolean mask to filter the dataframe.
        top_n (int): Number of top models to include in the filtered dataframe. -1 for all models.

    Returns:
        pd.DataFrame: Sorted and filtered dataframe.
    """
    df_filtered = df[mask].sort_values(by="f1_score", ascending=False)
    if top_n > 0:
        df_filtered = df_filtered.head(top_n)
    return df_filtered


def to_table(df: pd.DataFrame, output_path: Path):
    """Converts the input dataframe to a latex table.
        - Uses just model and f1_score columns

    Args:
        df (pd.DataFrame): Dataframe to convert to a latex table.
        output_path (Path): Path to save the latex table.
    """
    df_table = df[["model", "f1_score"]].copy()
    df_table.to_latex(
        output_path, index=False, float_format="%.4f", caption="Model Performance"
    )


def show_params(params: dict):
    """Utility function to print the parameters of a model in a readable format.

    Args:
        params (dict): Dictionary containing the parameters of a model.
    """
    console.print("\nParameters for best model:")
    for param, value in params.items():
        console.print(f"\t- {param}: {value}")


def show_top_models(df: pd.DataFrame, show_variant: bool = False):
    """Utility function to print the top models in a readable format.

    Args:
        df (pd.DataFrame): Dataframe containing the models to print.
        show_variant (bool, optional): Whether to show the variant for each model. Defaults to False.
    """
    for idx, (_, row) in enumerate(df.iterrows()):
        if show_variant:
            console.print(
                f"\t{idx + 1}. {row['model'].upper()}, Variant: {row['variant']}, F1 Score: {row['f1_score']:.4f}"
            )
        else:
            console.print(
                f"\t{idx + 1}. {row['model'].upper()}, F1 Score: {row['f1_score']:.4f}"
            )


def eval_group(
    df_result: pd.DataFrame,
    top_n: int,
    variant: TVariant,
    features: TFeatures,
    output_dir: Path,
):
    """Evaluates the performance of all models on a given variant and features.

    Args:
        df_result (pd.DataFrame): Dataframe with results.
        top_n (int): Number of top models to include.
        variant (TVariant): Variant to evaluate.
        features (TFeatures): Features to evaluate.
        output_dir (Path): Directory to save tables.
    """
    mask = (df_result["variant"] == variant) & (df_result["features"] == features)
    df_group = sort_and_filter(df_result, mask, top_n)

    console.print("-" * 100)

    if df_group.empty:
        console.print(f"No rows for variant '{variant}' and features '{features}'.")
        return

    console.print(
        f"Top {len(df_group)} models for variant '{variant}' and features '{features}':"
    )

    show_top_models(df_group)
    params = loads(df_group.iloc[0]["parameters"])
    show_params(params)
    to_table(df_group, output_dir / f"{variant}_{features}.tex")


def eval_features_absolute(
    df_result: pd.DataFrame,
    top_n: int,
    features: TFeatures,
    output_dir: Path,
):
    """Evaluates the performance of all models on given features across both variants.
       Just takes the max across both variants, e.g. doesn't average or anything.

    Args:
        df_result (pd.DataFrame): Dataframe with results.
        top_n (int): Number of top models to include.
        features (TFeatures): Features to evaluate.
        output_dir (Path): Directory to save tables.
    """
    mask = df_result["features"] == features
    df_group = sort_and_filter(df_result, mask, top_n)

    console.print("-" * 100)

    if df_group.empty:
        console.print(f"No rows for features '{features}'.")
        return

    console.print(
        f"Top {len(df_group)} models for features '{features}' across both variants:"
    )

    show_top_models(df_group, show_variant=True)
    params = loads(df_group.iloc[0]["parameters"])
    show_params(params)
    to_table(df_group, output_dir / f"both_{features}.tex")


def eval_features_average(
    df_result: pd.DataFrame,
    top_n: int,
    features: TFeatures,
    output_dir: Path,
):
    """Evaluates the performance of all models on given features across both variants.
       Averages the f1_score across both variants for each model.

    Args:
        df_result (pd.DataFrame): Dataframe with results.
        top_n (int): Number of top models to include.
        features (TFeatures): Features to evaluate.
        output_dir (Path): Directory to save tables.
    """
    mask = df_result["features"] == features
    df_group = df_result[mask].copy()

    if df_group.empty:
        console.print(f"No rows for features '{features}'.")
        return

    # Average across variants
    df_group = df_group.groupby("model").agg({"f1_score": "mean"}).reset_index()

    # Top n
    df_group = df_group.sort_values(by="f1_score", ascending=False)
    if top_n > 0:
        df_group = df_group.head(top_n)

    console.print("-" * 100)
    console.print(
        f"Top {len(df_group)} models for features '{features}' across both variants (averaged):"
    )
    show_top_models(df_group)

    to_table(df_group, output_dir / f"both_{features}_average.tex")


def run_eval(df_result: pd.DataFrame, top_n: int, output_dir: Path):
    """Runs the full evaluation.

    Args:
        df_result (pd.DataFrame): Dataframe with results.
        top_n (int): Number of top models to include.
        output_dir (Path): Directory to save tables.
    """
    for variant, features in product(
        get_args(TVariant), (all_features := get_args(TFeatures))
    ):
        eval_group(df_result, top_n, variant, features, output_dir)

    for features in all_features:
        eval_features_absolute(df_result, top_n, features, output_dir)
        eval_features_average(df_result, top_n, features, output_dir)
        to_feature_table(df_result, features, output_dir / f"feature_{features}.tex")


def show_best_params(df_result: pd.DataFrame):
    """Shows the best parameters for each model.

    Args:
        df_result (pd.DataFrame): Dataframe with results.
    """
    df_sorted = df_result.sort_values(by="model")
    for result in df_sorted.itertuples():
        console.print(
            f"Model: {str(result.model).upper()}, Variant: {result.variant}, Features: {result.features}"
        )
        params = loads(str(result.parameters))
        show_params(params)
        console.print("-" * 50)


def to_feature_table(df: pd.DataFrame, features: TFeatures, output_path: Path):
    """Converts the input dataframe to a latex table for the given features.
    - Each table has columns variant model f1_score, where variant is the index

    Args:
        df (pd.DataFrame): Dataframe to convert to a latex table.
        features (TFeatures): Features to filter the dataframe by.
        output_path (Path): Path to save the latex table.
    """
    mask = df["features"] == features
    df_table = df[mask][["variant", "model", "f1_score"]].copy()

    # Make the values look nicer
    model_map = {
        "logistic_regression": "Logistic Regression",
        "knn": "KNN",
        "random_forest": "Random Forest",
        "slovakbert": "SlovakBERT",
        "modernbert": "ModernBERT",
        "naive_bayes": "Multinomial Naive Bayes",
        "svm": "SVM",
        "mlp": "MLP",
        "transformer": "Transformer",
    }

    col_map = {
        "variant": "Variant",
        "model": "Model",
        "f1_score": "F1 Score",
    }

    def _map_func(model_name: str) -> str:
        for key in model_map.keys():
            if key in model_name:
                return model_map[key]
        return model_name

    df_table["model"] = df_table["model"].map(_map_func)
    df_table.rename(columns=col_map, inplace=True)
    df_table = df_table.sort_values(by=["Variant", "Model"]).set_index(
        ["Variant", "Model"]
    )

    df_table.to_latex(
        output_path,
        index=True,
        multirow=True,
        float_format="%.4f",
        caption=f"Validation results on {features.title()} features",
    )

    print(df_table)


def main():
    """Script entry point."""
    args = parse_args()
    if args.show_best_params:
        show_best_params(pd.read_csv(args.input))
        return

    if (output := args.output) is None:
        console.print("No output directory specified.")
        return

    (output_dir := Path(output)).mkdir(parents=True, exist_ok=True)

    df_result = pd.read_csv(args.input)
    if not validate_df(df_result):
        return

    run_eval(df_result, args.top_n, output_dir)


if __name__ == "__main__":
    main()
