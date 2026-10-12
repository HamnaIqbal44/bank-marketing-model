from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "bank-full.csv"
)


def download_dataset():
    """
    Download the UCI Bank Marketing dataset.

    Dataset ID 222 = Bank Marketing.
    """

    from ucimlrepo import fetch_ucirepo

    bank = fetch_ucirepo(id=222)

    # Predictor variables
    X = bank.data.features.copy()

    # Target variable
    y = bank.data.targets.copy()

    # y normally arrives as a DataFrame.
    # We only need its first column.
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Combine predictors and target.
    df = X.copy()
    df["y"] = y.values

    # Create data folder if needed.
    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save a local CSV.
    df.to_csv(
        DATA_PATH,
        index=False,
    )

    return df


def load_raw_data():
    """
    Load bank-full.csv.

    If the file does not exist,
    download it automatically.
    """

    if not DATA_PATH.exists():
        return download_dataset()

    # Original UCI CSV uses semicolon separators.
    # Our automatically generated CSV uses commas.

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        first_line = file.readline()

    if first_line.count(";") > first_line.count(","):
        separator = ";"
    else:
        separator = ","

    return pd.read_csv(
        DATA_PATH,
        sep=separator,
    )


def clean_data(df):
    """
    Clean the dataset before modelling.
    """

    cleaned = df.copy()

    # Remove duration because it creates data leakage.
    if "duration" in cleaned.columns:
        cleaned = cleaned.drop(
            columns=["duration"]
        )

    # Convert target:
    #
    # yes -> 1
    # no  -> 0

    cleaned["y"] = cleaned["y"].map(
        {
            "yes": 1,
            "no": 0,
        }
    )

    cleaned["y"] = cleaned["y"].astype(int)

    return cleaned


def split_features_target(df):
    """
    Separate predictors X from target y.
    """

    X = df.drop(
        columns=["y"]
    )

    y = df["y"]

    # Numerical columns
    numeric_features = (
        X.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    # Categorical columns
    categorical_features = (
        X.select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    return (
        X,
        y,
        numeric_features,
        categorical_features,
    )