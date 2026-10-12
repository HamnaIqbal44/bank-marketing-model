from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


def build_preprocessor(
    numeric_features,
    categorical_features,
):

    # Numerical variables are scaled.
    numeric_transformer = StandardScaler()

    # Categorical variables are converted
    # into numerical indicator columns.
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    return preprocessor