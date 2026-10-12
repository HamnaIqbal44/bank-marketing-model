import numpy as np
import pandas as pd


def get_feature_names(pipeline):

    preprocessor = (
        pipeline.named_steps[
            "preprocessing"
        ]
    )

    return (
        preprocessor
        .get_feature_names_out()
    )


def explain_model(
    pipeline,
    top_n=20,
):

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = get_feature_names(
        pipeline
    )

    # --------------------------------
    # Logistic Regression
    # --------------------------------

    if hasattr(model, "coef_"):

        coefficients = model.coef_[0]

        df = pd.DataFrame(
            {
                "Feature":
                    feature_names,

                "Coefficient":
                    coefficients,

                "Absolute importance":
                    np.abs(coefficients),
            }
        )

        df = df.sort_values(
            by="Absolute importance",
            ascending=False,
        )

        return (
            "coefficient",
            df.head(top_n),
        )

    # --------------------------------
    # Decision Tree / Random Forest
    # --------------------------------

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = (
            model.feature_importances_
        )

        df = pd.DataFrame(
            {
                "Feature":
                    feature_names,

                "Importance":
                    importance,
            }
        )

        df = df.sort_values(
            by="Importance",
            ascending=False,
        )

        return (
            "importance",
            df.head(top_n),
        )

    return None, None