import pandas as pd

from sklearn.base import clone

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def create_pipeline(
    preprocessor,
    model,
):

    return Pipeline(
        steps=[
            (
                "preprocessing",
                clone(preprocessor),
            ),
            (
                "model",
                clone(model),
            ),
        ]
    )


def train_and_evaluate(
    X,
    y,
    preprocessor,
    model,
):

    # ----------------------------------
    # 1. Split the dataset
    # ----------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # ----------------------------------
    # 2. Create preprocessing + model
    # ----------------------------------

    pipeline = create_pipeline(
        preprocessor,
        model,
    )

    # ----------------------------------
    # 3. Train
    # ----------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # ----------------------------------
    # 4. Make predictions
    # ----------------------------------

    predictions = pipeline.predict(
        X_test
    )

    # Probability of class 1
    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # ----------------------------------
    # 5. Evaluation metrics
    # ----------------------------------

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions,
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "ROC-AUC":
            roc_auc_score(
                y_test,
                probabilities,
            ),
    }

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
    )

    # ROC curve values
    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            y_test,
            probabilities,
        )
    )

    roc_data = pd.DataFrame(
        {
            "False Positive Rate":
                false_positive_rate,

            "True Positive Rate":
                true_positive_rate,

            "Threshold":
                thresholds,
        }
    )

    return {

        "pipeline": pipeline,

        "metrics": metrics,

        "confusion_matrix": cm,

        "roc_data": roc_data,

        "X_test": X_test,

        "y_test": y_test,

        "predictions": predictions,

        "probabilities": probabilities,
    }


def compare_models(
    X,
    y,
    preprocessor,
    models,
):

    results = []

    for model_name, model in models.items():

        result = train_and_evaluate(
            X,
            y,
            preprocessor,
            model,
        )

        row = {
            "Model": model_name
        }

        row.update(
            result["metrics"]
        )

        results.append(row)

    results_df = pd.DataFrame(
        results
    )

    return results_df.sort_values(
        by="ROC-AUC",
        ascending=False,
    )


def cross_validate_model(
    X,
    y,
    preprocessor,
    model,
):

    pipeline = create_pipeline(
        preprocessor,
        model,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    return scores


def confusion_matrix_dataframe(cm):

    return pd.DataFrame(
        cm,
        index=[
            "Actual No",
            "Actual Yes",
        ],
        columns=[
            "Predicted No",
            "Predicted Yes",
        ],
    )