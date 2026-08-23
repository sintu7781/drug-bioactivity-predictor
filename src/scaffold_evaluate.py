from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    CURATED_DATA_FILE,
    FEATURE_FILE,
    MODEL_FILE,
    RANDOM_STATE,
)
from .scaffold_split import scaffold_split


def main() -> None:

    print("Loading curated dataset...")

    df = pd.read_csv(
        CURATED_DATA_FILE
    )

    data = np.load(
        FEATURE_FILE
    )

    X = data["X"]
    y = data["y"]

    if len(df) != len(X):

        raise RuntimeError(
            "Feature matrix and curated "
            "dataset have different lengths."
        )

    print(
        f"Compounds: {len(df)}"
    )

    print(
        "Generating Bemis-Murcko scaffolds..."
    )

    train_indices, test_indices = (
        scaffold_split(
            df,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )
    )

    X_train = X[
        train_indices
    ]

    X_test = X[
        test_indices
    ]

    y_train = y[
        train_indices
    ]

    y_test = y[
        test_indices
    ]

    print(
        f"Training compounds: "
        f"{len(train_indices)}"
    )

    print(
        f"Test compounds: "
        f"{len(test_indices)}"
    )

    print(
        f"Training active fraction: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Test active fraction: "
        f"{y_test.mean():.4f}"
    )

    # ----------------------------------------------------
    # Load the selected Random Forest model
    # ----------------------------------------------------

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]

    # IMPORTANT:
    # The previously saved model was trained using
    # the random split. We need a NEW model trained
    # only on the scaffold-training set.
    #
    # Recreate the model from the saved artifact's
    # parameters.

    from sklearn.base import clone

    scaffold_model = clone(
        model
    )

    print(
        "\nTraining model on scaffold "
        "training set..."
    )

    scaffold_model.fit(
        X_train,
        y_train,
    )

    predictions = (
        scaffold_model.predict(
            X_test
        )
    )

    probabilities = (
        scaffold_model.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = {
        "split": "scaffold",
        "model": artifact[
            "model_name"
        ],
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
        "train_size": len(
            train_indices
        ),
        "test_size": len(
            test_indices
        ),
    }

    print(
        "\nScaffold split results:"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    output = (
        "reports/"
        "scaffold_evaluation.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()