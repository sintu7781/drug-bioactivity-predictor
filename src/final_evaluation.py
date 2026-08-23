from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import (
    CURATED_DATA_FILE,
    FEATURE_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from .scaffold_split import scaffold_split


def calculate_metrics(
    model,
    X_test,
    y_test,
) -> dict:

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
        ).ravel()
    )

    return {
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
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:

    print(
        "Loading data..."
    )

    df = pd.read_csv(
        CURATED_DATA_FILE
    )

    feature_data = np.load(
        FEATURE_FILE
    )

    X = feature_data["X"]
    y = feature_data["y"]

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]

    report = {
        "dataset": {
            "compounds": len(df),
            "features": X.shape[1],
            "active": int(
                y.sum()
            ),
            "inactive": int(
                (y == 0).sum()
            ),
            "active_fraction": float(
                y.mean()
            ),
        },
        "model": {
            "name": artifact[
                "model_name"
            ],
        },
    }

    # ====================================================
    # Random stratified split
    # ====================================================

    (
        X_train,
        X_test,
        _y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # The saved model was already trained on the
    # same random training partition.

    random_metrics = (
        calculate_metrics(
            model,
            X_test,
            y_test,
        )
    )

    report[
        "random_split"
    ] = {
        "train_size": len(
            X_train
        ),
        "test_size": len(
            X_test
        ),
        "metrics": random_metrics,
    }

    # ====================================================
    # Scaffold split
    # ====================================================

    (
        scaffold_train,
        scaffold_test,
    ) = scaffold_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_scaffold_train = X[
        scaffold_train
    ]

    X_scaffold_test = X[
        scaffold_test
    ]

    y_scaffold_train = y[
        scaffold_train
    ]

    y_scaffold_test = y[
        scaffold_test
    ]

    from sklearn.base import clone

    scaffold_model = clone(
        model
    )

    scaffold_model.fit(
        X_scaffold_train,
        y_scaffold_train,
    )

    scaffold_metrics = (
        calculate_metrics(
            scaffold_model,
            X_scaffold_test,
            y_scaffold_test,
        )
    )

    report[
        "scaffold_split"
    ] = {
        "train_size": len(
            scaffold_train
        ),
        "test_size": len(
            scaffold_test
        ),
        "train_active_fraction": float(
            y_scaffold_train.mean()
        ),
        "test_active_fraction": float(
            y_scaffold_test.mean()
        ),
        "metrics": scaffold_metrics,
    }

    # ====================================================
    # Generalization gap
    # ====================================================

    random_auc = (
        random_metrics["roc_auc"]
    )

    scaffold_auc = (
        scaffold_metrics["roc_auc"]
    )

    report[
        "generalization"
    ] = {
        "roc_auc_random": random_auc,
        "roc_auc_scaffold": scaffold_auc,
        "roc_auc_gap": (
            random_auc
            - scaffold_auc
        ),
    }

    # ====================================================
    # Save
    # ====================================================

    output = Path(
        "reports/model_validation.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL MODEL VALIDATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: {artifact['model_name']}"
    )

    print(
        f"Random ROC-AUC: "
        f"{random_auc:.4f}"
    )

    print(
        f"Scaffold ROC-AUC: "
        f"{scaffold_auc:.4f}"
    )

    print(
        f"Generalization gap: "
        f"{random_auc - scaffold_auc:.4f}"
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()