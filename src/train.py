from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (
    CV_FOLDS,
    FEATURE_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)


def build_models():
    """
    Build the baseline classification models.
    """

    return {
        "logistic_regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_metrics(
    model,
    X,
    y,
) -> dict[str, float]:
    """
    Calculate classification metrics on a held-out test set.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),
    }


def main() -> None:

    print(
        "Loading feature matrix..."
    )

    data = np.load(
        FEATURE_FILE
    )

    X = data["X"]
    y = data["y"]

    print(
        f"Feature matrix: {X.shape}"
    )

    print(
        f"Target vector: {y.shape}"
    )

    # ----------------------------------------------------
    # Train/test split
    # ----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # ----------------------------------------------------
    # Cross-validation
    # ----------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    results: list[dict] = []

    best_model = None
    best_model_name = None
    best_roc_auc = -np.inf

    # ----------------------------------------------------
    # Train models
    # ----------------------------------------------------

    for name, model in models.items():

        print()
        print(
            "=" * 60
        )

        print(
            f"Training {name}..."
        )

        print(
            "=" * 60
        )

        # Cross-validation ROC-AUC.
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=None,
        )

        print(
            "Cross-validation ROC-AUC:"
        )

        print(
            f"  Mean: {cv_scores.mean():.4f}"
        )

        print(
            f"  Std : {cv_scores.std():.4f}"
        )

        # Fit final model on all training data.
        model.fit(
            X_train,
            y_train,
        )

        # Evaluate on completely held-out test set.
        test_metrics = calculate_metrics(
            model,
            X_test,
            y_test,
        )

        result = {
            "model": name,
            "accuracy": test_metrics[
                "accuracy"
            ],
            "precision": test_metrics[
                "precision"
            ],
            "recall": test_metrics[
                "recall"
            ],
            "f1": test_metrics[
                "f1"
            ],
            "roc_auc": test_metrics[
                "roc_auc"
            ],
            "pr_auc": test_metrics[
                "pr_auc"
            ],
            "cv_roc_auc_mean": float(
                cv_scores.mean()
            ),
            "cv_roc_auc_std": float(
                cv_scores.std()
            ),
        }

        results.append(
            result
        )

        print()
        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        # Select the model using held-out ROC-AUC.
        if (
            test_metrics["roc_auc"]
            > best_roc_auc
        ):

            best_roc_auc = (
                test_metrics["roc_auc"]
            )

            best_model = model

            best_model_name = name

    # ----------------------------------------------------
    # Save comparison
    # ----------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    MODEL_COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    # ----------------------------------------------------
    # Save best model
    # ----------------------------------------------------

    if best_model is None:
        raise RuntimeError(
            "No model was successfully trained."
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": best_model,
        "model_name": best_model_name,

        "feature_count": int(
            X.shape[1]
        ),

        "random_state": RANDOM_STATE,

        "target": "EGFR",

        "target_chembl_id": "CHEMBL203",

        "activity_type": "IC50",

        "activity_unit": "nM",

        "active_threshold_nM": 1000.0,

        "morgan_radius": 2,

        "morgan_bits": 2048,

        "descriptor_count": 8,

        "metrics": results,
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
        compress=3,
    )

    # ----------------------------------------------------
    # Final report
    # ----------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "MODEL TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Best model : {best_model_name}"
    )

    print(
        f"ROC-AUC    : {best_roc_auc:.4f}"
    )

    print(
        f"Model file : {MODEL_FILE}"
    )

    print(
        f"Comparison : {MODEL_COMPARISON_FILE}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()