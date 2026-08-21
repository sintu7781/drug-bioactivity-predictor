from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
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
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (
    CV_FOLDS,
    FEATURE_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)


def metrics(
    model,
    X,
    y,
) -> dict[str, float]:
    
    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    return {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
    }


def build_models():

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
        "random_forest": (
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }
    

def main() -> None:
    
    data = np.load(
        FEATURE_FILE
    )
    
    X = data["X"]
    y = data["y"]
    
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
    
    models = build_models()

    results = []

    best_model = None
    best_name = None
    best_score = -np.inf
    
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for name, model in models.items():
        
        print(
            f"\nTraining {name}..."
        )
        
        cv_scores = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=None,
        )

        model.fit(
            X_train,
            y_train,
        )

        test_metrics = metrics(
            model,
            X_test,
            y_test,
        )
        
        row = {
            "model": name,
            **test_metrics,
            "cv_roc_auc_mean": (
                cv_scores.mean()
            ),
            "cv_roc_auc_std": (
                cv_scores.std()
            ),
        }

        results.append(
            row
        )

        print(
            json.dumps(
                row,
                indent=2,
            )
        )

        if (
            test_metrics["roc_auc"] 
            > best_score
        ):
            best_score = (
                test_metrics["roc_auc"]
            )
            best_model = model
            best_name = name
            
    comparison = pd.DataFrame(
        results
    )
            
    comparison.to_csv(
        "reports/model_comparison.csv",
        index=False,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_count": X.shape[1],
        "random_state": RANDOM_STATE,
        "target": "EGFR",
        "target_chembl_id": "CHEMBL203",
        "activity_type": "IC50",
        "activity_unit": "nM",
        "active_threshold_nM": 1000.0,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "metrics": results,
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
        compress=3,
    )

    print(
        f"\nBest model: {best_name}"
    )

    print(
        f"Test ROC-AUC: {best_score:.4f}"
    )
    
    print(
        f"Saved: {MODEL_FILE}"
    )
    
    
if __name__ == "__main__":
    main()

