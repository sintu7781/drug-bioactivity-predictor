import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


def evaluate_model(
    model,
    X_test,
    y_test,
):
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


def main() -> None:

    data = np.load(PROCESSED_DATA_DIR / "egfr_features.npz")

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
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = {
        "logistic_regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
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

    results = {}

    best_model = None
    best_name = None
    best_score = -1

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results[name] = metrics

        print(
            classification_report(
                y_test,
                model.predict(X_test),
                target_names=[
                    "INACTIVE",
                    "ACTIVE",
                ],
            )
        )

        print(metrics)

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model
            best_name = name

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_count": X.shape[1],
        "random_state": RANDOM_STATE,
        "metrics": results,
    }

    joblib.dump(
        artifact,
        MODELS_DIR / "bioactivity_model.joblib",
        compress=3,
    )

    print(f"\nBest model: {best_name}")

    print(f"ROC-AUC: {best_score:.4f}")


if __name__ == "__main__":
    main()
