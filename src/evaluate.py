from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    train_test_split,
)

from .config import (
    FEATURE_FILE,
    FIGURES_DIR,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)


def main() -> None:

    data = np.load(
        FEATURE_FILE
    )

    X = data["X"]
    y = data["y"]

    (
        _X_train,
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

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=[
            "INACTIVE",
            "ACTIVE",
        ],
    )

    plt.title(
        "EFGR Bioactivity Confusion Matrix"
    )
    
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / 
        "confusion_matrix.png",
        dpi=200,
    )

    plt.close()

    # ROC
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title(
        "EFGR Bioactivity ROC Curve"
    )
    
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / 
        "roc_curve.png",
        dpi=200,
    )

    plt.close()
    
    # PR
    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )
    
    plt.title(
        "EFGR Bioactivity Precision-Recall Curve"
    )
        
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / 
        "precision_recall_curve.png",
        dpi=200,
    )

    plt.close()
    
    print(
        "Evaluation figures generated."
    )


if __name__ == "__main__":
    main()
