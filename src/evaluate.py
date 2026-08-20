import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

from .config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


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

    artifact = joblib.load(MODELS_DIR / "bioactivity_model.joblib")

    model = artifact["model"]

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=[
            "INACTIVE",
            "ACTIVE",
        ],
    )

    plt.title("Bioactivity Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=200,
    )

    plt.close()

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("Bioactivity ROC Curve")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=200,
    )

    plt.close()


if __name__ == "__main__":
    main()
