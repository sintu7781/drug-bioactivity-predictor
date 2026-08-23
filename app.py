from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.predict import BioactivityPredictor


# =======================================================
# Paths
# =======================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "bioactivity_model.joblib"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_validation.json"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


# =======================================================
# Page configuration
# =======================================================

st.set_page_config(
    page_title="Drug Bioactivity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =======================================================
# Styling
# =======================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #666;
            margin-bottom: 1.5rem;
        }

        .prediction-active {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #2e7d32;
            background-color: #e8f5e9;
        }

        .prediction-inactive {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #c62828;
            background-color: #ffebee;
        }

        .disclaimer {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f5f5f5;
            color: #555;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =======================================================
# Load model
# =======================================================

@st.cache_resource
def load_predictor() -> BioactivityPredictor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found.\n\n"
            "Expected:\n"
            f"{MODEL_PATH}\n\n"
            "Run the training pipeline first."
        )

    return BioactivityPredictor(
        MODEL_PATH
    )


try:
    predictor = load_predictor()

except Exception as exc:
    st.error(
        f"Unable to load the trained model: {exc}"
    )
    st.stop()


# =======================================================
# Utility functions
# =======================================================

def render_molecule(
    smiles: str,
):
    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return None

    return Draw.MolToImage(
        molecule,
        size=(500, 400),
    )


def format_probability(
    probability: float,
) -> str:
    return f"{probability * 100:.2f}%"


def prediction_class(
    prediction: str,
) -> str:
    return prediction.upper()


# =======================================================
# Header
# =======================================================

st.markdown(
    '<div class="main-title">'
    "🧬 Drug Bioactivity Predictor"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Machine-learning prediction of EGFR bioactivity "
    "from molecular structure."
    "</div>",
    unsafe_allow_html=True,
)


# =======================================================
# Sidebar
# =======================================================

with st.sidebar:

    st.header(
        "Model Information"
    )

    st.write(
        "**Target:** EGFR"
    )

    st.write(
        "**ChEMBL ID:** CHEMBL203"
    )

    st.write(
        "**Endpoint:** IC50"
    )

    st.write(
        "**Activity threshold:** ≤ 1000 nM"
    )

    st.divider()

    st.write(
        "**Model:** Random Forest"
    )

    st.write(
        "**Features:** 2,056"
    )

    st.write(
        "• 8 molecular descriptors"
    )

    st.write(
        "• 2,048 Morgan fingerprint bits"
    )

    st.divider()

    st.caption(
        "Model trained on curated ChEMBL "
        "bioactivity data."
    )


# =======================================================
# Tabs
# =======================================================

single_tab, batch_tab, performance_tab = st.tabs(
    [
        "🧪 Single Compound",
        "📊 Batch Prediction",
        "📈 Model Performance",
    ]
)


# =======================================================
# Single compound prediction
# =======================================================

with single_tab:

    st.header(
        "Predict a Single Compound"
    )

    st.write(
        "Enter a valid SMILES representation "
        "of a chemical compound."
    )

    smiles = st.text_area(
        "SMILES",
        placeholder=(
            "Example: CCO"
        ),
        height=100,
    )

    predict_button = st.button(
        "🔬 Predict Bioactivity",
        type="primary",
        width="stretch",
    )

    if predict_button:

        if not smiles.strip():

            st.warning(
                "Please enter a SMILES string."
            )

        else:

            try:

                cleaned_smiles = (
                    smiles.strip()
                )

                result = predictor.predict(
                    cleaned_smiles
                )

                prediction = (
                    prediction_class(
                        result["prediction"]
                    )
                )

                active_probability = (
                    result[
                        "active_probability"
                    ]
                )

                inactive_probability = (
                    result[
                        "inactive_probability"
                    ]
                )

                descriptors = (
                    result["descriptors"]
                )

                st.divider()

                # ---------------------------------------
                # Prediction banner
                # ---------------------------------------

                if prediction == "ACTIVE":

                    st.success(
                        f"Prediction: {prediction}"
                    )

                else:

                    st.warning(
                        f"Prediction: {prediction}"
                    )

                # ---------------------------------------
                # Probability metrics
                # ---------------------------------------

                col1, col2, col3 = st.columns(
                    3
                )

                with col1:

                    st.metric(
                        "Active Probability",
                        format_probability(
                            active_probability
                        ),
                    )

                with col2:

                    st.metric(
                        "Inactive Probability",
                        format_probability(
                            inactive_probability
                        ),
                    )

                with col3:

                    st.metric(
                        "Model",
                        result["model"],
                    )

                st.progress(
                    active_probability,
                    text=(
                        "Probability of ACTIVE"
                    ),
                )

                st.divider()

                # ---------------------------------------
                # Structure + properties
                # ---------------------------------------

                structure_col, properties_col = (
                    st.columns(
                        [1, 1]
                    )
                )

                with structure_col:

                    st.subheader(
                        "Molecular Structure"
                    )

                    molecule_image = (
                        render_molecule(
                            cleaned_smiles
                        )
                    )

                    if molecule_image is not None:

                        st.image(
                            molecule_image,
                            caption=(
                                "RDKit molecular structure"
                            ),
                        )

                with properties_col:

                    st.subheader(
                        "Molecular Properties"
                    )

                    properties = pd.DataFrame(
                        {
                            "Property": [
                                "Molecular Weight",
                                "LogP",
                                "H-Bond Donors",
                                "H-Bond Acceptors",
                                "Rotatable Bonds",
                                "TPSA",
                                "Ring Count",
                                "Heavy Atom Count",
                            ],
                            "Value": [
                                f"{descriptors['MolWt']:.2f}",
                                f"{descriptors['LogP']:.2f}",
                                f"{int(descriptors['HBD'])}",
                                f"{int(descriptors['HBA'])}",
                                f"{int(descriptors['RotatableBonds'])}",
                                f"{descriptors['TPSA']:.2f}",
                                f"{int(descriptors['RingCount'])}",
                                f"{int(descriptors['HeavyAtomCount'])}",
                            ],
                        }
                    )

                    st.dataframe(
                        properties,
                        hide_index=True,
                        width="stretch",
                    )

                st.divider()

                st.subheader(
                    "Prediction Details"
                )

                st.json(
                    {
                        "smiles": cleaned_smiles,
                        "prediction": prediction,
                        "active_probability": (
                            active_probability
                        ),
                        "inactive_probability": (
                            inactive_probability
                        ),
                        "model": result[
                            "model"
                        ],
                    }
                )

            except ValueError as exc:

                st.error(
                    f"Invalid molecule: {exc}"
                )

            except Exception as exc:

                st.error(
                    "Prediction failed."
                )

                st.exception(
                    exc
                )


# =======================================================
# Batch prediction
# =======================================================

with batch_tab:

    st.header(
        "Batch Prediction"
    )

    st.write(
        "Upload a CSV file containing a "
        "`smiles` column."
    )

    st.code(
        "smiles\nCCO\nCCN\nc1ccccc1",
        language="text",
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            if "smiles" not in batch_df.columns:

                st.error(
                    "CSV must contain a "
                    "'smiles' column."
                )

                st.stop()

            if batch_df.empty:

                st.warning(
                    "The uploaded CSV is empty."
                )

                st.stop()

            st.write(
                f"Loaded **{len(batch_df):,} "
                "compounds**."
            )

            if len(batch_df) > 10_000:

                st.warning(
                    "Large batch detected. "
                    "Processing may take some time."
                )

            results = []

            progress_bar = st.progress(
                0,
                text="Starting prediction...",
            )

            total = len(
                batch_df
            )

            for index, smiles_value in enumerate(
                batch_df["smiles"]
            ):

                smiles_value = str(
                    smiles_value
                ).strip()

                try:

                    result = predictor.predict(
                        smiles_value
                    )

                    descriptors = (
                        result[
                            "descriptors"
                        ]
                    )

                    results.append(
                        {
                            "prediction": (
                                result[
                                    "prediction"
                                ]
                            ),
                            "active_probability": (
                                result[
                                    "active_probability"
                                ]
                            ),
                            "inactive_probability": (
                                result[
                                    "inactive_probability"
                                ]
                            ),
                            "molecular_weight": (
                                descriptors[
                                    "MolWt"
                                ]
                            ),
                            "logp": (
                                descriptors[
                                    "LogP"
                                ]
                            ),
                            "hbd": (
                                descriptors[
                                    "HBD"
                                ]
                            ),
                            "hba": (
                                descriptors[
                                    "HBA"
                                ]
                            ),
                            "tpsa": (
                                descriptors[
                                    "TPSA"
                                ]
                            ),
                            "status": "OK",
                        }
                    )

                except Exception as exc:

                    results.append(
                        {
                            "prediction": (
                                "INVALID"
                            ),
                            "active_probability": (
                                None
                            ),
                            "inactive_probability": (
                                None
                            ),
                            "molecular_weight": (
                                None
                            ),
                            "logp": None,
                            "hbd": None,
                            "hba": None,
                            "tpsa": None,
                            "status": str(
                                exc
                            ),
                        }
                    )

                progress_bar.progress(
                    (index + 1) / total,
                    text=(
                        f"Processing "
                        f"{index + 1:,} / "
                        f"{total:,}"
                    ),
                )

            result_df = pd.concat(
                [
                    batch_df.reset_index(
                        drop=True
                    ),
                    pd.DataFrame(
                        results
                    ),
                ],
                axis=1,
            )
            
            result_df = result_df.convert_dtypes()

            progress_bar.empty()

            st.success(
                "Batch prediction completed."
            )

            # -------------------------------------------
            # Summary
            # -------------------------------------------

            successful = int(
                (
                    result_df["status"]
                    == "OK"
                ).sum()
            )

            active_count = int(
                (
                    result_df["prediction"]
                    == "ACTIVE"
                ).sum()
            )

            inactive_count = int(
                (
                    result_df["prediction"]
                    == "INACTIVE"
                ).sum()
            )

            invalid_count = int(
                (
                    result_df["prediction"]
                    == "INVALID"
                ).sum()
            )

            c1, c2, c3, c4 = st.columns(
                4
            )

            with c1:
                st.metric(
                    "Total",
                    len(result_df),
                )

            with c2:
                st.metric(
                    "Active",
                    active_count,
                )

            with c3:
                st.metric(
                    "Inactive",
                    inactive_count,
                )

            with c4:
                st.metric(
                    "Invalid",
                    invalid_count,
                )

            st.divider()

            # -------------------------------------------
            # Results
            # -------------------------------------------

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True,
            )

            # -------------------------------------------
            # Download
            # -------------------------------------------

            csv_data = (
                result_df
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name=(
                    "bioactivity_predictions.csv"
                ),
                mime="text/csv",
                width="stretch",
            )

        except Exception as exc:

            st.error(
                "Could not process the uploaded CSV."
            )

            st.exception(
                exc
            )


# =======================================================
# Model performance
# =======================================================

with performance_tab:

    st.header(
        "Model Performance"
    )

    # ---------------------------------------------------
    # Validation summary
    # ---------------------------------------------------

    if VALIDATION_PATH.exists():

        import json

        validation = json.loads(
            VALIDATION_PATH.read_text(
                encoding="utf-8"
            )
        )

        random_metrics = (
            validation[
                "random_split"
            ]["metrics"]
        )

        scaffold_metrics = (
            validation[
                "scaffold_split"
            ]["metrics"]
        )

        generalization = (
            validation[
                "generalization"
            ]
        )

        st.subheader(
            "Final Validation"
        )

        c1, c2, c3 = st.columns(
            3
        )

        with c1:

            st.metric(
                "Random Split ROC-AUC",
                f"{random_metrics['roc_auc']:.4f}",
            )

        with c2:

            st.metric(
                "Scaffold Split ROC-AUC",
                f"{scaffold_metrics['roc_auc']:.4f}",
            )

        with c3:

            st.metric(
                "Generalization Gap",
                f"{generalization['roc_auc_gap']:.4f}",
            )

        st.divider()

        # -----------------------------------------------
        # Scaffold metrics
        # -----------------------------------------------

        st.subheader(
            "Scaffold Split Metrics"
        )

        scaffold_table = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC-AUC",
                    "PR-AUC",
                ],
                "Score": [
                    scaffold_metrics[
                        "accuracy"
                    ],
                    scaffold_metrics[
                        "precision"
                    ],
                    scaffold_metrics[
                        "recall"
                    ],
                    scaffold_metrics[
                        "f1"
                    ],
                    scaffold_metrics[
                        "roc_auc"
                    ],
                    scaffold_metrics[
                        "pr_auc"
                    ],
                ],
            }
        )

        scaffold_table["Score"] = (
            scaffold_table["Score"]
            .map(
                lambda x: f"{x:.4f}"
            )
        )

        st.dataframe(
            scaffold_table,
            hide_index=True,
            width="stretch",
        )

    else:

        st.warning(
            "Final validation report not found."
        )

    # ---------------------------------------------------
    # Model comparison
    # ---------------------------------------------------

    if COMPARISON_PATH.exists():

        st.subheader(
            "Model Comparison"
        )

        comparison = pd.read_csv(
            COMPARISON_PATH
        )

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            comparison.set_index(
                "model"
            )[
                [
                    "roc_auc",
                    "pr_auc",
                    "f1",
                ]
            ]
        )

    # ---------------------------------------------------
    # Figures
    # ---------------------------------------------------

    st.subheader(
        "Evaluation Visualizations"
    )

    figure_files = [
        (
            "ROC Curve",
            "roc_curve.png",
        ),
        (
            "Precision-Recall Curve",
            "precision_recall_curve.png",
        ),
        (
            "Confusion Matrix",
            "confusion_matrix.png",
        ),
    ]

    for title, filename in figure_files:

        path = (
            FIGURES_DIR
            / filename
        )

        if path.exists():

            st.write(
                f"### {title}"
            )

            st.image(
                str(path)
            )


# =======================================================
# Disclaimer
# =======================================================

st.divider()

st.markdown(
    """
    <div class="disclaimer">
    <strong>Scientific disclaimer:</strong>
    This application provides computational predictions of EGFR
    bioactivity based on a machine-learning model trained on curated
    experimental bioactivity data. Predictions are not experimental
    measurements and do not establish clinical efficacy, safety,
    therapeutic suitability, or regulatory approval.
    </div>
    """,
    unsafe_allow_html=True,
)