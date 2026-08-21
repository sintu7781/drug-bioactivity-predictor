from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit.Chem import Draw
from rdkit import Chem

from src.predict import (
    BioactivityPredictor,
)

PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models" 
    / "bioactivity_model.joblib"
)

st.set_page_config(
    page_title=(
        "Drug Bioactivity Predictor"
    ),
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    
    if not MODEL_PATH.exists():
        
        raise FileNotFoundError(
            "Model file not found. "
            "Run the training pipeline first."
        )

    return BioactivityPredictor(
        MODEL_PATH
    )
    

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
        size=(450, 350),
    )


st.title(
    "🧬 Drug Bioactivity Predictor"
)

st.caption(
    "Machine-learning prediction of "
    "EGFR bioactivity from molecular SMILES."
)

try:
    
    predictor = load_predictor()
    
except Exception as exc:
    
    st.error(
        str(exc)
    )
    
    st.stop()


tab_single, tab_batch, tab_model = (
    st.tabs(
        [
            "Single Compound",
            "Batch Prediction",
            "Model Performance",
        ]
    )
)


# ---------------------------------------------------------
# Single prediction
# ---------------------------------------------------------

with tab_single:
    st.subheader(
        "Single Compound Prediction"
    )

    smiles = st.text_input(
        "Enter SMILES",
        placeholder=(
            "Example: CCO"
        ),
    )

    if st.button(
        "Predict Bioactivity",
        type="primary",
        use_container_width=True,
    ):
        
        if not smiles.strip():
            
            st.warning(
                "Enter a SMILES string."
            )

        else:
            
            try:
                
                result = (
                    predictor.predict(
                        smiles
                    )
                )

                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:
                    
                    st.metric(
                        "Prediction",
                        result[
                            "prediction"
                        ],
                    )

                with col2:
                    
                    st.metric(
                        "Active Probability",
                        (
                        f"{result['active_probability'] * 100:.2f}%",
                        ),
                    )
                    
                with col3:
                    
                    st.metric(
                        "Model",
                        result["model"],
                    )
                    
                st.divider()

                
                left, right = (
                    st.columns(2)
                )
                
                with left:
                    
                    st.subheader(
                        "Molecule Structure"
                    )
                    
                    image = (
                        render_molecule(
                            smiles
                        )
                    )
                    
                    if image:
                        
                        st.image(
                            image
                        )
                    
                with right:
                    
                    st.subheader(
                        "Molecule Properties"
                    )
                    
                    properties = pd.DataFrame(
                        [
                            result[
                                "descriptors"
                            ]
                        ]
                    )
                    
                    st.dataframe(
                        properties,
                        use_container_width=True,
                        hide_index=True,
                    )
                
                st.info(
                    "This is a computational "
                    "model estimate. It does not "
                    "establish clinical efficacy, "
                    "safety, or therapeutic suitability."
                )

            except ValueError as exc:
                
                st.error(
                    str(exc)
                )
                

# ---------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------

with tab_batch:
    
    st.subheader(
        "Batch Prediction"
    )
    
    st.write(
        "Upload a CSV containing a "
        "`smiles` column."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV", 
        type=["csv"],
    )

    if uploaded_file:
        
        df = pd.read_csv(
            uploaded_file
        )

        if "smiles" not in df.columns:
            
            st.error(
                "The CSV must contain "
                "a 'smiles' column."
            )

        else:
            
            results = []

            progress = st.progress(
                0
            )
            
            total = len(df)

            for index, smiles in enumerate(
                df["smiles"]
            ):
                
                try:
                    
                    result = (
                        predictor.predict(
                            str(smiles)
                        )
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
                            "molecular_weight": (
                                result[
                                    "descriptors"
                                ]["MolWt"]
                            ),
                            "logP": (
                                result[
                                    "descriptors"
                                ]["logP"]
                            ),
                        }
                    )

                except ValueError:
                    
                    results.append(
                        {
                            "prediction": (
                                "INVALID_SMILES"
                            ),
                            "active_probability": (
                                None
                            ),
                            "molecular_weight": (
                                None
                            ),
                            "logP": (
                                None
                            ),
                        }
                    )
                
                progress.progress(
                    (index + 1) / total
                )

            result_df = pd.concat(
                [
                    df.reset_index(
                        drop=True
                    ),
                    pd.DataFrame(
                        results
                    ),
                ],
                axis=1,
            )

            st.dataframe(
                result_df,
                use_container_width=True,
            )

            csv = (
                result_df
                .to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                "Download Predictions",
                data=csv,
                file_name=(
                    "predictions.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------

with tab_model:
    
    st.subheader(
        "Model Information"
    )

    comparison_path = (
        PROJECT_ROOT
        / "reports"
        / "model_comparison.csv"
    )
    
    if comparison_path.exists():
        
        comparison = pd.read_csv(
            comparison_path
        )
        
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )
        
        st.subheader(
            "Evaluation Curves"
        )
        
        figures = [
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
                "confusion_matrix.png"
            ),
        ]
        
        for title, filename in figures:
            
            figure_path = (
                PROJECT_ROOT
                / "reports"
                / "figures"
                / filename
            )
            
            if figure_path.exists():
                
                st.write(title)
                
                st.image(
                    str(figure_path)
                )
    
    else:
        
        st.warning(
            "Model evaluation results "
            "are not available yet."
        )
                
