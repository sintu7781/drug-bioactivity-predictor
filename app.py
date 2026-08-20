from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import BioactivityPredictor

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "bioactivity_model.joblib"
)

st.set_page_config(
    page_title="Drug Bioactivity Predictor",
    page_icon="🧬",
    layout="wide",
)

@st.cache_resource
def load_predictor():
    
    return BioactivityPredictor(
        MODEL_PATH
    )

predictor = load_predictor()


st.title(
    "🧬 Drug Bioactivity Predictor"
)

st.caption(
    "Machine-learning prediction of EGFR "
    "bioactivity from molecular SMILES."
)


tab_single, tab_batch, tab_about = st.tabs(
    [
        "Single Compound",
        "Batch Prediction",
        "About Model",
    ]
)


with tab_single:
    
    st.subheader(
        "Single Compound Prediction"
    )
    
    smiles = st.text_input(
        "Enter SMILES",
        placeholder="CCO"
    )
    
    if st.button(
        "Predict Bioactivity",
        type="primary",
    ):
        
        if not smiles.strip():
            
            st.warning(
                "Please enter a SMILES string."
            )
        
        else:
            
            try:
                
                result = predictor.predict(
                    smiles
                )
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    
                    st.metric(
                        "Prediction",
                        result["prediction"],
                    )
                
                with col2:
                    
                    st.metric(
                        "Active Probability",
                        f"{result['active_probability'] * 100:.2f}%",
                    )
                
                st.subheader(
                    "Molecule Properties"
                )
                
                properties = pd.DataFrame(
                    [
                        result["descriptors"]
                    ]
                )
                
                st.dataframe(
                    properties,
                    use_container_width=True,
                    hide_index=True,
                )
                
            except ValueError as exc:
                
                st.error(
                    str(exc)
                )
                
with tab_batch:
    
    st.subheader(
        "Batch Prediction"
    )
    
    uploaded_file = st.file_uploader(
        "Upload CSV",
        "`smiles`."
    )
    
    if uploaded_file:
        
        df = pd.read_csv(
            uploaded_file
        )
        
        if "smiles" not in df.columns:
            
            st.error(
                "CSV must contain a 'smiles' column."
            )
            
        else:
            
            predictions = []
            
            probabilities = []
            
            for smiles in df["smiles"]:
                
                try:
                    
                    result = predictor.predict(
                        str(smiles)
                    )
                    
                    predictions.append(
                        result["prediction"]
                    )
                    
                    probabilities.append(
                        result["active_probability"]
                    )
                    
                except ValueError:
                    
                    predictions.append(
                        "INVALID_SMILES"
                    )
                    
                    probabilities.append(
                        None
                    )
            
            df["prediction"] = predictions
            
            df["active_probability"] = probabilities
            
            st.dataframe(
                df,
                use_container_width=True,
            )
            
            csv = df.to_csv(
                index=False
            ).encode("utf-8")
            
            st.download_button(
                "Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
            
with tab_about:
    
    st.subheader(
        "Model Information"
    )
    
    st.markdown(
    """
    **Target:** EGFR
    
    **ChEMBL Target:** CHEMBL203
    
    **Activity:** IC50
    
    **Features:**
    - Molecular descriptors
    - Morgan fingerprints
    
    **Models evaluated:**
    - Logistic Regression
    - Random Forest
    - XGBoost
    
    **Metrics:**
    - Accuracy
    - Precission
    - Recall
    - F1
    - ROC-AUC
    """
    )