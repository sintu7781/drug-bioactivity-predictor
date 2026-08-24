from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .features import (
    calculate_descriptors,
    calculate_morgan_fingerprint,
)

# ============================================================
# Hugging Face configuration
# ============================================================

HF_MODEL_REPO = "prime7781/drug-bioactivity-predictor"
HF_MODEL_FILENAME = "bioactivity_model.joblib"


# ============================================================
# Model resolution
# ============================================================

def resolve_model_path(
    model_path: str | Path | None = None,
) -> Path:
    """
    Resolve the trained model path.

    Priority:

    1. Explicit local model path.
    2. Hugging Face model download.

    This allows the same code to work locally and on
    Streamlit Community Cloud.
    """

    # --------------------------------------------------------
    # Local model explicitly supplied
    # --------------------------------------------------------

    if model_path is not None:

        local_path = Path(model_path)

        if not local_path.exists():

            raise FileNotFoundError(
                f"Model not found: {local_path}"
            )

        if not local_path.is_file():

            raise ValueError(
                f"Model path is not a file: {local_path}"
            )

        return local_path

    # --------------------------------------------------------
    # Download from Hugging Face
    # --------------------------------------------------------

    try:

        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILENAME,
            repo_type="model",
        )

        return Path(downloaded_path)

    except Exception as exc:

        raise RuntimeError(
            "Unable to load the trained model from Hugging Face. "
            f"Repository: {HF_MODEL_REPO}. "
            f"Filename: {HF_MODEL_FILENAME}"
        ) from exc


# ============================================================
# Predictor
# ============================================================

class BioactivityPredictor:
    """
    Production inference wrapper for the EGFR bioactivity
    classification model.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
    ) -> None:

        self.model_path = resolve_model_path(
            model_path
        )

        artifact = joblib.load(
            self.model_path
        )

        if not isinstance(
            artifact,
            dict,
        ):

            raise TypeError(
                "Invalid model artifact. "
                "Expected a dictionary."
            )

        required_keys = {
            "model",
            "model_name",
            "feature_count",
        }

        missing_keys = (
            required_keys
            - set(artifact.keys())
        )

        if missing_keys:

            raise ValueError(
                "Model artifact is missing "
                f"required fields: {missing_keys}"
            )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = artifact["model"]

        self.model_name = str(
            artifact["model_name"]
        )

        self.feature_count = int(
            artifact["feature_count"]
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.target = str(
            artifact.get(
                "target",
                "EGFR",
            )
        )

        self.target_chembl_id = str(
            artifact.get(
                "target_chembl_id",
                "CHEMBL203",
            )
        )

        self.activity_type = str(
            artifact.get(
                "activity_type",
                "IC50",
            )
        )

        self.activity_unit = str(
            artifact.get(
                "activity_unit",
                "nM",
            )
        )

        self.active_threshold_nM = float(
            artifact.get(
                "active_threshold_nM",
                1000.0,
            )
        )

        self.morgan_radius = int(
            artifact.get(
                "morgan_radius",
                2,
            )
        )

        self.morgan_bits = int(
            artifact.get(
                "morgan_bits",
                2048,
            )
        )

        self.descriptor_count = int(
            artifact.get(
                "descriptor_count",
                8,
            )
        )

    # ========================================================
    # SMILES validation
    # ========================================================

    @staticmethod
    def validate_smiles(
        smiles: str,
    ) -> Chem.Mol:
        """
        Validate a SMILES string and return an RDKit molecule.
        """

        if not isinstance(
            smiles,
            str,
        ):

            raise TypeError(
                "SMILES must be a string."
            )

        smiles = smiles.strip()

        if not smiles:

            raise ValueError(
                "SMILES cannot be empty."
            )

        molecule = Chem.MolFromSmiles(
            smiles
        )

        if molecule is None:

            raise ValueError(
                "Invalid SMILES string."
            )

        return molecule

    # ========================================================
    # Feature generation
    # ========================================================

    def _build_features(
        self,
        molecule: Chem.Mol,
    ) -> np.ndarray:
        """
        Generate the exact feature representation expected
        by the trained model.
        """

        descriptors = calculate_descriptors(
            molecule
        )

        fingerprint = calculate_morgan_fingerprint(
            molecule
        )

        descriptor_vector = np.asarray(
            [
                descriptors[name]
                for name in (
                    (
                        "MolWt",
                        "LogP",
                        "HBD",
                        "HBA",
                        "RotatableBonds",
                        "TPSA",
                        "RingCount",
                        "HeavyAtomCount",
                    )
                )
            ],
            dtype=np.float64,
        )

        fingerprint_vector = np.asarray(
            fingerprint,
            dtype=np.float64,
        )

        features = np.concatenate(
            [
                descriptor_vector,
                fingerprint_vector,
            ]
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if len(features) != self.feature_count:

            raise RuntimeError(
                "Feature count mismatch. "
                f"Expected {self.feature_count}, "
                f"got {len(features)}."
            )

        return features.reshape(
            1,
            -1,
        )

    # ========================================================
    # Molecular properties
    # ========================================================

    @staticmethod
    def _get_properties(
        molecule: Chem.Mol,
    ) -> dict[str, float | int]:

        return {
            "MolWt": float(
                Descriptors.MolWt(
                    molecule
                )
            ),
            "LogP": float(
                Crippen.MolLogP(
                    molecule
                ),
            ),
            "HBD": int(
                Lipinski.NumHDonors(
                    molecule
                )
            ),
            "HBA": int(
                Lipinski.NumHAcceptors(
                    molecule
                )
            ),
            "RotatableBonds": int(
                Lipinski.NumRotatableBonds(
                    molecule
                )
            ),
            "TPSA": float(
                rdMolDescriptors.CalcTPSA(
                    molecule
                )
            ),
            "RingCount": int(
                Lipinski.RingCount(
                    molecule
                )
            ),
            "HeavyAtomCount": int(
                molecule.GetNumHeavyAtoms()
            ),
        }

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        smiles: str,
    ) -> dict[str, Any]:
        """
        Predict bioactivity for a single compound.
        """

        molecule = self.validate_smiles(
            smiles
        )

        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
        )

        features = self._build_features(
            molecule
        )

        prediction_value = int(
            self.model.predict(
                features
            )[0]
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        if not hasattr(
            self.model,
            "predict_proba",
        ):

            raise RuntimeError(
                "The trained model does not "
                "support probability prediction."
            )

        probabilities = self.model.predict_proba(
            features
        )[0]

        classes = list(
            self.model.classes_
        )

        try:

            inactive_index = classes.index(0)
            active_index = classes.index(1)

        except ValueError as exc:

            raise RuntimeError(
                "Model classes must contain "
                "both 0 and 1."
            ) from exc

        inactive_probability = float(
            probabilities[inactive_index]
        )

        active_probability = float(
            probabilities[active_index]
        )

        prediction = (
            "ACTIVE"
            if prediction_value == 1
            else "INACTIVE"
        )

        # ----------------------------------------------------
        # Molecular descriptors
        # ----------------------------------------------------

        properties = self._get_properties(
            molecule
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "smiles": smiles.strip(),
            "canonical_smiles": canonical_smiles,
            "prediction": prediction,
            "prediction_label": prediction_value,
            "active_probability": active_probability,
            "inactive_probability": inactive_probability,
            "model": self.model_name,
            "target": self.target,
            "target_chembl_id": self.target_chembl_id,
            "activity_type": self.activity_type,
            "activity_unit": self.activity_unit,
            "active_threshold_nM": self.active_threshold_nM,
            "descriptors": properties,
        }