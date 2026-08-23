from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)

from .features import (
    calculate_descriptors,
    generate_morgan_fingerprint,
)


class BioactivityPredictor:
    """
    Production inference wrapper for the trained
    EGFR bioactivity classifier.

    The trained model expects:

        8 molecular descriptors
        +
        2048-bit Morgan fingerprint
        =
        2056 features
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
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

        missing = (
            required_keys
            - set(artifact.keys())
        )

        if missing:
            raise ValueError(
                "Model artifact is missing "
                f"required fields: {sorted(missing)}"
            )

        self.model = artifact["model"]

        self.model_name = (
            str(
                artifact["model_name"]
            )
        )

        self.feature_count = int(
            artifact["feature_count"]
        )

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

        # ------------------------------------------------
        # Validate the model/feature contract.
        # ------------------------------------------------

        expected_feature_count = (
            self.descriptor_count
            + self.morgan_bits
        )

        if self.feature_count != (
            expected_feature_count
        ):
            raise ValueError(
                "Model feature configuration "
                "is inconsistent. "
                f"Model expects "
                f"{self.feature_count} features, "
                f"but configuration produces "
                f"{expected_feature_count}."
            )

        if self.morgan_radius != 2:
            raise ValueError(
                "This predictor currently expects "
                "Morgan radius 2 to match the "
                "trained model."
            )

        if self.morgan_bits != 2048:
            raise ValueError(
                "This predictor currently expects "
                "2048 Morgan fingerprint bits "
                "to match the trained model."
            )

    # ====================================================
    # SMILES validation
    # ====================================================

    @staticmethod
    def validate_smiles(
        smiles: str,
    ) -> Chem.Mol:
        """
        Validate a SMILES string and return
        the corresponding RDKit molecule.
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
                f"Invalid SMILES string: {smiles}"
            )

        return molecule

    # ====================================================
    # Feature generation
    # ====================================================

    def _build_features(
        self,
        molecule: Chem.Mol,
    ) -> np.ndarray:
        """
        Build the exact feature vector expected
        by the trained model.
        """

        descriptors = (
            calculate_descriptors(
                molecule
            )
        )

        # Keep descriptor ordering deterministic.
        descriptor_names = [
            "MolWt",
            "LogP",
            "HBD",
            "HBA",
            "RotatableBonds",
            "TPSA",
            "RingCount",
            "HeavyAtomCount",
        ]

        descriptor_vector = np.asarray(
            [
                descriptors[name]
                for name in descriptor_names
            ],
            dtype=np.float32,
        )

        fingerprint_vector = (
            generate_morgan_fingerprint(
                molecule
            ).astype(
                np.float32
            )
        )

        features = np.concatenate(
            [
                descriptor_vector,
                fingerprint_vector,
            ]
        )

        # ------------------------------------------------
        # Hard safety check.
        # ------------------------------------------------

        if features.ndim != 1:
            raise RuntimeError(
                "Feature vector must be one-dimensional."
            )

        if len(features) != (
            self.feature_count
        ):
            raise RuntimeError(
                "Feature count mismatch. "
                f"Expected {self.feature_count}, "
                f"got {len(features)}."
            )

        # Model expects shape:
        #
        # (1, 2056)
        #
        return features.reshape(
            1,
            -1,
        )

    # ====================================================
    # Molecular properties
    # ====================================================

    @staticmethod
    def _get_properties(
        molecule: Chem.Mol,
    ) -> dict[str, float | int]:
        """
        Calculate human-readable molecular properties.
        """

        return {
            "MolWt": float(
                Descriptors.MolWt(
                    molecule
                )
            ),
            "LogP": float(
                Crippen.MolLogP(
                    molecule
                )
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

    # ====================================================
    # Prediction
    # ====================================================

    def predict(
        self,
        smiles: str,
    ) -> dict[str, Any]:
        """
        Predict EGFR bioactivity for a SMILES string.
        """

        # ------------------------------------------------
        # Validate molecule.
        # ------------------------------------------------

        molecule = (
            self.validate_smiles(
                smiles
            )
        )

        # ------------------------------------------------
        # Canonical SMILES.
        # ------------------------------------------------

        canonical_smiles = (
            Chem.MolToSmiles(
                molecule,
                canonical=True,
            )
        )

        # ------------------------------------------------
        # Generate model features.
        # ------------------------------------------------

        features = (
            self._build_features(
                molecule
            )
        )

        # ------------------------------------------------
        # Model prediction.
        # ------------------------------------------------

        prediction_value = int(
            self.model.predict(
                features
            )[0]
        )

        probabilities = (
            self.model.predict_proba(
                features
            )[0]
        )

        inactive_probability = float(
            probabilities[0]
        )

        active_probability = float(
            probabilities[1]
        )

        prediction = (
            "ACTIVE"
            if prediction_value == 1
            else "INACTIVE"
        )

        # ------------------------------------------------
        # Molecular properties.
        # ------------------------------------------------

        properties = (
            self._get_properties(
                molecule
            )
        )

        # ------------------------------------------------
        # Return structured result.
        # ------------------------------------------------

        return {
            "smiles": smiles.strip(),

            "canonical_smiles": (
                canonical_smiles
            ),

            "prediction": (
                prediction
            ),

            "prediction_label": (
                prediction_value
            ),

            "active_probability": (
                active_probability
            ),

            "inactive_probability": (
                inactive_probability
            ),

            "model": (
                self.model_name
            ),

            "target": (
                self.target
            ),

            "target_chembl_id": (
                self.target_chembl_id
            ),

            "activity_type": (
                self.activity_type
            ),

            "activity_unit": (
                self.activity_unit
            ),

            "active_threshold_nM": (
                self.active_threshold_nM
            ),

            "feature_count": (
                self.feature_count
            ),

            "descriptors": (
                properties
            ),
        }