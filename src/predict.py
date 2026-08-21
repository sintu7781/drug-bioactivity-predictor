from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import (
    calculate_descriptors,
    featurize_smiles,
    smiles_to_mol,
)


class BioactivityPredictor:
    
    def __init__(
        self,
        model_path: Path,
    ) -> None:

        artifact = joblib.load(
            model_path
        )

        self.model = artifact["model"]
        self.model_name = artifact[
            "model_name"
        ]

    def predict(
        self,
        smiles: str,
    ) -> dict:

        molecule = smiles_to_mol(
            smiles
        )

        vector = featurize_smiles(
            smiles
        )

        X = vector.reshape(
            1,
            -1,
        )

        prediction = int(
            self.model.predict(X)[0]
        )

        probabilities = (
            self.model.predict_proba(X)[0]
        )
        
        active_probability = float(
            probabilities[1]
        )

        descriptors = (
            calculate_descriptors(
                molecule
            )
        )

        return {
            "prediction": (
                "ACTIVE" 
                if prediction == 1
                else "INACTIVE"
            ),
            "active_probability": (
                active_probability
            ),
            "inactive_probability": float(
                probabilities[0]
            ),
            "model": self.model_name,
            "descriptors": descriptors,
        }
