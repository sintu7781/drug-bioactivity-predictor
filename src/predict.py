from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import (
    calculate_descriptors,
    smiles_to_mol,
    featurize_smiles,
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
        self.model_name = artifact["model_name"]
        
    def predict(
        self,
        smiles: str,
    ) -> dict:
        
        molecule = smiles_to_mol(smiles)
        
        features = featurize_smiles(
            smiles
        )
        
        X = features.reshape(
            1,
            -1,
        )
        
        prediction = int(
            self.model.predict(X)[0]
        )
        
        probability = float(
            self.model.predict_proba(X)[0, 1]
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
            "active_probability": probability,
            "inactive_probability": (
                1.0 - probability
            ),
            "model": self.model_name,
            "descriptors": descriptors,
        }