import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from features import featurize_smiles


def main() -> None:
    input_path = (
        PROCESSED_DATA_DIR /
        "egfr_ic50_curated.csv"
    )
    
    output_path = (
        PROCESSED_DATA_DIR /
        "egfr_features.npz"
    )
    
    df = pd.read_csv(input_path)
    
    features = []
    
    valid_rows = []
    
    for index, row in df.iterrows():
        
        try:
            vector = featurize_smiles(
                row["canonical_smiles"]
            )
            
            features.append(vector)
            valid_rows.append(index)
        
        except ValueError as exc:
            print(
                f"Skipping row {index}: {exc}"
            )
        
    df = df.loc[valid_rows].reset_index(drop=True)
    
    X = np.vstack(features)
    
    y = df["activity_label"].to_numpy(
        dtype=np.int64
    )
    
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
    )
    
    df.to_csv(
        PROCESSED_DATA_DIR /
        "egfr_model_metadata.csv",
        index=False,
    )
    
    print("Feature matrix:", X.shape)
    print("Target vector:", y.shape)
    

if __name__ == "__main__":
    main()
    