import pandas as pd

from src.preprocessing import (
    canonicalize_smiles,
    curate_data,
)


def test_canonicalize_smiles():

    result = canonicalize_smiles(
        "C(C)O"
    )

    assert result == "CCO"


def test_invalid_smiles():

    result = canonicalize_smiles(
        "invalid"
    )

    assert result is None


def test_curation():

    df = pd.DataFrame(
        {
            "molecule_chembl_id": [
                "CHEMBL1",
                "CHEMBL2",
                "CHEMBL3",
            ],
            "standard_type": [
                "IC50",
                "IC50",
                "IC50",
            ],
            "standard_units": [
                "nM",
                "nM",
                "nM",
            ],
            "standard_value": [
                100,
                5000,
                250,
            ],
            "canonical_smiles": [
                "CCO",
                "CCN",
                "C(C)O",
            ],
        }
    )

    curated, quality = curate_data(
        df
    )

    assert len(curated) == 3

    assert set(
        curated["activity_label"]
    ) == {0, 1}

    assert (
        quality["final_compounds"]
        == 3
    )