import pandas as pd
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = [
    'Ligand SMILES',
    'BindingDB Target Chain Sequence 1',
    'Ki (nM)',
]

file = "data/raw/BindingDB_All.tsv"

def load_bindingdb(
        file_path: str,
        chunk_size: int = 100_000,
):
    """
    load the BindingDB dataset using only
    the columns required by the DTI pipeline.
    
    :param file_path: Description
    :type file_path: str
    """

    df = pd.read_csv(
        file_path,
        sep="\t",
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
        chunksize=chunk_size
    )

    return df


FEATURE_DIR = Path( "data/processed/features")

def load_training_data(
    feature_dir: Path = FEATURE_DIR,
) -> dict[str, np.ndarray]:

    print("\nLoading generated features...")

    
    # Unique ligand feature table
    # Shape:
    # (number_of_unique_ligands, 1024)
    ligand_features = np.load(
        feature_dir / "ligand_features.npy",
        mmap_mode="r",
    )

    
    # Unique protein ESM feature table
    # Shape:
    # (number_of_unique_proteins, 320)
    protein_esm_features = np.load(
        feature_dir / "protein_esm_features.npy",
        mmap_mode="r",
    )

    
    # Unique protein token table
    # Shape:
    # (number_of_unique_proteins, 500)
    protein_token_features = np.load(
        feature_dir / "protein_token_features.npy",
        mmap_mode="r",
    )

    
    # Mapping from each DTI sample -> unique ligand
    ligand_index = np.load(
        feature_dir / "ligand_index.npy"
    )

    
    # Mapping from each DTI sample -> unique protein
    protein_index = np.load(
        feature_dir / "protein_index.npy"
    )

    # Regression target: pKi
    y = np.load(
        feature_dir / "y.npy"
    ).astype(
        np.float32,
        copy=False,
    )

    # Validation
    number_of_samples = len(y)

    if len(ligand_index) != number_of_samples:
        raise RuntimeError(
            "ligand_index and y have different lengths."
        )

    if len(protein_index) != number_of_samples:
        raise RuntimeError(
            "protein_index and y have different lengths."
        )

    if ligand_features.shape[1] != 1024:
        raise RuntimeError(
            f"Unexpected ligand shape: "
            f"{ligand_features.shape}"
        )

    if protein_esm_features.shape[1] != 320:
        raise RuntimeError(
            f"Unexpected ESM shape: "
            f"{protein_esm_features.shape}"
        )

    if protein_token_features.shape[1] != 500:
        raise RuntimeError(
            f"Unexpected protein token shape: "
            f"{protein_token_features.shape}"
        )

    if not np.isfinite(y).all():
        raise RuntimeError(
            "Target contains NaN or infinity."
        )

    print(
        f"Samples: {number_of_samples:,}"
    )

    print(
        f"Ligand feature table: "
        f"{ligand_features.shape}"
    )

    print(
        f"ESM feature table: "
        f"{protein_esm_features.shape}"
    )

    print(
        f"Protein token table: "
        f"{protein_token_features.shape}"
    )

    print(
        f"Target: {y.shape}"
    )

    return {
        "ligand_features": ligand_features,
        "protein_esm_features": protein_esm_features,
        "protein_token_features": protein_token_features,
        "ligand_index": ligand_index,
        "protein_index": protein_index,
        "y": y,
    }