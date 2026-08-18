from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.ligand import generate_ligand_features
from src.features.protein import ESMProteinEncoder
from src.features.protine_tokenizer import ProtineTokenizer

# Configuration
INPUT_FILE = Path(
    "data/processed/processed_bindingdb.csv"
)

FEATURE_DIR = Path(
    "data/processed/features"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Ligand configuration
MORGAN_RADIUS = 2
FINGERPRINT_SIZE = 1024


# Protein configuration
MAX_PROTEIN_LENGTH = 500
ESM_REPRESENTATION_LAYER = 6

# Start small for MPS.
# Increase later if memory allows.
ESM_BATCH_SIZE = 16



# ESM feature generation
def generate_esm_features(
    sequences: list[str],
    encoder: ESMProteinEncoder,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Generate ESM-2 embeddings in mini-batches.

    Only unique proteins should be supplied to this function.

    Returns
    -------
    np.ndarray
        Shape:
        (number_of_unique_proteins, 320)
    """

    if len(sequences) == 0:
        raise ValueError(
            "No protein sequences were provided."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    all_embeddings = []

    for start in tqdm(
        range(
            0,
            len(sequences),
            batch_size,
        ),
        desc="Generating ESM-2 embeddings",
    ):

        end = min(
            start + batch_size,
            len(sequences),
        )

        batch_sequences = (
            sequences[start:end]
        )

        batch_embeddings = (
            encoder.encode_batch(
                batch_sequences
            )
        )

        all_embeddings.append(
            batch_embeddings
        )

    return np.concatenate(
        all_embeddings,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )



# Validation
def validate_features(
    *,
    number_of_samples: int,
    unique_ligands: list[str],
    unique_proteins: list[str],
    ligand_features: np.ndarray,
    protein_esm_features: np.ndarray,
    protein_token_features: np.ndarray,
    ligand_index: np.ndarray,
    protein_index: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate generated feature arrays before saving them.
    """

    
    # Row mappings
    if len(ligand_index) != number_of_samples:
        raise RuntimeError(
            "Ligand index length does not match dataset."
        )

    if len(protein_index) != number_of_samples:
        raise RuntimeError(
            "Protein index length does not match dataset."
        )

    if len(y) != number_of_samples:
        raise RuntimeError(
            "Target length does not match dataset."
        )

    
    # Unique feature counts
    if ligand_features.shape[0] != len(
        unique_ligands
    ):
        raise RuntimeError(
            "Ligand feature count mismatch."
        )

    if protein_esm_features.shape[0] != len(
        unique_proteins
    ):
        raise RuntimeError(
            "ESM feature count mismatch."
        )

    if protein_token_features.shape[0] != len(
        unique_proteins
    ):
        raise RuntimeError(
            "Protein token count mismatch."
        )

    
    # Feature dimensions
    if ligand_features.shape[1] != FINGERPRINT_SIZE:
        raise RuntimeError(
            "Unexpected ligand feature dimension: "
            f"{ligand_features.shape}"
        )

    if protein_esm_features.shape[1] != 320:
        raise RuntimeError(
            "Unexpected ESM-2 feature dimension: "
            f"{protein_esm_features.shape}"
        )

    if protein_token_features.shape[1] != MAX_PROTEIN_LENGTH:
        raise RuntimeError(
            "Unexpected protein token dimension: "
            f"{protein_token_features.shape}"
        )

    
    # Numerical validation
    if not np.isfinite(
        ligand_features
    ).all():
        raise RuntimeError(
            "Ligand features contain NaN or infinity."
        )

    if not np.isfinite(
        protein_esm_features
    ).all():
        raise RuntimeError(
            "ESM features contain NaN or infinity."
        )

    if not np.isfinite(y).all():
        raise RuntimeError(
            "Target contains NaN or infinity."
        )

    
    # Index validation
    if ligand_index.min() < 0:
        raise RuntimeError(
            "Invalid ligand index detected."
        )

    if ligand_index.max() >= len(
        unique_ligands
    ):
        raise RuntimeError(
            "Ligand index exceeds feature table."
        )

    if protein_index.min() < 0:
        raise RuntimeError(
            "Invalid protein index detected."
        )

    if protein_index.max() >= len(
        unique_proteins
    ):
        raise RuntimeError(
            "Protein index exceeds feature table."
        )



# Main
def main() -> None:

   
    # 1. Load processed dataset
    print(
        "\nLoading processed BindingDB dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "canonical_smiles",
            "sequence",
            "pKi",
        ],
    )

    if df.empty:
        raise RuntimeError(
            "Processed dataset is empty."
        )

    print(
        f"DTI samples: {len(df):,}"
    )

   
    # 2. Factorize ligands
    #
    # This creates:
    #
    # ligand_index:
    #     one ligand ID for every DTI row
    #
    # unique_ligands:
    #     each canonical SMILES occurs only once
    #
    # This prevents duplicate Morgan feature generation.
   

    ligand_index, unique_ligands_index = (
        pd.factorize(
            df["canonical_smiles"],
            sort=False,
        )
    )

    unique_ligands = (
        unique_ligands_index
        .astype(str)
        .tolist()
    )

    ligand_index = (
        ligand_index.astype(
            np.int32
        )
    )

    print(
        f"Unique ligands: "
        f"{len(unique_ligands):,}"
    )

   
    # 3. Factorize proteins
    protein_index, unique_proteins_index = (
        pd.factorize(
            df["sequence"],
            sort=False,
        )
    )

    unique_proteins = (
        unique_proteins_index
        .astype(str)
        .tolist()
    )

    protein_index = (
        protein_index.astype(
            np.int32
        )
    )

    print(
        f"Unique proteins: "
        f"{len(unique_proteins):,}"
    )

   
    # 4. Target
    y = (
        df["pKi"]
        .to_numpy(
            dtype=np.float32
        )
    )

    # Dataset dataframe is no longer needed.
    del df

   
    # 5. Generate ligand features
    print(
        "\nGenerating Morgan fingerprints..."
    )

    ligand_features = (
        generate_ligand_features(
            unique_ligands,
            radius=MORGAN_RADIUS,
            fp_size=FINGERPRINT_SIZE,
        )
    )

    # Your ligand function currently returns float32.
    ligand_features = (
        ligand_features.astype(
            np.float32,
            copy=False,
        )
    )

    print(
        "Ligand features:",
        ligand_features.shape,
    )

   
    # 6. Generate ESM-2 protein features
    print(
        "\nInitialising ESM-2..."
    )

    protein_encoder = (
        ESMProteinEncoder(
            max_length=MAX_PROTEIN_LENGTH,
            representation_layer=(
                ESM_REPRESENTATION_LAYER
            ),
        )
    )

    print(
        f"ESM device: "
        f"{protein_encoder.device}"
    )

    protein_esm_features = (
        generate_esm_features(
            sequences=unique_proteins,
            encoder=protein_encoder,
            batch_size=ESM_BATCH_SIZE,
        )
    )

    print(
        "ESM-2 features:",
        protein_esm_features.shape,
    )

   
    # 7. Free ESM model memory
    del protein_encoder

   
    # 8. Generate protein tokens
    print(
        "\nGenerating protein token representations..."
    )

    protein_tokenizer = (
        ProtineTokenizer(
            max_length=MAX_PROTEIN_LENGTH
        )
    )

    protein_token_features = (
        protein_tokenizer.transform(
            unique_proteins
        )
    )

    # Values are only:
    #
    # 0    = PAD
    # 1-20 = amino acids
    # 21   = UNKNOWN
    #
    # uint8 therefore saves considerable disk space.
    protein_token_features = (
        protein_token_features.astype(
            np.uint8,
            copy=False,
        )
    )

    print(
        "Protein tokens:",
        protein_token_features.shape,
    )

   
    # 9. Validate everything
    print(
        "\nValidating generated features..."
    )

    validate_features(
        number_of_samples=len(y),
        unique_ligands=unique_ligands,
        unique_proteins=unique_proteins,
        ligand_features=ligand_features,
        protein_esm_features=(
            protein_esm_features
        ),
        protein_token_features=(
            protein_token_features
        ),
        ligand_index=ligand_index,
        protein_index=protein_index,
        y=y,
    )

    print(
        "Feature validation passed."
    )

   
    # 10. Save NumPy arrays
    print(
        "\nSaving features..."
    )

    np.save(
        FEATURE_DIR
        / "ligand_features.npy",
        ligand_features,
    )

    np.save(
        FEATURE_DIR
        / "protein_esm_features.npy",
        protein_esm_features,
    )

    np.save(
        FEATURE_DIR
        / "protein_token_features.npy",
        protein_token_features,
    )

    np.save(
        FEATURE_DIR
        / "ligand_index.npy",
        ligand_index,
    )

    np.save(
        FEATURE_DIR
        / "protein_index.npy",
        protein_index,
    )

    np.save(
        FEATURE_DIR
        / "y.npy",
        y,
    )

   
    # 11. Save lookup tables
    ligand_lookup = pd.DataFrame(
        {
            "ligand_id": np.arange(
                len(unique_ligands),
                dtype=np.int32,
            ),
            "canonical_smiles": (
                unique_ligands
            ),
        }
    )

    ligand_lookup.to_csv(
        FEATURE_DIR
        / "ligand_lookup.csv",
        index=False,
    )

    protein_lookup = pd.DataFrame(
        {
            "protein_id": np.arange(
                len(unique_proteins),
                dtype=np.int32,
            ),
            "sequence": unique_proteins,
        }
    )

    protein_lookup.to_csv(
        FEATURE_DIR
        / "protein_lookup.csv",
        index=False,
    )

   
    # 12. Save metadata
    metadata = {

        "dataset": {
            "number_of_samples": int(
                len(y)
            ),
            "number_of_unique_ligands": int(
                len(unique_ligands)
            ),
            "number_of_unique_proteins": int(
                len(unique_proteins)
            ),
        },

        "ligand": {
            "representation": (
                "Morgan fingerprint"
            ),
            "radius": MORGAN_RADIUS,
            "fingerprint_size": (
                FINGERPRINT_SIZE
            ),
        },

        "protein_esm": {
            "model": (
                "esm2_t6_8M_UR50D"
            ),
            "representation_layer": (
                ESM_REPRESENTATION_LAYER
            ),
            "embedding_dimension": 320,
            "max_sequence_length": (
                MAX_PROTEIN_LENGTH
            ),
            "pooling": (
                "mean residue pooling"
            ),
        },

        "protein_tokens": {
            "max_sequence_length": (
                MAX_PROTEIN_LENGTH
            ),
            "padding_index": (
                protein_tokenizer.PAD_INDEX
            ),
            "unknown_index": (
                protein_tokenizer.UNKNOWN_INDEX
            ),
            "vocabulary_size": (
                protein_tokenizer.vocab_size
            ),
        },

        "target": {
            "name": "pKi",
            "dtype": "float32",
        },
    }

    with open(
        FEATURE_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

   
    # 13. Summary
    print(
        "FEATURE GENERATION COMPLETED"
    )

    print(
        f"Samples: "
        f"{len(y):,}"
    )

    print(
        f"Unique ligands: "
        f"{len(unique_ligands):,}"
    )

    print(
        f"Unique proteins: "
        f"{len(unique_proteins):,}"
    )

    print(
        f"Ligand features: "
        f"{ligand_features.shape}"
    )

    print(
        f"ESM features: "
        f"{protein_esm_features.shape}"
    )

    print(
        f"Protein tokens: "
        f"{protein_token_features.shape}"
    )

    print(
        f"Targets: "
        f"{y.shape}"
    )

    print(
        f"\nFeatures saved to: "
        f"{FEATURE_DIR}"
    )



# Entry point
if __name__ == "__main__":
    main()