from __future__ import annotations

import numpy as np
import pandas as pd

from tqdm import tqdm

from src.features.ligand import LigandFeaturizer
from src.features.protein import ProteinEncoder
from src.features.protine_tokenizer import ProteinTokenizer

class FeatureTransformer:
    """
    Transform a cleaned DTI dataframe into model-ready features.

    Outputs
    -------
    X_ligand:
        Morgan fingerprints
        shape: (N, 1024)

    X_protein_esm:
        ESM-2 protein embeddings
        shape: (N, 320)

    X_protein_tokens:
        Integer-encoded protein sequences
        shape: (N, max_length)

    y:
        Binding-affinity target
        shape: (N,)
    """

    def __init__(
        self,
        fingerprint_size: int = 1024,
        fingerprint_radius: int = 2,
        max_protein_length: int = 500,
        esm_batch_size: int = 16,
    ) -> None:

        self.esm_batch_size = esm_batch_size

        self.ligand_encoder = LigandFeaturizer(
            fingerprint_size=fingerprint_size,
            radius=fingerprint_radius,
        )

        self.protein_encoder = ProteinEncoder(
            max_length=max_protein_length,
        )

        self.protein_tokenizer = ProteinTokenizer(
            max_length=max_protein_length,
        )

    
    # Ligand features
    def transform_ligands(
        self,
        smiles: pd.Series,
    ) -> np.ndarray:

        unique_smiles = (
            smiles
            .drop_duplicates()
            .tolist()
        )

        print(
            f"Generating fingerprints for "
            f"{len(unique_smiles):,} unique ligands..."
        )

        ligand_cache = {}

        for sm in tqdm(
            unique_smiles,
            desc="Morgan fingerprints",
        ):
            ligand_cache[sm] = (
                self.ligand_encoder.process_smiles(
                    sm
                )
            )

        features = [
            ligand_cache[sm]
            for sm in smiles
        ]

        return np.stack(
            features
        ).astype(
            np.float32
        )

    
    # ESM protein features
    def transform_protein_esm(
        self,
        sequences: pd.Series,
    ) -> np.ndarray:

        unique_sequences = (
            sequences
            .drop_duplicates()
            .tolist()
        )

        print(
            f"Generating ESM-2 embeddings for "
            f"{len(unique_sequences):,} unique proteins..."
        )

        unique_embeddings = (
            self.protein_encoder.transform(
                unique_sequences,
                batch_size=self.esm_batch_size,
            )
        )

        protein_cache = {
            sequence: embedding
            for sequence, embedding
            in zip(
                unique_sequences,
                unique_embeddings,
            )
        }

        features = [
            protein_cache[sequence]
            for sequence in sequences
        ]

        return np.stack(
            features
        ).astype(
            np.float32
        )

    
    # Protein tokens for DTIRegressor
    def transform_protein_tokens(
        self,
        sequences: pd.Series,
    ) -> np.ndarray:

        print(
            "Generating protein tokens "
            "for DTIRegressor..."
        )

        tokens = (
            self.protein_tokenizer.transform(
                sequences.tolist()
            )
        )

        return tokens.astype(
            np.int64
        )

    
    # Complete transformation
    def transform(
        self,
        df: pd.DataFrame,
        smiles_column: str = "canonical_smiles",
        protein_column: str = "protein_sequence",
        target_column: str = "pKi",
    ) -> dict[str, np.ndarray]:

        required_columns = [
            smiles_column,
            protein_column,
            target_column,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing required columns: "
                f"{missing_columns}"
            )

        print(
            f"Transforming {len(df):,} DTI samples..."
        )

        
        # Ligand representation
        X_ligand = (
            self.transform_ligands(
                df[smiles_column]
            )
        )

        
        # ESM representation
        X_protein_esm = (
            self.transform_protein_esm(
                df[protein_column]
            )
        )

        
        # Integer-token representation
        X_protein_tokens = (
            self.transform_protein_tokens(
                df[protein_column]
            )
        )

        
        # Target
        y = (
            df[target_column]
            .to_numpy(
                dtype=np.float32
            )
        )

        
        # Sanity checks
        n_samples = len(df)

        assert X_ligand.shape[0] == n_samples
        assert X_protein_esm.shape[0] == n_samples
        assert X_protein_tokens.shape[0] == n_samples
        assert y.shape[0] == n_samples

        return {
            "X_ligand": X_ligand,
            "X_protein_esm": X_protein_esm,
            "X_protein_tokens": X_protein_tokens,
            "y": y,
        }