from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

DEFAULT_RADIUS = 2
DEFAULT_FP_SIZE = 1024

def create_morgan_generator(
        radius: int = DEFAULT_RADIUS,
        fp_size: int = DEFAULT_FP_SIZE
):
    
    if radius <=0 :
        raise ValueError(
            "Morgan fingerprint radius must be greater than zero."
        )
    
    if fp_size <= 0:
        raise ValueError(
            "Fingerprint size must be greater than zero."
        )
    
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size
    )


def smiles_to_morgan(
        smiles: str,
        generator = None
):
    if not isinstance(smiles, str):
        raise TypeError(
            "SMILES cannnot be string."
        )
    
    smiles = smiles.strip()

    if not smiles:
        raise ValueError(
            "SMILES cannot be empty."
        )
    
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:

        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )
    
    if generator is None:
        generator = create_morgan_generator()

    fingureprint = generator.GetFingerprint(molecule)

    return np.asarray(
        fingureprint,
        dtype=np.float32
    )

def generate_ligand_features(
        smiles_list,
        radius: int = DEFAULT_RADIUS,
        fp_size: int = DEFAULT_FP_SIZE
):
    generator = create_morgan_generator(
        radius=radius,
        fp_size=fp_size,
    )

    features = [
        smiles_to_morgan(
            smiles,
            generator=generator
        )

        for smiles in smiles_list
    ]

    return np.stack(
        features,
        axis=0
    )
