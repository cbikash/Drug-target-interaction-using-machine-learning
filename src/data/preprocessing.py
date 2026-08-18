from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from rdkit import Chem




# Constants


# 20 standard amino acids + X for unknown residues.
#
# With padding index 0, this is compatible with a vocabulary
# size of 22:
#
#   0      -> padding
#   1-20   -> standard amino acids
#   21     -> X / unknown
#

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWYX"
)

# Reject censored affinity measurements because their exact
# experimental value is unknown.

CENSORED_PATTERN = re.compile(
    r"[<>≤≥]"
)

# Accept normal decimal and scientific notation.
NUMERIC_PATTERN = re.compile(
    r"""
    ^
    [+-]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+-]?\d+)?
    $
    """,
    re.VERBOSE,
)

@dataclass
class PreprocessingReport:
    """
    Summary of samples removed during preprocessing.
    """

    initial_rows: int = 0

    missing_required: int = 0

    invalid_or_censored_ki: int = 0

    non_positive_ki: int = 0

    invalid_smiles: int = 0

    invalid_protein: int = 0

    duplicates_removed: int = 0

    final_rows: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
    

def parse_ki_nm(
        value,
) -> float:
    """
    Parse a BindingDB Ki value expressed in nM.

    Rules
    -----
    - Missing values are rejected.
    - Censored values containing <, >, <=, >=, ≤, ≥
      are rejected.
    - Approximation symbols such as ~ and ≈ are removed.
    - Commas are removed.
    - Only a single numeric value is accepted.
    - Non-numeric values return NaN.

    Examples
    --------
    "100"       -> 100.0
    "~100"      -> 100.0
    "≈ 100"     -> 100.0
    "1,000"     -> 1000.0
    "<100"      -> NaN
    ">500"      -> NaN
    "10-20"     -> NaN
    """

    if pd.isna(value):
        return np.nan
    
    # Already numeric
    if isinstance(value, (int,float,np.integer, np.floating)):
        value = float(value)

        return (value if np.isfinite(value) else np.nan)
    
    value = str(value).strip()

    if not value:
        return np.nan
    
    # Remove censored observations
    if CENSORED_PATTERN.search(value):
        return np.nan
    
    # Remove approximation markers
    value = (
        value
        .replace("~", "")
        .replace("≈", "")
        .replace("∼", "")
        .replace(",", "")
        .strip()
    )

    # Require one valid numeric value
    if not NUMERIC_PATTERN.fullmatch(value):
        return np.nan
    
    try:
        value = float(value)
    except ValueError:
        return np.nan
    
    if not np.isfinite(value):
        return np.nan
    
    return value



def ki_nm_to_pki(
    ki_num
) -> np.ndarray:
    """
    Convert Ki measured in nanomolar (nM) to pKi.

    Formula
    -------
    pKi = 9 - log10(Ki[nM])
    """

    ki_num = np.asarray(
        ki_num,
        dtype=np.float64
    )

    # Check for NaN or infinity
    if np.any(~np.isfinite(ki_num)):
        raise ValueError(
            "Ki contains non-finite values."
        )

    # Ki must be positive
    if np.any(ki_num <= 0):
        raise ValueError(
            "Ki values must be greater than zero."
        )

    return 9.0 - np.log10(ki_num)


def canonicalize_smile(smiles):
    """
    Validate and canonicalise a SMILES string using RDKit.

    Invalid SMILES return None.
    """

    if pd.isna(smiles):
        return None
    
    smiles = str(smiles).strip()
    
    if not smiles:
        return None 
    
    try: 
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            return None
        
        canonical_smiles = (Chem.MolToSmiles(molecule, canonical=True))

        if not canonical_smiles:
            return None
        
        return canonical_smiles
    
    except Exception:
        return None
    

# Protein preprocessing

def normalize_protine_sequence(sequence) -> str | None:
    """
    Normalize a protine sequence.

    - Convert to upercase.
    - Removes withespace/newlines.
    - Reject emty sequence.
    """

    if pd.isna(sequence):
        return None
    
    sequence = str(sequence).upper()

    sequence = re.sub(
        r"\s+",
        "",
        sequence
    )

    if not sequence:
        return None

    return sequence


def is_valid_protine_sequence(
        sequence,
)-> bool:
    """
    Docstring for is_valid_protine_sequence
    
    :param sequence: Description
    :return: Description
    :rtype: bool
    """

    if not isinstance(sequence,str):
        return False
    
    if not sequence:
        return False 
    
    return all(amino_acid in VALID_AMINO_ACIDS for amino_acid in sequence)


def preprocessing_bindingdb(
        dataframe: pd.DataFrame,
        *,
        smiles_column: str,
        protein_column: str,
        ki_column: str
):
    """
    Preprocess BindingDB Ki measurements for DTI regression.

    Parameters
    ----------
    dataframe:
        Raw BindingDB dataframe.

    smiles_column:
        Name of the SMILES column.

    protein_column:
        Name of the target protein sequence column.

    ki_column:
        Name of the Ki column. Values must represent nM.

    Returns
    -------
    cleaned_dataframe:
        Clean modelling dataframe containing:

        - canonical_smiles
        - protein_sequence
        - ki_nm
        - pKi

    report:
        Number of rows removed at each preprocessing stage.
    """

    required_columns = [
        smiles_column,
        protein_column,
        ki_column
    ]

    missing_columns = [
        column 
        for column in required_columns 
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )
    
    report = PreprocessingReport(initial_rows=len(dataframe))

    df = dataframe[
        required_columns
    ].copy()

    df = df.rename(
        columns={
            smiles_column: "smiles",
            protein_column: "sequence",
            ki_column: 'ki'
        }
    )

    # 1. Remove missing required values

    before = len(df)

    df = df.dropna(
        subset=[
            "smiles",
            "sequence",
            "ki"
        ]
    )

    report.missing_required = (before - len(df))

    # 2. Parse Ki

    df['ki_nm'] = (
        df['ki'].map(parse_ki_nm)
    )

    invalid_ki_mask = (df['ki_nm'].isna())

    report.invalid_or_censored_ki = int(invalid_ki_mask.sum())

    df = df.loc[
        ~invalid_ki_mask
    ].copy()


    # 3. Remove non-positive ki

    positive_ki_mask = (
        df['ki_nm'] > 0
    )

    report.non_positive_ki = int(
        (~positive_ki_mask).sum()
    )

    df = df.loc[
        positive_ki_mask
    ].copy()

    # 4. Canonicalise SMILES

    df['canonical_smiles'] = (
        df['smiles'].map(canonicalize_smile)
    )

    invalid_smiles_mask = (
        df['canonical_smiles'].isna()
    )

    report.invalid_smiles = int(
        invalid_smiles_mask.sum()
    )

    df = df.loc[
        ~ invalid_smiles_mask
    ].copy()

    # 5. Normalise protein sequences

    df['sequence'] = (
        df['sequence'].map(normalize_protine_sequence)
    )

    valid_protien_mask = (
        df['sequence'].map(is_valid_protine_sequence)
    )

    report.invalid_protein = int((~valid_protien_mask).sum())

    df = df.loc[
        valid_protien_mask
    ].copy()

    # 6. convert Ki -> pKi
    df['pKi'] = ki_nm_to_pki(
        df["ki_nm"].to_numpy()
    )

    # 7.Remove exact dublicate
    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "canonical_smiles",
            "sequence",
            "ki_nm"
        ]
    )

    report.duplicates_removed = (
        before - len(df)
    )

    df = df[[
        "canonical_smiles",
        "sequence",
        "ki_nm",
        'pKi'
    ]].copy()

    df = df.reset_index(
        drop=True
    )

    report.final_rows = len(df)

    return df, report