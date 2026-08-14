import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    def clean_affinity(self, val):
        if pd.isna(val):
            return None

        val = str(val).strip()

        # Exclude censored limits (inequality qualifiers) 
        # as they are not exact observations
        if val.startswith((">", "<", ">=", "<=", "≥", "≤")):
            return None
        
        # Remove approximation symbols (e.g., ~100 becomes 100)
        # Approximations are usually kept as they represent a central estimate
        if val.startswith("~"):
            val = val[1:]

        try:
            return float(val)
        except:
            return None

    def convert_to_pKi(self, ki_nm):
        return -np.log10(ki_nm * 1e-9)  # Convert nM to M and then take -log10
    
    def drop_na(self, df, cols):
        return df.dropna(subset=cols)
    
    def build_affinity(self, df):
        """
        Build a cleaned drug-target affinity dataset using only Ki measurements.

        The Ki values are expected in nanomolar (nM) and are converted to pKi:

            pKi = -log10(Ki × 10^-9)
        """

        required_columns = [
            'Ligand SMILES',
            'BindingDB Target Chain Sequence 1',
            'Ki (nM)',
        ]

        # Validate required columns
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing required columns: {missing_columns}"
            )

        # Select only the required fields
        df = df[required_columns].copy()

        # Remove rows without ligand, protein, or Ki values
        df = df.dropna(
            subset=[
                'Ligand SMILES',
                'BindingDB Target Chain Sequence 1',
                'Ki (nM)',
            ]
        )

        # Clean Ki values, for example "<10", ">1000", or invalid strings
        df['Ki (nM)'] = df['Ki (nM)'].apply(self.clean_affinity)

        # Convert cleaned values to numeric
        df['Ki (nM)'] = pd.to_numeric(
            df['Ki (nM)'],
            errors='coerce'
        )

        # Remove invalid, missing, zero, and negative Ki values
        df = df[
            df['Ki (nM)'].notna() &
            (df['Ki (nM)'] > 0)
        ].copy()

        # Convert Ki in nM to pKi
        df['affinity'] = df['Ki (nM)'].apply(self.convert_to_pKi)

        # Standardise column names
        df = df.rename(
            columns={
                'Ligand SMILES': 'smiles',
                'BindingDB Target Chain Sequence 1': 'sequence',
            }
        )

        # Remove duplicate drug-target-affinity records
        df = df.drop_duplicates(
            subset=['smiles', 'sequence', 'affinity']
        ).reset_index(drop=True)

        return df[['smiles', 'sequence', 'affinity']]