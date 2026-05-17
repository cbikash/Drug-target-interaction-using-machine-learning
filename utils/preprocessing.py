import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    def clean_affinity(self, val):
        if pd.isna(val):
            return None

        val = str(val).strip()

        # Remove inequality symbols
        if val.startswith(">"):
            return None   # discard weak binding
        if val.startswith("<"):
            val = val[1:]
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

        df = df[[
                'Ligand SMILES',
                'BindingDB Target Chain Sequence 1',
                'Ki (nM)',
                'IC50 (nM)',
                'Kd (nM)',
            ]].copy()
        
        df.dropna(subset=['Ligand SMILES', 'BindingDB Target Chain Sequence 1'], inplace=True)


        df['affinity'] = df['affinity'] = (
            df['Ki (nM)']
            .fillna(df['Kd (nM)'])
            .fillna(df['IC50 (nM)']))
        
        df = df.dropna(subset=['affinity'])
        df['affinity'] = df['affinity'].apply(self.clean_affinity)
        df['affinity'] = pd.to_numeric(df['affinity'], errors='coerce')

        df = df[df['affinity'].notna()]
        df = df[df['affinity'] > 0]

        df['affinity'] = df['affinity'].apply(self.convert_to_pKi)

        df.rename(columns={
            'Ligand SMILES': 'smiles',
            'BindingDB Target Chain Sequence 1': 'sequence'
        }, inplace=True)

        return df[['smiles', 'sequence', 'affinity']]