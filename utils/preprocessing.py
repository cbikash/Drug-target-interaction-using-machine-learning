import pandas as pd
import numpy as np
import tqdm

def clean_affinity(val):
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
    

def convert_to_pKi(ki_nm):
    return -np.log10(ki_nm * 1e-9)  # Convert nM to M and then take -log10
    
def preprocess_chunk(chunk):
    df = chunk[[
        'Ligand SMILES',
        'BindingDB Target Chain Sequence 1',
        'Ki (nM)',
        'IC50 (nM)',
        'Kd (nM)',
    ]].copy()

    df = df.dropna(subset=['Ligand SMILES', 'BindingDB Target Chain Sequence 1'])

    # prortize Ki > IC50 > Kd
    df['affinity'] = (
    df['Ki (nM)']
    .fillna(df['Kd (nM)'])
    .fillna(df['IC50 (nM)']))

    df['affinity'] = df['affinity'].apply(clean_affinity)
    df = df.dropna(subset=['affinity'])

    df['pKi'] = df['affinity'].apply(convert_to_pKi)

    return df[['Ligand SMILES', 'BindingDB Target Chain Sequence 1', 'pKi']]


def preprocess_full(file):
    chunks = pd.read_csv(file, sep='\t', chunksize=100000)
    all_data = []

    for chunk in chunks:
        processed = preprocess_chunk(chunk)
        all_data.append(processed)

    df = pd.concat(all_data)

    df = df.groupby(['Ligand SMILES', 'BindingDB Target Chain Sequence 1']).mean().reset_index()

    return df


if __name__ == "__main__":
    file = "data/BindingDB_All.tsv"
    df = preprocess_full(file)
    df.to_csv("data/preprocessed_bindingdb.csv", index=False)
