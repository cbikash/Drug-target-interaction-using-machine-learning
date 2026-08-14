import pandas as pd
import numpy as np

df = pd.read_csv('data/bindingdb_highconf_pKd_clean.csv')

print("Initial dataset shape:", df.shape)
print("Columns in the dataset:", df.columns)
print("Number of unique compounds:", df['compound_iso_smiles'].nunique())
print("Number of unique targets:", df['target_sequence'].nunique())
