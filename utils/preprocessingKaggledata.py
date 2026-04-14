import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
import esm

from transformers import BertModel, BertTokenizer

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

df = pd.read_csv('/Users/bikaschaudharytharu/python/Drug-target-interaction-using-machine-learning/data/bindingdb_highconf_pKd_clean.csv', nrows=50000)

FP_SIZE = 1024
MAX_LEN = 300
OUTPUT_DIR = 'data/processed_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius = 2,
    fpSize=FP_SIZE,
)

def smile_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.uint8)

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
model = BertModel.from_pretrained('Rostlab/prot_bert').to(device)
model.eval()


def get_protbert_embedding(sequence):
    sequence = " ".join(list(sequence))
    inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=MAX_LEN).to(device)
    inputs = {k:v for k,v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding.astype(np.float32)


smiles_cache = {}
for sm in tqdm(df['compound_iso_smiles'].unique()):
    smiles_cache[sm] = smile_to_fp(sm)

sequence_cache = {}
for seq in tqdm(df['target_sequence'].unique()):
    sequence_cache[seq] = get_protbert_embedding(seq)


df['ligand'] = df['compound_iso_smiles'].map(smiles_cache)
df['protein'] = df['target_sequence'].map(sequence_cache)

df = df.dropna(subset=['ligand', 'protein'])

# update label to binary classification based on pKd threshold
THRESHOLD = 7
df['label'] = df['affinity'].apply(lambda x: 1 if x >= THRESHOLD else 0)

# -----------------------------
# SAVE NUMPY FILES
# -----------------------------
X_lig = np.stack(df["ligand"].values)
X_prot = np.stack(df["protein"].values)
y = df["label"].values

np.save(f"{OUTPUT_DIR}/X_lig.npy", X_lig)
np.save(f"{OUTPUT_DIR}/X_prot.npy", X_prot)
np.save(f"{OUTPUT_DIR}/y.npy", y)

print("Saved preprocessed data successfully!")