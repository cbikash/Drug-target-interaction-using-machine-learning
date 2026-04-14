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

df = pd.read_csv("./data/preprocessed_bindingdb.csv", nrows=50000)

df = df.rename(columns={
    "BindingDB Target Chain Sequence 1": "Target Sequence"
})

df = df.dropna(subset=["Ligand SMILES", "Target Sequence", "pKi"])
df["pKi"] = df["pKi"].astype("float32")


# -----------------------------
# CLEAN pKi (avoid NaN/inf)
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["pKi"])

# -----------------------------
# CONVERT TO CLASSIFICATION
# -----------------------------
THRESHOLD = 7
df["label"] = (df["pKi"] >= THRESHOLD).astype(np.float32)

print("Class distribution:")
print(df["label"].value_counts())

FP_SIZE = 1024
MAX_LEN = 300
OUTPUT_DIR = 'data/processed_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=FP_SIZE,
)

def smile_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.uint8)


# #protine encoding
# print("Loading BERT tokenizer and model...")

# tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
# model = BertModel.from_pretrained('Rostlab/prot_bert')
# model.to(device)
# model.eval()


# AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# AA_TO_IDX = {a: i + 1 for i, a in enumerate(AMINO_ACIDS)}
# def encode_protein_sequence1(sequence):
#     seq = sequence[:MAX_LEN]
#     encoded = [AA_TO_IDX.get(a, 0) for a in seq]

#     if len(encoded) < MAX_LEN:
#         encoded += [0] * (MAX_LEN - len(encoded))
    
#     return np.array(encoded, dtype=np.int16)
    


#load model esm, alphabet
print("Loading ESM model and alphabet...")
model_esm, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

model_esm = model_esm.to(device)
model_esm.eval()

batch_converter = alphabet.get_batch_converter()

print("ESM model and alphabet loaded successfully!")

def preprocess_sequence(sequence):
    sequence = sequence.upper()
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join([c for c in sequence if c in valid])
    return seq[:500]  # truncate

def get_esm_embedding(sequence):
    sequence = preprocess_sequence(sequence)
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)

    tokens = tokens.to(device)

    with torch.no_grad():
        results = model_esm(tokens, repr_layers=[6])
        token_embeddings = results['representations'][6]

    token_embeddings = token_embeddings[:, 1:-1, :]
    # Average pooling over the sequence length dimension

    protein_embedding = token_embeddings.mean(dim =1)
    print(protein_embedding.shape, 'protein embedding shape')
    return protein_embedding.squeeze().cpu().numpy()



# def encode_protein_sequence(sequence):

#     sequence = " ".join(list(sequence))
#     inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=MAX_LEN)
#     inputs = {key: val.to(device) for key, val in inputs.items()}
    
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Use the [CLS] token representation as the protein embedding
#     embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Average pooling over the sequence length
#     return embedding.cpu().numpy().astype(np.float32)

# print("Processing data and generating features...")
# smiles_cache = {}

# for sm in tqdm(df["Ligand SMILES"].unique(), desc="Processing SMILES"):
#     smiles_cache[sm] = smile_to_fp(sm)

# print("preprocessing protine with ProtBERT (slow, one-time)")
# seq_cache = {}

# for seq in tqdm(df["Target Sequence"].unique(), desc="Processing Protein Sequences"):
#     try:
#         seq_cache[seq] = encode_protein_sequence1(seq)
#     except Exception as e:
#         print(f"Error processing sequence: {seq[:30]}... - {e}")
#         seq_cache[seq] = None



print("Building SMILES cache...")
smiles_cache = {}
for sm in tqdm(df["Ligand SMILES"].unique()):
    smiles_cache[sm] = smile_to_fp(sm)

print("Building Protein cache...")
seq_cache = {}
for seq in tqdm(df["Target Sequence"].unique()):
    seq_cache[seq] = get_esm_embedding(seq)



df["ligand"] = df["Ligand SMILES"].map(smiles_cache)
df["protein"] = df["Target Sequence"].map(seq_cache)

df = df.dropna(subset=["ligand", "protein"])


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

# # map the features back to the dataframe
# df['ligand'] = df['Ligand SMILES'].map(smiles_cache)
# df['protein'] = df['Target Sequence'].map(seq_cache)

# # drop rows with None features
# df = df.dropna(subset=['ligand', 'protein'])

# # save the processed features
# X_ligand = np.stack(df['ligand'].values)
# X_protein = np.stack(df['protein'].values)
# y = df['pKi'].values.astype(np.float32)

# np.save(os.path.join(OUTPUT_DIR, 'X_ligand.npy'), X_ligand)
# np.save(os.path.join(OUTPUT_DIR, 'X_protein.npy'), X_protein)
# np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)

# print("Saved processed features to 'data/processed_features/' directory.")

    
