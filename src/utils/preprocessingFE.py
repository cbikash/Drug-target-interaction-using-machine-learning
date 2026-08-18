import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
import esm
import datetime

today = datetime.datetime.now().strftime("%Y-%m-%d")

class PreprocessorFeatures:
    def __init__(self, output_filename="output_"):

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.model_esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()

        self.model_esm = self.model_esm.to(self.device)
        self.model_esm.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        self.FP_SIZE = 1024
        self.MAX_LEN = 300
        self.OUTPUT_DIR = 'data/processed_features'
        self.output_filename = output_filename
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def process_smiles(self, smiles):
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=self.FP_SIZE,
        )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fb = morgan_gen.GetFingerprint(mol)
        return np.array(fb, dtype=np.uint8)
    

    def get_esm_embedding(self, sequence):
        sequence = self.process_sequence(sequence)
        data = [("protein", sequence)]
        _, _, tokens = self.batch_converter(data)

        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.model_esm(tokens, repr_layers=[6])
            token_embeddings = results['representations'][6]

        token_embeddings = token_embeddings[:, 1:-1, :]
        # Average pooling over the sequence length dimension

        protein_embedding = token_embeddings.mean(dim=1)
        return protein_embedding.squeeze().cpu().numpy()


    def process_sequence(self, sequence):
        sequence = sequence.upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        seq = "".join([c for c in sequence if c in valid])
        return seq[:500]  # truncate
    
    def build_features(self, df):
        smiles_cache = {}
        
        for sm in tqdm(df["smiles"].unique()):
            smiles_cache[sm] = self.process_smiles(sm)

        print("Building Protein cache...")
        seq_cache = {}
        for seq in tqdm(df["sequence"].unique()):
            seq_cache[seq] = self.get_esm_embedding(seq)

        df["ligand"] = df["smiles"].map(smiles_cache)
        df["protein"] = df["sequence"].map(seq_cache)
        df = df.dropna(subset=["ligand", "protein"])

        return df[["ligand", "protein", "affinity"]]
    

    def save_features(self, df_data):
        df = self.build_features(df_data)
        X_lig = np.stack(df["ligand"].values)
        X_prot = np.stack(df["protein"].values)
        y = df["affinity"].values

        np.save(f"{self.OUTPUT_DIR}/{self.output_filename}{today}_X_lig.npy", X_lig)
        np.save(f"{self.OUTPUT_DIR}/{self.output_filename}{today}_X_prot.npy", X_prot)
        np.save(f"{self.OUTPUT_DIR}/{self.output_filename}{today}_y.npy", y)

        print("Saved preprocessed data successfully!")