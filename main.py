from utils.preprocessing import Preprocessor
from utils.preprocessingFE import PreprocessorFeatures
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CNNDTIModel, MLPDTIModel 

def preprocess_rawdata():
    preprocessor = Preprocessor()
    file = "data/BindingDB_All.tsv"

    all_data = pd.read_csv(file, sep='\t', chunksize=100000, low_memory=False, dtype=str)

    processed_chunks = []
    df = pd.DataFrame()

    for chunk in all_data:
        processed = preprocessor.build_affinity(chunk)
        processed_chunks.append(processed)

    df = pd.concat(processed_chunks, ignore_index=True)
    df = df.groupby(['smiles', 'sequence']).mean().reset_index()

    df.to_csv("data/processed/preprocessed_bindingdb.csv", index=False)

def preprocess_features():
    df_raw = pd.read_csv("data/processed/preprocessed_bindingdb.csv", nrows=10000)
    preprocessor = PreprocessorFeatures()
    preprocessor.save_features(df_raw)

def load_features():
    X_lig = np.load("data/processed_features/X_lig.npy")
    X_prot = np.load("data/processed_features/X_prot.npy")
    y = np.load("data/processed_features/y.npy")
    return X_lig, X_prot, y

class DTI_Dataset(torch.utils.data.Dataset):
    def __init__(self, X_lig, X_prot, y):
        self.X_lig = X_lig
        self.X_prot = X_prot
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_lig[idx], self.X_prot[idx], self.y[idx]

if __name__ == "__main__":
    # Both steps can be run independently, but typically you'd run preprocess_rawdata() once to create the cleaned CSV, and then run preprocess_features() to generate the feature files for modeling.
    # preprocess_rawdata() # This will create the preprocessed CSV file with cleaned affinities
    # preprocess_features() # This will read the preprocessed CSV, build features, and save them as .npy files

    X_lig, X_prot, y = load_features() # This will load the preprocessed feature arrays for use in model training or evaluation

    # Split the data into training and testing sets
    X_train_lig, X_test_lig, X_train_prot, X_test_prot, y_train, y_test = train_test_split(X_lig, X_prot, y, test_size=0.2, random_state=42)

    # Create DataLoader for training and testing
    train_loader = DataLoader(DTI_Dataset(X_train_lig, X_train_prot, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(DTI_Dataset(X_test_lig, X_test_prot, y_test), batch_size=32)

