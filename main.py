from utils.preprocessing import Preprocessor
from utils.preprocessingFE import PreprocessorFeatures
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CNNDTIModel, MLPDTIModel 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
print("Success!")

import datetime

today = datetime.date.today().strftime("%Y-%m-%d")

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

    df.to_csv(f"data/processed/{today}_preprocessed_bindingdb.csv", index=False)

def preprocess_features():
    df_raw = pd.read_csv(f"data/processed/bindingdb_subset_150k.csv", low_memory=False)
    preprocessor = PreprocessorFeatures(output_filename="bindingdb_all_")
    print("Processing features for the dataset...", preprocessor)
    preprocessor.save_features(df_raw)

def load_features():
    X_lig = np.load(f"data/processed_features/{today}_X_lig.npy")
    X_prot = np.load(f"data/processed_features/{today}_X_prot.npy")
    y = np.load(f"data/processed_features/{today}_y.npy")
    return X_lig, X_prot, y

def ml_data_prepare(data_loader):
    X = []
    y = []

    for ligand, protein, label in data_loader:

        feature = torch.cat((ligand, protein), dim=1)

        X.append(feature)
        y.append(label)

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    return X.numpy(), y.numpy()


class DTI_Dataset(torch.utils.data.Dataset):
    def __init__(self, X_lig, X_prot, y):
        self.X_lig = X_lig
        self.X_prot = X_prot
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_lig[idx], dtype=torch.float32),
            torch.tensor(self.X_prot[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

if __name__ == "__main__":

    print("datetime:", today)
    # Both steps can be run independently, but typically you'd run preprocess_rawdata() once to create the cleaned CSV, and then run preprocess_features() to generate the feature files for modeling.
    # preprocess_rawdata() # This will create the preprocessed CSV file with cleaned affinities
    preprocess_features() # This will read the preprocessed CSV, build features, and save them as .npy files

    print("Preprocessing completed. Features saved to 'data/processed_features/' directory.")

    # X_lig, X_prot, label = load_features() # This will load the preprocessed feature arrays for use in model training or evaluation

    # # Split the data into training and testing sets
    # X_train_lig, X_test_lig, X_train_prot, X_test_prot, y_train, y_test = train_test_split(X_lig, X_prot, label, test_size=0.2, random_state=42)

    # # Create DataLoader for training and testing
    # train_loader = DataLoader(DTI_Dataset(X_train_lig, X_train_prot, y_train), batch_size=32, shuffle=True)
    # test_loader = DataLoader(DTI_Dataset(X_test_lig, X_test_prot, y_test), batch_size=32)
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

   
    # # X_train, y_train = ml_data_prepare(train_loader)
    # # X_test, y_test = ml_data_prepare(test_loader)
    

    # # rf = RandomForestRegressor(
    # #     n_estimators=200,
    # #     max_depth=20,
    # #     random_state=42
    # # )

    # # rf.fit(X_train, y_train)
    # # y_pred = rf.predict(X_test)
    # # print(mean_squared_error(y_pred, y_test))

    # import joblib

    # # joblib.dump(rf, "models/random_forest_model.pkl")

    # # y_pred_train = rf.predict(y_train)

    # # print(mean_squared_error(y_pred_train, y_train))

    # # print(test)


    # print(train_loader.dataset[0][0].shape)


    # learning_rate = 0.01
    # epochs = 1

    # model = CNNDTIModel(
    #     drug_input_dim= X_train_lig.shape[1],
    #     target_input_dim=X_train_prot.shape[1],
    #     hidden_dim=256,
    #     output_dim=1
    # ).to(device)

    # print(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.MSELoss()



    # for epoch in range(epochs):
    #     total_epoch_loss = 0.0
    #     for drug_x, target_x, labels in train_loader:
    #         drug_x, target_x, labels = drug_x.to(device), target_x.to(device), labels.to(device)

    #         # print(drug_x)
    #         # print(drug_x.shape, target_x.shape, labels.shape)

    #         # break  # Remove this break statement to run the full training loop

    #         optimizer.zero_grad()
    #         outputs = model(drug_x, target_x)
    #         loss = criterion(outputs.squeeze(), labels)
    #         loss.backward()
    #         optimizer.step()

    #         total_epoch_loss += loss.item()
       
    #     avg_epoch_loss = total_epoch_loss / len(train_loader)
    #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")


    # model.eval()
    # joblib.dump(model, "models/cnn_model.pkl")

    # with torch.no_grad():
    #     y_pred = []
    #     y_true = []

    #     for drug_x, target_x, labels in test_loader:
    #         drug_x, target_x = drug_x.to(device), target_x.to(device)
    #         outputs = model(drug_x, target_x)
    #         y_pred.extend(outputs.squeeze().cpu().numpy())
    #         y_true.extend(labels.numpy())
