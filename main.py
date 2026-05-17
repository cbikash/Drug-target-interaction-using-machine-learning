from utils.preprocessing import Preprocessor
from utils.preprocessingFE import PreprocessorFeatures
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    # Both steps can be run independently, but typically you'd run preprocess_rawdata() once to create the cleaned CSV, and then run preprocess_features() to generate the feature files for modeling.
    preprocess_rawdata() # This will create the preprocessed CSV file with cleaned affinities
    preprocess_features() # This will read the preprocessed CSV, build features, and save them as .npy files

