import os
import numpy as np
from sklearn.model_selection import train_test_split


# CONFIGURATION
FEATURE_DIR = "data/processed/features"

SPLIT_FILE = os.path.join(
    FEATURE_DIR,
    "fixed_split_indices.npz",
)

RANDOM_STATE = 42



# LOAD FEATURES
ligand_features = np.load(
    os.path.join(FEATURE_DIR, "ligand_features.npy"),
    mmap_mode="r",
)

protein_esm_features = np.load(
    os.path.join(FEATURE_DIR, "protein_esm_features.npy"),
    mmap_mode="r",
)

protein_token_features = np.load(
    os.path.join(FEATURE_DIR, "protein_token_features.npy"),
    mmap_mode="r",
)

ligand_index = np.load(
    os.path.join(FEATURE_DIR, "ligand_index.npy")
)

protein_index = np.load(
    os.path.join(FEATURE_DIR, "protein_index.npy")
)

y = np.load(
    os.path.join(FEATURE_DIR, "y.npy")
)



# CREATE / LOAD FIXED SPLIT
if os.path.exists(SPLIT_FILE):

    print("Loading existing split...")

    split = np.load(SPLIT_FILE)

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

else:

    print("Creating fixed split...")

    indices = np.arange(len(y))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    np.savez(
        SPLIT_FILE,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )



# MACHINE LEARNING DATA
def get_ml_data():

    X_train = np.concatenate(
        [
            ligand_features[ligand_index[train_idx]],
            protein_esm_features[protein_index[train_idx]],
        ],
        axis=1,
    )

    X_val = np.concatenate(
        [
            ligand_features[ligand_index[val_idx]],
            protein_esm_features[protein_index[val_idx]],
        ],
        axis=1,
    )

    X_test = np.concatenate(
        [
            ligand_features[ligand_index[test_idx]],
            protein_esm_features[protein_index[test_idx]],
        ],
        axis=1,
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    return (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,
    )



# DEEP LEARNING DATA
def get_deep_learning_data():

    X_lig_train = ligand_features[
        ligand_index[train_idx]
    ]

    X_lig_val = ligand_features[
        ligand_index[val_idx]
    ]

    X_lig_test = ligand_features[
        ligand_index[test_idx]
    ]


    X_prot_train = protein_token_features[
        protein_index[train_idx]
    ]

    X_prot_val = protein_token_features[
        protein_index[val_idx]
    ]

    X_prot_test = protein_token_features[
        protein_index[test_idx]
    ]


    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]


    return (
        X_lig_train,
        X_prot_train,
        y_train,

        X_lig_val,
        X_prot_val,
        y_val,

        X_lig_test,
        X_prot_test,
        y_test,
    )