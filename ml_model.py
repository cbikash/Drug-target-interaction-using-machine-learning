import os
import time
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
import xgboost as xgb


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/processed_features"
RESULTS_FILE = "bindingdb_ml_results.csv"

# Prefix used when loading saved NumPy files
output_filename = ""

RANDOM_STATE = 42

TEST_SIZE = 0.20

# 12.5% of the remaining 80% gives:
# 70% training, 10% validation, 20% testing
VALIDATION_SIZE = 0.125

# Approximate SVM configuration
SVM_MAX_TRAINING_SAMPLES = 20_000
NYSTROEM_COMPONENTS = 256

# Prevent prediction from creating very large temporary arrays
PREDICTION_BATCH_SIZE = 10_000


# ============================================================
# Evaluation
# ============================================================

class ModelEvaluator:
    def __init__(self):
        self.results = []

    def evaluate(
        self,
        model_name,
        y_true,
        y_pred,
        train_time,
        inference_time
    ):
        """
        Calculate regression metrics and store the results.
        """

        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Pearson correlation is undefined for constant arrays
        if (
            len(y_true) > 1
            and np.std(y_true) > 0
            and np.std(y_pred) > 0
        ):
            pearson_corr, _ = pearsonr(y_true, y_pred)
        else:
            pearson_corr = 0.0

        result = {
            "Model": model_name,
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2_Score": float(r2),
            "Pearson_Correlation": float(pearson_corr),
            "Train_Time_sec": float(train_time),
            "Inference_Time_sec": float(inference_time)
        }

        self.results.append(result)

        print(f"\n--- {model_name} Evaluation ---")
        print(
            f"MSE: {mse:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"MAE: {mae:.4f}"
        )
        print(
            f"R²: {r2:.4f} | "
            f"Pearson: {pearson_corr:.4f}"
        )
        print(
            f"Train time: {train_time:.2f}s | "
            f"Inference time: {inference_time:.4f}s"
        )

    def get_results_dataframe(self):
        """
        Return model results ordered by RMSE.
        Lower RMSE indicates better performance.
        """

        if not self.results:
            return pd.DataFrame()

        return (
            pd.DataFrame(self.results)
            .sort_values(by="RMSE", ascending=True)
            .reset_index(drop=True)
        )

    def print_summary(self):
        results_df = self.get_results_dataframe()

        if results_df.empty:
            print("No model results are available.")
            return

        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(results_df.to_string(index=False))

    def save_to_csv(self, filepath=RESULTS_FILE):
        """
        Save all accumulated results to a CSV file.
        """

        results_df = self.get_results_dataframe()

        if results_df.empty:
            print("No results to save.")
            return

        output_directory = os.path.dirname(filepath)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        results_df.to_csv(filepath, index=False)

        print(f"\nResults successfully saved to: {filepath}")


# ============================================================
# Data Loading and Validation
# ============================================================

def load_and_prepare_data(data_dir=DATA_DIR):
    """
    Load ligand features, protein features and affinity targets.

    Returns a 70% training, 10% validation and 20% testing split.
    """

    print("Loading data...")

    ligand_path = os.path.join(
        data_dir,
        f"{output_filename}X_lig.npy"
    )

    protein_path = os.path.join(
        data_dir,
        f"{output_filename}X_prot.npy"
    )

    target_path = os.path.join(
        data_dir,
        f"{output_filename}y.npy"
    )

    X_ligand = np.load(ligand_path).astype(
        np.float32,
        copy=False
    )

    X_protein = np.load(protein_path).astype(
        np.float32,
        copy=False
    )

    y = np.load(target_path).astype(
        np.float32,
        copy=False
    ).ravel()

    # Validate the number of samples
    if not (
        len(X_ligand) == len(X_protein) == len(y)
    ):
        raise ValueError(
            "Ligand features, protein features and targets "
            "have different numbers of samples."
        )

    # Remove rows containing NaN or infinity
    valid_mask = (
        np.isfinite(y)
        & np.all(np.isfinite(X_ligand), axis=1)
        & np.all(np.isfinite(X_protein), axis=1)
    )

    invalid_rows = np.sum(~valid_mask)

    if invalid_rows > 0:
        print(
            f"Removing {invalid_rows:,} rows containing "
            "NaN or infinite values."
        )

        X_ligand = X_ligand[valid_mask]
        X_protein = X_protein[valid_mask]
        y = y[valid_mask]

    print(f"Ligand feature shape: {X_ligand.shape}")
    print(f"Protein feature shape: {X_protein.shape}")
    print(f"Target shape: {y.shape}")

    # Concatenate Morgan fingerprints and protein embeddings
    X = np.concatenate(
        [X_ligand, X_protein],
        axis=1
    )

    # Release the separate arrays after concatenation
    del X_ligand
    del X_protein

    print(f"Combined dataset shape: {X.shape}")
    print(f"Estimated feature memory: {X.nbytes / 1e9:.2f} GB")

    # First create the untouched test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Create a validation set from the training portion
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Release intermediate arrays
    del X
    del X_train_full
    del y_train_full

    print("\nDataset split:")
    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_validation):,}")
    print(f"Testing samples:    {len(X_test):,}")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ============================================================
# Utility Functions
# ============================================================

def predict_in_batches(
    model,
    X,
    batch_size=PREDICTION_BATCH_SIZE
):
    """
    Predict in smaller batches to reduce peak memory usage.
    """

    predictions = []

    for start_index in range(0, len(X), batch_size):
        end_index = min(
            start_index + batch_size,
            len(X)
        )

        batch_predictions = model.predict(
            X[start_index:end_index]
        )

        predictions.append(
            np.asarray(
                batch_predictions,
                dtype=np.float32
            ).ravel()
        )

    return np.concatenate(predictions)


def train_and_evaluate(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    evaluator
):
    """
    Train and evaluate a standard scikit-learn-compatible model.
    """

    print(f"\nTraining {model_name}...")

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    predictions = predict_in_batches(
        model,
        X_test
    )
    inference_time = time.perf_counter() - start_time

    evaluator.evaluate(
        model_name=model_name,
        y_true=y_test,
        y_pred=predictions,
        train_time=train_time,
        inference_time=inference_time
    )

    return model


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    evaluator = ModelEvaluator()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = load_and_prepare_data()

    number_of_features = X_train.shape[1]

    # ========================================================
    # 1. Naïve Mean Baseline
    # ========================================================
    #
    # This model always predicts the mean training affinity.
    # Every trained model should outperform this baseline.
    #

    dummy_model = DummyRegressor(
        strategy="mean"
    )

    dummy_model = train_and_evaluate(
        model=dummy_model,
        model_name="Mean Baseline",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        evaluator=evaluator
    )

    # ========================================================
    # 2. Linear Ridge Baseline
    # ========================================================
    #
    # This determines whether nonlinear models genuinely
    # outperform a regularised linear model.
    #

    ridge_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=10.0,
                solver="lsqr",
                max_iter=5_000,
                tol=1e-3
            )
        )
    ])

    ridge_pipeline = train_and_evaluate(
        model=ridge_pipeline,
        model_name="Ridge Regression",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        evaluator=evaluator
    )

    # ========================================================
    # 3. XGBoost
    # ========================================================
    #
    # The validation set is used for early stopping.
    # The test set remains untouched until final evaluation.
    #

    print("\nTraining XGBoost...")

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",

        # A larger upper limit is safe because early stopping
        # will normally stop training much earlier
        n_estimators=1_000,
        learning_rate=0.05,

        max_depth=6,
        min_child_weight=5,

        subsample=0.80,
        colsample_bytree=0.70,

        reg_alpha=0.05,
        reg_lambda=1.0,

        tree_method="hist",
        max_bin=256,

        early_stopping_rounds=30,

        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    start_time = time.perf_counter()

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False
    )

    xgb_train_time = time.perf_counter() - start_time

    start_time = time.perf_counter()

    xgb_predictions = predict_in_batches(
        xgb_model,
        X_test
    )

    xgb_inference_time = (
        time.perf_counter() - start_time
    )

    evaluator.evaluate(
        model_name="XGBoost",
        y_true=y_test,
        y_pred=xgb_predictions,
        train_time=xgb_train_time,
        inference_time=xgb_inference_time
    )

    if hasattr(xgb_model, "best_iteration"):
        print(
            f"Best XGBoost iteration: "
            f"{xgb_model.best_iteration}"
        )

    # ========================================================
    # 4. Random Forest
    # ========================================================
    #
    # max_samples and max_features reduce training cost.
    # min_samples_leaf helps reduce overfitting.
    #

    random_forest_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,

        min_samples_split=4,
        min_samples_leaf=2,

        max_features=0.20,
        max_samples=0.80,

        bootstrap=True,

        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )

    random_forest_model = train_and_evaluate(
        model=random_forest_model,
        model_name="Random Forest",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        evaluator=evaluator
    )

    # ========================================================
    # 5. Approximate RBF SVM
    # ========================================================
    #
    # A random subset is used instead of selecting the first
    # 20,000 records, which could introduce ordering bias.
    #

    print("\nPreparing Approximate RBF SVM data...")

    svm_sample_size = min(
        SVM_MAX_TRAINING_SAMPLES,
        len(X_train)
    )

    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    svm_indices = random_generator.choice(
        len(X_train),
        size=svm_sample_size,
        replace=False
    )

    X_train_svm = X_train[svm_indices]
    y_train_svm = y_train[svm_indices]

    # More stable than using a fixed gamma such as 0.05
    nystroem_gamma = 1.0 / number_of_features

    svm_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "nystroem",
            Nystroem(
                kernel="rbf",
                gamma=nystroem_gamma,
                n_components=NYSTROEM_COMPONENTS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0,
                solver="lsqr",
                max_iter=5_000,
                tol=1e-3
            )
        )
    ])

    svm_pipeline = train_and_evaluate(
        model=svm_pipeline,
        model_name="Approximate RBF SVM",
        X_train=X_train_svm,
        y_train=y_train_svm,
        X_test=X_test,
        y_test=y_test,
        evaluator=evaluator
    )

    # Release SVM subset
    del X_train_svm
    del y_train_svm
    del svm_indices

    # ========================================================
    # Final Comparison
    # ========================================================

    evaluator.print_summary()
    evaluator.save_to_csv(RESULTS_FILE)