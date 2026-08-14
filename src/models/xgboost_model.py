import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os
import datetime

today = datetime.datetime.now().strftime("%Y-%m-%d")

class XGBoostDTA:
    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        
    def load_processed_features(self, base_filename):
        """Load processed features from your PreprocessorFeatures"""
        feature_dir = 'data/processed_features'
        
        X_lig = np.load(f"{feature_dir}/{base_filename}{today}_X_lig.npy")
        X_prot = np.load(f"{feature_dir}/{base_filename}{today}_X_prot.npy")
        y = np.load(f"{feature_dir}/{base_filename}{today}_y.npy")
        
        # Combine ligand and protein features
        X = np.concatenate([X_lig, X_prot], axis=1)
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def train_with_grid_search(self, X_train, y_train, cv=5):
        """Perform grid search to find best hyperparameters"""
        
        # Define pipeline with scaling and XGBoost
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                verbosity=1
            ))
        ])
        
        # Hyperparameter grid for XGBoost
        param_grid = {
            'xgb__n_estimators': [100, 200, 300],
            'xgb__max_depth': [3, 5, 7, 10],
            'xgb__learning_rate': [0.01, 0.1, 0.2],
            'xgb__subsample': [0.8, 0.9, 1.0],
            'xgb__colsample_bytree': [0.8, 0.9, 1.0],
            'xgb__reg_alpha': [0, 0.1, 1],
            'xgb__reg_lambda': [0, 0.1, 1]
        }
        
        # Grid search with early stopping
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return grid_search
    
    def train_with_early_stopping(self, X_train, X_val, y_train, y_val):
        """Train with early stopping using validation set"""
        
        # Create XGBoost matrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train with early stopping
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance"""
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n=== Model Performance ===")
        print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"Train RMSE: {np.sqrt(train_mse):.4f}, Test RMSE: {np.sqrt(test_mse):.4f}")
        print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        # Check for overfitting
        if test_r2 < train_r2 - 0.1:
            print("⚠️  Warning: Possible overfitting detected!")
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse)
        }
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance from trained model"""
        if isinstance(self.model.named_steps['xgb'], xgb.XGBRegressor):
            # For XGBRegressor
            booster = self.model.named_steps['xgb'].get_booster()
            importances = booster.get_score(importance_type='weight')
        else:
            # For directly trained XGBoost model
            importances = self.model.get_score(importance_type='weight')
        
        if feature_names is None:
            # Generate generic names if not provided
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Convert to sorted list of tuples
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        print("\n=== Top 20 Most Important Features ===")
        for i in range(min(20, len(sorted_importances))):
            feat_name, importance = sorted_importances[i]
            print(f"{i+1}. {feat_name}: {importance}")
        
        return importances
    
    def save_model(self, filename=None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        if filename is None:
            filename = f"xgb_dta_model_{today}.pkl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if hasattr(self.model, 'named_steps'):
            # For pipeline model
            joblib.dump(self.model, filepath)
        else:
            # For direct XGBoost model
            self.model.save_model(filepath.replace('.pkl', '.json'))
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        if filepath.endswith('.json'):
            # For direct XGBoost model
            self.model = xgb.Booster()
            self.model.load_model(filepath)
        else:
            # For pipeline model
            self.model = joblib.load(filepath)
        
        print(f"Model loaded from {filepath}")

# Alternative implementation without pipeline (for early stopping)
class XGBoostDirect:
    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
    
    def train_direct(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost directly with early stopping"""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return self.model


# Example usage
if __name__ == "__main__":
    # Initialize the XGBoost trainer
    xgb_trainer = XGBoostDTA()
    
    # Load your processed features (use same base filename as preprocessing)
    base_filename = "output_"  # Replace with your actual filename prefix
    X, y = xgb_trainer.load_processed_features(base_filename)
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = xgb_trainer.prepare_data(X, y)
    
    # Further split train into train/validation for early stopping
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train_sub.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Option 1: Grid search with pipeline
    grid_search = xgb_trainer.train_with_grid_search(X_train_sub, y_train_sub)
    
    # Option 2: Direct training with early stopping (uncomment to use)
    # xgb_direct = XGBoostDirect()
    # xgb_direct.train_direct(X_train_sub, y_train_sub, X_val, y_val)
    # xgb_trainer.model = xgb_direct.model  # Transfer model
    
    # Evaluate model
    metrics = xgb_trainer.evaluate_model(X_train_sub, X_test, y_train_sub, y_test)
    
    # Get feature importance
    # You can provide actual feature names if available
    n_morgan = 1024  # Your FP_SIZE
    n_esm = X.shape[1] - n_morgan
    feature_names = ([f'morgan_{i}' for i in range(n_morgan)] + 
                    [f'esm_{i}' for i in range(n_esm)])
    
    importances = xgb_trainer.get_feature_importance(feature_names)
    
    # Save model
    xgb_trainer.save_model()
    
    # Example: Make predictions on new data
    # predictions = xgb_trainer.model.predict(new_X)