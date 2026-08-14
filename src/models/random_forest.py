import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
import os
import datetime

today = datetime.datetime.now().strftime("%Y-%m-%d")

class AdvancedDTAModel:
    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models = {}
        self.preprocessors = {}
        self.best_model_name = None
        
    def load_and_preprocess_features(self, base_filename):
        """Load and preprocess features"""
        feature_dir = 'data/processed_features'
        
        X_lig = np.load(f"{feature_dir}/X_lig.npy")
        X_prot = np.load(f"{feature_dir}/X_prot.npy")
        y = np.load(f"{feature_dir}/y.npy")
        
        # Combine features
        X = np.concatenate([X_lig, X_prot], axis=1)
        
        # Handle potential NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def advanced_preprocessing(self, X_train, X_test, y_train, 
                              feature_reduction_ratio=0.5,
                              pca_variance_retained=0.95):
        """
        Advanced preprocessing with feature engineering
        """
        print("Applying advanced preprocessing...")
        
        # 1. Feature selection - keep only most relevant features
        n_features_select = max(100, int(X_train.shape[1] * feature_reduction_ratio))
        print(f"Selecting top {n_features_select} features...")
        
        selector = SelectKBest(score_func=f_regression, k=n_features_select)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # 2. PCA - reduce dimensionality while retaining variance
        pca = PCA(n_components=pca_variance_retained, random_state=42)
        X_train_pca = pca.fit_transform(X_train_selected)
        X_test_pca = pca.transform(X_test_selected)
        
        print(f"PCA reduced features from {X_train_selected.shape[1]} to {X_train_pca.shape[1]}")
        print(f"PCA retained {pca.explained_variance_ratio_.sum():.3f} variance")
        
        # 3. Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)
        
        # Store preprocessors
        self.preprocessors = {
            'selector': selector,
            'pca': pca,
            'scaler': scaler
        }
        
        return X_train_scaled, X_test_scaled
    
    def train_multiple_models(self, X_train, y_train):
        """
        Train multiple models and compare performance
        """
        print("Training multiple models...")
        
        models_config = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,      # More trees
                    max_depth=20,          # Deeper trees
                    min_samples_split=2,   # Less restrictive
                    min_samples_leaf=1,    # Smaller leaves
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    oob_score=True
                ),
                'description': 'Optimized Random Forest'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'description': 'Gradient Boosting Regressor'
            }
        }
        
        for name, config in models_config.items():
            print(f"Training {config['description']}...")
            model = config['model']
            model.fit(X_train, y_train)
            
            # Store model
            self.models[name] = {
                'model': model,
                'description': config['description'],
                'type': name
            }
            
            # Print OOB score for Random Forest
            if name == 'random_forest':
                print(f"OOB Score: {model.oob_score_:.4f}")
    
    def evaluate_all_models(self, X_train, X_test, y_train, y_test):
        """
        Evaluate all trained models
        """
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Correlations
            from scipy.stats import pearsonr
            pearson_corr, _ = pearsonr(y_test, y_test_pred)
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'pearson_corr': pearson_corr
            }
            
            print(f"\n=== {model_info['description']} ===")
            print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            print(f"Test Pearson Corr: {pearson_corr:.4f}")
        
        # Find best model based on test R²
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} with Test R²: {results[best_model_name]['test_r2']:.4f}")
        
        return results
    
    def save_best_model(self, filename=None):
        """Save the best performing model and preprocessors"""
        if self.best_model_name is None:
            raise ValueError("No model has been evaluated yet!")
        
        model_package = {
            'model': self.models[self.best_model_name]['model'],
            'preprocessors': self.preprocessors,
            'model_name': self.best_model_name,
            'model_description': self.models[self.best_model_name]['description']
        }
        
        if filename is None:
            filename = f"best_dta_model_{today}.pkl"
        
        filepath = os.path.join(self.output_dir, filename)
        joblib.dump(model_package, filepath)
        print(f"Best model saved to {filepath}")
        
        return model_package
    
    def predict_with_best_model(self, X):
        """Make predictions using the best model"""
        if self.best_model_name is None:
            raise ValueError("No model has been selected as best!")
        
        # Apply same preprocessing steps
        X_selected = self.preprocessors['selector'].transform(X)
        X_pca = self.preprocessors['pca'].transform(X_selected)
        X_scaled = self.preprocessors['scaler'].transform(X_pca)
        
        # Make predictions
        predictions = self.models[self.best_model_name]['model'].predict(X_scaled)
        return predictions

# Quick optimization version
class FastAdvancedDTA:
    def __init__(self):
        self.model = None
        self.preprocessor = None
    
    def quick_train(self, X_train, y_train):
        """Very fast training with good parameters"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Simple preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Fast but effective model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        self.model = model
        self.preprocessor = scaler
        
        return model
    
    def predict(self, X):
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict(X_scaled)

# Example usage
if __name__ == "__main__":
    print("Advanced DTA Model Training")
    
    # Initialize advanced model
    advanced_model = AdvancedDTAModel()
    
    # Load features
    base_filename = "output_"  # Replace with your actual filename
    X, y = advanced_model.load_and_preprocess_features(base_filename)
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = advanced_model.prepare_data(X, y)
    
    # Apply advanced preprocessing
    X_train_proc, X_test_proc = advanced_model.advanced_preprocessing(
        X_train, X_test, y_train,
        feature_reduction_ratio=0.3,  # Keep 30% of original features
        pca_variance_retained=0.90    # Retain 90% variance
    )
    
    # Train multiple models
    advanced_model.train_multiple_models(X_train_proc, y_train)
    
    # Evaluate all models
    results = advanced_model.evaluate_all_models(
        X_train_proc, X_test_proc, y_train, y_test
    )
    
    # Save best model
    best_model_package = advanced_model.save_best_model()
    
    print(f"\nModel training completed!")
    print(f"Best model achieved Test R²: {results[advanced_model.best_model_name]['test_r2']:.4f}")
    print(f"Best model: {advanced_model.best_model_name}")