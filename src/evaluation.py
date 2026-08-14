import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    explained_variance_score
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import os

class DTAEvaluationMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Calculate comprehensive evaluation metrics for DTA prediction
        """
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        
        # Correlation coefficients
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        # Concordance Index (CI)
        ci = self.concordance_index(y_true, y_pred)
        
        # Fraction of predictions within tolerance
        tolerance_05 = self.fraction_within_tolerance(y_true, y_pred, tolerance=0.5)
        tolerance_10 = self.fraction_within_tolerance(y_true, y_pred, tolerance=1.0)
        
        # Store metrics
        self.metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': ev,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'concordance_index': ci,
            'fraction_within_0.5': tolerance_05,
            'fraction_within_1.0': tolerance_10,
            'n_samples': len(y_true)
        }
        
        return self.metrics
    
    def concordance_index(self, y_true, y_pred):
        """
        Calculate Concordance Index (CI) - commonly used in DTA prediction
        """
        ind = np.argsort(y_true)
        y_true_sorted = y_true[ind]
        y_pred_sorted = y_pred[ind]
        
        pairs = 0
        correct_pairs = 0
        
        for i in range(len(y_true_sorted)):
            for j in range(i + 1, len(y_true_sorted)):
                if y_true_sorted[i] != y_true_sorted[j]:
                    pairs += 1
                    if y_pred_sorted[i] <= y_pred_sorted[j]:
                        correct_pairs += 1
        
        if pairs == 0:
            return 0.0
        
        return correct_pairs / pairs
    
    def fraction_within_tolerance(self, y_true, y_pred, tolerance=0.5):
        """
        Calculate fraction of predictions within specified tolerance
        """
        abs_errors = np.abs(y_true - y_pred)
        return np.mean(abs_errors <= tolerance)
    
    def print_metrics(self):
        """
        Print formatted metrics report
        """
        print(f"\n{'='*50}")
        print(f"EVALUATION METRICS REPORT - {self.metrics['model_name']}")
        print(f"{'='*50}")
        print(f"Number of samples: {self.metrics['n_samples']}")
        print("-" * 30)
        print(f"MSE:              {self.metrics['mse']:.4f}")
        print(f"RMSE:             {self.metrics['rmse']:.4f}")
        print(f"MAE:              {self.metrics['mae']:.4f}")
        print(f"R² Score:         {self.metrics['r2']:.4f}")
        print(f"Explained Variance: {self.metrics['explained_variance']:.4f}")
        print(f"Pearson Corr:     {self.metrics['pearson_corr']:.4f}")
        print(f"Spearman Corr:    {self.metrics['spearman_corr']:.4f}")
        print(f"Concordance Index: {self.metrics['concordance_index']:.4f}")
        print(f"Fraction within ±0.5: {self.metrics['fraction_within_0.5']:.4f}")
        print(f"Fraction within ±1.0: {self.metrics['fraction_within_1.0']:.4f}")
        print("="*50)
    
    def plot_predictions(self, y_true, y_pred, title="Predicted vs Actual Values"):
        """
        Plot predicted vs actual values
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title}\nR² = {self.metrics["r2"]:.3f}')
        
        # Residuals plot
        residuals = y_true - y_pred
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Histogram of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        
        # Q-Q plot
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.show()
    
    def cross_validation_report(self, model, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform cross-validation and return detailed report
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"\nCross-Validation Report ({cv}-fold):")
        print(f"RMSE scores: {cv_rmse}")
        print(f"Mean RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        print(f"Std RMSE: {cv_rmse.std():.4f}")
        
        return {
            'cv_scores': cv_scores,
            'cv_rmse': cv_rmse,
            'mean_cv_rmse': cv_rmse.mean(),
            'std_cv_rmse': cv_rmse.std()
        }

# Function to compare multiple models
def compare_models(results_dict):
    """
    Compare metrics across multiple models
    """
    df = pd.DataFrame(results_dict).T
    print("\nModel Comparison:")
    print(df.round(4))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = ['rmse', 'mae', 'r2', 'pearson_corr', 'spearman_corr', 'concordance_index']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//3, i%3]
        models = df.index
        values = df[metric].values
        bars = ax.bar(models, values)
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(metric)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return df

# Example usage
if __name__ == "__main__":
    # Example with dummy data
    np.random.seed(42)
    y_true = np.random.normal(6.0, 1.0, 100)  # Simulated pKd values
    y_pred = y_true + np.random.normal(0, 0.3, 100)  # Add some noise
    
    # Initialize evaluator
    evaluator = DTAEvaluationMetrics()
    
    # Calculate metrics
    metrics = evaluator.calculate_all_metrics(y_true, y_pred, "Example Model")
    
    # Print metrics
    evaluator.print_metrics()
    
    # Plot results
    evaluator.plot_predictions(y_true, y_pred)
    
    # Example of comparing multiple models
    model_results = {
        'Random Forest': {
            'rmse': 0.45,
            'mae': 0.35,
            'r2': 0.85,
            'pearson_corr': 0.92,
            'spearman_corr': 0.90,
            'concordance_index': 0.88,
            'fraction_within_0.5': 0.75,
            'fraction_within_1.0': 0.92
        },
        'XGBoost': {
            'rmse': 0.42,
            'mae': 0.32,
            'r2': 0.87,
            'pearson_corr': 0.93,
            'spearman_corr': 0.91,
            'concordance_index': 0.89,
            'fraction_within_0.5': 0.78,
            'fraction_within_1.0': 0.94
        },
        'Neural Network': {
            'rmse': 0.48,
            'mae': 0.37,
            'r2': 0.83,
            'pearson_corr': 0.91,
            'spearman_corr': 0.89,
            'concordance_index': 0.86,
            'fraction_within_0.5': 0.72,
            'fraction_within_1.0': 0.90
        }
    }
    
    comparison_df = compare_models(model_results)