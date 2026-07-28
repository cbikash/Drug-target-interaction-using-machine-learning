import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

class ExplainableDTICNN(nn.Module):
    def __init__(self, ligand_dim=1024, protein_dim=320, conv_channels=[64, 128], 
                 fc_dims=[256, 128, 1], dropout_rate=0.3):
        super(ExplainableDTICNN, self).__init__()
        
        # Ligand processing
        self.ligand_conv = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Protein processing
        self.protein_fc = nn.Sequential(
            nn.Linear(protein_dim, conv_channels[0]),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(conv_channels[0], conv_channels[1]),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate dimensions
        self.ligand_flat_dim = conv_channels[1] * 256  # 128 * 256 = 32768
        self.protein_flat_dim = conv_channels[1]       # 128
        
        # Combined fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.ligand_flat_dim + self.protein_flat_dim, fc_dims[0]),
            nn.BatchNorm1d(fc_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.BatchNorm1d(fc_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(fc_dims[1], fc_dims[2])
        )
    
    def forward(self, ligand_features, protein_features):
        # Handle tensor dimensions
        if ligand_features.dim() == 4:
            ligand_features = ligand_features.squeeze(1).squeeze(1)
        elif ligand_features.dim() == 3 and ligand_features.size(1) == 1:
            ligand_features = ligand_features.squeeze(1)
        
        # Ligand processing
        lig_x = ligand_features.unsqueeze(1)
        lig_x = self.ligand_conv(lig_x)
        lig_x = lig_x.view(lig_x.size(0), -1)
        
        # Protein processing
        prot_x = self.protein_fc(protein_features)
        
        # Combine features
        combined = torch.cat((lig_x, prot_x), dim=1)
        
        # Final prediction
        output = self.fc_layers(combined)
        return output

class LIMEXAIAnalyzer:
    def __init__(self, model, device):
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not available. Please install: pip install lime")
        
        self.model = model
        self.device = device
        
        # Create feature names for interpretation
        self.ligand_feature_names = [f'ligand_{i}' for i in range(1024)]
        self.protein_feature_names = [f'protein_{i}' for i in range(320)]
        self.all_feature_names = self.ligand_feature_names + self.protein_feature_names
        
        # Convert PyTorch model to function for LIME
        def predict_fn(input_array):
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_array).to(self.device)
                ligand_part = input_tensor[:, :1024]
                protein_part = input_tensor[:, 1024:]
                predictions = self.model(ligand_part, protein_part)
                return predictions.cpu().numpy()
        
        self.predict_fn = predict_fn
        
        # Initialize LIME explainer with sample training data
        print("Initializing LIME explainer...")
        sample_training_data = np.random.randn(100, 1024 + 320)  # Sample training data
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=sample_training_data,
            feature_names=self.all_feature_names,
            mode='regression',
            discretize_continuous=True,
            random_state=42
        )
        print("LIME explainer initialized successfully!")
    
    def explain_instance(self, ligand_features, protein_features, instance_idx=0, num_features=20):
        """Explain a single instance"""
        # Combine ligand and protein features
        combined_features = np.concatenate([ligand_features, protein_features], axis=1)
        
        # Select specific instance
        instance = combined_features[instance_idx]
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance,
            self.predict_fn,
            num_features=num_features,
            top_labels=1
        )
        
        return explanation
    
    def get_feature_importance(self, explanation):
        """Extract feature importance from LIME explanation"""
        features, values = zip(*explanation.as_list())
        return dict(zip(features, values))
    
    def visualize_explanation(self, explanation, title="LIME Feature Importance"):
        """Visualize LIME explanation"""
        features, values = zip(*explanation.as_list())
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if v < 0 else 'blue' for v in values]
        bars = plt.barh(range(len(features)), values, color=colors, alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Contribution to Prediction')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, v in enumerate(values):
            plt.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', 
                    va='center', ha='left' if v >= 0 else 'right')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_ligand_protein_importance(self, explanation):
        """Analyze and compare ligand vs protein feature importance"""
        feature_importance = self.get_feature_importance(explanation)
        
        ligand_importance = {}
        protein_importance = {}
        
        for feature, value in feature_importance.items():
            if feature.startswith('ligand_'):
                ligand_importance[feature] = value
            elif feature.startswith('protein_'):
                protein_importance[feature] = value
        
        # Calculate total importance
        total_ligand_importance = sum(abs(v) for v in ligand_importance.values())
        total_protein_importance = sum(abs(v) for v in protein_importance.values())
        
        # Visualize comparison
        plt.figure(figsize=(10, 6))
        categories = ['Ligand Features', 'Protein Features']
        values = [total_ligand_importance, total_protein_importance]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.7)
        plt.ylabel('Total Absolute Contribution')
        plt.title('Ligand vs Protein Feature Importance')
        
        # Add value annotations
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return {
            'ligand_importance': total_ligand_importance,
            'protein_importance': total_protein_importance,
            'ligand_features': ligand_importance,
            'protein_features': protein_importance
        }
    
    def get_top_features(self, explanation, n=10):
        """Get top contributing features"""
        feature_importance = self.get_feature_importance(explanation)
        
        # Sort by absolute importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_features[:n]
    
    def analyze_multiple_instances(self, ligand_features, protein_features, 
                                 instance_indices, num_features=15):
        """Analyze multiple instances and compare results"""
        results = {}
        
        for idx in instance_indices:
            print(f"Analyzing instance {idx}...")
            explanation = self.explain_instance(
                ligand_features, protein_features, 
                instance_idx=idx, 
                num_features=num_features
            )
            
            feature_importance = self.get_feature_importance(explanation)
            results[idx] = {
                'explanation': explanation,
                'importance': feature_importance,
                'prediction': self.predict_fn(ligand_features[idx:idx+1])[0][0]
            }
        
        return results

def demonstrate_lime_analysis():
    """Demonstrate LIME analysis with sample data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ExplainableDTICNN().to(device)
    
    # Create sample data
    sample_ligand = np.random.randn(10, 1024).astype(np.float32)
    sample_protein = np.random.randn(10, 320).astype(np.float32)
    
    print("Initializing LIME analyzer...")
    lime_analyzer = LIMEXAIAnalyzer(model, device)
    
    # Analyze first instance
    print("Generating explanation for instance 0...")
    explanation = lime_analyzer.explain_instance(
        sample_ligand, 
        sample_protein, 
        instance_idx=0,
        num_features=20
    )
    
    # Visualize the explanation
    lime_analyzer.visualize_explanation(explanation, "LIME Explanation - Instance 0")
    
    # Analyze ligand vs protein importance
    importance_analysis = lime_analyzer.analyze_ligand_protein_importance(explanation)
    
    # Get top contributing features
    top_features = lime_analyzer.get_top_features(explanation, n=10)
    print("\nTop 10 Contributing Features:")
    for i, (feature, value) in enumerate(top_features, 1):
        print(f"{i:2d}. {feature}: {value:.4f}")
    
    # Analyze multiple instances
    print("\nAnalyzing multiple instances...")
    multi_results = lime_analyzer.analyze_multiple_instances(
        sample_ligand, sample_protein, 
        instance_indices=[0, 1, 2, 3, 4], 
        num_features=10
    )
    
    # Compare predictions and feature importance
    print("\nInstance Predictions and Top Features:")
    for idx, result in multi_results.items():
        print(f"\nInstance {idx}: Prediction = {result['prediction']:.4f}")
        top_3 = sorted(result['importance'].items(), 
                      key=lambda x: abs(x[1]), reverse=True)[:3]
        for feature, value in top_3:
            print(f"  {feature}: {value:.4f}")
    
    return lime_analyzer, explanation, importance_analysis

def lime_batch_analysis(model, test_loader, device, n_samples=3):
    """Perform LIME analysis on multiple samples from test loader"""
    lime_analyzer = LIMEXAIAnalyzer(model, device)
    
    all_results = []
    
    for i, (ligand_batch, protein_batch, target_batch) in enumerate(test_loader):
        if i >= n_samples:
            break
            
        print(f"Analyzing batch {i+1}/{min(n_samples, len(test_loader))}")
        
        # Convert to numpy for LIME
        ligand_np = ligand_batch.cpu().numpy()
        protein_np = protein_batch.cpu().numpy()
        
        # Analyze first instance in batch
        explanation = lime_analyzer.explain_instance(
            ligand_np, protein_np, instance_idx=0, num_features=15
        )
        
        # Get importance analysis
        importance_analysis = lime_analyzer.analyze_ligand_protein_importance(explanation)
        
        # Get top features
        top_features = lime_analyzer.get_top_features(explanation, n=10)
        
        result = {
            'batch_index': i,
            'explanation': explanation,
            'importance_analysis': importance_analysis,
            'top_features': top_features,
            'target_value': target_batch[0].item()
        }
        
        all_results.append(result)
    
    return all_results

def plot_comparison_heatmap(lime_analyzer, ligand_features, protein_features, 
                          instance_indices, num_features=10):
    """Create heatmap comparing feature importance across multiple instances"""
    feature_importance_matrix = []
    feature_names_list = []
    
    for idx in instance_indices:
        explanation = lime_analyzer.explain_instance(
            ligand_features, protein_features, 
            instance_idx=idx, 
            num_features=num_features
        )
        
        feature_importance = lime_analyzer.get_feature_importance(explanation)
        
        # Get top features for this instance
        top_features = lime_analyzer.get_top_features(explanation, n=num_features)
        top_feature_names = [feat for feat, val in top_features]
        
        # Create importance vector for this instance
        importance_vector = []
        for feat_name in top_feature_names:
            importance_vector.append(feature_importance[feat_name])
        
        feature_importance_matrix.append(importance_vector)
        feature_names_list.append(top_feature_names)
    
    # Create heatmap
    feature_importance_matrix = np.array(feature_importance_matrix)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(feature_importance_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='RdBu_r', 
                center=0,
                xticklabels=top_feature_names if len(top_feature_names) <= 15 else top_feature_names[:15],
                yticklabels=[f'Instance {idx}' for idx in instance_indices])
    
    plt.title('LIME Feature Importance Heatmap Across Instances')
    plt.xlabel('Top Features')
    plt.ylabel('Instances')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    if not LIME_AVAILABLE:
        print("LIME is not available. Please install it with:")
        print("pip install lime")
        exit(1)
    
    try:
        print("Starting LIME analysis...")
        lime_analyzer, explanation, importance_analysis = demonstrate_lime_analysis()
        
        print("\nLIME analysis completed successfully!")
        print("Key insights:")
        print(f"- Total ligand feature importance: {importance_analysis['ligand_importance']:.4f}")
        print(f"- Total protein feature importance: {importance_analysis['protein_importance']:.4f}")
        
        # Create sample data for heatmap
        sample_ligand = np.random.randn(10, 1024).astype(np.float32)
        sample_protein = np.random.randn(10, 320).astype(np.float32)
        
        print("\nCreating comparison heatmap...")
        plot_comparison_heatmap(lime_analyzer, sample_ligand, sample_protein, 
                              instance_indices=[0, 1, 2, 3, 4], num_features=8)
        
    except Exception as e:
        print(f"Error in LIME analysis: {e}")
        print("Make sure to install LIME: pip install lime")