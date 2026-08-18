# Explainable Multimodal Learning for Drug–Target Binding Affinity Prediction

This repository contains the implementation of a drug–target binding affinity prediction framework developed as part of an MRes Artificial Intelligence dissertation project.

The project formulates drug–target interaction prediction as a **supervised regression task**, where experimentally measured inhibition constants (\(K_i\)) are transformed into \(pK_i\) values. It compares conventional machine-learning baselines with a multimodal deep-learning model and applies **SHAP** for model explainability.

---

## Project Overview

The study investigates whether combining molecular and protein representations can improve drug–target binding affinity prediction.

The workflow includes:

- BindingDB data acquisition and preprocessing
- \(K_i\)-only affinity filtering
- Molecular validation and canonicalisation using RDKit
- \(pK_i\) transformation
- Morgan fingerprint generation for ligands
- ESM-2 protein embeddings for conventional ML models
- Integer-encoded protein sequences for the deep-learning model
- Fixed train/validation/test splitting
- Conventional ML baseline training
- Multimodal deep-learning model training
- Regression and ranking evaluation
- SHAP-based global and local explainability

---

## Research Task

The prediction target is continuous drug–target binding affinity expressed as:

\[
pK_i = 9 - \log_{10}(K_i \text{ in nM})
\]

Higher \(pK_i\) values indicate stronger reported binding affinity.

---

## Dataset

The primary dataset is derived from **BindingDB**.

Only records containing valid, exact and positive \(K_i\) measurements are retained. Records are removed when they contain:

- Missing ligand SMILES
- Missing protein sequences
- Missing or non-numeric \(K_i\)
- Censored values such as `<`, `>`, `≤`, or `≥`
- Zero or negative \(K_i\) values
- Invalid SMILES strings
- Unsupported protein sequences
- Duplicate drug–target–affinity records

The final modelling dataset contains:

| Split | Samples | Percentage |
|---|---:|---:|
| Training | 298,715 | 64% |
| Validation | 74,679 | 16% |
| Test | 93,349 | 20% |
| **Total** | **466,743** | **100%** |

Fixed split indices are reused across models to support a consistent comparison.

> Raw BindingDB data are not included in this repository. Download the required dataset directly from BindingDB and run the preprocessing pipeline locally.

---

## Feature Representation

### Ligand Representation

Ligands are represented using:

- **Morgan fingerprints**
- Radius: **2**
- Fingerprint size: **1,024 bits**
- Toolkit: **RDKit**

### Protein Representation for Conventional ML

Protein sequences are represented using:

- **ESM-2**
- Checkpoint: `esm2_t6_8M_UR50D`
- Representation layer: final layer
- Mean pooling
- Output size: **320 dimensions**

The ligand and protein representations are concatenated:

```text
1024 Morgan features + 320 ESM-2 features = 1344 features
```

### Protein Representation for Deep Learning

The deep-learning model uses:

- Integer-encoded amino-acid sequences
- Trainable embedding layer
- One-dimensional convolutional layers
- Batch normalisation
- ReLU activation
- Dropout
- Adaptive max pooling

The ligand and protein branches are fused before final regression.

---

## Models

### Conventional Machine-Learning Baselines

The following models are evaluated:

- Mean Baseline
- Ridge Regression
- Random Forest
- XGBoost
- Approximate RBF Kernel Ridge

### Proposed Deep-Learning Model

The deep-learning model contains two branches.

#### Ligand branch

```text
Morgan Fingerprint
      |
    1024
      |
     512
      |
     256
      |
     128
```

Batch normalisation, ReLU and dropout are applied between hidden layers.

#### Protein branch

```text
Protein Sequence
      |
Integer Encoding
      |
Embedding
      |
1D Convolution
      |
1D Convolution
      |
Adaptive Max Pooling
      |
128-dimensional representation
```

#### Fusion

```text
Ligand Representation
        +
Protein Representation
        |
    Concatenation
        |
 Fully Connected Layers
        |
     pKi Output
```

---

## Training Configuration

The deep-learning model uses:

| Parameter | Value |
|---|---|
| Loss | Huber Loss |
| Huber delta | 1.0 |
| Optimiser | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Dropout | `0.30` |
| Batch size | `16` |
| Maximum epochs | `100` |
| Early stopping patience | `15` |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor | `0.5` |
| Scheduler patience | `5` |
| Gradient clipping | `5.0` |
| Random seed | `42` |

---

## Evaluation Metrics

Models are evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (\(R^2\))
- Pearson correlation
- Concordance Index (CI)
- Training time
- Inference time

---

## Latest Experimental Results

### Conventional Machine-Learning Models

| Model | RMSE | MAE | R² | Pearson | CI |
|---|---:|---:|---:|---:|---:|
| Random Forest | **0.9343** | **0.7241** | **0.6167** | **0.7952** | **0.7952** |
| XGBoost | 1.0043 | 0.7859 | 0.5572 | 0.7564 | 0.7741 |
| Ridge Regression | 1.2512 | 0.9919 | 0.3126 | 0.5591 | 0.6832 |
| Approx. RBF Kernel Ridge | 1.3040 | 1.0408 | 0.2534 | 0.5055 | 0.6612 |
| Mean Baseline | 1.5092 | 1.2073 | ≈0.0000 | 0.0000 | 0.5000 |

Random Forest achieved the strongest conventional baseline performance.

### Deep-Learning Model

| Split | RMSE | MAE | R² | Pearson | CI |
|---|---:|---:|---:|---:|---:|
| Training | 0.5460 | 0.3840 | 0.8683 | 0.9327 | 0.8934 |
| Validation | 0.8032 | 0.5786 | 0.7155 | 0.8495 | 0.8319 |
| Test | **0.8067** | **0.5829** | **0.7142** | **0.8489** | **0.8314** |

The deep-learning model achieved the strongest overall predictive performance under the fixed random split.

---

## Explainability

The project uses **Kernel SHAP** to analyse model behaviour.

Kernel SHAP was selected because the complete deep-learning model combines continuous ligand fingerprint inputs with discrete integer-encoded protein sequence inputs.

The explainability workflow includes:

- Global SHAP feature importance
- Local SHAP waterfall explanations
- Ligand feature attribution
- Protein sequence-position attribution
- Prediction reconstruction checks

The SHAP results indicate that both molecular fingerprint features and protein sequence positions contribute to model predictions.

> SHAP values describe model behaviour and should not be interpreted as evidence of biological causality without additional structural or experimental validation.

---

## Project Structure

```text
Drug-target-interaction-using-machine-learning/
│
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── processed_features/
│
├── artifacts/
│   ├── models/
│   ├── scalers/
│   ├── metrics/
│   ├── plots/
│   └── shap/
│
│
├── src/
│   ├── data/
│   │   ├── loading.py
│   │   ├── preprocessing.py
│   │   └── splitting.py
│   │
│   ├── features/
│   │   ├── ligand.py
│   │   ├── protein.py
|   |   ├── transformer.py
│   │   └── protein_tokenizer.py
│   │
│
├── baselinemodel.ipynb
├── dl.ipynb
├── script_features_vactorization.ipynb
├── script_preprocess_bindingdb.ipynb
├── version.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The exact structure may vary slightly depending on the final repository organisation.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd dti-affinity-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:

```text
numpy
pandas
scikit-learn
xgboost
torch
rdkit
fair-esm
shap
matplotlib
scipy
joblib
```

---

## Running the Pipeline

A typical workflow is:

### 1. Prepare the BindingDB dataset

Place the downloaded BindingDB file inside:

```text
data/raw/
```

### 2. Run preprocessing

The preprocessing stage should:

```text
BindingDB
   ↓
Select required columns
   ↓
Filter Ki measurements
   ↓
Remove censored/invalid values
   ↓
Validate SMILES
   ↓
Validate protein sequences
   ↓
Remove duplicates
   ↓
Convert Ki to pKi
   ↓
Save processed dataset
```

### 3. Generate features

Generate:

- Morgan fingerprints
- ESM-2 embeddings
- Integer-encoded protein sequences

### 4. Generate or load fixed splits

The split indices are stored in:

```text
data/processed_features/fixed_split_indices.npz
```

### 5. Train baseline models

Train and evaluate:

```text
Mean Baseline
Ridge Regression
Random Forest
XGBoost
Approximate RBF Kernel Ridge
```

### 6. Train the deep-learning model

Train using the fixed training and validation indices and retain the checkpoint with the best validation loss.

### 7. Evaluate on the test set

The test set should remain unused during model development and hyperparameter selection.

### 8. Generate SHAP explanations

Run the explainability pipeline after the final model has been selected.

---

## Reproducibility

The project uses several measures to improve reproducibility:

- Fixed random seed: `42`
- Saved train/validation/test indices
- Automated preprocessing
- Saved model checkpoints
- Saved preprocessing reports
- Saved evaluation metrics
- Consistent evaluation functions
- Cached protein embeddings
- Version-controlled source code

Example preprocessing report location:

```text
artifacts/metrics/preprocessing_report.csv
```

Example model checkpoint location:

```text
artifacts/models/
```

---

## Limitations

The current study has several limitations:

1. The principal evaluation uses a fixed random split.
2. Random splitting does not guarantee completely unseen drugs, molecular scaffolds, or protein targets.
3. Protein sequences may require truncation for computational processing.
4. Morgan fingerprints do not explicitly capture 3D ligand structure or ligand–target spatial interactions.
5. BindingDB measurements may contain experimental heterogeneity and noise.
6. SHAP explanations are model-specific and have not been experimentally validated as causal biological mechanisms.

---

## Future Work

Future extensions should include:

- Scaffold-based evaluation
- Cold-drug evaluation
- Cold-target evaluation
- Repeated controlled seeds or partitions
- External dataset validation
- Graph-based molecular representations
- Transformer-based or pretrained protein encoders in the end-to-end model
- Hyperparameter optimisation
- Stronger regularisation
- SHAP stability analysis
- Mapping fingerprint attributions back to explicit molecular substructures
- Mapping protein attributions to structural or experimentally validated residues

---

## Dissertation Context

This repository supports the dissertation:

**Explainable Multimodal Learning for Drug–Target Binding Affinity Prediction Using Molecular and Protein Representations**

The project investigates the use of machine learning, deep learning and explainable AI to support computational prioritisation of drug–target pairs for further experimental investigation.

---

## Data Availability

BindingDB data should be obtained from the official BindingDB source.

This repository should contain only:

- Source code
- Configuration files
- Preprocessing scripts
- Feature-generation scripts
- Model definitions
- Evaluation scripts
- Generated metrics
- Selected figures

Large raw datasets, cached embeddings and model checkpoints may be excluded from Git version control where appropriate.

---

## Author

**Bikas Chaudhary Tharu**  
MRes Artificial Intelligence  
University of Wolverhampton

---


## Disclaimer

This project is intended for academic research purposes. The predictions generated by the models are computational estimates and should not be treated as substitutes for experimental binding assays, biological validation, clinical evidence, or regulatory assessment.
