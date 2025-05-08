
# SNP Weight Network for Phenotype Prediction

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)

An attention-based neural network for predicting phenotypic traits from SNP data with interpretable weights.

## Features
- **Attention Mechanism**: Learns SNP importance weights 
- **Residual Connection**: Preserves original SNP information
- **Mixed Precision Training**: Supports FP16/FP32 via `autocast` 
- **Early Stopping**: Prevents overfitting with configurable patience
- **Visualization**: Automatic plotting of training metrics

## Installation
We provided a pre-packaged Conda environment for directly running SNPWeightNet.

```bash
conda env create -f environment.yml
```

## Run SNPWeightNet
We provide a step-by-step guide for running SNPWeightNet.

### Step 2: Train model
train your own model

```bash
python train.py --snp_path path_to_your_snp.tsv --pheno_path path_to_your_pheno.tsv --result_dir path_to_save_result --start_col Starting phenotype column index(1-based) --end_col Ending phenotype column index (1-based)
```
you need to provide two input files: SNP.tsv, pheno.tsv


#### Requirements
- **Input Files**:
  - `SNP.tsv`: Tab-separated SNP data file
    - First row: headers
    - First column: sample IDs
    - Data values: 0, 1, 2 (genotypes) or -1 (missing values)
  
  - `pheno.tsv`: Tab-separated phenotype data file
    - First row: headers (phenotype names)
    - First column: sample IDs (must match SNP file)
    - Subsequent columns: phenotype values

- **Output Directory**:
  - Will automatically create two subdirectories:
    - `txt_results/`: Contains model weights (`best_model.pth`) and text results
    - `pdf_plots/`: Contains training visualization plots

#### Examples
1. Train on single phenotype column (column 2):
```bash
python train.py --snp_path data/snps.tsv --pheno_path data/phenos.tsv --result_dir results --start_col 2 --end_col 2
```

2. Train on multiple consecutive columns (columns 1-5):
```bash
python train.py --snp_path data/snps.tsv --pheno_path data/phenos.tsv --result_dir results --start_col 1 --end_col 5
```

### Step 2: Make Predictions (After Training)

```bash
python predict.py --model_path path_to_trained_model.pth  --snp_path path_to_your_snp.tsv  --pheno_path path_to_your_pheno.tsv  --result_dir path_to_save_predictions --col Phenotype_column_index_to_predict(1-based)
```

#### Output Files
- `predictions_col_X_[phenotype_name].csv`: Contains true values, predicted values and absolute errors
- `att_weights_col_X_[phenotype_name].csv`: Attention weights matrix
- `evaluation_col_X_[phenotype_name].txt`: Evaluation metrics (R² and PCC)

## Hyperparameter Tuning Guide

### Default Configuration
```python
config = {
    'batch_size': 64,          # Number of samples per batch
    'weights_units': (64, 32), # Architecture for attention weights network
    'regressor_units': (64, 32), # Architecture for prediction network  
    'dropout': 0.3,           # Dropout rate for regularization
    'lr': 1e-3,               # Learning rate
    'grad_accum': 8,          # Gradient accumulation steps
    'epochs': 500,            # Maximum training epochs
    'precision': 'mixed',     # Training precision (mixed or full)
    'test_size': 0.2,         # Validation set ratio
    'random_state': 42,       # Random seed for reproducibility
    'patience': 50,           # Early stopping patience (epochs)
    'delta': 0.001,           # Minimum improvement threshold
    'weight_decay': 0.1,      # L2 regularization strength
}




### Tuning Recommendations

#### 1. Network Architecture
| Parameter | Recommended Range | Tips |
|-----------|-------------------|------|
| `weights_units` | (32-128, 16-64) | Deeper for complex traits |
| `regressor_units` | (32-128, 16-64) | Match to phenotype complexity |
| `dropout` | 0.2-0.5 | Increase if overfitting |

#### 2. Training Process
| Parameter | Recommended Range | Effect |
|-----------|-------------------|--------|
| `batch_size` | 32-256 | Larger for stable gradients |
| `lr` | 1e-4 to 1e-2 | Use learning rate warmup |
| `grad_accum` | 4-16 | Simulates larger batch size |
| `epochs` | 200-1000 | Monitor early stopping |

#### 3. Regularization
| Parameter | Adjustment Strategy |
|-----------|----------------------|
| `weight_decay` | Increase (0.01-0.5) to prevent overfitting |
| `patience` | Reduce for faster convergence |
| `delta` | 0.0001-0.01 (smaller for precise traits) |


