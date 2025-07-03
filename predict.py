# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 13:05:11 2025

@author: admin
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import os
from torchmetrics import R2Score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler

def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# ----------------- Model Definition -------------------
class SimpleSNPModel(nn.Module):
    def __init__(self, num_snps, config):
        """Initialize model with attention and regressor modules"""
        super().__init__()
        self.config = config
        self.attention = self._build_attention_module(num_snps)
        self.regressor = self._build_regressor_module(num_snps)

    def _build_attention_module(self, num_snps):
        """Build attention module with configurable architecture"""
        layers = []
        prev_size = num_snps
        weights_units = self.config['weights_units']
        for i, h_size in enumerate(weights_units):
            layers.append(nn.Linear(prev_size, h_size))
            if i < len(weights_units) - 1:
                layers.append(nn.GELU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, num_snps))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _build_regressor_module(self, num_snps):
        """Build regressor module with configurable architecture"""
        layers = []
        prev_size = num_snps
        regressor_units = self.config['regressor_units']
        for i, h_size in enumerate(regressor_units):
            layers.append(nn.Linear(prev_size, h_size))
            if i < len(regressor_units) - 1:
                layers.append(nn.LayerNorm(h_size))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.config['dropout']))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass with attention weights and residual connection"""
        pre_sigmoid_weights = self.attention[:-1](x) 
        att_weights = self.attention(x) 
        weighted = x * att_weights + x  # Residual connection
        return self.regressor(weighted).squeeze(), pre_sigmoid_weights    

# ----------------- Data Preprocessing (same as training code) -----------------
def preprocess_snp(snp_matrix):
    """Handle missing SNP values and normalize"""
    mask = snp_matrix == -1
    snp_matrix[mask] = np.random.choice([0, 1, 2], size=np.sum(mask))
    return (snp_matrix - 1.0) / 1.0

def load_data(snp_path, pheno_path, col):
    """Load and preprocess SNP and phenotype data"""
    # Read SNP data
    snp_data = pd.read_csv(snp_path, header=None, sep='\t', dtype=str, low_memory=False)
    snp_df = snp_data.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
    snp_df = snp_df.fillna(-1)
    snp_values = snp_df.values.astype(np.float32)

    # Read phenotype data
    pheno_data = pd.read_csv(pheno_path, header=None, sep='\t', dtype=str, engine='python')
    phenotype_series = pheno_data.iloc[1:, col].copy()
    phenotype_df = phenotype_series.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    fill_value = phenotype_df.mean()
    phenotype_values = phenotype_df.fillna(fill_value).values.astype(np.float32)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    snp_scaled = scaler.fit_transform(snp_values)

    X = torch.FloatTensor(snp_scaled)
    y = torch.FloatTensor(phenotype_values)

    return X, y, pheno_data.iloc[0, col]

# ----------------- Prediction Function -----------------
def predict(model_path, snp_path, pheno_path, result_dir, col):
    """Make predictions using trained model"""
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    # Load data
    X, y, phenotype_name = load_data(snp_path, pheno_path, col)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleSNPModel(num_snps=X.shape[1]).to(device)
    
    # Load trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Move data to device
    X = X.to(device)
    y = y.to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions, att_weights = model(X)
    
    # Calculate metrics
    r2_metric = R2Score().to(device)
    r2 = r2_metric(predictions, y).item()
    pcc = pearsonr(predictions.cpu().numpy().flatten(), y.cpu().numpy().flatten())[0]
    
    # Prepare result data
    results_df = pd.DataFrame({
        'True_Value': y.cpu().numpy(),
        'Predicted_Value': predictions.cpu().numpy(),
        'Absolute_Error': torch.abs(predictions - y).cpu().numpy()
    })
    
    # Save results
    result_file = os.path.join(result_dir, f'predictions_col_{col}_{phenotype_name}.csv')
    results_df.to_csv(result_file, index=False)
    
    # Save attention weights
    att_weights_file = os.path.join(result_dir, f'att_weights_col_{col}_{phenotype_name}.csv')
    pd.DataFrame(att_weights.cpu().numpy()).to_csv(att_weights_file, index=False)
    
    # Print and save evaluation results
    evaluation_results = f"""
============================================================
Prediction Results for {phenotype_name} (Column {col})
============================================================
R² Score: {r2:.4f}
Pearson Correlation Coefficient (PCC): {pcc:.4f}
============================================================

"""
    print(evaluation_results)
    
    # Save evaluation results to text file
    eval_file = os.path.join(result_dir, f'evaluation_col_{col}_{phenotype_name}.txt')
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(evaluation_results)
    
    return results_df, att_weights

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='SNP Weight Network Prediction')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model (.pth file)')
    parser.add_argument('--snp_path', type=str, required=True, 
                       help='Path to SNP data file')
    parser.add_argument('--pheno_path', type=str, required=True, 
                       help='Path to phenotype data file')
    parser.add_argument('--result_dir', type=str, required=True, 
                       help='Directory to save prediction results')
    parser.add_argument('--col', type=int, required=True, 
                       help='Phenotype column index (1-based) to predict')
    
    args = parser.parse_args()
    
    # Run prediction
    predict(args.model_path, args.snp_path, args.pheno_path, args.result_dir, args.col)

if __name__ == "__main__":
    main()
