# -*- coding: utf-8 -*-
"""
Enhanced Prediction Module with Robust Error Handling
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
import traceback
from torchmetrics import R2Score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# ----------------- Error Handling Utilities -------------------
class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

class DataLoadingError(Exception):
    """Custom exception for data loading failures"""
    pass

class PredictionError(Exception):
    """Custom exception for prediction failures"""
    pass

def validate_file_path(path, description):
    """Validate file path exists and is readable"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found at {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read {description} at {path}")
    return True

def safe_create_directory(dir_path):
    """Safely create directory with error handling"""
    try:
        os.makedirs(dir_path, exist_ok=True)
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"No write permissions for {dir_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {dir_path}: {str(e)}")

# ----------------- Data Loading & Preprocessing --------------
def preprocess_snp(snp_matrix):
    """Handle missing SNP values with validation"""
    try:
        if not isinstance(snp_matrix, np.ndarray):
            raise TypeError("Input must be numpy array")
            
        mask = snp_matrix == -1
        if mask.sum() > 0:
            snp_matrix[mask] = np.random.choice([0, 1, 2], size=np.sum(mask))
        return (snp_matrix - 1.0) / 1.0
    except Exception as e:
        raise DataLoadingError(f"SNP preprocessing failed: {str(e)}")

def load_data(snp_path, pheno_path, col):
    """Load and preprocess data with comprehensive validation"""
    try:
        # Validate inputs
        validate_file_path(snp_path, "SNP data file")
        validate_file_path(pheno_path, "phenotype data file")
        
        if not isinstance(col, int) or col < 0:
            raise ValueError("Column index must be non-negative integer")

        # Read SNP data
        try:
            snp_data = pd.read_csv(snp_path, header=None, sep='\t', 
                                  dtype=str, low_memory=False, on_bad_lines='error')
            if snp_data.shape[0] < 2:
                raise DataLoadingError("SNP file has insufficient rows")
                
            snp_df = snp_data.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
            snp_df = snp_df.fillna(-1)
            snp_values = snp_df.values.astype(np.float32)
            
            if snp_values.size == 0:
                raise DataLoadingError("No valid SNP data after preprocessing")
                
            # Preprocess SNP data
            snp_values = preprocess_snp(snp_values)
        except Exception as e:
            raise DataLoadingError(f"SNP data loading failed: {str(e)}")

        # Read phenotype data
        try:
            pheno_data = pd.read_csv(pheno_path, header=None, sep='\t', 
                                   dtype=str, engine='python', on_bad_lines='error')
            if pheno_data.shape[0] < 2:
                raise DataLoadingError("Phenotype file has insufficient rows")
                
            if col >= pheno_data.shape[1]:
                raise IndexError(f"Column index {col} out of bounds for phenotype data")
                
            phenotype_series = pheno_data.iloc[1:, col].copy()
            phenotype_df = phenotype_series.apply(lambda x: pd.to_numeric(x, errors='coerce')
            
            if phenotype_df.isna().all():
                raise DataLoadingError(f"All values in phenotype column {col} are invalid")
                
            fill_value = phenotype_df.mean()
            phenotype_values = phenotype_df.fillna(fill_value).values.astype(np.float32)
            
            # Get phenotype name
            phenotype_name = pheno_data.iloc[0, col] if col < pheno_data.shape[1] else f"col_{col}"
        except Exception as e:
            raise DataLoadingError(f"Phenotype data loading failed: {str(e)}")

        # Validate dimensions
        if snp_values.shape[0] != phenotype_values.shape[0]:
            raise DataLoadingError(
                f"Sample count mismatch: SNPs={snp_values.shape[0]}, phenotypes={phenotype_values.shape[0]}"
            )

        # Normalize data
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            snp_scaled = scaler.fit_transform(snp_values)
        except Exception as e:
            raise DataLoadingError(f"Data normalization failed: {str(e)}")

        # Convert to tensors
        X = torch.FloatTensor(snp_scaled)
        y = torch.FloatTensor(phenotype_values)

        return X, y, phenotype_name
        
    except Exception as e:
        raise DataLoadingError(f"Data loading failed: {str(e)}")

# ----------------- Prediction Function -----------------
def predict(model_path, snp_path, pheno_path, result_dir, col):
    """Make predictions with comprehensive error handling"""
    try:
        # Validate inputs
        validate_file_path(model_path, "model file")
        safe_create_directory(result_dir)
        
        # Load data with validation
        X, y, phenotype_name = load_data(snp_path, pheno_path, col)
        
        # Determine device with fallback
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize model with validation
        try:
            model = SimpleSNPModel(num_snps=X.shape[1]).to(device)
        except Exception as e:
            raise PredictionError(f"Model initialization failed: {str(e)}")
        
        # Load trained model weights with validation
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except Exception as e:
            raise PredictionError(f"Failed to load model weights: {str(e)}")
        
        # Move data to device
        X = X.to(device)
        y = y.to(device)
        
        # Make predictions with validation
        try:
            with torch.no_grad():
                predictions, att_weights = model(X)
                
            if predictions.dim() != 1:
                raise PredictionError(f"Unexpected prediction shape: {predictions.shape}")
            if att_weights.dim() != 2:
                raise PredictionError(f"Unexpected attention weights shape: {att_weights.shape}")
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")
        
        # Calculate metrics with validation
        try:
            r2_metric = R2Score().to(device)
            r2 = r2_metric(predictions, y).item()
            pcc = pearsonr(predictions.cpu().numpy().flatten(), y.cpu().numpy().flatten())[0]
        except Exception as e:
            raise PredictionError(f"Metric calculation failed: {str(e)}")
        
        # Prepare result data
        try:
            results_df = pd.DataFrame({
                'True_Value': y.cpu().numpy(),
                'Predicted_Value': predictions.cpu().numpy(),
                'Absolute_Error': torch.abs(predictions - y).cpu().numpy()
            })
        except Exception as e:
            raise PredictionError(f"Result dataframe creation failed: {str(e)}")
        
        # Save results with validation
        try:
            # Save predictions
            result_file = os.path.join(result_dir, f'predictions_col_{col}_{phenotype_name}.csv')
            results_df.to_csv(result_file, index=False)
            if not os.path.exists(result_file):
                raise PredictionError(f"Failed to verify saved predictions at {result_file}")
            
            # Save attention weights
            att_weights_file = os.path.join(result_dir, f'att_weights_col_{col}_{phenotype_name}.csv')
            pd.DataFrame(att_weights.cpu().numpy()).to_csv(att_weights_file, index=False)
            if not os.path.exists(att_weights_file):
                raise PredictionError(f"Failed to verify saved attention weights at {att_weights_file}")
        except Exception as e:
            raise PredictionError(f"Result saving failed: {str(e)}")
        
        # Generate evaluation report
        evaluation_results = f"""
============================================================
Prediction Results for {phenotype_name} (Column {col})
============================================================
R² Score: {r2:.4f}
Pearson Correlation Coefficient (PCC): {pcc:.4f}
============================================================
"""
        print(evaluation_results)
        
        # Save evaluation results with validation
        try:
            eval_file = os.path.join(result_dir, f'evaluation_col_{col}_{phenotype_name}.txt')
            with open(eval_file, 'w', encoding='utf-8') as f:
                f.write(evaluation_results)
            if not os.path.exists(eval_file):
                raise PredictionError(f"Failed to verify saved evaluation at {eval_file}")
        except Exception as e:
            raise PredictionError(f"Evaluation saving failed: {str(e)}")
        
        return results_df, att_weights
        
    except Exception as e:
        raise PredictionError(f"Prediction pipeline failed: {str(e)}")

# ----------------- Main Execution ----------------------------
def main():
    """Main execution with top-level error handling"""
    try:
        # Parse arguments with validation
        parser = argparse.ArgumentParser(
            description='SNP Weight Network Prediction with Enhanced Error Handling',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
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
        
        # Validate column index
        if args.col < 1:
            raise ValueError("Column index must be ≥1")
        
        # Run prediction with error handling
        try:
            predict(
                args.model_path, 
                args.snp_path, 
                args.pheno_path, 
                args.result_dir, 
                args.col
            )
        except Exception as e:
            print(f"\nERROR during prediction: {str(e)}")
            print(traceback.format_exc())
            sys.exit(1)
            
        print("\nPrediction completed successfully")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
