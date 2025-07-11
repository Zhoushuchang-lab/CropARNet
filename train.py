# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:48:03 2025

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

# ----------------- Early Stopping ----------------------
class EarlyStopping:
    def __init__(self, config, patience=None, delta=None, verbose=True):
        """Initialize early stopping tracker"""
        self.patience = patience if patience is not None else config['patience']
        self.delta = delta if delta is not None else config['delta']
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_r2 = -np.Inf
        self.best_pcc = -np.Inf

    def __call__(self, val_loss, model, r2=None, pcc=None, path='./'):
        """Check if training should stop early"""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_r2 = r2
            self.best_pcc = pcc
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_r2 = r2
            self.best_pcc = pcc
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, col):
        """Save model checkpoint"""
        torch.save(model.state_dict(), path + f'best_model{col}.pth')

# ----------------- Data Preprocessing -----------------
def preprocess_snp(snp_matrix):
    """Handle missing SNP values and normalize"""
    mask = snp_matrix == -1
    snp_matrix[mask] = np.random.choice([0, 1, 2], size=np.sum(mask))
    return (snp_matrix - 1.0) / 1.0

def load_data(snp_path, pheno_path, col, config):
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

    X = torch.FloatTensor(snp_scaled).to(device)
    y = torch.FloatTensor(phenotype_values).to(device)

    return train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])

# ----------------- Training Function -----------------
def train_model(snp_path, pheno_path, result_txt_path, result_pdf_path, col, config):
    """Train model on SNP and phenotype data"""
    X_train, X_test, y_train, y_test = load_data(snp_path, pheno_path, col, config)
    
    # Get phenotype name
    pheno_data = pd.read_csv(pheno_path, header=None, sep='\t', dtype=str, engine='python')
    phenotype_name = pheno_data.iloc[0, col]

    model = SimpleSNPModel(num_snps=X_train.shape[1], config=config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    early_stopping = EarlyStopping(config)
    r2_metric = R2Score().to(device)
    scaler = GradScaler(enabled=config['precision'] == 'mixed')

    # Track metrics
    train_losses, test_losses = [], []
    train_r2, test_r2 = [], []
    train_pcc, test_pcc = [], []

    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        optimizer.zero_grad()

        # Mixed precision training
        with autocast(enabled=config['precision'] == 'mixed'):
            outputs, _ = model(X_train)
            loss = criterion(outputs, y_train)

        # Gradient scaling and update
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Validation phase
        model.eval()
        with torch.no_grad():
            test_pred, test_att_weights = model(X_test)
            test_loss = criterion(test_pred, y_test)
            test_r2_value = r2_metric(test_pred, y_test).item()
            test_pcc_value = pearsonr(test_pred.cpu().numpy().flatten(), y_test.cpu().numpy().flatten())[0]

            early_stopping(test_loss.item(), model, r2=test_r2_value, pcc=test_pcc_value)

            if early_stopping.early_stop:
                end_time = time.time()
                log_message = f'''
phenotype: {phenotype_name}
============================================================
Best Test Loss: {early_stopping.val_loss_min:.4f}
Best R²: {early_stopping.best_r2:.4f}
Best PCC: {early_stopping.best_pcc:.4f}
============================================================'''
                print(log_message)
                with open(result_txt_path, 'a', encoding='utf-8') as f:
                    f.write(log_message + '\n\n')
                with open(result_txt_path, 'a', encoding='utf-8') as f:
                    f.write(f"Training time for phenotype column {col}: {end_time - start_time:.2f} seconds\n\n")
                # Save attention weights
                final_model_path = result_txt_path.replace(".txt",  f'final_model_col_{col}.pth')
                torch.save(model.state_dict(), final_model_path)
                print(f"Final model saved to {final_model_path}")
                att_weights_path = result_txt_path.replace(".txt", f"_att_weights_{phenotype_name}.csv")
                pd.DataFrame(test_att_weights.cpu().detach().numpy()).to_csv(att_weights_path, index=False)
                print(f"Attention weights saved to {att_weights_path}")
                break

            test_losses.append(test_loss.item())
            test_r2.append(test_r2_value)
            test_pcc.append(test_pcc_value)

        # Calculate training metrics
        with torch.no_grad():
            train_losses.append(loss.item())
            train_r2_value = r2_metric(outputs, y_train).item()
            train_pcc_value = pearsonr(outputs.cpu().numpy().flatten(), y_train.cpu().numpy().flatten())[0]
            train_r2.append(train_r2_value)
            train_pcc.append(train_pcc_value)
            r2_metric.reset()
        
        # Print progress periodically
        if (epoch+1) % 20 == 0:
            epoch_message=f'''Epoch {epoch+1:03d} | 
            Train Loss: {loss.item():.4f} | Train R²: {train_r2[-1]:.4f} |Train PCC: {train_pcc[-1]:.4f}
            Test Loss: {test_loss.item():.4f}  |Test R²: {test_r2[-1]:.4f}   | Test PCC: {test_pcc[-1]:.4f}'''
            print(epoch_message)

    # Visualization function
    def plot_metrics():
        """Plot training and validation metrics"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.title("Loss Curves")
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(train_r2, label='Train')
        plt.plot(test_r2, label='Test')
        plt.title("R² Scores")
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(train_pcc, label='Train')
        plt.plot(test_pcc, label='Test')
        plt.title("Pearson Correlation")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(result_pdf_path.replace(".pdf", f"_{phenotype_name}.pdf"), 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    plot_metrics()

def main():
    """Main function to handle command line arguments and training loop"""
    parser = argparse.ArgumentParser(description='SNP Weight Network Training')
    parser.add_argument('--snp_path', type=str, required=True, help='Path to SNP data file')
    parser.add_argument('--pheno_path', type=str, required=True, help='Path to phenotype data file')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save results (will create txt/pdf subdirectories)')
    parser.add_argument('--start_col', type=int, default=1, help='Starting phenotype column index (1-based)')
    parser.add_argument('--end_col', type=int, required=True, help='Ending phenotype column index (1-based)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create result directories
    os.makedirs(args.result_dir, exist_ok=True)
    txt_dir = os.path.join(args.result_dir, 'txt_results')
    pdf_dir = os.path.join(args.result_dir, 'pdf_plots')
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model for each phenotype column
    for col in range(args.start_col, args.end_col+1):
        # Generate timestamped result filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"phenotype_col_{col}_{timestamp}"
        
        # Build full file paths
        result_txt_path = os.path.join(txt_dir, f"{base_name}.txt")
        result_pdf_path = os.path.join(pdf_dir, f"{base_name}.pdf")
        
        train_model(args.snp_path, args.pheno_path, result_txt_path, result_pdf_path, col, config)

if __name__ == "__main__":
    main()
