
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
import traceback
from torchmetrics import R2Score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# ----------------- Error Handling Utilities -------------------
class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

class DataLoadingError(Exception):
    """Custom exception for data loading failures"""
    pass

class TrainingError(Exception):
    """Custom exception for training failures"""
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

# ----------------- Configuration Loading ----------------------
def load_config(config_path='config.json'):
    """Load configuration with enhanced error handling"""
    try:
        validate_file_path(config_path, "config file")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required configuration parameters
        required_params = {
            'weights_units': (list, "list of integers"),
            'regressor_units': (list, "list of integers"),
            'dropout': ((float, int), "float between 0 and 1"),
            'patience': (int, "positive integer"),
            'test_size': (float, "float between 0 and 1"),
            'random_state': (int, "integer"),
            'lr': (float, "positive float"),
            'weight_decay': (float, "non-negative float"),
            'epochs': (int, "positive integer"),
            'precision': (str, "either 'mixed' or 'full'")
        }
        
        for param, (expected_type, description) in required_params.items():
            if param not in config:
                raise ConfigError(f"Missing required parameter: {param} ({description})")
            if not isinstance(config[param], expected_type):
                raise ConfigError(f"Invalid type for {param}: expected {expected_type.__name__}, got {type(config[param]).__name__}")
            if expected_type == float and config[param] <= 0:
                raise ConfigError(f"Invalid value for {param}: must be positive")
            if param == 'dropout' and not (0 <= config[param] <= 1):
                raise ConfigError("Dropout must be between 0 and 1")
            if param == 'precision' and config[param] not in ['mixed', 'full']:
                raise ConfigError("Precision must be either 'mixed' or 'full'")
                
        return config
        
    except json.JSONDecodeError:
        raise ConfigError(f"Invalid JSON format in config file {config_path}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration: {str(e)}")

# ----------------- Model Definition ---------------------------
class SimpleSNPModel(nn.Module):
    def __init__(self, num_snps, config):
        super().__init__()
        try:
            self.config = config
            if not isinstance(num_snps, int) or num_snps <= 0:
                raise ValueError(f"num_snps must be positive integer, got {num_snps}")
                
            self.attention = self._build_attention_module(num_snps)
            self.regressor = self._build_regressor_module(num_snps)
        except Exception as e:
            raise ValueError(f"Model initialization failed: {str(e)}")

    def _build_attention_module(self, num_snps):
        """Build attention module with error checking"""
        try:
            layers = []
            prev_size = num_snps
            for i, h_size in enumerate(self.config['weights_units']):
                if not isinstance(h_size, int) or h_size <= 0:
                    raise ValueError(f"Invalid hidden size {h_size} in attention layer {i}")
                layers.append(nn.Linear(prev_size, h_size))
                if i < len(self.config['weights_units']) - 1:
                    layers.append(nn.GELU())
                prev_size = h_size
            layers.append(nn.Linear(prev_size, num_snps))
            layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)
        except Exception as e:
            raise ValueError(f"Attention module construction failed: {str(e)}")

    def _build_regressor_module(self, num_snps):
        """Build regressor module with error checking"""
        try:
            layers = []
            prev_size = num_snps
            for i, h_size in enumerate(self.config['regressor_units']):
                if not isinstance(h_size, int) or h_size <= 0:
                    raise ValueError(f"Invalid hidden size {h_size} in regressor layer {i}")
                layers.append(nn.Linear(prev_size, h_size))
                if i < len(self.config['regressor_units']) - 1:
                    layers.append(nn.LayerNorm(h_size))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(self.config['dropout']))
                prev_size = h_size
            layers.append(nn.Linear(prev_size, 1))
            return nn.Sequential(*layers)
        except Exception as e:
            raise ValueError(f"Regressor module construction failed: {str(e)}")

    def forward(self, x):
        """Forward pass with dimension checking"""
        try:
            if x.dim() != 2:
                raise ValueError(f"Input must be 2D tensor, got {x.dim()}D")
                
            pre_sigmoid_weights = self.attention[:-1](x) 
            att_weights = self.attention(x) 
            weighted = x * att_weights + x  # Residual connection
            return self.regressor(weighted).squeeze(), pre_sigmoid_weights
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}")

# ----------------- Early Stopping -----------------------------
class EarlyStopping:
    def __init__(self, config, patience=None, delta=None, verbose=True):
        """Initialize with validation"""
        try:
            self.patience = patience if patience is not None else config['patience']
            self.delta = delta if delta is not None else config['delta']
            self.verbose = verbose
            
            if not isinstance(self.patience, int) or self.patience <= 0:
                raise ValueError("Patience must be positive integer")
            if not isinstance(self.delta, (float, int)) or self.delta < 0:
                raise ValueError("Delta must be non-negative number")
                
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.best_r2 = -np.Inf
            self.best_pcc = -np.Inf
        except Exception as e:
            raise ValueError(f"Early stopping initialization failed: {str(e)}")

    def __call__(self, val_loss, model, r2=None, pcc=None, path='./'):
        """Check early stopping conditions"""
        try:
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
        except Exception as e:
            raise RuntimeError(f"Early stopping check failed: {str(e)}")

    def save_checkpoint(self, val_loss, model, path, col):
        """Save model with error handling"""
        try:
            save_path = os.path.join(path, f'best_model_col_{col}.pth')
            torch.save(model.state_dict(), save_path)
            if not os.path.exists(save_path):
                raise RuntimeError(f"Failed to verify saved model at {save_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model checkpoint: {str(e)}")

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

def load_data(snp_path, pheno_path, col, config):
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
            phenotype_df = phenotype_series.apply(lambda x: pd.to_numeric(x, errors='coerce'))
            
            if phenotype_df.isna().all():
                raise DataLoadingError(f"All values in phenotype column {col} are invalid")
                
            fill_value = phenotype_df.mean()
            phenotype_values = phenotype_df.fillna(fill_value).values.astype(np.float32)
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.FloatTensor(snp_scaled).to(device)
        y = torch.FloatTensor(phenotype_values).to(device)

        return train_test_split(
            X, y, 
            test_size=config['test_size'], 
            random_state=config['random_state']
        )
        
    except Exception as e:
        raise DataLoadingError(f"Data loading failed: {str(e)}")

# ----------------- Training Function --------------------------
def train_model(snp_path, pheno_path, result_txt_path, result_pdf_path, col, config):
    """Training process with comprehensive error handling"""
    try:
        # Validate and create output directories
        safe_create_directory(os.path.dirname(result_txt_path))
        safe_create_directory(os.path.dirname(result_pdf_path))
        
        # Get phenotype name for logging
        try:
            pheno_data = pd.read_csv(pheno_path, header=None, sep='\t', 
                                   dtype=str, engine='python', nrows=1)
            phenotype_name = pheno_data.iloc[0, col] if col < pheno_data.shape[1] else f"col_{col}"
        except:
            phenotype_name = f"col_{col}"

        # Initialize logging
        log_header = f"""
{'='*80}
Training CropARNet on: {phenotype_name}
SNP file: {snp_path}
Phenotype file: {pheno_path}
Column index: {col}
Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}"""
        
        with open(result_txt_path, 'a', encoding='utf-8') as f:
            f.write(log_header + '\n')

        # Load data
        X_train, X_test, y_train, y_test = load_data(snp_path, pheno_path, col, config)

        # Initialize model
        model = SimpleSNPModel(num_snps=X_train.shape[1], config=config).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=config['lr'], 
                                    weight_decay=config['weight_decay'])
        early_stopping = EarlyStopping(config)
        r2_metric = R2Score().to(device)
        scaler = GradScaler(enabled=config['precision'] == 'mixed')

        # Track metrics
        train_losses, test_losses = [], []
        train_r2, test_r2 = [], []
        train_pcc, test_pcc = [], []

        for epoch in range(config['epochs']):
            try:
                start_time = time.time()
                
                # Training phase
                model.train()
                optimizer.zero_grad()

                with autocast(enabled=config['precision'] == 'mixed'):
                    outputs, _ = model(X_train)
                    loss = criterion(outputs, y_train)

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
                    test_pcc_value = pearsonr(test_pred.cpu().numpy().flatten(), 
                                            y_test.cpu().numpy().flatten())[0]

                    early_stopping(test_loss.item(), model, 
                                 r2=test_r2_value, pcc=test_pcc_value,
                                 path=os.path.dirname(result_txt_path), col=col)

                    if early_stopping.early_stop:
                        end_time = time.time()
                        log_message = f"""
{'='*80}
Early stopping triggered at epoch {epoch+1}
Best Test Loss: {early_stopping.val_loss_min:.4f}
Best R²: {early_stopping.best_r2:.4f}
Best PCC: {early_stopping.best_pcc:.4f}
Training time: {end_time - start_time:.2f} seconds
{'='*80}"""
                        print(log_message)
                        with open(result_txt_path, 'a') as f:
                            f.write(log_message + '\n\n')
                        break

                    test_losses.append(test_loss.item())
                    test_r2.append(test_r2_value)
                    test_pcc.append(test_pcc_value)

                # Training metrics
                with torch.no_grad():
                    train_losses.append(loss.item())
                    train_r2_value = r2_metric(outputs, y_train).item()
                    train_pcc_value = pearsonr(outputs.cpu().numpy().flatten(), 
                                             y_train.cpu().numpy().flatten())[0]
                    train_r2.append(train_r2_value)
                    train_pcc.append(train_pcc_value)
                    r2_metric.reset()

                # Periodic logging
                if (epoch+1) % 20 == 0:
                    epoch_message = f"""Epoch {epoch+1:03d} | 
Train Loss: {loss.item():.4f} | Train R²: {train_r2[-1]:.4f} | Train PCC: {train_pcc[-1]:.4f}
Test Loss: {test_loss.item():.4f} | Test R²: {test_r2[-1]:.4f} | Test PCC: {test_pcc[-1]:.4f}"""
                    print(epoch_message)
                    with open(result_txt_path, 'a') as f:
                        f.write(epoch_message + '\n')

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    error_msg = "CUDA out of memory - try reducing batch size or model complexity"
                    print(error_msg)
                    with open(result_txt_path, 'a') as f:
                        f.write(f"\nERROR: {error_msg}\n")
                    raise
                else:
                    raise
            except Exception as e:
                error_msg = f"Error during epoch {epoch+1}: {str(e)}"
                print(error_msg)
                with open(result_txt_path, 'a') as f:
                    f.write(f"\nERROR: {error_msg}\n")
                raise

        # Save final outputs
        try:
            # Save attention weights
            att_weights_path = result_txt_path.replace(".txt", f"_att_weights_{phenotype_name}.csv")
            pd.DataFrame(test_att_weights.cpu().detach().numpy()).to_csv(att_weights_path, index=False)
            
            # Save final model
            final_model_path = result_txt_path.replace(".txt", f"_final_model_col_{col}.pth")
            torch.save(model.state_dict(), final_model_path)
            
            # Plot metrics
            plot_metrics(train_losses, test_losses, train_r2, test_r2, train_pcc, test_pcc, 
                        result_pdf_path, phenotype_name)
                        
        except Exception as e:
            raise TrainingError(f"Failed to save outputs: {str(e)}")

    except Exception as e:
        raise TrainingError(f"Training failed: {str(e)}")

# ----------------- Visualization -----------------------------
def plot_metrics(train_losses, test_losses, train_r2, test_r2, train_pcc, test_pcc, 
                result_pdf_path, phenotype_name):
    """Generate and save metric plots with error handling"""
    try:
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
        
        plt.suptitle(f"Training Metrics - {phenotype_name}")
        plt.tight_layout()
        
        plot_path = result_pdf_path.replace(".pdf", f"_{phenotype_name}.pdf")
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate plots: {str(e)}")

# ----------------- Main Execution ----------------------------
def main():
    """Main execution with top-level error handling"""
    try:
        # Parse arguments with validation
        parser = argparse.ArgumentParser(
            description='CropARNet: Genomic Prediction with Attention and Residual Networks',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--snp_path', type=str, required=True,
                          help='Path to SNP data file (tab-separated)')
        parser.add_argument('--pheno_path', type=str, required=True,
                          help='Path to phenotype data file (tab-separated)')
        parser.add_argument('--result_dir', type=str, required=True,
                          help='Directory to save results')
        parser.add_argument('--start_col', type=int, default=1,
                          help='Starting phenotype column index (1-based)')
        parser.add_argument('--end_col', type=int, required=True,
                          help='Ending phenotype column index (1-based)')
        parser.add_argument('--config', type=str, default='config.json',
                          help='Path to JSON config file')
        
        args = parser.parse_args()
        
        # Validate arguments
        if args.start_col < 1:
            raise ValueError("Start column must be ≥1")
        if args.start_col > args.end_col:
            raise ValueError("Start column cannot be greater than end column")
            
        # Load configuration
        config = load_config(args.config)
            
        # Setup device
        global device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create result directories
        safe_create_directory(args.result_dir)
        txt_dir = os.path.join(args.result_dir, 'txt_results')
        pdf_dir = os.path.join(args.result_dir, 'pdf_plots')
        safe_create_directory(txt_dir)
        safe_create_directory(pdf_dir)
            
        # Process each phenotype column
        for col in range(args.start_col, args.end_col + 1):
            print(f"\n{'='*80}")
            print(f"Processing phenotype column {col}")
            print('='*80)
            
            try:
                # Generate timestamped filenames
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = f"phenotype_col_{col}_{timestamp}"
                result_txt_path = os.path.join(txt_dir, f"{base_name}.txt")
                result_pdf_path = os.path.join(pdf_dir, f"{base_name}.pdf")
                
                # Run training
                train_model(
                    args.snp_path, args.pheno_path, 
                    result_txt_path, result_pdf_path, 
                    col, config
                )
                
            except Exception as e:
                print(f"\nERROR processing column {col}: {str(e)}")
                print(traceback.format_exc())
                continue
                
        print("\nProcessing completed")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
