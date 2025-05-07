# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:06:03 2025
@author: admin
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import R2Score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler
# 超参数配置
config = {
    'batch_size': 64,
    'weights_units': (64, 32),
    'regressor_units': (64, 32),
    'dropout': 0.3,
    'lr': 1e-3,
    'grad_accum': 8,
    'epochs': 500,
    'precision': 'mixed',
    'test_size': 0.2,
    'random_state': 42,
    'patience': 50,
    'delta': 0.001,
    'weight_decay': 0.1,
}

# ----------------- 模型定义 -------------------
class SimpleSNPModel(nn.Module):
    def __init__(self, num_snps):
        super().__init__()
        self.attention = self._build_attention_module(num_snps)
        self.regressor = self._build_regressor_module(num_snps)

    def _build_attention_module(self, num_snps):
        layers = []
        prev_size = num_snps
        weights_units = config['weights_units']
        for i, h_size in enumerate(weights_units):
            layers.append(nn.Linear(prev_size, h_size))
            if i < len(weights_units) - 1:
                layers.append(nn.GELU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, num_snps))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _build_regressor_module(self, num_snps):
        layers = []
        prev_size = num_snps
        regressor_units = config['regressor_units']
        for i, h_size in enumerate(regressor_units):
            layers.append(nn.Linear(prev_size, h_size))
            if i < len(regressor_units) - 1:
                layers.append(nn.LayerNorm(h_size))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config['dropout']))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)


# ----------------- 早停 ----------------------
class EarlyStopping:
    def __init__(self, patience=config['patience'], delta=config['delta'], verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_r2 = -np.Inf
        self.best_pcc = -np.Inf

    def __call__(self, val_loss, model, r2=None, pcc=None, path='./'):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_r2 = r2
            self.best_pcc = pcc
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_r2 = r2
            self.best_pcc = pcc
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path + 'best_model.pth')

# ----------------- 数据预处理 -----------------
def preprocess_snp(snp_matrix):
    mask = snp_matrix == -1
    snp_matrix[mask] = np.random.choice([0, 1, 2], size=np.sum(mask))
    return (snp_matrix - 1.0) / 1.0

def load_data(snp_path, pheno_path, col):
    # 读取SNP数据
    snp_data = pd.read_csv(snp_path, header=None, sep='\t', dtype=str, low_memory=False)
    snp_df = snp_data.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
    snp_df = snp_df.fillna(-1)
    snp_values = snp_df.values.astype(np.float32)

    # 读取表型数据
    pheno_data = pd.read_csv(pheno_path, header=None, sep='\t', dtype=str, engine='python')
    phenotype_series = pheno_data.iloc[1:, col].copy()
    phenotype_df = phenotype_series.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    fill_value = phenotype_df.mean()
    phenotype_values = phenotype_df.fillna(fill_value).values.astype(np.float32)

    # 数据标准化

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    snp_scaled = scaler.fit_transform(snp_values)

    X = torch.FloatTensor(snp_scaled).to(device)
    y = torch.FloatTensor(phenotype_values).to(device)

    return train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])

# ----------------- 训练函数 -----------------
def train_model(snp_path, pheno_path, result_txt_path, result_pdf_path, col):
    X_train, X_test, y_train, y_test = load_data(snp_path, pheno_path, col)
    
    # 获取表型名称
    pheno_data = pd.read_csv(pheno_path, header=None, sep='\t', dtype=str, engine='python')
    phenotype_name = pheno_data.iloc[0, col]

    model = SimpleSNPModel(num_snps=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    early_stopping = EarlyStopping()
    r2_metric = R2Score().to(device)
    scaler = GradScaler(enabled=config['precision'] == 'mixed')

    train_losses, test_losses = [], []
    train_r2, test_r2 = [], []
    train_pcc, test_pcc = [], []

    for epoch in range(config['epochs']):
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

        model.eval()
        with torch.no_grad():
            test_pred, test_att_weights = model(X_test)
            test_loss = criterion(test_pred, y_test)
            test_r2_value = r2_metric(test_pred, y_test).item()
            test_pcc_value = pearsonr(test_pred.cpu().numpy().flatten(), y_test.cpu().numpy().flatten())[0]

            early_stopping(test_loss.item(), model, r2=test_r2_value, pcc=test_pcc_value)

            if early_stopping.early_stop:
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
                
                # 保存注意力权重
                att_weights_path = result_txt_path.replace(".txt", f"_att_weights_{phenotype_name}.csv")
                pd.DataFrame(test_att_weights.cpu().detach().numpy()).to_csv(att_weights_path, index=False)
                print(f"Attention weights saved to {att_weights_path}")
                break

            test_losses.append(test_loss.item())
            test_r2.append(test_r2_value)
            test_pcc.append(test_pcc_value)

        with torch.no_grad():
            train_losses.append(loss.item())
            train_r2_value = r2_metric(outputs, y_train).item()
            train_pcc_value = pearsonr(outputs.cpu().numpy().flatten(), y_train.cpu().numpy().flatten())[0]
            train_r2.append(train_r2_value)
            train_pcc.append(train_pcc_value)
            r2_metric.reset()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1:03d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')

    # 可视化
    def plot_metrics():
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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    snp_path = r"D:\doc\SNPWeightNet\01data\Expression\20DPA_eQTL_PCG.tsv"
    pheno_path = r"D:\doc\SNPWeightNet\01data\T_20DPA.tsv"
    result_txt_path = r"D:\doc\SNPWeightNet\03result\expression_20dpa\result.txt"
    result_pdf_path = r"D:\doc\SNPWeightNet\03result\expression_20dpa\training_metrics.pdf"

    for col in range(1, 82):  # 假设处理第1到第9列表型
        start_time = time.time()
        train_model(snp_path, pheno_path, result_txt_path, result_pdf_path, col)
        end_time = time.time()
        with open(result_txt_path, 'a', encoding='utf-8') as f:
            f.write(f"Training time for phenotype {col}: {end_time - start_time:.2f} seconds\n\n")
