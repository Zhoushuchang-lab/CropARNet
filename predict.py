# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:51:26 2025

@author: admin
"""

# -*- coding: utf-8 -*-
"""
SNP Weight Network Prediction Script
@author: admin
"""

import torch
import numpy as np
import pandas as pd
from torchmetrics import R2Score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import argparse
import os

# 保持与训练代码相同的配置
config = {
    'batch_size': 64,
    'weights_units': (64, 32),
    'regressor_units': (64, 32),
    'dropout': 0.3,
}

# ----------------- 模型定义 (与训练代码相同) -------------------
class SimpleSNPModel(torch.nn.Module):
    def __init__(self, num_snps):
        super().__init__()
        self.attention = self._build_attention_module(num_snps)
        self.regressor = self._build_regressor_module(num_snps)

    def _build_attention_module(self, num_snps):
        layers = []
        prev_size = num_snps
        weights_units = config['weights_units']
        for i, h_size in enumerate(weights_units):
            layers.append(torch.nn.Linear(prev_size, h_size))
            if i < len(weights_units) - 1:
                layers.append(torch.nn.GELU())
            prev_size = h_size
        layers.append(torch.nn.Linear(prev_size, num_snps))
        layers.append(torch.nn.Sigmoid())
        return torch.nn.Sequential(*layers)

    def _build_regressor_module(self, num_snps):
        layers = []
        prev_size = num_snps
        regressor_units = config['regressor_units']
        for i, h_size in enumerate(regressor_units):
            layers.append(torch.nn.Linear(prev_size, h_size))
            if i < len(regressor_units) - 1:
                layers.append(torch.nn.LayerNorm(h_size))
                layers.append(torch.nn.GELU())
                layers.append(torch.nn.Dropout(config['dropout']))
            prev_size = h_size
        layers.append(torch.nn.Linear(prev_size, 1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        pre_sigmoid_weights = self.attention[:-1](x) 
        att_weights = self.attention(x) 
        weighted = x * att_weights + x  # 残差连接
        return self.regressor(weighted).squeeze(), pre_sigmoid_weights

# ----------------- 数据预处理 (与训练代码相同) -----------------
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

    X = torch.FloatTensor(snp_scaled)
    y = torch.FloatTensor(phenotype_values)

    return X, y,  pheno_data .iloc[0, col]

# ----------------- 预测函数 -----------------
def predict(model_path, snp_path, pheno_path, result_dir, col):
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载数据
    X, y, phenotype_name = load_data(snp_path, pheno_path, col)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化模型
    model = SimpleSNPModel(num_snps=X.shape[1]).to(device)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 将数据移动到设备
    X = X.to(device)
    y = y.to(device)
    
    # 进行预测
    with torch.no_grad():
        predictions, att_weights = model(X)
    
    # 计算指标
    r2_metric = R2Score().to(device)
    r2 = r2_metric(predictions, y).item()
    pcc = pearsonr(predictions.cpu().numpy().flatten(), y.cpu().numpy().flatten())[0]
    
    # 准备结果数据
    results_df = pd.DataFrame({
        'True_Value': y.cpu().numpy(),
        'Predicted_Value': predictions.cpu().numpy(),
        'Absolute_Error': torch.abs(predictions - y).cpu().numpy()
    })
    
    # 保存结果
    result_file = os.path.join(result_dir, f'predictions_col_{col}_{phenotype_name}.csv')
    results_df.to_csv(result_file, index=False)
    
    # 保存注意力权重
    att_weights_file = os.path.join(result_dir, f'att_weights_col_{col}_{phenotype_name}.csv')
    pd.DataFrame(att_weights.cpu().numpy()).to_csv(att_weights_file, index=False)
    
    # 打印并保存评估结果
    evaluation_results = f"""
============================================================
Prediction Results for {phenotype_name} (Column {col})
============================================================
R² Score: {r2:.4f}
Pearson Correlation Coefficient (PCC): {pcc:.4f}
============================================================

"""
    print(evaluation_results)
    
    # 将评估结果保存到文本文件
    eval_file = os.path.join(result_dir, f'evaluation_col_{col}_{phenotype_name}.txt')
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(evaluation_results)
    
    return results_df, att_weights

def main():
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
    
    # 运行预测
    predict(args.model_path, args.snp_path, args.pheno_path, args.result_dir, args.col)

if __name__ == "__main__":
    main()