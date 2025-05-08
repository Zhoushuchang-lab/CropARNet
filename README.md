# SNPWeightNet
a mlp with weight Modulator and Residual connetions
# SNP Weight Network for Phenotype Prediction

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)

An attention-based neural network for predicting phenotypic traits from SNP data with interpretable weights.

## 🚀 Features
- **Attention Mechanism**: Learns SNP importance weights 
- **Residual Connection**: Preserves original SNP information
- **Mixed Precision Training**: Supports FP16/FP32 via `autocast` 
- **Early Stopping**: Prevents overfitting with configurable patience
- **Visualization**: Automatic plotting of training metrics

## 🛠️ Installation
```bash
conda create -n snpnet python=3.8
conda activate snpnet
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pandas matplotlib scikit-learn torchmetrics
