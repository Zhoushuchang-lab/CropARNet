
# SNP Weight Network for Phenotype Prediction

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)

An attention-based neural network for predicting phenotypic traits from SNP data with interpretable weights.

## 🚀 Features
- **Attention Mechanism**: Learns SNP importance weights 
- **Residual Connection**: Preserves original SNP information
- **Mixed Precision Training**: Supports FP16/FP32 via `autocast` 
- **Early Stopping**: Prevents overfitting with configurable patience
- **Visualization**: Automatic plotting of training metrics

## 🛠️ Installation
We provided a pre-packaged Conda environment for directly running SNPWeightNet.

```bash
conda env create -f environment.yml
```

## Run SNPWeightNet
We provide a step-by-step guide for running SNPWeightNet.

### step1: Train model
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

#### Command
```bash
python predict.py \
    --model_path path_to_trained_model.pth \
    --snp_path path_to_your_snp.tsv \
    --pheno_path path_to_your_pheno.tsv \
    --result_dir path_to_save_predictions \
    --col Phenotype_column_index_to_predict(1-based)
```

