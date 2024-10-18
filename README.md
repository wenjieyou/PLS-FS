# PLS-based Feature Selection (plsfs)

This Python package provides Partial Least Squares (PLS) based feature selection methods for high-dimensional data. The package includes four main algorithms:

1. **PLS-based feature ranker for classification (plsfrc)**: A weak selector for feature ranking.
2. **PLS-based recursive feature elimination for classification (plsrfec)**: A robust feature selector for classification tasks.
3. **PLS-based local recursive feature elimination for classification (plslrfec)**: Suitable for multi-category data.
4. **PLS-based multi-perturbations ensemble gene selection (mpegs_pls)**: Generates diverse feature subsets, identifying weak signals genes. It can be further expanded into PLS-based gene subset augmentation (PLSGSA).

## Usage

The package provides a simple interface for feature selection. Here's a brief overview of each method:

### 1. PLS-based Feature Ranker for Classification (plsfrc)

```python
rank_feat, vip_feat = plsfrc(trn, ytrn, nfac)
```

### 2. PLS-based Recursive Feature Elimination for Classification (plsrfec)

```python
rfe_ind = plsrfec(trn, ytrn, nfac=4)
```

### 3. PLS-based Local Recursive Feature Elimination for Classification (plslrfec)

```python
rfe_ind = plslrfec(trn, ytrn)
```

### 4. PLS-based multi-Perturbations Ensemble Gene Selection (mpegs_pls)

```python
idx_feat, w_feat = mpegs_pls(trn, ytrn, nit=2000, nfac=2)
```

## Example

```python
from scipy.io import loadmat

data = loadmat('dat/SRBCT.mat')
trn = data['trn']
ytrn = data['ytrn']

# Feature selector (test our algorithm)
rank_feat, vip_feat = plsfrc(trn, ytrn, 2)
```

## References

1. W. You, Z. Yang, and G. Ji, "PLS-based gene subset augmentation and tumor-specific gene identification," Computers in Biology and Medicine, Volume 174, 2024, 108434, [Link](https://doi.org/10.1016/j.compbiomed.2024.108434)
2. W. You, Z. Yang, and G. Ji, "PLS-based Recursive Feature Elimination for High-dimensional Small Sample," Knowledge-Based Systems, Vol. 55, 2014, Pages 15-28, [Link](https://www.sciencedirect.com/science/article/pii/S0950705113003158)
3. W. You, Z. Yang, and G. Ji, Feature selection for high-dimensional multi-category data using PLS-based local recursive feature elimination, Expert Systems with Applications, Vol. 41, Issue 4, Part 1, 2014, Pages 1463-1475, [Link](https://www.sciencedirect.com/science/article/pii/S0957417413006647)
4. G. Ji, Z. Yang, and W. You, PLS-based Gene Selection and Identification of Tumor-Specific Genes, IEEE Transactions on Systems, Man, Cybernetics C, Application Review, vol. 41, no. 6, 2011, pp.830-841, [Link](https://ieeexplore.ieee.org/abstract/document/5607317)

