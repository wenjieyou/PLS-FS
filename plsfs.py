# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:42:54 2022

PLS-based feature selection (plsfs)
    1. PLS-based feature ranking algorithm (weak selector, plsfrc)
    2. PLS-based recursive feature elimination (plsrfec)
    3. PLS-based local recursive feature elimination (plslrfec, suitable for multi-category data)
    4. Multi-perturbations ensemble feature selection (mpegs_pls)
    The result returns the ensemble pls-vip value (evip) for each variable. Note: there are two methods: by weight and by rank.
    Sampling follows the class distribution in the population.

@author: wenjie
"""

import numpy as np
import scipy.io as sio
from sklearn.utils.validation import check_random_state
from sklearn.utils.extmath import randomized_svd
import random
np.seterr(divide='ignore',invalid='ignore')


def mpegs_pls(dat, ylab, nit, nfac):
    """
     PLS-based Multi-perturbations Ensemble Gene Selection (samples with replacement, local feature sampling).
     The result returns the ensemble vip value (evip) for each variable.
     Sampling follows the class distribution in the population.

    Parameters:
    dat (np.ndarray): Examples. Rows correspond to observations, columns to features.
    ylab (np.ndarray): Labels. A column vector of response values or class labels.
    nit (int): Number of resampling.   
    nfac (int): Number of latent variables (factors). 

    Returns:     
    IDX_FEAT  -  indices of columns in TRN ordered by feature importance.
    W_FEAT - feature weights with large positive weights assigned to important feature.

    Reference:
    W. You, Z. Yang, and G. Ji, PLS-based gene subset augmentation and tumor-specific gene 
    identification, Computers in Biology and Medicine, Volume 174, 2024, 108434, 
    https://doi.org/10.1016/j.compbiomed.2024.108434..
    """
    s_num, f_num = dat.shape
    D = int(np.sqrt(f_num))
    
    V = np.zeros((1,f_num))   # VIP coefficients for each iteration based on PLS
    nsel = np.ones((1,f_num)) # number of selections (initialized as 1 to avoid zero in denominator)
    
    for k in range(nit):
        # Perturb features, randomly sample from the feature space, with a rate of sqrt()
        f_sel = random.sample(range(f_num),D)
        # f_sel = f_sel.sort()
        dat_tmp = dat[:,f_sel]
        
        # Perturb samples, perform random sampling with replacement from the sample set, 
        class_label = np.unique(ylab)    # Total number of class labels
        s_sel= []
        # Sampling ensures class balance, i.e., the sampled class distribution matches the overall distribution
        for i in range(class_label.shape[0]):
            s_tmp = np.argwhere(ylab.reshape(s_num) == class_label[i])
            s_tmp_num = s_tmp.shape[0]
            sid_tmp = np.random.choice(s_tmp.reshape(s_tmp_num), 
                                       s_tmp_num, replace=True)  # Sampling with replacement
            s_sel = s_sel + sid_tmp.tolist()
            
        X = dat_tmp[s_sel,:]   
        y = ylab[s_sel]
        
        vip = plsvip(X, y, nfac)   # Based on feature weights
        V[:,f_sel] = V[:,f_sel] + vip
        nsel[:,f_sel] = nsel[:,f_sel] + 1
        
    evip = (V/nsel).ravel()   
    w_feat = np.sort(evip)[::-1]
    idx_feat = np.argsort(evip)[::-1]
        
    return idx_feat, w_feat    
  

def plsrfec(trn_data, trn_cls, nfac=None, alpha=0.5):
    """
    PLS-based Recursive Feature Elimination for classification

    Parameters:
    trn_data (np.ndarray): Training examples. Rows correspond to observations, columns to features.
    trn_cls (np.ndarray): Training labels. A column vector of response values or class labels.
    nfac (int, optional): Number of latent variables (factors). Defaults to number of unique classes.
    alpha (float, optional): Reduction factor for feature elimination. Defaults to 0.5.

    Return:  
    rfe_ind (np.ndarray): Indices of columns in trn_data ordered by feature importance.
    
    Code by: Wenjie, You, 2022.09.22    
    
    Reference:
    [1] W. You, Z. Yang, and G. Ji, PLS-based Recursive Feature Elimination
          for High-imensional Small Sample, Knowledge-Based Systems, Vol.55, 
          2014, Pages 15-28,
          (https://www.sciencedirect.com/science/article/pii/S0950705113003158)

    """
    if nfac is None:
        nfac = len(np.unique(trn_cls))
        
    feat_dim = trn_data.shape[1]
    
    # Calling plsfrc, ranking of features    
    ind_feat, _ = plsfrc(trn_data, trn_cls, nfac)
    ind_feat2 = np.copy(ind_feat)
    
    k = feat_dim - 1
    while k >= nfac:
        # Select the current feature indices to keep
        orig_inx = ind_feat[:k] 
        
        # Extract the submatrix of training data with the selected features
        rfe_data = trn_data[:, orig_inx]
        
        # Call plsfrc on the submatrix and re-rank the remaining features
        ind2, _ = plsfrc(rfe_data, trn_cls, nfac)
        
        # Update global feature ranking by modifying ind_feat according to ind2
        for m, inx in enumerate(ind2):
            ind_feat[m] = ind_feat2[inx]
        
        # Copy the updated feature ranking to ind_feat2 for use in the next iteration
        ind_feat2 = np.copy(ind_feat)

        if k >= 100:
            k = round(k * alpha)
        else:
            k -= 1

    return ind_feat


def plslrfec(trn_data, trn_cls, nfac=2):
    """    
    PLSLRFEC - PLS-based local recursive feature elimination for classification
    Based on PLS-OVA-RFE for multi-class feature selection.

    Reference:
    [1] W. You, Z. Yang, and G. Ji, Feature selection for high-dimensional multi-category data using PLS-based local recursive feature elimination,
        Expert Systems with Applications, Vol. 41, Issue 4, Part 1, 2014, Pages 1463-1475,
        (https://www.sciencedirect.com/science/article/pii/S0957417413006647)
    """
    feat_dim = trn_data.shape[1]    
    
    ind_feat, _ = PLS_OVA(trn_data, trn_cls, nfac)
       
    ind_feat2 = np.zeros((feat_dim, 2))
    ind_feat2[:, 0] = np.arange(feat_dim) 
    ind_feat2[:, 1] = ind_feat
    
    k = feat_dim - 1
    
    while k >= nfac:        
        orig_inx = ind_feat[:k]         
        RFE_data = trn_data[:, orig_inx]        
        
        ind2, _ = PLS_OVA(RFE_data, trn_cls, nfac)         

        for m in range(len(ind2)):
            inx = ind2[m]
            ind_feat[m] = ind_feat2[inx, 1]        
        
        ind_feat2[:, 0] = np.arange(feat_dim)
        ind_feat2[:, 1] = ind_feat        
        
        if k >= 100:
            k = round(k * 0.5)
        else:
            k = k - 1
    
    return ind_feat


def PLS_OVA(trn, ytrn, nfac=None):
    """
    PLS_OVA - PLS-based One-Versus-All feature ranking for classification.
        This algorithm is suitable for multi-category classification tasks
    """
    if nfac is None:
        nfac = 2
        
    tmp = ytrn  # Save original ytrn labels
    class_label = np.unique(ytrn)  # Get unique class labels
    rr = np.zeros((len(class_label), trn.shape[1]))  # Initialize matrix to store feature ranking results

    # Perform One-Versus-All strategy for each class label
    for i in range(len(class_label)):
        ytrn[ytrn != class_label[i]] = class_label[i] + 1  # Set non-current class labels to class_label[i]+1
        rank_feat = plsvip(trn, ytrn, nfac)  # Call plsvip to compute feature scores
        rr[i, :] = rank_feat  # Store the results in the rr matrix
        ytrn = tmp  # Restore ytrn labels

    res = np.mean(rr, axis=0)  # Calculate the average score for each feature
    rank_feat = np.sort(res)[::-1]  # Sort the scores in descending order
    ind_feat = np.argsort(res)[::-1]  # Return the sorted feature indices

    return ind_feat, rank_feat

def plsfrc(trn, ytrn, nfac):    
    """
    PLSFRC - PLS-based Feature Ranker for Classification
            
    Parameters
    ----------
   TRN  -  training examples
   YTRN - training labels
   NFAC - number of  latent variables (factors), NFAC defaults to 'number of categories'.
   TRN is a data matrix whose rows correspond to points (or observations) and whose
   columns correspond to features (or predictor variables). YTRN is a column vector
   of response values or class labels for each observations in TRN.  TRN and YTRN
   must have the same number of rows.

   Return:  
   IDX_FEAT  -  indices of columns in TRN ordered by feature importance.
   VIP_FEAT - feature weights with large positive weights assigned to important feature.

   Code by: Wenjie, You, 2022.09.22
   
   Example:       
       data = sio.loadmat('dat/BCdat.mat')
       trn = data['trn']    
       ytrn = data['ytrn']
       rank_feat, vip_feat = plsfrc(trn,ytrn,2)
   
   Reference:
   [1] G. Ji, Z. Yang, and W. You, PLS-based Gene Selection and Identification
         of Tumor-Specific Genes, IEEE Transactions on Systems, Man, Cybernetics C,
         Application Review, vol. 41, no. 6, pp. 830-841, Nov. 2011.
         https://ieeexplore.ieee.org/abstract/document/5607317
   [2] https://github.com/rmarkello/pyls.git

    """
        
    m = ytrn.shape[0]
    
    if nfac is None:
        nfac = np.unique(ytrn).shape[0]
        
    # Encode class labels into dummy variables matrix Y
    class_label = np.unique(ytrn)   
    Y = np.zeros((m, class_label.shape[0]-1),dtype=int)    
    # Convert class labels to dummy variables (0/1)    
    for i in range(class_label.shape[0]-1):
        cls_label_vec = np.tile(class_label[i], m)
        Y[:,i] = (ytrn.reshape(m) == cls_label_vec)
        
    X = trn
    # Call plsreg to get explained variance (pctvar) and weights (W)
    pctvar, W = plsreg(X, Y, nfac); 

    # Calculate VIP (Variable Importance in Projection) weights       
    vip = X.shape[1]*pctvar[1]@((W**2).T)/np.sum(pctvar[1])
    
    # Sort the features based on VIP weights
    vip_feat = np.sort(vip)[::-1]
    idx_feat = np.argsort(vip)[::-1]
        
    # Return sorted feature indices and VIP weights
    return idx_feat, vip_feat

def plsvip(trn, ytrn, nfac):    
    """    
    VIP indicator is calculated using PLS, 
    where ytrn implements category label encoding.
    The result returns vip (vip value for each variable),    
    """    
        
    m = ytrn.shape[0]
    
    if nfac is None:
        nfac = np.unique(ytrn).shape[0]
        
    class_label = np.unique(ytrn)   
    Y = np.zeros((m, class_label.shape[0]-1),dtype=int)    
    for i in range(class_label.shape[0]-1):
        cls_label_vec = np.tile(class_label[i], m)
        Y[:,i] = (ytrn.reshape(m) == cls_label_vec)
        
    X = trn
    pctvar, W = plsreg(X, Y, nfac);        
    vip = X.shape[1]*pctvar[1]@((W**2).T)/np.sum(pctvar[1])
    
    return vip


def plsreg(X, Y, ncomp):   
    
    X0 = (X - X.mean(axis=0, keepdims=True))
    Y0 = (Y - Y.mean(axis=0, keepdims=True))       
    dict0 = simpls(X0, Y0, ncomp)
    x_loadings = dict0['x_loadings']
    y_loadings = dict0['y_loadings']
    W = dict0['x_weights']
    # percent variance explained for both X and Y for all components
    pctvar = [np.sum(x_loadings ** 2, axis=0) / np.sum(X0 ** 2),
              np.sum(y_loadings ** 2, axis=0) / np.sum(Y0 ** 2)
              ]
    return pctvar, W


def simpls(X, Y, n_components=None, seed=0):
    """
    Performs partial least squares regression with the SIMPLS algorithm

    Parameters
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is observations and `B` is features
    Y : (S, T) array_like
        Input data matrix, where `S` is observations and `T` is features
    n_components : int, optional
        Number of components to estimate. If not specified will use the
        smallest of the input X and Y dimensions. Default: None
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed to use for random number generation. Helps ensure reproducibility
        of results. Default: None

    Returns
    -------
    results : dict
        Dictionary with x- and y-loadings / x-weights       

    """

    X, Y = np.asanyarray(X), np.asanyarray(Y)
    if n_components is None:
        n_components = min(len(X) - 1, X.shape[1])

    # center variables and calculate covariance matrix
    X0 = (X - X.mean(axis=0, keepdims=True))
    Y0 = (Y - Y.mean(axis=0, keepdims=True))
    Cov = X0.T @ Y0

    # to store outputs
    x_loadings = np.zeros((X.shape[1], n_components))
    y_loadings = np.zeros((Y.shape[1], n_components))
    x_weights = np.zeros((X.shape[1], n_components))
    V = np.zeros((X.shape[1], n_components))

    for comp in range(n_components):
        ci, si, ri = svd(Cov, n_components=1, seed=seed)
        ti = X0 @ ri
        normti = np.linalg.norm(ti)
        x_weights[:, [comp]] = ri / normti        
        ti /= normti        
        x_loadings[:, [comp]] = X0.T @ ti  # == X0.T @ X0 @ x_weights
        qi = Y0.T @ ti
        y_loadings[:, [comp]] = qi 
        
        vi = x_loadings[:, [comp]]
        for repeat in range(2):
            for j in range(comp):
                vj = V[:, [j]]
                vi = vi - ((vj.T @ vi) * vj)
        vi /= np.linalg.norm(vi)
        V[:, [comp]] = vi
   
        Cov = Cov - (vi @ (vi.T @ Cov))
        Vi = V[:, :comp]
        Cov = Cov - (Vi @ (Vi.T @ Cov))
   
    return dict(
        x_weights=x_weights,
        x_loadings=x_loadings,
        y_loadings=y_loadings,
    )


def svd(crosscov, n_components=None, seed=None):
    """
    Calculates the SVD of `crosscov` and returns singular vectors/values

    Parameters
    ----------
    crosscov : (B, T) array_like
        Cross-covariance (or cross-correlation) matrix to be decomposed
    n_components : int, optional
        Number of components to retain from decomposition
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    U : (B, L) `numpy.ndarray`
        Left singular vectors from singular value decomposition
    d : (L, L) `numpy.ndarray`
        Diagonal array of singular values from singular value decomposition
    V : (J, L) `numpy.ndarray`
        Right singular vectors from singular value decomposition
    """

    seed = check_random_state(seed)
    crosscov = np.asanyarray(crosscov)

    if n_components is None:
        n_components = min(crosscov.shape)
    elif not isinstance(n_components, int):
        raise TypeError('Provided `n_components` {} must be of type int'
                        .format(n_components))

    # run most computationally efficient SVD
    if crosscov.shape[0] <= crosscov.shape[1]:
        U, d, V = randomized_svd(crosscov.T, n_components=n_components,
                                 random_state=seed, transpose=False)
        V = V.T
    else:
        V, d, U = randomized_svd(crosscov, n_components=n_components,
                                 random_state=seed, transpose=False)
        U = U.T

    return U, np.diag(d), V


if __name__ == '__main__':
    
    data = sio.loadmat('dat/SRBCT.mat')
    trn = data['trn']
    ytrn = data['ytrn']
    
    # Feature selector (test our algorithm)
    # Weak feature ranker based on PLS (fast)   
    rank_feat, vip_feat = plsfrc(trn,ytrn,2)
    
    # PLS-based recursive feature elimination (robust)
    # rfe_ind = plsrfec(trn, ytrn, 4)
     
    # 基于PLS局部递归特征消除的特征选择 (适合于多分类)
    # PLS-based local recursive feature elimination (multi-category classification)
    # rfe_ind = plslrfec(trn, ytrn)    
    
    # PLS-based multi-perturbations ensemble gene (feature) selection
    # (generated feature subsets have diversity)
    # idx_feat, w_feat = mpegs_pls(trn,ytrn,2000,2)
