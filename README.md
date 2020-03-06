# PLS-FS
Partial least squares based feature selection
%===============================================
%   PLSRanking - PLS-based feature ranking
%===============================================
% @Input
%  TRN  -  training examples
%  YTRN - training labels
%   NFAC - number of  latent variables (factors), NFAC defaults to 'number of categories'.
%   TRN is a data matrix whose rows correspond to points (or observations) and whose
%   columns correspond to features (or predictor variables). YTRN is a column vector
%   of response values or class labels for each observations in TRN.  TRN and YTRN
%   must have the same number of rows.
%
% @Output
%   IND_FEAT  -  indices of columns in TRN ordered by feature importance.
%   RANK_FEAT - feature weights with large positive weights assigned to important feature.
%
%   Code by: Wenjie, You, 2012.07.25
%   wenjie.you@hotmail.com
%
%   Example:
%   load BCdat.mat
%   [ind_feat, rank_feat] = PLSRanking(trn, ytrn, 2);
%================================================
%   Reference:
%   [1] G. Ji, Z. Yang, and W. You, PLS-based Gene Selection and Identification
%         of Tumor-Specific Genes, IEEE Transactions on Systems, Man, Cybernetics C,
%         Application Review, vol. 41, no. 6, pp. 830-841, Nov. 2011.
%
%   [2] W. You, Z. Yang, and G. Ji, PLS-based Recursive Feature Elimination
%         for High-imensional Small Sample, Knowledge-based systems, vol. 55, pp.15-28 2014.
%=================================================


 
