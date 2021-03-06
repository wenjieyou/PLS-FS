function [ind_feat,rank_feat]=PLSRanking(trn,ytrn,nfac)
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
%   [2] W. You, Z. Yang, and G. Ji, PLS-based Recursive Feature Elimination
%         for High-imensional Small Sample, Knowledge-Based Systems, vol. 55, pp. 15-28, 2014.
%=================================================

if nargin<3 || isempty(nfac)
    nfac=length(unique(ytrn));
end
class_label = unique(ytrn);         %类别标签编码
Y=zeros(size(ytrn,1),size(class_label,1));
for i=1:size(class_label)-1
    Y(:,i) = ytrn == class_label(i);
end
X=trn;
[pctvar,W]=plsreg(X,Y,nfac);
vip=size(X,2)*pctvar(2,:)*(W.^2)'/sum(pctvar(2,:));
[rank_feat, ind_feat]=sort(vip,'descend');
end

function [pctVar,W] = plsreg(X,Y,ncomp)
meanX = mean(X,1);
meanY = mean(Y,1);
X0 = bsxfun(@minus, X, meanX);
Y0 = bsxfun(@minus, Y, meanY);
[Xloadings,Yloadings,Weights] = simpls(X0,Y0,ncomp);
pctVar = [sum(Xloadings.^2,1) ./ sum(sum(X0.^2,1));
    sum(Yloadings.^2,1) ./ sum(sum(Y0.^2,1))];
W = Weights;
end

function [Xloadings,Yloadings,Weights] = simpls(X0,Y0,ncomp)
dx = size(X0,2);
dy = size(Y0,2);
outClass = superiorfloat(X0,Y0);
Xloadings = zeros(dx,ncomp,outClass);
Yloadings = zeros(dy,ncomp,outClass);
Weights = zeros(dx,ncomp,outClass);
V = zeros(dx,ncomp);
Cov = X0'*Y0;
for i = 1:ncomp
    [ri,si,ci] = svd(Cov,'econ'); ri = ri(:,1); ci = ci(:,1); si = si(1);
    ti = X0*ri;
    normti = norm(ti); ti = ti ./ normti;
    Xloadings(:,i) = X0'*ti;
    qi = si*ci/normti;
    Yloadings(:,i) = qi;
    Weights(:,i) = ri ./ normti;
    vi = Xloadings(:,i);
    for repeat = 1:2
        for j = 1:i-1
            vj = V(:,j);
            vi = vi - (vi'*vj)*vj;
        end
    end
    vi = vi ./ norm(vi);
    V(:,i) = vi;
    Cov = Cov - vi*(vi'*Cov);
    Vi = V(:,1:i);
    Cov = Cov - Vi*(Vi'*Cov);
end
end
