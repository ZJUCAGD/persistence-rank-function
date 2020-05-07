function [mappedX, mapping] = Supervised_Isomap(X, labels, no_dims, k, alpha)
%ISOMAP Runs the Isomap algorithm
%
%   [mappedX, mapping] = isomap(X, no_dims, k); 
%
% The functions modified from the original matlab isomap tool,
% by using cosine distance instead of euclidean distance

    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
    if ~exist('k', 'var')
        k = 12;
    end
    
    label_eq=(labels==labels');%if l(y_i)==l(y_j)
    N=length(labels);
    
    sum_X = sum(X .^ 2, 2);
    DD = bsxfun(@plus, sum_X', bsxfun(@plus, sum_X,  -2 * (X * X')));
    
    %beta=sum(sum(sqrt(DD)))/(N*N-N);
    beta=sum(sum(DD))/(N*N-N);
    tmp=DD/beta;
    Q=sqrt(1-exp(-tmp)).*label_eq+(sqrt(exp(tmp))-alpha).*(~label_eq);
    Q(Q<1e-12)=0;
    Q=real(Q);   
    [mappedX, mapping] = A_Isomap(Q, no_dims, k);
    mapping.X = X;
    mapping.name='IsoKRR4';
    