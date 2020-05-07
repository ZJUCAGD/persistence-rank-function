function [D, ni] = find_NN_supervised(Q, k)
%FIND_NN Finds k nearest neigbors for all datapoints in the dataset
% using cosine distance
%
% This file is modified from the Matlab Toolbox for Dimensionality Reduction
% [D, ni] = find_nn(X, k)
% (C) Laurens van der Maaten, Delft University of Technology
 
    [Q, ind] = sort(abs(Q), 2, 'ascend');
    D = Q(:,2:k + 1);
    ni = ind(:,2:k + 1);
    
    D(D == 0) = 1e-9;
    
    
    n = size(Q, 1);
    Dout = sparse(n, n);
    idx = repmat(1:n, [1 k])';
    Dout(sub2ind([n, n], idx,   ni(:))) = D;
    Dout(sub2ind([n, n], ni(:), idx))   = D;
    D = Dout;



    