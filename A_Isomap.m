function [mappedX, mapping] = A_Isomap(Q, no_dims, k)
%ISOMAP Runs the Isomap algorithm
%
%   [mappedX, mapping] = isomap(X, no_dims, k); 
%
% The functions modified from the original matlab isomap tool,
% by using cosine distance instead of euclidean distance
%
% (C) Laurens van der Maaten, Delft University of Technology


    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
    if ~exist('k', 'var')
        k = 12;
    end

    % Construct neighborhood graph
    disp('Constructing neighborhood graph...'); 
    D = real(find_NN_supervised(Q, k));
   
    
    % Select largest connected component
%     count = zeros(1, max(blocks));
%     for i=1:max(blocks)
%         count(i) = length(find(blocks == i));
%     end
%     
%      [count, block_no] = max(count);
%      conn_comp = find(blocks == block_no);    
%      D = D(conn_comp, conn_comp);
    
    D=connect_components(D,Q);
    
    mapping.D = D;
    n = size(D, 1);

    % Compute shortest paths
    disp('Computing shortest paths...');
    D = dijkstra(D, 1:n);
    mapping.DD = D;
    
    % Performing MDS using eigenvector implementation
    disp('Constructing low-dimensional embedding...');
    D = D .^ 2;
    M = -.5 .* (bsxfun(@minus, bsxfun(@minus, D, sum(D, 1)' ./ n), sum(D, 1) ./ n) + sum(D(:)) ./ (n .^ 2));
    M(isnan(M)) = 0;
    M(isinf(M)) = 0;
    [vec, val] = eig(M);
	if size(vec, 2) < no_dims
		no_dims = size(vec, 2);
		warning(['Target dimensionality reduced to ' num2str(no_dims) '...']);
	end
	
    % Computing final embedding
    [val, ind] = sort(real(diag(val)), 'descend'); 
    vec = vec(:,ind(1:no_dims));
    val = val(1:no_dims);
    mappedX = real(bsxfun(@times, vec, sqrt(val)'));
    
    % Store data for out-of-sample extension
    mapping.k = k;
    mapping.vec = vec;
    mapping.val = val;
    mapping.no_dims = no_dims;
    