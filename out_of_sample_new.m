function t_point = out_of_sample(point, mapping)

    switch mapping.name
        case 'IsoKRR'
            % Precomputations for speed
            if strcmp(mapping.name, 'IsoKRR')
                invVal = inv(diag(mapping.val));
                [val, index] = sort(mapping.val, 'descend');
                mapping.landmarks = 1:size(mapping.X, 1);
            else
                val = mapping.beta .^ (1 / 2);
                [val, index] = sort(real(diag(val)), 'descend');
            end
            val = val(1:mapping.no_dims);
            meanD1 = mean(mapping.DD .^ 2, 1);
            meanD2 = mean(mean(mapping.DD .^ 2));
            
            % Process all points (notice that in this implementation 
            % out-of-sample points are not used as landmark points)
            points = point;
            t_point = repmat(0, [size(point, 1) mapping.no_dims]);
            for i=1:size(points, 1)
                
                % Compute distance of new sample to training points
                point = points(i,:);
                tD = L2_distance(point', mapping.X');
                [tmp, ind] = sort(tD); 
                tD(ind(mapping.k + 2:end)) = 0;
                tD = sparse(tD);
                tD = dijkstra([0 tD; tD' mapping.D], 1);
                tD = tD(mapping.landmarks + 1) .^ 2;

                % Compute point embedding
                subB = -.5 * (bsxfun(@minus, tD, mean(tD, 2)) - meanD1 - meanD2);
                if strcmp(mapping.name, 'LandmarkIsomap')
                    vec = subB * mapping.alpha * mapping.invVal;
                    vec = vec(:,index(1:mapping.no_dims));
                else
                    vec = subB * mapping.vec * invVal;
                    vec = vec(:,index(1:mapping.no_dims));
                end
                t_point(i,:) = real(vec .* sqrt(val)');
            end
            
        case 'IsoKRR2'
            lambda=1e-4;
            N=size(mapping.X,1);
            tmp=inv(mapping.X*mapping.X'+lambda*eye(N));
            Z=real(bsxfun(@times, mapping.vec, sqrt(mapping.val)'));
            t_point=(Z'*tmp*mapping.X*point')';
        case 'IsoKRR3'
            lambda=1e-4;
            N=size(mapping.X,1);
            Z=real(bsxfun(@times, mapping.vec, sqrt(mapping.val)'));
            
            sqrt_X = sqrt(sum(mapping.X .^ 2, 2)); 
            sqrt_kx=sqrt(sum(point .^ 2, 2)); 
            K=1-(mapping.X*mapping.X')./(sqrt_X*sqrt_X');
            tmp=inv(K+lambda*eye(N));
            kx=1-(mapping.X*point')./(sqrt_X*sqrt_kx');
            t_point=(Z'*tmp*kx)';    
        case 'IsoKRR4'
            lambda=1e-4; 
            N=size(mapping.X,1);
            Z=real(bsxfun(@times, mapping.vec, sqrt(mapping.val)'));
                    
            beta=50;
            G=L2_distance(mapping.X', mapping.X');
            K=exp(-sqrt(G)/beta);%Gram matrix
            tmp=inv(K+lambda*eye(N));
            kx=exp(-sqrt(L2_distance(mapping.X',point'))/beta);
            t_point=(Z'*tmp*kx)';
        case 'S-LP'
             % Initialize some other variables
            n = size(mapping.X, 1);
            
            % Compute embeddings
            t_point = repmat(0, [size(point, 1) numel(mapping.val)]);
            for i=1:size(point, 1)
                
                % Compute Gaussian kernel between test point and training points
                K = (sqrt(L2_distance(point(i,:)', mapping.X')))';
                
                k=12;
                [foo, ind] = sort(K, 'ascend');            
                K(ind(k+1:end)) = 0;
                
                K(K ~= 0) = exp(-K(K ~= 0) / mapping.beta);

                % Normalize kernel
                K = (1 ./ n) .* (K ./ sqrt(mean(K) .* mean(mapping.K, 2)));

                % Compute embedded point
                t_point(i,:) = sum(mapping.vec .* repmat(K, [1 size(mapping.vec, 2)]), 1);
            end
            
        otherwise
            error(['An out-of-sample extension for ' mapping.name ' is not available in the toolbox. You might consider using OUT_OF_SAMPLE_EST instead.']);
    end
    