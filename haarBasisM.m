function haar_basis = haarBasisM(n,num_sam)
%%
% This function generates a matrix of 2^n haar bases.
% Each row represents a haar basis from 0 to 1 with 1/2^num_sam as an
% interval

%%
if n>num_sam
    print('n is more than the number of intervals');
    return;
end
basis_num = 2^n;
sam_sum = 2^num_sam;
haar_basis = zeros(basis_num,sam_sum);

% haar0-1
haar_basis(1,:) = ones(1,sam_sum);

index = 2;
for p = 1:n
    for k = 1:2^(p-1)
        for i = (2*k-2)*fix(sam_sum/2^p)+1:(2*k-1)*fix(sam_sum/2^p)
            haar_basis(index,i) = sqrt(2^(p-1));
        end
        for i = (2*k-1)*fix(sam_sum/2^p)+1:(2*k)*fix(sam_sum/2^p)
            haar_basis(index,i) = -sqrt(2^(p-1));
        end
        index = index + 1;
    end
end
% t = 0:1/2^num_sam:1-1/2^num_sam;
% plot(t,haar_basis(16,:))
end
