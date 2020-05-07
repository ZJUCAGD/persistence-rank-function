function lambda_v = haarDecomposition2DFunc(z,haar_basis)
%%
% This function returns the coefficient vector of a 2D haar basis
% It is used to decompose a function represented by a matrix with its size
% 2^sam_num * 2^sam_num
%%
 [basis_num,share] = size(haar_basis);
 lambda_v = zeros(basis_num^2,1);
 index = 1;
 for i = 1:basis_num
     for j = 1:basis_num
         tensor_basis = haar_basis(i,:)'*haar_basis(j,:);
         lambda_v(index) = (1/share)^2*sum(sum(tensor_basis.*z));
         index = index + 1;
     end
 end

end