function z = PersBettiSur(b,d,bd_xy,sam_num)
%%
% This function is to generate the extended PRF by using a barcode.
% Input: 
% b: a vector to store the birth indices of a barcode
% d: a vector to store the death indices of a barcode
% bd_xy: the bound where the PRF is considered, and those not in
% [0,bd_xy]^2 are truncated.
% sam_num: an integer to discrete the region (into 2^sam_num x 2^sam_num)
%
% Output:
% z: a matrix with its size 2^sam_num x 2^sam_num to store the discrete PRF
%
%%
% To restrict the domain of PD into [0,bd_xy]^2
num = length(b);
for i = 1:num
   if d(i) > bd_xy 
       d(i) = bd_xy;
   end
   if b(i) > bd_xy
       b(i) = bd_xy;
   end
end

% To normalize PD 
b = b/bd_xy;
d = d/bd_xy;
p = d - b;

delta_xy = 1/2^sam_num;
n = floor(1/delta_xy);
z = zeros(n);


for i = 1:num
   n_b = floor(b(i)/delta_xy)+1;
   n_d = floor((b(i)+p(i))/delta_xy);
   for j = n_b:n_d
       for k = n_b:n_d
            z(j,k) = z(j,k) + 1;
       end
   end
end


end