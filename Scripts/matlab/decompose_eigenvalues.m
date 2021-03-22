function [rlambda, clambda_eta, clambda_beta] = decompose_eigenvalues(lambda)
%Decompose eigenvalues lambda into 3 components:
%   rlambda      == real eigenvalues
%   clambda_eta  == real component of complex conjugate pairs
%   clambda_beta == +imag component of complex conjugate pairs

ilambda = imag(lambda);
ilambda(abs(ilambda) < 1e-14) = 0; % Use this as the threshhold for what's real and what's not!
realI = find(ilambda == 0);
rlambda = sort(lambda(realI));
clambda = lambda;
clambda(realI) = [];

if mod(numel(clambda),2) ~= 0
    error('Why isn''t there an even number of conjugate pairs?')
end

clambda_eta = real(clambda);
clambda_beta = imag(clambda);


[clambda_eta, I] = sort(clambda_eta);
clambda_eta = clambda_eta(1:2:end);
clambda_beta = clambda_beta(I);
clambda_beta = abs(clambda_beta(1:2:end));

end