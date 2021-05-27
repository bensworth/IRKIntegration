% LU routine that doesn't use pivoting. MATLAB's LU always uses pivoting.
% This was taken from:
% https://stackoverflow.com/questions/41150997/perform-lu-decomposition-without-pivoting-in-matlab

function [L, U] = lu_nopivot(A)

n = size(A, 1); % Obtain number of rows (should equal number of columns)
L = eye(n); % Start L off as identity and populate the lower triangular half slowly
for k = 1 : n
    % For each row k, access columns from k+1 to the end and divide by
    % the diagonal coefficient at A(k ,k)
    L(k + 1 : n, k) = A(k + 1 : n, k) / A(k, k);

    % For each row k+1 to the end, perform Gaussian elimination
    % In the end, A will contain U
    for l = k + 1 : n
        A(l, :) = A(l, :) - L(l, k) * A(k, :);
    end
end
U = A;

end