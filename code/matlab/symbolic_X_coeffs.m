% Compute symbolically the coefficients of polynomials X that are inner
% products between b0tilde and columns of the adjugate of M = A-z*I.
%
% The output can be copied and pasted straight into C/C++ code.
%
% No chance of messing up these very complicated formulae when copying 
% them by hand if it's just all automated...
%
%NOTES: This extends the code in 'symbolic_adjugate_coeffs.m' to directly
% compute the things that we need.

clc
clear

s = 1; % Number of stages. Choose this to be whatever you like!
X_coeffs     = 'm_XCoeffs';    % Actual variable name of output
invA_label   = 'm_invA0';      % Actual variable name of matrix A above
btilde_label = 'm_b0tilde';    % Actual variable name of vector \tilde{b}

% Symbolic matrix. If you do it this way, the matrix elements are indexed from 1, which is not what we want...
%A = sym('A_%d__%d___', [s, s]);

% Create like this to get the 0's indexing right...
for i = 1:s
    for j  = 1:s
        A(i,j) = sym(sprintf('A_%d__%d___',  i-1, j-1));
    end
end

% btilde vector
for i = 1:s
    btilde(i) = sym(sprintf('b_%d____',  i-1));
end
btilde =  btilde(:); % Ensure column vector!
assume(btilde,'real')

syms z;
M = A - z*eye(s);  % M takes the form of A-z*I
adjM = adjoint(M); % Symbolic adjoint of M

% Compute inner product over columns of adjM
X = btilde' * adjM;


for i = 1:s
    % Pull out coefficients for polynomial in z
    q = coeffs(X(i), 'z');

    %q = vpa(q); % Convert numbers from symbolic to doubles
    q = arrayfun(@char, q, 'uniform', 0);
    
    % Replace the place holders in symbolic variables with meaningful things
    q = strrep(q, '____', ']');
    q = strrep(q, '___', ']'); % Replace the ___ on the RHS of each matrix element by ']'
    q = strrep(q, '__', ']['); % Replace the __ in the middle of each matrix element by ']['
    q = strrep(q, '_', '[');   % Replace the _ on the LHS of each matrix element by '['
    
    q = strrep(q, 'b', sprintf('%s', btilde_label));
    q = strrep(q, 'A', invA_label); % Replace A with proper label
    q = strrep(q, ' ', '');          % Remove any spaces

    fprintf('/* s=%d: Coefficients for polynomial X_{%d}(z) */\n', s, i)
    for k = 1:s
        if numel(q) < k
            fprintf('%s[%d][%d][%d]=+0.0;\n', X_coeffs, i-1, k-1)
        else
            if strcmp(q{k}(1), '-')
                fprintf('%s[%d][%d]=%s;\n', X_coeffs, i-1, k-1, q{k})
            else
                fprintf('%s[%d][%d]=+%s;\n', X_coeffs, i-1, k-1, q{k})
            end
        end
    end
    fprintf('\n')
end