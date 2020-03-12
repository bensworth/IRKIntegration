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
X_coeffs     = 'X';    % Actual variable name of output
% invA_label   = 'm_invA0';      % Actual variable name of matrix A above
% d_label = 'm_d0';    % Actual variable name of vector d0 == b0^\top * inv(A0)
invA_label   = 'B';      % Actual variable name of matrix A above
d_label = 'd';    % Actual variable name of vector d0 == b0^\top * inv(A0)

outformat2 = @(i, j, data) fprintf('Set(%s, %d, %d, %s);\n', X_coeffs, i, j, data);

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
    d(i) = sym(sprintf('b_%d____',  i-1));
end
d =  d(:); % Ensure column vector!
assume(d,'real')

syms z;
M = A - z*eye(s);  % M takes the form of A-z*I
adjM = adjoint(M); % Symbolic adjoint of M

% Compute inner product over columns of adjM
X = d' * adjM;


for i = 1:s
    % Pull out coefficients for polynomial in z
    q = coeffs(X(i), 'z');

    %q = vpa(q); % Convert numbers from symbolic to doubles
    q = arrayfun(@char, q, 'uniform', 0);
    
    % Replace the place holders in symbolic variables with meaningful things
    q = strrep(q, '____', ')');
    q = strrep(q, '___', ')'); % Replace the ___ on the RHS of each matrix element by ']'
    q = strrep(q, '__', ','); % Replace the __ in the middle of each matrix element by ']['
    q = strrep(q, '_', '(');   % Replace the _ on the LHS of each matrix element by '['
    
    q = strrep(q, 'b', sprintf('%s', d_label));
    q = strrep(q, 'A', invA_label); % Replace A with proper label
    q = strrep(q, ' ', '');          % Remove any spaces

    fprintf('/* s=%d: Coefficients for polynomial X_{%d}(z) */\n', s, i)
    for k = 1:s
        if numel(q) < k
            q{k} = '+0.0';
        else
            if ~strcmp(q{k}(1), '-')
                q{k} = ['+', q{k}];
            end
        end
        outformat2(k-1, i-1, q{k})
    end
    fprintf('\n')
end