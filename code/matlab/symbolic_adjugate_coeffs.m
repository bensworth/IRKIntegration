% Compute the symbolic adjugate of M = A-z*I and output the formulae so they
% can be copied and pasted straight into C/C++ code.
%
% No chance of messing up these very complicated formulae when copying 
% them by hand if it's just all automated...

clc
clear

s = 4; % Number of stages. Choose this to be whatever you like!
adjMCoeffs_label = 'm_adjMCoeffs'; % Actual variable name of output
invA0_label      = 'm_invA0';       % Actual variable name of matrix A above

% Symbolic matrix. If you do it this way, the matrix elements are indexed from 1, which is not what we want...
%A = sym('A_%d__%d___', [s, s]);

% Create like this to get the 0's indexing right...
for i = 1:s
    for j  = 1:s
        A(i,j) = sym(sprintf('A_%d__%d___',  i-1, j-1));
    end
end


syms z;
M = A - z*eye(s);  % M takes the form of A-z*I
adjM = adjoint(M); % Symbolic adjoint of M

%adjM

% These are the coefficients for the polynomials in z if you want to look 
% at them... But we just extract the sets of coefficient one at a time below
% Qcoefficients = cell(s);
% for i = 1:s
%     for j = 1:s
%         Qcoefficients{i,j} = coeffs(adjM(i,j), 'z');
%     end
% end


for i = 1:s
    for j = 1:s  
        % Pull out coefficients for polynomial in z
        q = coeffs(adjM(i,j), 'z');
        
        q = vpa(q); % Convert numbers from symbolic to doubles
        q = arrayfun(@char, q, 'uniform', 0);
        q = strrep(q, '___', ']'); % Replace the ___ on the RHS of each matrix element by ']'
        q = strrep(q, '__', ']['); % Replace the __ in the middle of each matrix element by ']['
        q = strrep(q, '_', '[');   % Replace the _ on the LHS of each matrix element by '['
        q = strrep(q, 'A', invA0_label); % Replace A with proper label
        q = strrep(q, ' ', '');          % Remove any spaces
        
        fprintf('/* s=%d:\tCoefficients for polynomial Q_{%d,%d}(z) */\n', s, i, j)
        for k = 1:s
            if numel(q) < k
                fprintf('%s[%d][%d][%d]=+0.0;\n', adjMCoeffs_label, i-1, j-1, k-1)
            else
                if strcmp(q{k}(1), '-')
                    fprintf('%s[%d][%d][%d]=%s;\n', adjMCoeffs_label, i-1, j-1, k-1, q{k})
                else
                    fprintf('%s[%d][%d][%d]=+%s;\n', adjMCoeffs_label, i-1, j-1, k-1, q{k})
                end
            end
        end
        fprintf('\n')
    end
end
