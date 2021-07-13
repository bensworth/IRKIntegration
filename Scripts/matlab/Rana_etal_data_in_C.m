% Print out Rana et al. (2021) LD and DU factorizations of A0 so they
% can be copied straight into C/C++ code
%
% Just uncomment the IRK ID you want below and run this script. All data will
% be printed to the console

clc
clear

% Choose the preconditioning option:
preconditioner = 'LD'; % The LOWER triangular preconditioner better suited to LEFT preconditioning
preconditioner = 'DU'; % The UPPER triangular preconditioner better suited to RIGHT preconditioning


% This sets the precision of all computations to 32 digits...
digits(32)

mystr = '';

%%% --- Set up the formatting of the way in which the data is printed to console --- %%%
% Label these things as they're defined in the code so the output makes sense...
A_label    = 'preconditionerCoefficients'; 


% Data will be output to console as. Print 15 digits of data...
outformat1 = @(label, i, data) myfprintf('%s(%d) = %+.15f;\n', label, i, data);
outformat2 = @(label, i, j, data) myfprintf('%s(%d, %d) = %+.15f;\n', label, i, j, data);

%%% --- Tables --- %%% 
% UNCOMMENT ONE OF ME AND HIT RUN!
IRK_ID = 'Gauss2';
IRK_ID =  'Gauss4';
IRK_ID =  'Gauss6';
IRK_ID =  'Gauss8';
IRK_ID =  'Gauss10';
% % % 
IRK_ID = 'RadauIIA3';
IRK_ID = 'RadauIIA5';
IRK_ID = 'RadauIIA7';
IRK_ID = 'RadauIIA9';
% % %  
IRK_ID =  'LobIIIC2';
IRK_ID =  'LobIIIC4';
IRK_ID =  'LobIIIC6';
IRK_ID =  'LobIIIC8';

% Get table
[A, ~, ~] = butcher_tableaux(IRK_ID);
s = size(A,1);

[L, U] = lu_nopivot(A); % A = L*U

if strcmp(preconditioner, 'LD')
    D = diag(diag(U));
    U = D \ U; % Scale U to be unit lower triangular.
    LD = L * D; % Now A = L*DU with unit lower triangular L
    precCoefficients = LD;

    % Just check the factorization works as intended.
    fprintf('Ensure I''m zero: ||LD*U - A||=%.1e\n\n', norm(LD*U - A))
    
elseif strcmp(preconditioner, 'DU')
    D = diag(diag(L));
    L = L * inv(D); % Scale L to be unit lower triangular
    DU = D*U; % Now A = L*DU with unit upper triangular L
    precCoefficients = DU;
    
    % Just check the factorization works as intended.
    fprintf('Ensure I''m zero: ||LD*U - A||=%.1e\n\n', norm(L*DU - A))
end
    

%%% --- Now print everything to console... --- %%%
myfprintf('/* %s coefficients. IRK ID: %s */\n', preconditioner ,IRK_ID)
for i = 1:s
    for j = 1:s
        outformat2(A_label, i-1, j-1, precCoefficients(i,j))
    end
end


% Print function that indents before printing.
function myfprintf(varargin)
    fprintf('                    ') % Indent 20 spaces
    fprintf(varargin{:})
end