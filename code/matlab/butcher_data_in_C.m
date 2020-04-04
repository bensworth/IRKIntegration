% Print out Butcher tables and associated information so that it can be
% copied straight into C/C++ code
%
% Just uncomment the ID you want below and run this script. All data will
% be printed to the console

clc
clear

mystr = '';

%%% --- Set up the formatting of the way in which the data is printed to console --- %%%
% Label these things as they're defined in the code so the output makes sense...
s_label        =  'm_s';
zetaSize_label =  'm_zetaSize';
etaSize_label  =  'm_etaSize';

A_label    = 'm_A0'; 
invA_label = 'm_invA0'; 
b_label    = 'm_b0';
c_label    = 'm_c0';
d_label    = 'm_d0';
zeta_label = 'm_zeta';
eta_label  = 'm_eta';
beta_label = 'm_beta';

sizingRoutineStr = sprintf('SizeButcherArrays();');

% Data will be output to console as
outformat1 = @(label, i, data) myfprintf('%s(%d) = %+.15f;\n', label, i, data);
outformat2 = @(label, i, j, data) myfprintf('%s(%d, %d) = %+.15f;\n', label, i, j, data);

%%% --- Tables --- %%% 
% UNCOMMENT ONE OF ME
% ID = 'SDIRK1';
% ID = 'SDIRK2';
% ID = 'SDIRK3';
% ID = 'SDIRK4';
% 
ID =  'Gauss4';
% ID =  'Gauss6';
% ID =  'Gauss8';
% ID =  'Gauss10';
% 
% ID = 'RadauIIA3';
% ID = 'RadauIIA5';
% ID = 'RadauIIA7';
% ID = 'RadauIIA9';
%  
% ID =  'LobIIIC2';
% ID =  'LobIIIC4';
% ID =  'LobIIIC6';

% Get table
[A, b, c] = butcher_tableaux(ID);

s  = size(A,1);
% Compute the quantities we don't already have
invA =  inv(A);
d = (A'\b)'; 

% Get eigenvalues of A^-1 and split into real + conjugate pairs
lambda = 1./eig(A);
[zeta, eta, beta] = decompose_eigenvalues(lambda);
zeta_size = numel(zeta);
eta_size  = numel(eta);
% rlambda
% clambda_eta
% clambda_beta


%%% --- Now print everything to console... --- %%%
myfprintf('/* ID: %s */\n', ID)
myfprintf('/* --- Dimensions --- */\n')
myfprintf('%s        = %d;\n', s_label, s);
myfprintf('%s = %d;\n', zetaSize_label, zeta_size);
myfprintf('%s  = %d;\n', etaSize_label, eta_size);
myfprintf('%s\n\n', sizingRoutineStr)


myfprintf('/* --- Tableau constants --- */\n')
% A
myfprintf('/* --- A --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(A_label, i-1, j-1, A(i,j))
    end
end
% invA
myfprintf('/* --- inv(A) --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(invA_label, i-1, j-1, invA(i,j))
    end
end
%fprintf('\n');
% b
myfprintf('/* --- b --- */\n');
for i = 1:s
    outformat1(b_label, i-1, b(i))
end
%fprintf('\n');
% c
myfprintf('/* --- c --- */\n');
for i = 1:s
    outformat1(c_label, i-1, c(i))
end
%fprintf('\n');
% d
myfprintf('/* --- d --- */\n');
for i = 1:s
    outformat1(d_label, i-1, d(i))
end
%fprintf('\n');
% zeta
myfprintf('/* --- zeta --- */\n');
for i = 1:zeta_size
    outformat1(zeta_label, i-1, zeta(i))
end
%fprintf('\n');
% eta
myfprintf('/* --- eta --- */\n');
for i = 1:eta_size
    outformat1(eta_label, i-1, eta(i))
end
%fprintf('\n');
%  beta
myfprintf('/* --- beta --- */\n');
for i = 1:eta_size
    outformat1(beta_label, i-1, beta(i))
end
myfprintf('/* -------------------------- */\n\n')

% Print function that indents before printing.
function myfprintf(varargin)
    fprintf('        ') % Indent 8 spaces
    fprintf(varargin{:})
end