% Print out Butcher tables and associated information so that it can be
% copied straight into C/C++ code

clc
clear


%%% --- Set up the formatting of the way in which the data is printed to console --- %%%
% Label these things as they're defined in the code so the output makes sense...
s_label  =  'm_s';
zetaSize_label  =  'm_zetaSize';
etaSize_label  =  'm_etaSize';

A_label    = 'A'; 
invA_label    = 'invA'; 
b_label    = 'b';
c_label    = 'c';
d_label    = 'd';
zeta_label = 'zeta';
eta_label  = 'eta';
beta_label = 'beta';

sizingRoutineStr = sprintf('SizeButcherArrays(%s, %s, %s, %s, %s, %s, %s, %s);', A_label, invA_label, b_label, c_label, d_label, zeta_label, eta_label, beta_label);

% Data will be output to console as
outformat1 = @(label, i, data) fprintf('Set(%s, %d, %+.15f);\n', label, i, data);
outformat2 = @(label, i, j, data) fprintf('Set(%s, %d, %d, %+.15f);\n', label, i, j, data);

%%% --- Tables --- %%%

% Backward Euler
A = 1.0;
b = 1.0;
c = 1.0;

% % 2nd-order SDIRK
% gamma   = 1.0 - 1.0 / sqrt(2.0);
% A       = [gamma, 0.0; ...
%            2*(1-gamma)-1, gamma];
% b       = [1/2; 1/2];
% c       = [gamma; 1-gamma];


% 2nd-order Lobatto IIIC. L-stable
% A = 0.5*[1, -1; 1, 1];
% b = 0.5*[1; 1];
% c = [0; 1];

% % SDIRK3 from Dobrev et al. (2017). Updated numbers from WIKI page.
% % Actually, see Butcher (2008), p. 262
% x = 0.43586652150845899942;
% p = 0.5*(1+x);
% q = 0.5*(1-x);
% y = -3/2*x^2 + 4*x - 1/4;
% z =  3/2*x^2 - 5*x + 5/4;
% c = [x;  p;  1];
% b = [y;  z;  x];
% A = [x,  0,  0;
%        q,  x,  0;
%        y,  z,  x];

% % implicit 4th-order method, Hammer & Hollingsworth (A-stable)
% % note: coincides with s=2-stage, p=2s-order Gauss method
% % %       see https://www.math.auckland.ac.nz/~butcher/ODE-book-2008/Tutorials/IRK.pdf
A = [0.25, 0.25-sqrt(3.0)/6.0; ...
     0.25+sqrt(3.0)/6.0, 0.25];
b = [0.5, 0.5]';
c = [0.5-sqrt(3.0)/6.0; ...
     0.5+sqrt(3.0)/6.0]';


% A 6th-order Gauss--Legendre method
A = [5/36, 2/9-sqrt(15)/15, 5/36-sqrt(15)/30; ...
    5/36+sqrt(15)/24, 2/9, 5/36-sqrt(15)/24; ...
    5/36+sqrt(15)/30, 2/9+sqrt(15)/15, 5/36];
b = [5/18; 4/9; 5/18];
c = [1/2-sqrt(15)/10; 1/2; 1/2+sqrt(15)/10];
 
 
 

s  = size(A,1);
% Compute the quantities we don't already have
invA =  inv(A);
%btilde = b' * invA;
d = (A'\b)'; % This is a better conditioned operation...


% Get eigenvalues of A^-1 and split into real + conjugate pairs
lambda = 1./eig(A);
[zeta, eta, beta] = decompose_eigenvalues(lambda);
zeta_size = numel(zeta);
eta_size = numel(eta);
% rlambda
% clambda_eta
% clambda_beta



%%% --- Now print everything to console... --- %%%
fprintf('/* --- Dimensions --- */\n')
%fprintf('/* ------------------ */\n')
fprintf('%s        = %d;\n', s_label, s);
fprintf('%s = %d;\n', zetaSize_label, zeta_size);
fprintf('%s  = %d;\n', etaSize_label, eta_size);
fprintf('%s%s\n', sizingRoutineStr, ' /* Set data arrays to correct dimensions */')
fprintf('/* ---------------- */\n\n')




fprintf('/* --- Tableaux constants --- */\n')
%fprintf('/* -------------------------- */\n')
% A
fprintf('/* --- A --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(A_label, i-1, j-1, A(i,j))
    end
end
% invA
fprintf('/* --- inv(A) --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(invA_label, i-1, j-1, invA(i,j))
    end
end
%fprintf('\n');
% b
fprintf('/* --- b --- */\n');
for i = 1:s
    outformat1(b_label, i-1, b(i))
end
%fprintf('\n');
% c
fprintf('/* --- c --- */\n');
for i = 1:s
    outformat1(c_label, i-1, c(i))
end
%fprintf('\n');
% d
fprintf('/* --- d --- */\n');
for i = 1:s
    outformat1(d_label, i-1, d(i))
end
%fprintf('\n');
% zeta
fprintf('/* --- zeta --- */\n');
for i = 1:zeta_size
    outformat1(zeta_label, i-1, zeta(i))
end
%fprintf('\n');
% eta
fprintf('/* --- eta --- */\n');
for i = 1:eta_size
    outformat1(eta_label, i-1, eta(i))
end
%fprintf('\n');
%  beta
fprintf('/* --- beta --- */\n');
for i = 1:eta_size
    outformat1(beta_label, i-1, beta(i))
end
fprintf('/* -------------------------- */\n\n')
