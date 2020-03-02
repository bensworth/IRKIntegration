% Print out Butcher tables and associated information so that it can be
% copied straight into C/C++ code

clc
clear

% Label these things as they're defined in the code so the output makes sense...
A_label      = 'm_A0'; 
b_label      = 'm_b0';
c_label      = 'm_c0';
invA_label   = 'm_invA0';
btilde_label = 'm_b0tilde';
beta_label   = 'm_beta';
eta_label    = 'm_eta';


% Backward Euler
A = 1.0;
b = 1.0;
c = 1.0;

% % implicit 4th-order method, Hammer & Hollingsworth (A-stable)
% % note: coincides with s=2-stage, p=2s-order Gauss method
% %       see https://www.math.auckland.ac.nz/~butcher/ODE-book-2008/Tutorials/IRK.pdf
% A = [0.25, 0.25-sqrt(3.0)/6.0; ...
%      0.25+sqrt(3.0)/6.0, 0.25];
% b = [0.5, 0.5]';
% c = [0.5-sqrt(3.0)/6.0; ...
%      0.5+sqrt(3.0)/6.0]';


% A 6th-order Gauss--Legendre method
A = [5/36, 2/9-sqrt(15)/15, 5/36-sqrt(15)/30; ...
    5/36+sqrt(15)/24, 2/9, 5/36-sqrt(15)/24; ...
    5/36+sqrt(15/30), 2/9+sqrt(15)/15, 5/36];
b = [5/18; 4/9; 5/18];
c = [1/2-sqrt(15)/10; 1/2; 1/2+sqrt(15)/10];
 
 
 
s  = size(A,1);

% Compute the quantities we don't already have
invA =  inv(A);
%btilde = b' * invA;
btilde = (A'\b)'; % This is a better conditioned operation...
lambda = 1./eig(A);
beta = imag(lambda);
eta  = real(lambda);

%%% --- Now print everything to console... --- %%%

% A
for i = 1:s
    for j = 1:s
        fprintf('%s[%d][%d]=%+.16f;\n', A_label, i-1, j-1, A(i,j));
    end
end
fprintf('\n');
% b
for i = 1:s
    fprintf('%s[%d]=%+.16f;\n', b_label, i-1, b(i));
end
fprintf('\n');
% c
for i = 1:s
    fprintf('%s[%d]=%+.16f;\n', c_label, i-1, c(i));
end
fprintf('\n');
% invA
for i = 1:s
    for j = 1:s
        fprintf('%s[%d][%d]=%+.16f;\n', invA_label, i-1, j-1, invA(i,j));
    end
end
fprintf('\n');
% btilde
for i = 1:s
    fprintf('%s[%d]=%+.16f;\n', btilde_label, i-1, btilde(i));
end
fprintf('\n');
%  beta
for i = 1:s
    fprintf('%s[%d]=%+.16f;\n', beta_label, i-1, beta(i));
end
fprintf('\n');
% eta
for i = 1:s
    fprintf('%s[%d]=%+.16f;\n', eta_label, i-1, eta(i));
end
