% Check algorithm for computing action of matrix polnomial on vector using
% a Horner-like scheme (see https://en.wikipedia.org/wiki/Horner%27s_method)
clc
clear
rng(2)

poly_deg = 4;                  % Degree of matrix polynomial
coeffs = rand(poly_deg+1, 1);   % Polynomial coefficients, from 0th-order up to poly_degth-order

n = 10;                 % Size of matrix
v = rand(n, 1);         % Vector to compute action on
L = rand(n);          % Matrix
L_action = @(v) L*v;    % Action of matrix

r_horner = horner_mat_action(coeffs, L_action,  v);
% Note the need to reverse order of coefficient array as per defn of polyvalm
r_true = polyvalm(flipud(coeffs), L) * v;

fprintf('Test:\t||r_horner - r_true||=%.2e\n', norm(r_horner-r_true));



function r = horner_mat_action(coeffs, L_action, v)
% Compute action of polynomial P(L) on vector v using a Horner-like scheme,
% r = P(L)*v
% P(L) = c(0)*L^0 + ... + c(n)*L^n

c = coeffs;
n = numel(c);
r = c(n)*v;

for ell = n-1:-1:1
    rcopy = r;
    r = c(ell) * v + L_action(rcopy);
end

end


