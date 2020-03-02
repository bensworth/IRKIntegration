% Just checking the adjugate/determinant formulation works how I think it does...

clc
clear

% Build a spatial disc. matrix
nx = 2;
Inx = eye(nx);
e = ones(nx, 1);
dx = 2/nx;
dt = 0.1*dx;
L = spdiags([-e, e], [-1, 0], nx, nx);
L(1, end) =  -1;
L = L / dx;
L = L * dt;


% Quantities associated with Butcher matrix A0
A0 = [1/4, 1/4-1/6*sqrt(3); 1/4+1/6*sqrt(3) 1/4];

A0 = [(88.0-7.0*sqrt(6.0))/360.0, (296.0-169*sqrt(9))/1800.0, (-2.0+3.0*sqrt(6.0))/225.0; ...
         (296.0+169*sqrt(9))/1800.0, (88.0+7.0*sqrt(6.0))/360.0, (-2.0-3.0*sqrt(6.0))/225.0; ...
         (16.0-sqrt(6.0))/36.0, (16.0+sqrt(6.0))/36.0, 1.0/9.0];

s = size(A0, 1);
Is = eye(s);
invA0 = inv(A0);
lambda = eig(A0);
lambda = 1./lambda; % Eigenvalues of inv(A0)


% The matrix we want to invert
Ms = kron(invA0, Inx) - kron(Is, L);

%%% First test using the universally true result
adjMs = double(adjoint(sym(Ms)));
invMs = 1/det(Ms) * adjMs;
fprintf('Test 1:\t||inv(Ms) * 1/det(Ms) * adjMs - I|| = %.2e\n\n', norm(Ms * invMs - eye(nx*s), inf))


%%% Second test using the fact that Ms is a matrix over commutative ring
% Symbolic calculation of adjugate of invA0 - z*I
syms z
adjMs_sym = adjoint(sym(invA0 - z*eye(s)));

% Sub L into symbolic adjugate matrix
gen_adjMs = zeros(s*nx);
for i = 1:s
    for j = 1:s
        qij = double(fliplr(coeffs(adjMs_sym(i,j))));
        gen_adjMs((i-1)*nx+1:i*nx, (j-1)*nx+1:j*nx) = polyvalm(qij, L);
    end
end

% Evaluate characteristic polynomial in expanded form
% Ps_coeffs = charpoly(invA0);
% Ps_L = polyvalm(Ps_coeffs, L);

% Evaluate the characteristic polynomial in factored form
Ps_L = eye(nx);
for i = 1:s
    Ps_L = Ps_L * (lambda(i) * eye(nx) - L);
end
gen_detMs = Ps_L;

invMs = kron(Is, inv(gen_detMs)) * gen_adjMs;
fprintf('Test 2:\t||Ms * inv(getdet(Ms)) * genadjMs - I|| = %.2e\n\n', norm(Ms * invMs - eye(nx*s), inf))

% This  doesn't quite work... 
% Check the adjugate of Ms is actually the adjugate of generalized adjugate
fprintf('Test 3:\t||adj(Ms) - adj(genadj(Ms))|| = %.2e\n\n', norm(adjMs - double(adjoint(sym(gen_adjMs))), inf))

% Check the determinant of Ms is actually det of generalized det
fprintf('Test 4:\t|det(Ms) - det(gendet(Ms))| = %.2e\n\n', abs(det(Ms) - det(gen_detMs)))













