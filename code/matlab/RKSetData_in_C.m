clc
clear

% A = Q*R*Q'

%%% --- Tables --- %%% 
% UNCOMMENT ONE OF ME AND HIT RUN!
%ID = 'ASDIRK3';
% ID = 'ASDIRK4';
% 
% ID = 'LSDIRK1';
% ID = 'LSDIRK2';
% ID = 'LSDIRK3';
ID = 'LSDIRK4';
% 
%ID = 'Gauss2';
%ID =  'Gauss4';
% ID =  'Gauss6';
ID =  'Gauss8';
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
% ID =  'LobIIIC8';

% This sets the precision of all computations to 32 digits...
%...digits(32) no it don't...


%%% --- Set up the formatting of the way in which the data is printed to console --- %%%
% Label these things as they're defined in the code so the output makes sense...
s_label    = 's';
s_eff_label = 's_eff';
A_label    = 'A0';
invA_label = 'invA0'; 
b_label    = 'b0';
c_label    = 'c0';
d_label    = 'd0';
zeta_label = 'zeta';
eta_label  = 'eta';
beta_label = 'beta';
Q_label    = 'Q0'; 
R_label    = 'R0'; 
R_block_sizes_label = 'R0_block_sizes';

[A, b, c] = butcher_tableaux(ID);

% Just check A is not singular!
if cond(A) > 1e4
    error('A is ill-conditioned?')
end

s = numel(b);
%rng(2)
%A = rand(s);
Ainv = inv(A);
%Ainv = A\eye(s);
%condeig(A)
%condeig(Ainv) % Interesting... A\I v.s., inv(A) results in a massive
%difference here for SDIRK schemes...
    
d = ((A')\b);

% Treat SDIRK differently: Schur decomposition is horribly ill-conditioned
% for these lower triangular matrices. That is to say, if we compute the
% eigenvalues/Schur decomposition using schur() then we get horrible
% results including eigenvalues w/ significant imaginary components
if contains(ID, 'SDIRK')
%     zeta_size = s;
%     zeta = 1/A(1,1)*ones(s, 1);
%     eta_size = 0;
%     eta = [];
%     beta = [];
    % Looking at numerical output, U takes the form of an identity Hankel
    % matrix with alternating sign along the diagonal.
    Q = zeros(s, s);
    for i = 1:s
        Q(s-i+1,i) = (-1)^i;
    end
    R = Q'*(A\Q); % Given U, we can easily compute T!
    
else
    [Q, R] = schur(Ainv, 'real');
    %lambda = ordeig(R);


%     % Re-order T so that:
%     %   1. CC eigs appear closest to top left, with smallest values of beta/eta
%     %       appearing closest to top left.
%     %   2. Real eigs appear furthest from top left, with largest values of zeta
%     %       appearing to top left.
%     %
%     % The idea here is that we do linear solves before quadratic solves, and 
%     %   within each group, we do harder solves before easier solves.
% 
%     % Clusters is a logical array, where its largest indices are placed closest 
%     % to the top left corner
%     clusters = zeros(s, 1);
% 
%     zeta_inds = find(abs(imag(lambda)) < 1e-15);
%     zeta = lambda(zeta_inds);
%     [~, I_zeta] = sort(zeta, 'ascend'); 
% 
%     cc_inds = find(abs(imag(lambda)) > 1e-15);
%     cc_inds = cc_inds(1:2:end); % Select every 2nd entry, cc-pairs appear next to each other
%     cc = lambda(cc_inds);
%     [~, I_cc] = sort(abs(imag(cc))./abs(real(cc)), 'descend'); 
% 
% 
%     % Real eigs are to be moved to bottom right, so they have smallest cluster
%     % indices
%     for i = 1:numel(zeta_inds)
%        clusters(zeta_inds(I_zeta(i))) = i; 
%     end
% 
%     % CC eigs are to be moved to top left, so they have the largest cluster
%     % indices
%     for i = 1:numel(cc_inds)
%        clusters([cc_inds(I_cc(i)), cc_inds(I_cc(i))+1]) = [i+numel(zeta), i+numel(zeta)]; 
%     end
% 
%     % Re-order the Schur decomposition
%     [US, TS] = ordschur(U, T, clusters);
%     U = US;
%     T = TS;
% 
%     % Just check this re-ordering worked as planned... 
%     lambda = ordeig(T);
%     lambda = lambda(:);
%     [lambda, abs(imag(lambda))./abs(real(lambda))];
% 
%     zeta_size = numel(zeta);
%     eta_size = numel(cc);
% 
%     lambda;
%     eta = real(lambda(1:2:2*eta_size));
%     beta = abs(imag(lambda(1:2:2*eta_size)));
%     zeta = lambda(2*eta_size+1:end);
% 
%     % Reverse order of eigenvalues into order we actually do solves 
%     eta = flipud(eta);
%     beta = flipud(beta);
%     zeta = flipud(zeta);
end


Rtrunc = tril(R, -1);
s_complex = nnz(Rtrunc); 
s_real = s - 2*s_complex;
s_eff = s_complex + s_real;

[rows, ~] = find(Rtrunc);
rows = rows-1; % Row starts of the 2x2 blocks

block_row_starts = (1:s)';
for i = 1:numel(rows)
    block_row_starts(rows(i)) = rows(i);
    block_row_starts(rows(i)+1) = -1;
end
block_row_starts(block_row_starts == -1) = [];
block_sizes = [block_row_starts(2:end); s+1] - block_row_starts;

block_row_starts = block_row_starts-1; % Convert to 0's based indexing
% R
% block_sizes
% block_row_starts


% Get eigenvalues of A^-1 and split into real + conjugate pairs
lambda = ordeig(R);
zeta = [];
eta = [];
beta = [];
i = 1;
for idx = 1:s_eff
   if block_sizes(idx) == 1
       zeta = [zeta; lambda(i)];
   else
       eta = [eta; real(lambda(i));];
       beta = [beta; abs(imag(lambda(i)));];
       i = i + 1;
   end
   i = i + 1;
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% --- Tests to make sure things make sense... --- %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TOL = 5e-13;
% %fprintf('------TESTS--------\n');
% r = norm(Ainv - (U*T*U'), inf);
% assert(r < TOL, 'schur FAIL: r=%.2e', r)
% r = norm(eye(s) - (U*U'), inf);
% assert(r < TOL, 'orthog FAIL: r=%.2e', r)
% 
% % Compute eigenvalues of inv(A) in most stable way...
% lambda_exact = 1./eig(A);
% 
% zeta_exact = lambda_exact(abs(imag(lambda_exact)) < 1e-15);
% zeta_exact = sort(zeta_exact, 'ascend');
% 
% cc = lambda_exact(abs(imag(lambda_exact)) > 1e-15);
% cc = sort(cc);
% cc = cc(1:2:end);
% [~, I] = sort(abs(imag(cc))./abs(real(cc)), 'descend');
% cc = cc(I);
% eta_exact = real(cc);
% beta_exact = abs(imag(cc));
% 
% r = norm(zeta_exact - zeta);
% assert(r < TOL, 'zeta FAIL: r=%.2e', 2)
% r = norm(eta_exact - eta);
% assert(r < TOL, 'eta FAIL: r=%.2e', 2)
% r = norm(beta_exact - beta);
% assert(r < TOL, 'beta FAIL: r=%.2e', 2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- Now print everything to console... --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data will be output to console as. Print 15 digits of data...
outformat1 = @(label, i, data) myfprintf('%s(%d) = %+.15f;\n', label, i, data);
outformat11 = @(label, i, data) myfprintf('%s[%d] = %d;\n', label, i, data);
outformat2 = @(label, i, j, data) myfprintf('%s(%d, %d) = %+.15f;\n', label, i, j, data);

%myfprintf('/* ID: %s */\n', ID)
myfprintf('%s = %d;\n', s_label, s);
myfprintf('%s = %d;\n', s_eff_label, s_eff);
myfprintf('SizeData();\n');

%myfprintf('/* --- Tableau constants --- */\n')
% A
myfprintf('/* --- A --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(A_label, i-1, j-1, A(i,j))
    end
end
% inv(A)
myfprintf('/* --- inv(A) --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(invA_label, i-1, j-1, Ainv(i,j))
    end
end
myfprintf('/* --- b --- */\n');
for i = 1:s
    outformat1(b_label, i-1, b(i))
end
% c
myfprintf('/* --- c --- */\n');
for i = 1:s
    outformat1(c_label, i-1, c(i))
end
% d
myfprintf('/* --- d --- */\n');
for i = 1:s
    outformat1(d_label, i-1, d(i))
end
%  zeta
myfprintf('/* --- zeta --- */\n');
for i = 1:numel(zeta)
    outformat1(zeta_label, i-1, zeta(i))
end
% eta
myfprintf('/* --- eta --- */\n');
for i = 1:numel(eta)
    outformat1(eta_label, i-1, eta(i))
end
%  beta
myfprintf('/* --- beta --- */\n');
for i = 1:numel(eta)
    outformat1(beta_label, i-1, beta(i))
end
% Q
myfprintf('/* --- Q --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(Q_label, i-1, j-1, Q(i,j))
    end
end
% R
myfprintf('/* --- R --- */\n');
for i = 1:s
    for j = 1:s
        outformat2(R_label, i-1, j-1, R(i,j))
    end
end
myfprintf('/* --- R block sizes --- */\n');
for i = 1:s_eff
    outformat11(R_block_sizes_label, i-1, block_sizes(i))
end


% Print function that indents before printing.
function myfprintf(varargin)
    fprintf('            ') % Indent 8 spaces
    fprintf(varargin{:})
end