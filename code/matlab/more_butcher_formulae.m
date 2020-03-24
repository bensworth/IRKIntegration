% evaluates stability function for implicit Runge-Kutta (IRK) method
% for given order p
% formulas from [Hairer, Norsett, Wanner (1987)] (third edition)
% and [Dobrev et al. (2017)]
function [rz, A, b, c] = irk_rz(z, p, k, eqn)
    switch(p)
        % L-stable methods
        case({1, 11})
            [A, b, c] = get_irk_p1_sdirk();
        case({2, 21})
            [A, b, c] = get_irk_p2_sdirk();
%             [A, b, c] = get_irk_p2_miller();
        case({3, 31})
            [A, b, c] = get_irk_p33_sdirk();
        case({4, 41})
%             [A, b, c] = get_irk_p4_lobatto();
            [A, b, c] = get_irk_p4_sdirk();
%             [A, b, c] = get_irk_p4_norsett();
        case({5,51})
%             [A, b, c] = get_irk_p5_radau();
            [A, b, c] = get_irk_p5_aai();
        % A-stable methods
        case(12)
            [A, b, c] = get_irk_p1_radau();
        case(22)
            [A, b, c] = get_irk_p2_midpoint();
%             [A, b, c] = get_irk_p2_trapezoidal();
        case(32)
            [A, b, c] = get_irk_p3_a();
%             [A, b, c] = get_irk_p3_alt();
        case(42)
%             [A, b, c] = get_irk_p4_hh();
%             [A, b, c] = get_irk_p4_alt();
            [A, b, c] = get_irk_p34_sdirk();
        case(52)
            [A, b, c] = get_irk_p5_CooperSayfy();
        case(62)
            [A, b, c] = get_irk_p6_gauss();
        % methods that are not A-/L-stable
        case(33)
            [A, b, c] = get_irk_p3_hh();
%             [A, b, c] = get_irk_p3_CesKunz();
        case(43)
            [A, b, c] = get_irk_p4_butcher();
%             [A, b, c] = get_irk_p4_CesKunz();
        case(44)
            [A, b, c] = get_irk_p4_ssp();
        otherwise
            error('Explicit Runge-Kutta method of order %i not implemented', p);
    end
    if strcmpi(eqn, 'Dahlquist')
%         rz  = 1.0 + sum(z * b * inv(eye(size(A)) - z * A));
        rz  = det(eye(size(A)) - z * A + z * ones(length(b), 1) * b) ...
            / det(eye(size(A)) - z * A);
    elseif strcmpi(eqn, 'Advection') ...
            || strcmpi(eqn, '1D-Diffusion') || strcmpi(eqn, '2D-Diffusion') ...
            || strcmpi(eqn, 'harmonic-oscillator') ...
            || strcmpi(eqn, 'unit-disk')
        rz  = zeros(size(z));
        for evalIdx = 1:length(z)
%             rz(evalIdx) = 1.0 + sum(z(evalIdx) * b * inv(eye(size(A)) - z(evalIdx) * A));
            rz(evalIdx) = det(eye(size(A)) - z(evalIdx) * A + z(evalIdx) * ones(length(b), 1) * b) ...
                / det(eye(size(A)) - z(evalIdx) * A);
        end
    else
        error('%s equation not implemented', eqn);
    end
end

%% get Butcher tableaux

% implicit Euler method, SDIRK1 (L-stable)
% [Dobrev et al. (2017)]
function [A, b, c] = get_irk_p1_sdirk()
    A = 1.0;
    b = 1.0;
    c = 1.0;
end

% implicit 2nd-order method, SDIRK2 (L-stable)
% [Dobrev et al. (2017)]
function [A, b, c] = get_irk_p2_sdirk()
    gamma   = 1.0 - 1.0 / sqrt(2.0);
    A       = [gamma, 0.0; ...
               1.0-gamma, gamma];
    b       = A(end, :);
    c       = [gamma; ...
               1.0];
end

% implicit 3rd-order method, SDIRK33 (L-stable)
function [A, b, c] = get_irk_p33_sdirk()
% [Dobrev et al. (2017)]
% see also talk by Butcher: https://www.math.auckland.ac.nz/~butcher/CONFERENCES/TRONDHEIM/trondheim.pdf
% see also MFEM, http://mfem.github.io/doxygen/html/ode_8cpp_source.html
    q   = 0.435866521508458999416019;
    r   = 1.20849664917601007033648;
    s   = 0.717933260754229499708010;
    A   = [q, 0.0, 0.0; ...
           s-q, q, 0.0; ...
           r, 1.0-q-r, q];
    b   = A(end, :);
    c   = [q; ...
           s; ...
           1.0];
    % note: see MFEM, http://mfem.github.io/doxygen/html/ode_8cpp_source.html
%     gamma   = 1.0 - 1.0 / sqrt(2.0);
%     A       = [gamma, 0.0; ...
%                1.0-2.0*gamma, gamma];
%     b       = [0.5, 0.5];
%     c       = [gamma; ...
%                1.0-gamma];
end

% implicit 4th-order method, SDIRK34 (A-stable, not L-stable)
function [A, b, c] = get_irk_p34_sdirk()
    % note: see MFEM, http://mfem.github.io/doxygen/html/ode_8cpp_source.html
    q   = 1.0 / sqrt(3.0 ) * cos(pi / 18.0) + 0.5;
    r   = 1.0 / (6.0 *(2.0 * q - 1.0) * (2.0 * q - 1.0));
    A   = [q, 0.0, 0.0; ...
           0.5-q, q, 0.0; ...
           2.0*q, 1.0-4.0*q, q];
    b   = [r, 1.0-2.0*r, r];
    c   = [q; 0.5; 1.0-q];
end

% implicit 4th-order method, SDIRK4 (L-stable)
% HairerWanner1996, Table IV.6.5
% also in DuarteDobbinsSmooke2016, Appendix C
function [A, b, c] = get_irk_p4_sdirk()
    A   = [0.25, 0.0, 0.0, 0.0, 0.0; ...
           0.5, 0.25, 0.0, 0.0, 0.0; ...
           17.0/50.0, -1.0/25.0, 0.25, 0.0, 0.0; ...
           371.0/1360.0, -137.0/2720.0, 15.0/544.0, 0.25, 0.0; ...
           25.0/24.0, -49.0/48.0, 125.0/16.0, -85.0/12.0, 0.25];
    b   = A(end, :);
    c   = [0.25; ...
           0.75; ...
           11.0/20.0; ...
           0.5; ...
           1.0];
end

% implicit 4th-order SSP Runge-Kutta method (L-stable?)
% KetchesonMacdonaldGottlieb2008
% see http://people.maths.ox.ac.uk/~macdonald/imssp.pdf
function [A, b, c] = get_irk_p4_ssp()
    mu      = [0.119309657880174,   0.000000000000000,  0.000000000000000,  0.000000000000000;  ...
               0.226141632153728,   0.070605579799433,  0.000000000000000,  0.000000000000000;  ...
               0.000000000000000,   0.180764254304414,  0.070606483961727,  0.000000000000000;  ...
               0.000000000000000,   0.000000000000000,  0.212545672537219,  0.119309875536981;  ...
               0.010888081702583,   0.034154109552284,  0.000000000000000,  0.181099440898861];
    lam     = [0.000000000000000,   0.000000000000000,  0.000000000000000,  0.000000000000000;  ...
               1.000000000000000,   0.000000000000000,  0.000000000000000,  0.000000000000000;  ...
               0.000000000000000,   0.799340893504885,  0.000000000000000,  0.000000000000000;  ...
               0.000000000000000,   0.000000000000000,  0.939878564212065,  0.000000000000000;  ...
               0.048147179264990,   0.151029729585865,  0.000000000000000,  0.800823091149145];
    mu0     = mu(1:end-1, :);
    mu1     = mu(end, :);
    lam0    = lam(1:end-1, :);
    lam1    = lam(end, :);
    A       = inv(eye(size(lam0)) - lam0) * mu0;
    b       = mu1 + lam1 * inv(eye(size(lam0)) - lam0) * mu0;
    c       = sum(A, 2);
end

% implicit 3rd-order Radau IIA method, A- and L-stable
% check: A nonsingular and last row of A coincides with b
function [A, b, c] = get_irk_p3_radau()
    A = [5.0/12.0, -1.0/12.0; ...
         3.0/4.0, 1.0/4.0];
    b = [3.0/4.0, 1.0/4.0];
    c = [1.0/3.0; ...
         1.0];
end

% implicit 2nd-order DIRK scheme (L-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 217
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p2_miller()
    A = [1.0/3.0, 0.0; ...
         0.75, 0.25];
    b = [0.75, 0.25];
    c = [1.0/3.0; ...
         1.0];
end

% implicit ??-order DIRK scheme (L-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 217
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_pU2_miller()
    A = [1.0, 0.0, 0.0; ...
         -1.0/12.0, 5.0/12.0, 0.0; ...
         0.0, 0.75, 0.25];
    b = [0.0, 0.75, 0.25];
    c = [1.0; ...
         1.0/3.0; ...
         1.0];
end

% implicit 4th-order block-diagonal IRK scheme (L-stable)
% JacksonNorsett1995, Figure 5
% see https://epubs.siam.org/doi/pdf/10.1137/0732002
function [A, b, c] = get_irk_p4_norsett()
    A = [5.0/12.0, 1.0/12.0-sqrt(3.0)/6.0, 0.0, 0.0; ...
         1.0/12.0+sqrt(3.0)/6.0, 5.0/12.0, 0.0, 0.0; ...
         0.0, 0.0, 0.5, -sqrt(3.0)/6.0; ...
         0.0, 0.0, sqrt(3.0)/6.0, 0.5];
    b = [1.5, 1.5, -1.0, -1.0];
    c = [0.5-sqrt(3.0)/6.0; ...
         0.5+sqrt(3.0)/6.0; ...
         0.5-sqrt(3.0)/6.0; ...
         0.5+sqrt(3.0)/6.0];
end

% implicit 5th-order method, Radau IIA (L-stable)
% HairerNorsettWanner1987, Table 7.7
function [A, b, c] = get_irk_p5_radau()
    A = [(88.0-7.0*sqrt(6.0))/360.0, (296.0-169*sqrt(9))/1800.0, (-2.0+3.0*sqrt(6.0))/225.0; ...
         (296.0+169*sqrt(9))/1800.0, (88.0+7.0*sqrt(6.0))/360.0, (-2.0-3.0*sqrt(6.0))/225.0; ...
         (16.0-sqrt(6.0))/36.0, (16.0+sqrt(6.0))/36.0, 1.0/9.0];
    b = [(16.0-sqrt(6.0))/36.0, (16.0+sqrt(6.0))/36.0, 1.0/9.0];
    c = [(4.0-sqrt(6.0))/10.0; ...
         (4.0+sqrt(6.0))/10.0; ...
         1.0];
end

% implicit 5th-order method (L-stable)
% AbabnehAhmadIsmail2009
% see https://pdfs.semanticscholar.org/b129/c01b99d671a852d13240eaabef43815e518c.pdf
function [A, b, c] = get_irk_p5_aai()
    a41 = (59765462671.0 - 2469071899.0 * sqrt(41.0)) / 269065110000.0;
    a42 = (26775007261.0 + 244199891.0 * sqrt(41.0)) / 89286180000.0;
    a43 = (889326089143.0 - 203592224167.0 * sqrt(41.0)) / 19910818140000.0;
    A   = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0; ...
           -1.0/12.0, 0.25, 0.0, 0.0, 0.0, 0.0; ...
           (73.0+12.0*sqrt(41.0))/150.0, (24.0-19.0*sqrt(41.0))/300.0, 0.25, 0.0, 0.0, 0.0; ...
           a41, a42, a43, 0.25, 0.0, 0.0; ...
           0.0, 15.0/37.0, (2091.0-879.0*sqrt(41.0))/12136.0, (2091.0+879.0*sqrt(41.0))/12136.0, 0.25, 0.0; ...
           0.0, 15.0/37.0, (2091.0-879.0*sqrt(41.0))/12136.0, (2091.0+879.0*sqrt(41.0))/12136.0, 0.25, 0.0];
    c   = [0.25; ...
           1.0/6.0; ...
           (49.0+sqrt(41.0))/60; ...
           (49.0-sqrt(41.0))/60; ...
           1.0; ...
           1.0];
    b = [0.0, 15.0/37.0, (2091.0-879.0*sqrt(41.0))/12136.0, (2091.0+879.0*sqrt(41.0))/12136.0, 0.25, 0.0];
    A = A(1:end-1, 1:end-1);
    b = b(1:end-1);
    c = c(1:end-1);
end

%% A-stable

% implicit 1st-order method, Radau IIA (A-stable but also L-stable as it coincides with Backward Euler..)
% see https://www.math.auckland.ac.nz/~butcher/ODE-book-2008/Tutorials/IRK.pdf
function [A, b, c] = get_irk_p1_radau()
    A   = 1.0;
    b   = 1.0;
    c   = 1.0;
end

% implicit 2nd-order midpoint method (A-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 213
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p2_midpoint()
    A = 0.5;
    b = 1.0;
    c = 0.5;
end

% implicit 2nd-order trapezoidal method, Lobatto IIIA (A-stable)
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p2_trapezoidal()
    A = [0.0, 0.0; ...
         0.5, 0.5];
    b = [0.5, 0.5];
    c = [0.0; ...
         1.0];
end

% implicit 3rd-order Kutta method (A-stable)
function [A, b, c] = get_irk_p3_a()
    g   = (3.0 + sqrt(3.0)) / 6.0;
    A   = [g, 0.0; ...
           1.0-2.0*g, g];
    b   = [0.5, 0.5];
    c   = [g; ...
           1.0-g];
% % AbabnehAhmad2009, see http://iopscience.iop.org/article/10.1088/0256-307X/26/8/080503/pdf
% % might be L-stable, not sure:
%     A   = [5.0/4.0, 0.0, 0.0; ...
%            (-24.0+sqrt(2.0))/28.0, 5.0/4.0, 0.0; ...
%            (14250.0-563.0*sqrt(2.0))/2583.0, 98.0*(-24.0+sqrt(2.0))/369.0, 5.0/4.0];
%     b   = [0.25, 0.75, 0.0];
%     c   = [];
end

% implicit 3rd-order EDIRK scheme (A-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 216
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p3_alt()
    A = [0.0, 0.0, 0.0; ...
         0.75, 0.75, 0.0; ...
         7.0/18.0, -4.0/18.0, 15.0/18.0];
    b = [7.0/18.0, -4.0/18.0, 15.0/18.0];
    c = [0.0; ...
         1.5; ...
         1.0];
end

% implicit 4th-order EDIRK scheme (A-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 216
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p4_alt()
    A = [0.0, 0.0, 0.0, 0.0; ...
         0.75, 0.75, 0.0, 0.0; ...
         447.0/675.0, -357.0/675.0, 855.0/675.0, 0.0; ...
         13.0/42.0, 84.0/42.0, -125.0/42.0, 70.0/42.0];
    b = [13.0/42.0, 84.0/42.0, -125.0/42.0, 70.0/42.0];
    c = [0.0; ...
         1.5; ...
         7.0/5.0; ...
         1.0];
end

% implicit 4th-order method, Hammer & Hollingsworth (A-stable)
% note: coincides with s=2-stage, p=2s-order Gauss method
%       see https://www.math.auckland.ac.nz/~butcher/ODE-book-2008/Tutorials/IRK.pdf
function [A, b, c] = get_irk_p4_hh()
    A = [0.25, 0.25-sqrt(3.0)/6.0; ...
         0.25+sqrt(3.0)/6.0, 0.25];
    b = [0.5, 0.5];
    c = [0.5-sqrt(3.0)/6.0; ...
         0.5+sqrt(3.0)/6.0];
end

% implicit 4th-order method, Lobatto IIIA (A-stable)
% HairerNorsettWanner1987, Table 7.7
function [A, b, c] = get_irk_p4_lobatto()
    A = [0.0, 0.0, 0.0; ...
         5.0/24.0, 1.0/3.0, -1.0/24.0; ...
         1.0/6.0, 2.0/3.0, 1.0/6.0];
    b = [1.0/6.0, 2.0/3.0, 1.0/6.0];
    c = [0.0; ...
         0.5; ...
         1.0];
end

% implicit 5th-order method (A-stable)
% CooperSayfy1979, p. 551
% see https://www.ams.org/journals/mcom/1979-33-146/S0025-5718-1979-0521275-1/S0025-5718-1979-0521275-1.pdf
% note: coefficents also given (some typos) in AbabnehAhmadIsmail2009
% https://pdfs.semanticscholar.org/b129/c01b99d671a852d13240eaabef43815e518c.pdf
function [A, b, c] = get_irk_p5_CooperSayfy()
    A = [(6.0-sqrt(6.0))/10.0, 0.0, 0.0, 0.0, 0.0, 0.0; ...
         (-6.0+5.0*sqrt(6.0))/14.0, (6.0-sqrt(6.0))/10.0, 0.0, 0.0, 0.0, 0.0; ...
         (888.0+607.0*sqrt(6.0))/2850.0, (126.0-161.0*sqrt(6.0))/1425.0, (6.0-sqrt(6.0))/10.0, 0.0, 0.0, 0.0; ...
         (3153.0-3082.0*sqrt(6.0))/14250.0, (3213.0+1148.0*sqrt(6.0))/28500.0, (-267.0+88.0*sqrt(6.0))/500.0, (6.0-sqrt(6.0))/10.0, 0.0, 0.0; ...
         (-32583.0+14638.0*sqrt(6.0))/71250.0, (-17199.0+364.0*sqrt(6.0))/142500.0, (1329.0-544.0*sqrt(6.0))/2500.0, (-96.0+131.0*sqrt(6.0))/625.0, (6.0-sqrt(6.0))/10.0, 0.0; ...
         0.0, 0.0, 1.0/9.0, (16.0-sqrt(6.0))/36.0, (16.0+sqrt(6.0))/36.0, 0.0];
    c = [(6.0-sqrt(6.0))/10.0, ...
         (6.0+9.0*sqrt(6.0))/35.0, ...
         1.0, ...
         (4.0-sqrt(6.0))/10.0, ...
         (4.0+sqrt(6.0))/10.0, ...
         1.0];
    b = [0.0, 0.0, 1.0/9.0, (16.0-sqrt(6.0))/36.0, (16.0+sqrt(6.0))/36.0, 0.0];
    A = A(1:end-1, 1:end-1);
    b = b(1:end-1);
    c = c(1:end-1);
end

% implicit 6th-order Gauss method (A-stable)
% see https://www.math.auckland.ac.nz/~butcher/ODE-book-2008/Tutorials/IRK.pdf
function [A, b, c] = get_irk_p6_gauss()
    A = [5.0/36.0, 2.0/9.0-sqrt(15.0)/15.0, 5.0/36.0-sqrt(15.0)/30.0; ...
         5.0/36.0+sqrt(15.0)/24.0, 2.0/9.0, 5.0/36.0-sqrt(15.0)/24.0; ...
         5.0/36.0+sqrt(15.0)/30.0, 2.0/9.0+sqrt(15.0)/15.0, 5.0/36.0];
    b = [5.0/18.0, 4.0/9.0, 5.0/18.0];
    c = [0.5-sqrt(15.0)/10.0; ...
         0.5; ...
         0.5+sqrt(15.0)/10.0];
end

%% neither A- nor L-stable

% implicit 3rd-order Kutta method (not A-/L-stable)
function [A, b, c] = get_irk_p3_noAL()
    % note: see MFEM, http://mfem.github.io/doxygen/html/ode_8cpp_source.html
    g   = (3.0 - sqrt(3.0)) / 6.0;
    A   = [g, 0.0; ...
           1.0-2.0*g, g];
    b   = [0.5, 0.5];
    c   = [g; ...
           1.0-g];
end

% implicit 3rd-order E(S)DIRK scheme (not A-/L-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 214
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p3_hh()
    A = [0.0, 0.0; ...
         1.0/3.0, 1.0/3.0];
    b = [0.25, 0.75];
    c = [0.0; ...
         2.0/3.0];
end

% implicit 4th-order EDIRK (Lobatto III) scheme (not A-/L-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 214
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p4_butcher()
    A = [0.0, 0.0, 0.0; ...
         0.25, 0.25, 0.0; ...
         0.0, 1.0, 0.0];
    b = [1.0/6.0, 4.0/6.0, 1.0/6.0];
    c = [0.0; ...
         0.5; ...
         1.0];
end

% implicit 3rd-order EDIRK scheme (not A-/L-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 215
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p3_CesKunz()
    A = [0.0, 0.0, 0.0; ...
         0.25, 0.25, 0.0; ...
         1.0/6.0, 4.0/6.0, 1.0/6.0];
    b = [1.0/6.0, 4.0/6.0, 1.0/6.0];
    c = [0.0; ...
         0.5; ...
         1.0];
end

% implicit 4th-order EDIRK (Lobatto III) scheme (not A-/L-stable)
% KennedyCarpenter2016, NASA/TM?2016?219173, Eqn. 215
% see https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf
function [A, b, c] = get_irk_p4_CesKunz()
    A = [0.0, 0.0, 0.0, 0.0; ...
         1.0/6.0, 1.0/6.0, 0.0, 0.0; ...
         1.0/12.0, 0.5, 1.0/12.0, 0.0; ...
         1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0];
    b = [1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0];
    c = [0.0; ...
         1.0/3.0; ...
         2.0/3.0; ...
         1.0];
end
