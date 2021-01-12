epsilon = 12; % Solve to relative tolerance of 10^-epsilon
iters = @(Z_gamma) -2*epsilon./log10(Z_gamma); % Number of GMRES iterations

% The variable (beta/eta)^2
x = linspace(0, 8, 1e3); % Note one of the bounds is only valid for x < 8


%%% SPSD case: GMRES bounds as a function of (beta/eta)^2
Z_spd = @(kappa) 1 - 1./kappa.^2; % General SPD GMRES bound for condition number kappa
kappa_gamma_star = @(x) 0.5*(1 + sqrt(1 + x));
kappa_eta = @(x) 1 + x;

figure(1)
semilogy(x, iters(Z_spd(kappa_gamma_star(x))), 'b', 'LineWidth', 2, 'DisplayName', 'SPSD: $\gamma = \gamma_*$')
hold on
semilogy(x, iters(Z_spd(kappa_eta(x))), 'b--', 'LineWidth', 2, 'DisplayName', 'SPSD: $\gamma = \eta$')

figure(2)
semilogy(x, iters(Z_spd(kappa_eta(x)))./iters(Z_spd(kappa_gamma_star(x))), 'b', 'LineWidth', 2, 'DisplayName', 'SPSD')
hold on



%%% SS case: GMRES bounds as a function of (beta/eta)^2
Z_gamma_star = @(x) 1 - 4./(x + 2).^2;
Z_eta = @(x) 9/64*x.*(7*x + 16)./(x + 1).^2; % Note this is only valid for x < 8!

figure(1)
hold on
semilogy(x, iters(Z_gamma_star(x)), 'r', 'LineWidth', 2, 'DisplayName', 'SS: $\gamma = \gamma_*$')
semilogy(x, iters(Z_eta(x)), 'r--', 'LineWidth', 2, 'DisplayName', 'SS: $\gamma = \eta$')

% General bound on condition number. Can use this directly for SPSD case
% where P_gamma is SPD
kappa_gamma_star_bound = @(x) 2 + 0.5*x - 0.5*x./(1 + x);
a = 20;
semilogy(x(1:a:end), iters(Z_spd(kappa_gamma_star_bound(x(1:a:end)))), 'g->', 'LineWidth', 1, 'DisplayName', 'SPSD: General bound for $\gamma = \gamma_*$')


figure(2)
hold on
semilogy(x, iters(Z_eta(x))./iters(Z_gamma_star(x)), 'r', 'LineWidth', 2, 'DisplayName', 'SS')


figure(1)
set(gca, 'TickLabelInterpreter', 'LaTeX', 'FontSize', 18)
fs = {'FontSize', 22, 'Interpreter', 'Latex'};
xlabel('$(\beta/\eta)^2$', fs{:})
ylabel('GMRES iterations', fs{:})
lh = legend();
lh.set(fs{:}, 'FontSize',  22, 'Location', 'Best')



figure(2)
set(gca, 'TickLabelInterpreter', 'LaTeX', 'FontSize', 18)
fs = {'FontSize', 22, 'Interpreter', 'Latex'};
xlabel('$(\beta/\eta)^2$', fs{:})
ylabel('GMRES($\eta$)/GMRES($\gamma_*$) iterations', fs{:})
lh = legend();
lh.set(fs{:}, 'FontSize',  22, 'Location', 'Best')

figure(1)
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
file_name = strcat('gmres_iters_spsd_ss', '.pdf');
saveas(gcf, file_name);


figure(2)
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
file_name = strcat('gmres_rel_iters_spsd_ss', '.pdf');
saveas(gcf, file_name);
