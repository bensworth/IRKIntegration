% The variable (beta/eta)^2
x = linspace(0, 10, 50);


%%% SPSD case: Graph condition numbers as a function of (beta/eta)^2
kappa_gamma_star = @(x) 0.5*(1 + sqrt(1 + x));
kappa_eta = @(x) 1 + x;
plot(x, kappa_gamma_star(x), 'b', 'LineWidth', 2, 'DisplayName', 'SPSD: $\gamma = \gamma_*$')
hold on
plot(x, kappa_eta(x), 'b--', 'LineWidth', 2, 'DisplayName', 'SPSD: $\gamma = \eta$')


%%% SS case: Graph condition numbers as a function of (beta/eta)^2
kappa_gamma_star = @(x) 0.5*(2 + x);
kappa_eta = @(x) 0.5*sqrt(4 + x).*(1 + x);
plot(x, kappa_gamma_star(x), 'r', 'LineWidth', 2, 'DisplayName', 'SS: $\gamma = \gamma_*$')
hold on
plot(x, kappa_eta(x), 'r--', 'LineWidth', 2, 'DisplayName', 'SS: $\gamma = \eta$')

set(gca, 'TickLabelInterpreter', 'LaTeX', 'FontSize', 18)
fs = {'FontSize', 22, 'Interpreter', 'Latex'};
xlabel('$(\beta/\eta)^2$', fs{:})
ylabel('Condition number $\kappa(\mathcal{P}_{\gamma})$', fs{:})
lh = legend();
lh.set(fs{:}, 'FontSize',  22, 'Location', 'Best')


% General upper bound using the theory Ben has for L1=L2.
%kappa_gamma_star_bound = @(x) 2 + 0.5*( x.^2 ./ (x + 1));
kappa_gamma_star_bound = @(x) 2 + 0.5*x - 0.5*x./(1 + x);
plot(x, kappa_gamma_star_bound(x), 'g->', 'LineWidth', 1, 'DisplayName', 'General bound for $\gamma = \gamma_*$')


% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% file_name = strcat('conds_spsd_ss', '.pdf');
% saveas(gcf, file_name);