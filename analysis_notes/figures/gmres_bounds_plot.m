% The variable (beta/eta)^2
x = linspace(0, 8, 50); % Note one of the bounds is only valid for x < 8


%%% SPSD case: GMRES bounds as a function of (beta/eta)^2
Z_spd = @(kappa) (kappa.^2 - 1)./kappa.^2; % General SPD GMRES bound for condition number kappa
kappa_gamma_star = @(x) 0.5*(1 + sqrt(1 + x));
kappa_eta = @(x) 1 + x;
plot(x, Z_spd(kappa_gamma_star(x)), 'b', 'LineWidth', 2, 'DisplayName', 'SPSD: $\gamma = \gamma_*$')
hold on
plot(x, Z_spd(kappa_eta(x)), 'b--', 'LineWidth', 2, 'DisplayName', 'SPSD: $\gamma = \eta$')

% Look at CG bounds. These don't really make much sense in this context,
% but still interesting to see how the different condition numbers play out
% CG_bound = @(kappa) (sqrt(kappa) - 1)./(sqrt(kappa) + 1);
% plot(x, CG_bound(kappa_gamma_star(x)), 'g', 'LineWidth', 2, 'DisplayName', 'SPSD: CG bound$(\mathcal{P}_{\gamma_*})$')
% plot(x, CG_bound(kappa_eta(x)), 'g--', 'LineWidth', 2, 'DisplayName', 'SPSD: CG bound$(\mathcal{P}_{\eta})$')


%%% SS case: GMRES bounds as a function of (beta/eta)^2
Z_gamma_star = @(x) x./(x + 2);
Z_eta = @(x) x.*(8*x + 17)./(8*(x + 1).^2); % Note this is only valid for x < 8!
plot(x, Z_gamma_star(x), 'r', 'LineWidth', 2, 'DisplayName', 'SS: $\gamma = \gamma_*$')
hold on
plot(x, Z_eta(x), 'r--', 'LineWidth', 2, 'DisplayName', 'SS: $\gamma = \eta$')


% General bound on condition number. Can use this directly for SPSD case
% where P_gamma is SPD
kappa_gamma_star_bound = @(x) 2 + 0.5*x - 0.5*x./(1 + x);
plot(x, Z_spd(kappa_gamma_star_bound(x)), 'g->', 'LineWidth', 1, 'DisplayName', 'SPSD: General bound for $\gamma = \gamma_*$')


set(gca, 'TickLabelInterpreter', 'LaTeX', 'FontSize', 18)
fs = {'FontSize', 22, 'Interpreter', 'Latex'};
xlabel('$(\beta/\eta)^2$', fs{:})
ylabel('GMRES bound $\mathcal{Z}_{\gamma}$', fs{:})
lh = legend();
lh.set(fs{:}, 'FontSize',  22, 'Location', 'Best')

% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% file_name = strcat('gmres_bounds_spsd_ss', '.pdf');
% saveas(gcf, file_name);
