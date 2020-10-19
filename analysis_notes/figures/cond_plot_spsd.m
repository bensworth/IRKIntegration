clc
clear


% % Values of (beta/eta)^2 taken from linear paper
% gauss = {[0.33]; [0; 0.91]; [1.59; 0.09]; [0; 2.46; 0.27]};
% 
% radau = {[0.5]; [0; 1.29]; [2.21; 0.11]; [0; 3.2; 0.32]};
% 
% lobatto = {[1]; [0; 2.21]; [3.51; 0.13]; [0; 4.88; 0.38]};
% 
% gamma_star = @(x) 0.5*(1 + sqrt(1 + x));
% gamma_eta = @(x) 1 + x;
% 
% scheme = gauss;
% scheme = radau;
% scheme = lobatto;
% 
% for idx = 1:numel(scheme)
%    %disp(gamma_star(scheme{idx}))
%    disp(gamma_eta(scheme{idx}))
% end


% The variable (beta/eta)^2
x = linspace(0, 10);

%%% Graph condition numbers as a function of (beta/eta)^2
kappa_gamma_star = @(x) 0.5*(1 + sqrt(1 + x));
kappa_eta = @(x) 1 + x;
plot(x, kappa_gamma_star(x), 'r-', 'LineWidth', 2, 'DisplayName', '$\gamma=\gamma_*$')
hold on
plot(x, kappa_eta(x), 'b--', 'LineWidth', 2, 'DisplayName', '$\gamma = \eta$')
set(gca, 'TickLabelInterpreter', 'LaTeX', 'FontSize', 18)

fs = {'FontSize', 22, 'Interpreter', 'Latex'};
xlabel('$(\beta/\eta)^2$', fs{:})
ylabel('Condition number $\kappa(\mathcal{P}_{\gamma})$', fs{:})
lh = legend();
lh.set(fs{:}, 'FontSize',  22, 'Location', 'Best')
% 
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% file_name = strcat('conds_spsd', '.pdf');
% saveas(gcf, file_name);
