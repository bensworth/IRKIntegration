close all;

% CG convergence as function of \beta^2/\eta^2
g = @(x) (sqrt(1+x)-1)./(sqrt(1+x)+1);
g2 = @(x,p) (sqrt((1+p+x)/(1-p))-1)./(sqrt((1+p+x)/(1-p))+1); % w/ 1-p
% g2 = @(x,p) (sqrt((1+p+x)/(1))-1)./(sqrt((1+p+x)/(1))+1);   % w/o 1-p


slowdown2 = @(x,p) log(g(x))./log(g2(x,p));

x = 0.1:0.01:10;


figure;
semilogx(x, g(x),'k'); hold on;
semilogx(x, g2(x,0.25),'b'); hold on;
semilogx(x, g2(x, 0.5),'r'); hold on;
semilogx(x, g2(x,0.75),'g'); hold on;

figure;
semilogx(x, slowdown2(x,0.25),'b'); hold on;
semilogx(x, slowdown2(x, 0.5),'r'); hold on;
semilogx(x, slowdown2(x,0.75),'g'); hold on;
ylim([0,5])

tol = log(1e-6);
iter = @(x) tol./log(g(x));
iter2 = @(x,p) tol./log(g2(x,p));

figure;
semilogx(x, iter(x),'k'); hold on;
semilogx(x, iter2(x,0.25),'b'); hold on;
semilogx(x, iter2(x, 0.5),'r'); hold on;
semilogx(x, iter2(x,0.75),'g'); hold on;


