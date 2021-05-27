% All of this data is taken from Appendix A "Optimal Coefficients" of
% "Preconditioning of fully implicit Runge-Kutta schemes for parabolic
% PDEs" by G. A. Staff et al. (2006), Modeling, identification and control,
% VOL. 27, No. 2, 109--123
%
% Here, the methods are denoted as X(p), which means it belongs to the
% family of methods X and is of order p.

function A0 = Staff_etal_2006_optimal_coefficients(method)

switch method
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% ------ Gauss--Legendre methods, Gauss(p) uses p/2 stages ------ %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'Gauss4'
        A0 = [0.25,     0;...
              0.488313, 0.25];
    
    case 'Gauss6'
        A0 = [0.138888, 0,        0;...
              0.224907, 0.222222, 0;
              0.143025, 0.387432, 0.138888];
    
    case 'Gauss8'
        A0 = [0.086963, 0,        0,        0;...
              0.171390, 0.163036, 0,        0;....
              0.192773, 0.273261, 0.163036, 0;...
              0.245927, 0.232027, 0.273809, 0.086963];
          
    case 'Gauss10'
        A0 = [0.059231, 0,        0,        0,        0;...
              0.094654, 0.119657, 0,        0,        0;...
              0.118474, 0.226545, 0.142222, 0,        0;...
              0.156695, 0.244621, 0.242734, 0.119657, 0;...
              0.108481, 0.287240, 0.227631, 0.206980, 0.059231]; 
      
          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% ------ Radau IIA methods, RadauIIA(p) uses (p+1)/2 stages ----- %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    case 'RadauIIA3'
        A0 = [0.416666, 0;...
              0.673076, 0.25];
          
    case 'RadauIIA5'
        A0 = [0.196815, 0,        0;...
              0.259583, 0.292073, 0;...
              0.194743, 0.41444,  0.111111]; % There's a typo in the middle entry of this line in the paper. Not sure what the number is supposed to be.
          
    case 'RadauIIA7'
        A0 = [0.112999, 0,        0,        0;...
              0.207430, 0.206892, 0,        0;...
              0.280581, 0.238590, 0.189036, 0;...
              0.321615, 0.194202, 0.255668, 0.0625];
          
    case 'RadauIIA9'
        A0 = [0.072998, 0,        0,        0,        0;...
              0.134217, 0.146214, 0,        0,        0;...
              0.166967, 0.191017, 0.167585, 0,        0;...
              0.181347, 0.188433, 0.174109, 0.128756, 0;...
              0.168265, 0.212583, 0.132551, 0.176719, 0.04];
          
          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% -- Lobatto IIIC methods, LobattoIIIC(p) uses (p+2)/2 stages --- %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    case 'LobIIIC2'
        A0 = [0.5, 0;...
              0, 0.5];
          
    case 'LobIIIC4'
        A0 = [ 0.1666666, 0,        0;...
              -0.125000,  0.416666, 0;...
              -0.1666666, 0.606060, 0.1666666];
         
    case 'LobIIIC6'
        A0 = [ 0.083333, 0,        0,        0;...
              -0.031715, 0.25,     0,        0;...
               0.070601, 0.508398, 0.25,     0;...
               0.132073, 0.522927, 0.483915, 0.083333];
          
    otherwise
        error('IRK method %s not implemented/recognised.', method)
              
end