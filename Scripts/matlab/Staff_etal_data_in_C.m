% Print out optimized preconditioner coefficients from Staff et al. (2006) 
% so that it can be copied straight into C/C++ code
%
% Just uncomment the ID you want below and run this script. All data will
% be printed to the console

clc
clear

% This sets the precision of all computations to 32 digits...
digits(32)

mystr = '';

%%% --- Set up the formatting of the way in which the data is printed to console --- %%%
% Label these things as they're defined in the code so the output makes sense...
A_label    = 'preconditionerCoefficients'; 

% Data will be output to console as. Print 15 digits of data...
outformat1 = @(label, i, data) myfprintf('%s(%d) = %+.15f;\n', label, i, data);
outformat2 = @(label, i, j, data) myfprintf('%s(%d, %d) = %+.15f;\n', label, i, j, data);

%%% --- Tables --- %%% 
% UNCOMMENT ONE OF ME AND HIT RUN!
ID = 'Gauss2';
ID =  'Gauss4';
ID =  'Gauss6';
ID =  'Gauss8';
ID =  'Gauss10';
% 
ID = 'RadauIIA3';
ID = 'RadauIIA5';
ID = 'RadauIIA7';
ID = 'RadauIIA9';
%  
ID =  'LobIIIC2';
ID =  'LobIIIC4';
ID =  'LobIIIC6';
%ID =  'LobIIIC8';

% Get table
[A] = Staff_etal_2006_optimal_coefficients(ID);
s = size(A,1);


%%% --- Now print everything to console... --- %%%
myfprintf('/* ID: %s */\n', ID)
for i = 1:s
    for j = 1:s
        outformat2(A_label, i-1, j-1, A(i,j))
    end
end

% Print function that indents before printing.
function myfprintf(varargin)
    fprintf('                    ') % Indent 20 spaces
    fprintf(varargin{:})
end