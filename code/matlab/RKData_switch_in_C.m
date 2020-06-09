% Print a switch statement that covers all IRK schemes

clear
clc

% Get list and description of the RK schemes we consider
schemes = RKSchemes();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAIN PART OF CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf4('switch(ID) {\n')
for i = 1:numel(schemes)
   fprintf8('// %s\n', schemes{i}{2}) 
   fprintf8('case Type::%s:\n\n', schemes{i}{1})
   fprintf12('break;\n\n')
end
fprintf8('default:\n')
fprintf12('%s\n', 'mfem_error("RKData:: Invalid Runge Kutta type.\n");')
fprintf4('}\n\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Print function that indents 4 space before printing.
function fprintf4(varargin)
    fprintf('    ') 
    fprintf(varargin{:})
end

% Print function that indents 8 space before printing.
function fprintf8(varargin)
    fprintf('        ') 
    fprintf(varargin{:})
end

% Print function that indents 12 spaces before printing.
function fprintf12(varargin)
    fprintf('            ') 
    fprintf(varargin{:})
end