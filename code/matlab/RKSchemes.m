function schemes = RKSchemes()

% List of all the IRK schemes (and their descriptions) we consider 
schemes = {
% A-stable SDIRK
{'ASDIRK3'; '2-stage 3rd-order A-stable SDIRK'};
{'ASDIRK4'; '3-stage 4th-order A-stable SDIRK'};
% L-stable SDIRK
{'LSDIRK1'; '1-stage 1st-order L-stable SDIRK'};
{'LSDIRK2'; '2-stage 2nd-order L-stable SDIRK'};
{'LSDIRK3'; '3-stage 3rd-order L-stable SDIRK'};
{'LSDIRK4'; '5-stage 4th-order L-stable SDIRK'};
% Gauss
{'Gauss2'; '1-stage 2nd-order Gauss--Legendre'};
{'Gauss4'; '2-stage 4th-order Gauss--Legendre'};
{'Gauss6'; '3-stage 6th-order Gauss--Legendre'};
{'Gauss8'; '4-stage 8th-order Gauss--Legendre'};
{'Gauss10'; '5-stage 10th-order Gauss--Legendre'};
% RadauIIA 
{'RadauIIA3'; '2-stage 3rd-order Radau IIA'};
{'RadauIIA5'; '3-stage 5th-order Radau IIA'};
{'RadauIIA7'; '4-stage 7th-order Radau IIA'};
{'RadauIIA9'; '5-stage 9th-order Radau IIA'};
% Lobatto IIIC  
{'LobIIIC2'; '2-stage 2nd-order Lobatto IIIC'};
{'LobIIIC4'; '3-stage 4th-order Lobatto IIIC'};
{'LobIIIC6'; '4-stage 6th-order Lobatto IIIC'};
{'LobIIIC8'; '5-stage 8th-order Lobatto IIIC'};
};

end