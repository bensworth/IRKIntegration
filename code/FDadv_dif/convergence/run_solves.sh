#!/bin/bash

# Run the "driver" executable in ../ for many different temporal/spatial refinements.
#
# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory within the current directory.


date
outdir=data # Where the data is sent to
np=4        # Number of processes


### --- PDE PARAMETERS
flux=1          # Nonlinear flux
problem=1       # ID of example problem to be solved
ax=0.85; ay=1.0 # Wave speeds
mx=0.3; my=0.25 # Diffusivities
tf=2            # Final integration time
dim=2           # Spatial dimension

### --- DISCRETIZATIONS
cfl=1; # dt = cfl*dx
IRK=12; space=2  
#IRK=14; space=4  

# Ramp up solver tolerances.
katol=1e-8; krtol=1e-8
nrtol=1e-8
use_AIR=0

# dt will go from [2^-dt_min_refine, ..., 2^-dt_max_refine]
l_min_refine=8
l_max_refine=8

save=1 # Save only the text file output from the problem and not the solution

# Name of file to be output... "^REFINE^" will be replaced with the actual refinement...
dir=$outdir/
out=SDIRK"$IRK"_l^REFINE^_d"$dim"_ex"$problem"

# Run solves at different temporal refinements...
echo "solving..."
for refine in `seq $l_min_refine $l_max_refine`
do
    echo "\n--------------------------------------"
    echo "Refinement = $refine"

    # Run solver at current refinement
    mpirun -np $np ../driver -f $flux -t $IRK -cfl $cfl -tf $tf -o $space \
        -l $refine -d $dim -ex $problem -katol $katol -krtol $krtol -nrtol $nrtol \
        -ax $ax -ay $ay -mx $mx -my $my \
        -air $use_AIR \
        -save $save -out $dir${out/^REFINE^/$refine} 

done
