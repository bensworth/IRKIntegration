#!/bin/bash

# Run the "driver" executable in ../IRK_class for many different temporal/spatial refinements.
#
# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory within the current directory.


date

outdir=data_test # Where the data is sent to

np=4        # Number of processes
# Newton parameters
newton_print=-1;
nrtol=1e-10; natol=1e-10;  
nmaxit=30
# Krylov parameters
kp=0; 
katol=1e-13; 
krtol=1e-13; 
kmaxit=50; kdim=50

# Problem parameters
dim=2  # Spatial dimension
ax=0.85; ay=1.0; mx=0.3; my=0.25 # PDE coefficients
flux=1 # Flux function
ex=1   # example problem to be solved
tf=2   # Final integration time


# IRK == IRK method; space == Order of spatial discretization

### --- ASDIRK --- ###
IRK=-14; space=4  

### --- LSDIRK --- ###
#IRK=1; space=2  
#IRK=2; space=2  
#IRK=3; space=4  
#IRK=4; space=4; 

### --- Gauss --- ###
#IRK=12; space=2  
#IRK=14; space=4  
#IRK=16; space=6; 
#IRK=18; space=8; 
#IRK=110; space=10; 

### --- RadauIIA# --- ###
#IRK=23; space=4; 
#IRK=25; space=6; 
#IRK=27; space=8; 
#IRK=29; space=10; 

### --- LobIIIC --- ###
#IRK=32; space=2  
#IRK=34; space=4  
#IRK=36; space=6; 
#IRK=38; space=8; 


# dt will go from [2^-dt_min_refine, ..., 2^-dt_max_refine]
dt_min_refine=2
dt_max_refine=5

save=1 # Save only the text file output from the problem and not the solution


# Name of file to be output... "^REFINE^" will be replaced with the actual refinement...
dir=$outdir/"$time_type"
out=IRK"$IRK"_dt^REFINE^_d"$dim"_ex"$ex" 

# Run solves at different temporal refinements...
echo "solving..."
for dt_refine in `seq $dt_min_refine $dt_max_refine`
do
    # Use Python to compute dt == 2^-dt_refine
    export dt_refine
    dt="$(python -c 'import os; print(2**-int(os.environ["dt_refine"]))')"    
    echo "\n--------------------------------------"
    echo "Temporal refinement = $dt_refine; dt = $dt"

    # Run solver at current refinement
    # Run with dt = 2*dx
    mpirun -np $np ../../IRK_nonlin_class/driver_adv_dif -t $IRK -dt $dt -tf $tf \
        -o $space -l $(($dt_refine+2)) \
        -d $dim -f $flux -ex $ex \
        -ax $ax -ay $ay -mx $mx -my $my \
        -nrtol $nrtol -natol $natol -nmaxit $nmaxit -np $newton_print \
        -katol $katol -krtol $krtol -kmaxit $kmaxit -kdim $kdim -kp $kp \
        -save $save -out $dir${out/^REFINE^/$dt_refine}

done
