#!/bin/bash

# Run the "driver" executable in ../IRK_class for many different temporal/spatial refinements.
#
# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory within the current directory.


date

max_amgiter_type2=5

outdir=data2 # Where the data is sent to

np=4        # Number of processes
atol=1e-15; rtol=1e-15

# IRK == IRK method; space == Order of spatial discretization

### --- SDIRK --- ###
# IRK=1; space=1  
# IRK=2; space=2  
# IRK=3; space=3  
IRK=4; space=4; atol=1e-15; rtol=1e-15    

### --- Gauss --- ###
#IRK=12; space=2  
#IRK=14; space=4  
#IRK=16; space=6; #atol=1e-11; rtol=1e-11    
IRK=18; space=8; atol=1e-15; rtol=1e-15  
#IRK=110; space=10; atol=1e-15; rtol=1e-15  

### --- RadauIIA# --- ###
# IRK=23; space=3; #atol=1e-6; rtol=1e-6   
# IRK=25; space=5; atol=1e-10; rtol=1e-10     
# IRK=27; space=7; #atol=1e-13; rtol=1e-13    
# IRK=29; space=9; atol=1e-15; rtol=1e-15    

### --- LobIIIC --- ###
# IRK=32; space=2  
# IRK=34; space=4  
# IRK=36; space=6; atol=1e-15; rtol=1e-15     
# IRK=38; space=8; atol=1e-15; rtol=1e-15   


# dt will go from [2^-dt_min_refine, ..., 2^-dt_max_refine]
dt_min_refine=3
dt_max_refine=7

tf=2   # Final integration time

dim=1  # Spatial dimension
FD=4   # ID of problem to be solved

save=1 # Save only the text file output from the problem and not the solution


# Name of file to be output... "^REFINE^" will be replaced with the actual refinement...
dir=$outdir/"$time_type"
out=IRK"$IRK"_dt^REFINE^_d"$dim"_FD"$FD"_cfl1

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
    mpirun -np $np ../IRK_class/driver -t $IRK -dt $dt -tf $tf -o $space -l $(($dt_refine+1)) \
        -d $dim -FD $FD -atol $atol -rtol $rtol -maxit 250 -kdim 50 -save $save \
        -out $dir${out/^REFINE^/$dt_refine} -maxit_AMG $max_amgiter_type2

done
