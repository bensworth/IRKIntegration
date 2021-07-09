#!/bin/bash

# Run the executable for many different space-time refinements.
#
# Also run each problem some number of times such that an average can be taken
#
# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory 
#       within the current directory.


date

# Name of executable
exe=../../driver_adv_dif_FD

# Number of processes
np=4 

# The number of times to run each problem
num_runs=3

# Dummy values to make sure there's a value passed through command line below
block_prec_ID=1; gamma=1

# IRK alg parameters, just choose one of IRK_alg = 0 or = 1. The shell code below will fix the name of the output appropriately
# CC preconditioned algorithm
IRK_alg=0; gamma=1 # Constant in preconditioner. 1 is optimal value 

# Block preconditioned algorithm
#IRK_alg=1; block_prec_ID=1 # Lower triangular Gauss--Seidel
#IRK_alg=1; block_prec_ID=4 # Rana et al. (2021)


# Krylov parameters
kp=2; 
katol=0.0;  
krtol=1e-13; 
kmaxit=200; kdim=30

#Problem parameters
outdir=data # Where the data is sent to
ax=0.85; ay=1.0; mx=0.3; my=0.25 # PDE coefficients

dim=2  # Spatial dimension
ex=1   # example problem to be solved
tf=2   # Final integration time
dt=-2  # Time step

# h will go from [2^-h_min_refine, ..., 2^-h_max_refine]
h_min_refine=3
h_max_refine=7


save=1 # Save only the text file output from the problem and not the solution

# IRK == IRK method; space == Order of spatial discretization

### --- Gauss --- ###
#IRK=14; space=4;  
#IRK=18; space=8; 

### --- RadauIIA --- ###
#IRK=23; space=4;  
#IRK=27; space=8;  

### --- LobIIIC --- ###
#IRK=34; space=4; 
#IRK=38; space=8; 

# Run each problem a number of times.
for run in `seq 1 $num_runs`
do
    
    # Name of file to be output... "^REFINE^" will be replaced with the actual refinement...
    if [ $IRK_alg == 0 ]; then
        #echo "IRK=0"
        out=$outdir/CC_IRK"$IRK"_h^REFINE^_d"$dim"_ex"$ex"_run"$run"
    else 
        #echo "IRK=1"
        out=$outdir/PREC"$block_prec_ID"_IRK"$IRK"_h^REFINE^_d"$dim"_ex"$ex"_run"$run"
    fi

    probinfo_out="$out.probinfo.out" # The probinfo struct
    command_line_out="$out.cl.out" # We'll send the command line output here
    
    # Run solves at different spatial and temporal refinements.
    echo "solving..."
    for h_refine in `seq $h_min_refine $h_max_refine`
    do
        mpirun -np $np $exe \
            -irk $IRK_alg -gamma $gamma -prec_ID $block_prec_ID \
            -t $IRK -dt $dt -tf $tf \
            -o $space -l $h_refine \
            -d $dim -ex $ex \
            -ax $ax -ay $ay -mx $mx -my $my \
            -katol $katol -krtol $krtol -kmaxit $kmaxit -kdim $kdim -kp $kp \
            -save $save -out ${probinfo_out/^REFINE^/$h_refine} \
            2>&1 | tee ${command_line_out/^REFINE^/$h_refine} 
            # The above will output the command line AND its contents to file
    done
done