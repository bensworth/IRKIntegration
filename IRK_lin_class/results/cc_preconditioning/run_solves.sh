#!/bin/bash

# Run the executable for many different temporal/spatial refinements.
#
# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory 
#       within the current directory.


date

# Name of executable
exe=../../driver_adv_dif_FD

np=4 # Number of processes

# IRK alg parameters
IRK_alg=0 # CC preconditioned algorithm
gamma=1 # Constant in preconditioner. 1 is optimal value

# Krylov parameters
kp=2; 
katol=0.0; 
krtol=1e-13; 
kmaxit=200; kdim=30

# Problem parameters
#outdir=example1 # Where the data is sent to
outdir=data # Where the data is sent to
dim=2  # Spatial dimension
ax=0.85; ay=1.0; mx=0.3; my=0.25 # PDE coefficients
ex=1   # example problem to be solved
tf=2   # Final integration time
dt=-2  # Time step

# h will go from [2^-h_min_refine, ..., 2^-h_max_refine]
h_min_refine=3
h_max_refine=7

save=1 # Save only the text file output from the problem and not the solution

# IRK == IRK method; space == Order of spatial discretization

### --- ASDIRK --- ###
#IRK=-13; space=4;  
#IRK=-14; space=4;  

### --- LSDIRK --- ###
#IRK=1; space=2;  
#IRK=2; space=2;  
#IRK=3; space=4;  
#IRK=4; space=4; 

### --- Gauss --- ###
#IRK=12; space=2;  
#IRK=14; space=4;  
#IRK=16; space=6; 
#IRK=18; space=8; 
#IRK=110; space=10; 

### --- RadauIIA --- ###
#IRK=23; space=4; 
#IRK=25; space=6; 
#IRK=27; space=8; 
#IRK=29; space=10; 

### --- LobIIIC --- ###
#IRK=32; space=2  
IRK=34; space=4; 
#IRK=36; space=6; 
#IRK=38; space=8; 

# Name of file to be output... "^REFINE^" will be replaced with the actual refinement...
dir=$outdir/"$time_type"
out=IRK"$IRK"_h^REFINE^_d"$dim"_ex"$ex" 

# Run solves at different temporal refinements...
echo "solving..."
for h_refine in `seq $h_min_refine $h_max_refine`
do
    mpirun -np $np $exe \
        -irk $IRK_alg -gamma $gamma \
        -t $IRK -dt $dt -tf $tf \
        -o $space -l $h_refine \
        -d $dim -ex $ex \
        -ax $ax -ay $ay -mx $mx -my $my \
        -katol $katol -krtol $krtol -kmaxit $kmaxit -kdim $kdim -kp $kp \
        -save $save -out $dir${out/^REFINE^/$h_refine}

done
