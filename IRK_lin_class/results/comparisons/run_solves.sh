#!/bin/bash

# Run the executable for many different GMRES relative residual refinements.
#
# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory 
#       within the current directory.


date

# Name of executable
exe=../../driver_adv_dif_FD

np=4 # Number of processes

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

# More advection-dominated problem. Struggle to converge...
# outdir=data2 # Where the data is sent to
# ax=4.25; ay=5.0; mx=0.03; my=0.025 # PDE coefficients

dim=2  # Spatial dimension
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
IRK=18; space=8; 
#IRK=110; space=10; 

### --- RadauIIA --- ###
#IRK=23; space=4; 
#IRK=25; space=6; 
#IRK=27; space=8; 
#IRK=29; space=10; 

### --- LobIIIC --- ###
#IRK=32; space=2  
#IRK=34; space=4; 
#IRK=36; space=6; 
#IRK=38; space=8; 

# Name of file to be output... "^REFINE^" will be replaced with the actual refinement...
if [ $IRK_alg == 0 ]; then
    #echo "IRK=0"
    out=$outdir/CC_IRK"$IRK"_h^REFINE^_d"$dim"_ex"$ex"
else 
    #echo "IRK=1"
    out=$outdir/PREC"$block_prec_ID"_IRK"$IRK"_h^REFINE^_d"$dim"_ex"$ex"
fi

probinfo_out="$out.probinfo.out" # The probinfo struct
command_line_out="$out.cl.out" # We'll send the command line output here
#echo $probinfo_out
#echo $command_line_out

#--AMG-theta 0.25 --AMG-relax 0 --AMG-use-air 1 \

# Run solves at different temporal refinements...
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
