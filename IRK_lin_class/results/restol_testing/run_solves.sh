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

# IRK alg parameters, just choose one of IRK_alg = 0 or = 1. The shell code below will fix the name of the output appropriately
IRK_alg=0 # CC preconditioned algorithm
IRK_alg=1 # Block preconditioned algorithm

gamma=1 # CC: Constant in preconditioner. 1 is optimal value
block_prec_ID=1 # Block preconditioner: Lower triangular Gauss--Seidel
block_prec_ID=4 # Block preconditioner: Rana et al. (2021)


# Krylov parameters
kp=2; 
katol=0.0;  
kmaxit=200; kdim=30
# reltol will go from [10^-rtol_min_refine, ..., 10^-rtol_max_refine]
reltol_min_refine=4
reltol_max_refine=12

#Problem parameters
outdir=data # Where the data is sent to
dim=2  # Spatial dimension
ax=0.85; ay=1.0; mx=0.3; my=0.25 # PDE coefficients
ex=1   # example problem to be solved
tf=2   # Final integration time
dt=-2  # Time step
l_refine=6 # 2^l_refine DOFs in each dimension


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
IRK=110; space=10; 

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
    out=$outdir/CC_IRK"$IRK"_l"$l_refine"_d"$dim"_ex"$ex"_r^REFINE^
else 
    #echo "IRK=1"
    out=$outdir/PREC"$block_prec_ID"_IRK"$IRK"_l"$l_refine"_d"$dim"_ex"$ex"_r^REFINE^
fi

probinfo_out="$out.probinfo.out" # The probinfo struct
command_line_out="$out.cl.out" # We'll send the command line output here
#echo $probinfo_out
#echo $command_line_out

# Run solves at different temporal refinements...
echo "solving..."
for reltol_refine in `seq $reltol_min_refine $reltol_max_refine`
do
    mpirun -np $np $exe \
        -irk $IRK_alg -gamma $gamma -prec_ID $block_prec_ID \
        -t $IRK -dt $dt -tf $tf \
        -o $space -l $l_refine \
        -d $dim -ex $ex \
        -ax $ax -ay $ay -mx $mx -my $my \
        -katol $katol -krtol 1e-$reltol_refine -kmaxit $kmaxit -kdim $kdim -kp $kp \
        -save $save -out ${probinfo_out/^REFINE^/$reltol_refine} \
        2>&1 | tee ${command_line_out/^REFINE^/$reltol_refine} 
        # The above will output the command line AND its contents to file
done
