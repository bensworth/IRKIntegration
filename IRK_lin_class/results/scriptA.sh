#!/bin/bash
# MSUB -l nodes=2
# MSUB -l partition=quartz
# MSUB -l walltime=00:60:00
# MSUB -q pdebug
# MSUB -V
# MSUB -o ./results/large.out

export OMP_NUM_THREADS=1

/g/g19/bs/julia-1.5.3/bin/julia runtest_A.jl >> results/ord_red_sdirkA.txt
