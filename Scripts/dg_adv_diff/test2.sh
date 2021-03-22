#!/bin/bash
# MSUB -l nodes=2
# MSUB -l partition=quartz
# MSUB -l walltime=00:60:00
# MSUB -q pdebug
# MSUB -V
# MSUB -o ./results/large.out

# h
# 0.00390625
# 0.0078125
# 0.015625
# 0.03125
# 0.0625
# 0.125

# p = 4
# 0.00390625
# 0.0078125
# 0.015625
# 0.03125
# 0.0625
# 0.125

# p = 7
# 0.0421
# 0.0625
# 0.0929
# 0.138
# 0.205
# 0.3048

# p = 8
# 0.0625
# 0.0884
# 0.125
# 0.1768
# 0.25
# 0.3536

# p = 9
# 0.0850
# 0.1157
# 0.1575
# 0.2143
# 0.2916
# 0.3969

# p = 10
# 0.1088
# 0.1436
# 0.1895
# 0.25
# 0.33
# 0.4353

# # 8th order accuracy
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 4 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 1e-6 -nov -gmres 0 >> results/dg_irk4_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 14 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 1e-6 -nov -gmres 0 >> results/dg_irk14_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 27 -o 4 -dt 0.0421 -air 1 -tf 0.35 -e 1e-6 -nov -gmres 0 >> results/dg_irk27_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 18 -o 4 -dt 0.0625 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk18_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 29 -o 4 -dt 0.085 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk29_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 110 -o 4 -dt 0.1088 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk110_o4_1e-6_nogmres_opt.txt

# srun -n144 ./dg_adv_diff  -rs 4 -rp 1 -i 4 -o 4 -dt 0.0078125 -air 1 -tf 0.065 -e 1e-6 -nov -gmres 0 >> results/dg_irk4_o4_1e-6_nogmres_opt.txt
# srun -n144 ./dg_adv_diff  -rs 4 -rp 1 -i 14 -o 4 -dt 0.0078125 -air 1 -tf 0.065 -e 1e-6 -nov -gmres 0 >> results/dg_irk14_o4_1e-6_nogmres_opt.txt
# srun -n144 ./dg_adv_diff  -rs 4 -rp 1 -i 27 -o 4 -dt 0.0625 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk27_o4_1e-6_nogmres_opt.txt
# srun -n144 ./dg_adv_diff  -rs 4 -rp 1 -i 18 -o 4 -dt 0.0884 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk18_o4_1e-6_nogmres_opt.txt
# srun -n144 ./dg_adv_diff  -rs 4 -rp 1 -i 29 -o 4 -dt 0.1157 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk29_o4_1e-6_nogmres_opt.txt
# srun -n144 ./dg_adv_diff  -rs 4 -rp 1 -i 110 -o 4 -dt 0.1436 -air 1 -tf 0.75 -e 1e-6 -nov -gmres 0 >> results/dg_irk110_o4_1e-6_nogmres_opt.txt

srun -n72 ./dg_adv_diff  -rs 4 -rp 0 -i 4 -o 4 -dt 0.015625 -air 1 -tf 0.15 -e 1e-6 -nov -gmres 0 >> results/dg_irk4_o4_1e-6_nogmres_opt.txt
srun -n72 ./dg_adv_diff  -rs 4 -rp 0 -i 14 -o 4 -dt 0.015625 -air 1 -tf 0.15 -e 1e-6 -nov -gmres 0 >> results/dg_irk14_o4_1e-6_nogmres_opt.txt
srun -n72 ./dg_adv_diff  -rs 4 -rp 0 -i 27 -o 4 -dt 0.0929 -air 1 -tf 0.75 -e 1e-6 -nov -gmres 0 >> results/dg_irk27_o4_1e-6_nogmres_opt.txt
srun -n72 ./dg_adv_diff  -rs 4 -rp 0 -i 18 -o 4 -dt 0.125 -air 1 -tf 1.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk18_o4_1e-6_nogmres_opt.txt
srun -n72 ./dg_adv_diff  -rs 4 -rp 0 -i 29 -o 4 -dt 0.1575 -air 1 -tf 1.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk29_o4_1e-6_nogmres_opt.txt
srun -n72 ./dg_adv_diff  -rs 4 -rp 0 -i 110 -o 4 -dt 0.1895 -air 1 -tf 1.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk110_o4_1e-6_nogmres_opt.txt

srun -n36 ./dg_adv_diff  -rs 3 -rp 0 -i 4 -o 4 -dt 0.03125 -air 1 -tf 0.35 -e 1e-6 -nov -gmres 0 >> results/dg_irk4_o4_1e-6_nogmres_opt.txt
srun -n36 ./dg_adv_diff  -rs 3 -rp 0 -i 14 -o 4 -dt 0.03125 -air 1 -tf 0.35 -e 1e-6 -nov -gmres 0 >> results/dg_irk14_o4_1e-6_nogmres_opt.txt
srun -n36 ./dg_adv_diff  -rs 3 -rp 0 -i 27 -o 4 -dt 0.138 -air 1 -tf 1.5 -e 1e-6 -nov -gmres 0 >> results/dg_irk27_o4_1e-6_nogmres_opt.txt
srun -n36 ./dg_adv_diff  -rs 3 -rp 0 -i 18 -o 4 -dt 0.1768 -air 1 -tf 1.5 -e 1e-6 -nov -gmres 0 >> results/dg_irk18_o4_1e-6_nogmres_opt.txt
srun -n36 ./dg_adv_diff  -rs 3 -rp 0 -i 29 -o 4 -dt 0.2143 -air 1 -tf 2.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk29_o4_1e-6_nogmres_opt.txt
srun -n36 ./dg_adv_diff  -rs 3 -rp 0 -i 110 -o 4 -dt 0.25 -air 1 -tf 2.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk110_o4_1e-6_nogmres_opt.txt

srun -n18 ./dg_adv_diff  -rs 2 -rp 0 -i 4 -o 4 -dt 0.0625 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk4_o4_1e-6_nogmres_opt.txt
srun -n18 ./dg_adv_diff  -rs 2 -rp 0 -i 14 -o 4 -dt 0.0625 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_irk14_o4_1e-6_nogmres_opt.txt
srun -n18 ./dg_adv_diff  -rs 2 -rp 0 -i 27 -o 4 -dt 0.205 -air 1 -tf 2.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk27_o4_1e-6_nogmres_opt.txt
srun -n18 ./dg_adv_diff  -rs 2 -rp 0 -i 18 -o 4 -dt 0.25 -air 1 -tf 2.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk18_o4_1e-6_nogmres_opt.txt
srun -n18 ./dg_adv_diff  -rs 2 -rp 0 -i 29 -o 4 -dt 0.2916 -air 1 -tf 2.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk29_o4_1e-6_nogmres_opt.txt
srun -n18 ./dg_adv_diff  -rs 2 -rp 0 -i 110 -o 4 -dt 0.33 -air 1 -tf 2.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk110_o4_1e-6_nogmres_opt.txt

srun -n9 ./dg_adv_diff  -rs 1 -rp 0 -i 4 -o 4 -dt 0.125 -air 1 -tf 1.25 -e 1e-6 -nov -gmres 0 >> results/dg_irk4_o4_1e-6_nogmres_opt.txt
srun -n9 ./dg_adv_diff  -rs 1 -rp 0 -i 14 -o 4 -dt 0.125 -air 1 -tf 1.25 -e 1e-6 -nov -gmres 0 >> results/dg_irk14_o4_1e-6_nogmres_opt.txt
srun -n9 ./dg_adv_diff  -rs 1 -rp 0 -i 27 -o 4 -dt 0.3048 -air 1 -tf 3.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk27_o4_1e-6_nogmres_opt.txt
srun -n9 ./dg_adv_diff  -rs 1 -rp 0 -i 18 -o 4 -dt 0.3536 -air 1 -tf 3.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk18_o4_1e-6_nogmres_opt.txt
srun -n9 ./dg_adv_diff  -rs 1 -rp 0 -i 29 -o 4 -dt 0.3969 -air 1 -tf 3.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk29_o4_1e-6_nogmres_opt.txt
srun -n9 ./dg_adv_diff  -rs 1 -rp 0 -i 110 -o 4 -dt 0.4353 -air 1 -tf 3.0 -e 1e-6 -nov -gmres 0 >> results/dg_irk110_o4_1e-6_nogmres_opt.txt







