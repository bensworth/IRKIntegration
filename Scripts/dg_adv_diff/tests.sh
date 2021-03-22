#!/bin/bash
# MSUB -l nodes=8
# MSUB -l partition=quartz
# MSUB -l walltime=00:60:00
# MSUB -q pdebug
# MSUB -V
# MSUB -o ./results/large.out

## 2nd order in space, Gauss6, diffusion dominated
# Inner GMRES
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 8 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 7 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 6 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 5 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 4 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 3 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 2 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 1 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-2_irk16_gmres_opt.txt

# # No inner GMRES
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 8 -tf 1.0 -e 1e-2 -nov -gmres -1 >> results/dg_o2_1e-2_irk16_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 7 -tf 1.0 -e 1e-2 -nov -gmres -1 >> results/dg_o2_1e-2_irk16_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 6 -tf 1.0 -e 1e-2 -nov -gmres -1 >> results/dg_o2_1e-2_irk16_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 5 -tf 1.0 -e 1e-2 -nov -gmres -1 >> results/dg_o2_1e-2_irk16_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 4 -tf 1.0 -e 1e-2 -nov -gmres -1 >> results/dg_o2_1e-2_irk16_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 3 -tf 1.0 -e 1e-2 -nov -gmres -1 >> results/dg_o2_1e-2_irk16_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 2 -tf 1.0 -e 1e-2 -nov -gmres -1 >> results/dg_o2_1e-2_irk16_nogmres_opt.txt

# Inner GMRES
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 8 -tf 1.0 -e 1e-2 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 7 -tf 1.0 -e 1e-6 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 6 -tf 1.0 -e 1e-6 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 5 -tf 1.0 -e 1e-6 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 4 -tf 1.0 -e 1e-6 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 3 -tf 1.0 -e 1e-6 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 2 -tf 1.0 -e 1e-6 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 1 -tf 1.0 -e 1e-6 -nov -gmres 1 >> results/dg_o2_1e-6_irk16_gmres_opt.txt

# No inner GMRES
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 8 -tf 1.0 -e 1e-6 -nov -gmres -1 >> results/dg_o2_1e-6_irk16_nogmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 7 -tf 1.0 -e 1e-6 -nov -gmres -1 >> results/dg_o2_1e-6_irk16_nogmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 6 -tf 1.0 -e 1e-6 -nov -gmres -1 >> results/dg_o2_1e-6_irk16_nogmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 5 -tf 1.0 -e 1e-6 -nov -gmres -1 >> results/dg_o2_1e-6_irk16_nogmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 4 -tf 1.0 -e 1e-6 -nov -gmres -1 >> results/dg_o2_1e-6_irk16_nogmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 3 -tf 1.0 -e 1e-6 -nov -gmres -1 >> results/dg_o2_1e-6_irk16_nogmres_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 4 -i 16 -o 2 -dt 0.1 -air 2 -tf 1.0 -e 1e-6 -nov -gmres -1 >> results/dg_o2_1e-6_irk16_nogmres_opt.txt



# # 4th order accuracy in space and time
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 04 -o 4 -dt 0.00425 -air 1 -tf 0.0425 -e 1e-6 -nov -gmres 0  >> results/dg_o4_rko4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 14 -o 4 -dt 0.00425 -air 1 -tf 0.0425 -e 1e-6 -nov -gmres 0 >> results/dg_o4_rko4_1e-6_nogmres_opt.txt

# # 4th order accuracy, same time step as 8th order
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 04 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_rko4_1e-6_bigdt_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 14 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_rko4_1e-6_bigdt_nogmres_opt.txt

# # 8th order accuracy
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 18 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 110 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 29 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 18 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 110 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 29 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt


# # Extra runs
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 27 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 27 -o 4 -dt 0.065 -air 1 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 18 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 110 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 29 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 18 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 110 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 29 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 27 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 1 >> results/dg_o4_1e-6_gmres_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 27 -o 4 -dt 0.065 -air 2 -tf 0.65 -e 1e-6 -nov -gmres 0 >> results/dg_o4_1e-6_nogmres_opt.txt
