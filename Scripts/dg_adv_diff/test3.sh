#!/bin/bash
# MSUB -l nodes=8
# MSUB -l partition=quartz
# MSUB -l walltime=00:20:00
# MSUB -q pdebug
# MSUB -V
# MSUB -o ./results/large.out

# p = 2 - 0.0000152
# p = 3 - 0.000615
# p = 4 - 0.00390625
# p = 5 - 0.01184
# p = 6 - 0.0248
# p = 7 - 0.0421
# p = 8 - 0.0625
# p = 9 - 0.0850
# p = 10 - 0.1088

# opt constant
srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 23 -o 4 -dt 0.000615 -air 1 -tf 0.0065 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 32 -o 4 -dt 0.0000152 -air 1 -tf 0.0001 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt


# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 4 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 14 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 34 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 25 -o 4 -dt 0.01184 -air 1 -tf 0.1 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 16 -o 4 -dt 0.0248 -air 1 -tf 0.2 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 36 -o 4 -dt 0.0248 -air 1 -tf 0.2 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 27 -o 4 -dt 0.0421 -air 1 -tf 0.35 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 18 -o 4 -dt 0.0625 -air 1 -tf 0.5 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 38 -o 4 -dt 0.0625 -air 1 -tf 0.5 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 29 -o 4 -dt 0.085 -air 1 -tf 0.65 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 110 -o 4 -dt 0.1088 -air 1 -tf 1.0 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_opt.txt


# eta constant
srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 23 -o 4 -dt 0.000615 -air 1 -tf 0.0065 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_eta.txt
srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 32 -o 4 -dt 0.0000152 -air 1 -tf 0.0001 -e 0 -nov -gmres 0 >> results/dg_irk_hyp_o4_eta.txt


# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 4 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 14 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 34 -o 4 -dt 0.00390625 -air 1 -tf 0.065 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 25 -o 4 -dt 0.01184 -air 1 -tf 0.1 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 16 -o 4 -dt 0.0248 -air 1 -tf 0.2 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 36 -o 4 -dt 0.0248 -air 1 -tf 0.2 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 27 -o 4 -dt 0.0421 -air 1 -tf 0.35 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 18 -o 4 -dt 0.0625 -air 1 -tf 0.5 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt
# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 38 -o 4 -dt 0.0625 -air 1 -tf 0.5 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 29 -o 4 -dt 0.085 -air 1 -tf 0.65 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt

# srun -n288 ./dg_adv_diff  -rs 4 -rp 2 -i 110 -o 4 -dt 0.1088 -air 1 -tf 1.0 -e 0 -nov -gmres 0 -mag 0 >> results/dg_irk_hyp_o4_eta.txt
