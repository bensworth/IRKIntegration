#!/bin/bash -l
#SBATCH --job-name=adv-diff-reaction
#SBATCH --nodes=4
#SBATCH --time=08:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=mpi_test_%j.log

### To specifically run on fast intel nodes, call
### sbatch -p skylake-gold imex-adv-diff-react-runs.sh 

source ~/modules.sh

# for DT in 0.16 0.08 0.04 0.02 0.01
for DT in 0.005 0.0025
do
	echo "dt = $DT"
	echo "Radau(3,0)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 123 -i 0 -nostar >> output2.txt
	echo "Radau(3,1)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 123 -i 1 -nostar >> output2.txt
	echo "Radau(3,2)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 123 -i 2 -nostar >> output2.txt

	echo "Radau(4,0)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 124 -i 0 -nostar >> output2.txt
	echo "Radau(4,1)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 124 -i 1 -nostar >> output2.txt
	# echo "Radau(4,2)" >> output2.txt
	# srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 124 -i 2 -nostar >> output2.txt

	echo "Radau(5,0)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 125 -i 0 -nostar >> output2.txt
	# echo "Radau(5,1)" >> output2.txt
	# srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 125 -i 1 -nostar >> output2.txt
	# echo "Radau(5,2)" >> output2.txt
	# srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 125 -i 2 -nostar >> output2.txt

	echo "Radau*(3,0)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 123 -i 0 -star >> output2.txt
	echo "Radau*(3,1)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 123 -i 1 -star >> output2.txt
	echo "Radau*(3,2)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 123 -i 2 -star >> output2.txt

	echo "Radau*(4,0)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 124 -i 0 -star >> output2.txt
	# echo "Radau*(4,1)" >> output2.txt
	# srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 124 -i 1 -star >> output2.txt
	# echo "Radau*(4,2)" >> output2.txt
	# srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 124 -i 2 -star >> output2.txt

	echo "Radau*(5,0)" >> output2.txt
	srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 125 -i 0 -star >> output2.txt
	# echo "Radau*(5,1)" >> output2.txt
	# srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 125 -i 1 -star >> output2.txt
	# echo "Radau*(5,2)" >> output2.txt
	# srun -n 64 ./IRKIntegration/IRK_polyIMEX_class/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -irk 125 -i 2 -star >> output2.txt

	echo "IMEX222" >> output2.txt
	srun -n 64 ./imex_mfem/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -imex 222 >> output2.txt
	echo "IMEX233" >> output2.txt
	srun -n 64 ./imex_mfem/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -imex 233 >> output2.txt
	echo "IMEX-43" >> output2.txt
	srun -n 64 ./imex_mfem/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -imex -43 >> output2.txt
	echo "IMEX443" >> output2.txt
	srun -n 64 ./imex_mfem/nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -fi -imex 443 >> output2.txt

	echo "RK-L2" >> output2.txt
	srun -n 64 ./imex_mfem/nl-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -rk 2 >> output2.txt
	echo "RK-A2" >> output2.txt
	srun -n 64 ./imex_mfem/nl-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -rk 22 >> output2.txt
	echo "RK-L3" >> output2.txt
	srun -n 64 ./imex_mfem/nl-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -rk 3 >> output2.txt
	echo "RK-A3" >> output2.txt
	srun -n 64 ./imex_mfem/nl-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -rk 23 >> output2.txt
	echo "RK-A4" >> output2.txt
	srun -n 64 ./imex_mfem/nl-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt $DT -eps 0.1 -eta 10 -r 2 -rk 24 >> output2.txt
done


# srun -n 64 nl-imex-dg-adv-diff-mms -rs 4 -rp 1 -o 3 -tf 1 -dt 0.16 -eps 0.1 -eta 10 -r 2 -fi -irk 123 -i 0 -nostar