#!/bin/bash

source ~/modules.sh
srun -n120 ./test-imex-dg-adv-diff -rs 4 -rp 1 -o 4 -e 10 -tf 2 -mdt 0.015 >> final_imex_tests_o4_e10_tf2.txt
