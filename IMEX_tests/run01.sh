#!/bin/bash

source ~/modules.sh
srun -n120 ./test-imex-dg-adv-diff -rs 4 -rp 1 -o 4 -e 0.1 -tf 2 -mdt 0.006 >> final_imex_tests_o4_e0.1_tf2.txt
