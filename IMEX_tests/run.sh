#!/bin/bash
srun -n120 ./test-imex-dg-adv-diff -rs 4 -rp 1 -o 4 -e 0.01 -tf 2 >> imex_tests_o4_e0.01_tf2_newbdf.txt
wait
srun -n120 ./test-imex-dg-adv-diff -rs 4 -rp 1 -o 4 -e 0.1 -tf 2 >> imex_tests_o4_e0.1_tf2_newbdf.txt
wait
srun -n120 ./test-imex-dg-adv-diff -rs 4 -rp 1 -o 4 -e 1 -tf 2 >> imex_tests_o4_e1_tf2_newbdf.txt
wait
srun -n120 ./test-imex-dg-adv-diff -rs 4 -rp 1 -o 4 -e 10 -tf 2 >> imex_tests_o4_e10_tf2_newbdf.txt
