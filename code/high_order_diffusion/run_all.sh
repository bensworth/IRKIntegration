#!/usr/bin/env bash

methods="01 02 03 04 12 14 16 18 110 23 25 27 29 32 34 36 38"

for i in $methods
do
   #./heat -r 1 -o 3 -i $i
   ./heat -m $DIR_mfem/data/star-q3.mesh -r 0 -o 9 -i $i
done
