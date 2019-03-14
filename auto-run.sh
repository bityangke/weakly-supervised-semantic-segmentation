#!/bin/bash

for sigma in $(seq 1 10)
do
    for lamda in $(seq -f "%f" 0.1 0.1 1)
    do
        file_name="sigma_"$sigma"_lamda_"$lamda"_result"
        python train.py --sigma $sigma --lamda $lamda
        python test_miou.py  >> $file_name
	echo $file_name
    done
done
