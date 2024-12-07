#!/bin/bash

for i in $(seq 1 $1)
do
    nohup python graph_constructor_glycan_only_inference.py $i $1 $2 $3 $4 $5 $6 $7 &
done
wait
