#!/usr/bin/bash
for i in $(seq 0 $1);
do
sbatch ff_bigbind_struct.sh $1 $i
done