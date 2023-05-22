#!/usr/bin/bash
for i in $(seq 0 $1);
do
sbatch dock_crossdocked.sh
done