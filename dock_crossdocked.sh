#!/bin/bash
#SBATCH -J dock_crossdocked 
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

source /nas/longleaf/home/mixarcid/.bashrc

conda activate mm

cd /nas/longleaf/home/mixarcid/plantain

python -m scripts.dock_crossdocked

