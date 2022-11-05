#!/bin/bash
#SBATCH -J vina_bigbind
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

__conda_setup="$('/nas/longleaf/home/mixarcid/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nas/longleaf/home/mixarcid/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/nas/longleaf/home/mixarcid/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nas/longleaf/home/mixarcid/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

cd /nas/longleaf/home/mixarcid/plantain

python vina_bigbind.py

