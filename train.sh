#!/bin/bash
#SBATCH -J train
#SBATCH -t 10-00:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/R-%x.%j.out
#SBATCH --error=jobs/R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --exclude=g0605
# ^^^ until they fix that node

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/users/m/i/mixarcid/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/users/m/i/mixarcid/mixarcid/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work/users/m/i/mixarcid/mixarcid/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/users/m/i/mixarcid/mixarcid/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

module load gcc/11.2.0
module load cuda/11.8
conda activate chem-py3.9

cd /nas/longleaf/home/mixarcid/plantain

rm -rf plantain
rm -rf plantain_pose
# rm -rf wandb

pip install --upgrade terrace
python train.py $@
