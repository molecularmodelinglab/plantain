#!/bin/bash
set -e
mamba create -n plantain-v2 python=3.10 -y
conda activate plantain-v2
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install -r requirements.txt
pip install -r dev_requirements.txt
# this last line is sometimes necessary
# pip uninstall nvidia_cublas_cu11
