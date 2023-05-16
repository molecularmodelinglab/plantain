#!/bin/bash
mamba create -n plantain-nightly python=3.10 -y &&
conda activate plantain-nightly &&
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia -y &&
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html &&
pip install -r requirements.txt &&
pip install -r dev_requirements.txt
# this last line is sometimes necessary
# pip uninstall nvidia_cublas_cu11