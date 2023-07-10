#!/bin/bash
conda create -n plantain python=3.10 -y &&
conda activate plantain &&
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia -y &&
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html &&
pip install -r requirements.txt &&
pip install -r dev_requirements.txt
