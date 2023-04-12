mamba create -n plantain python=3.10 -y
conda activate plantain
mamba install pytorch==1.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge -y
# mamba install -c dglteam dgl-cuda11.6 -y
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
# this last line is sometimes necessary
# pip uninstall nvidia_cublas_cu11
