# todo: super hacky. Fix!
export REQUESTS_CA_BUNDLE=/work/users/m/i/mixarcid/miniconda3/ssl/cacert.pem
mamba create -n mm python=3.7 -y
conda activate mm
mamba install pytorch==1.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge -y
mamba install -c conda-forge -c omnia openmm openff-toolkit openff-forcefields openmmforcefields -y
pip install -r requirements.txt
pip install -r requirements_mm.txt
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
