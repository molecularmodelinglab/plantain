mamba create -n mm python=3.7 -y
conda activate mm
mamba install -c conda-forge -c omnia openmm openff-toolkit openff-forcefields openmmforcefield -y
pip install -r requirements.txt