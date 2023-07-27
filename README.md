# PLANTAIN

Implementation of PLANTAIN (Predicting LigANd pose wiTh an AI scoring functioN). PLANTAIN is a new AI docking algorithm that achieves state-of-the-art results while being significantly faster than alternatives. Read more in the [paper](https://arxiv.org/abs/2307.12090)! This repository contains everything necessary to run PLANTAIN on your own complexes.

If you have any questions, notice any issues, or just want to say hi, please reach out! Email me at [mixarcid@unc.edu](mailto:mixarcid@unc.edu).

## Getting started

Start by setting up your environment. Once you have [conda](https://docs.conda.io) installed, simply run:
```bash
./create_env.sh
```
This will create a new `plantain` environment, which you can use to run everything below.

## Running with pretrained weights

Now you're ready to run PLANTAIN. First, create a PDB file containing the pocket residues of your target protein. Then create a smi file with the SMILES strings of the compounds you want to screen (one smiles string per line). Now run the following:

```bash
python inference.py compounds.smi pocket.pdb --out predictions
```

This will predict the pose of each molecule in `compounds.smi` and save them to numbered SDF files in `predictions/`. For instance, the `predictions/0.sdf` will correspond to the first compound in the file. These SDF files contain each of the 16 poses proposed by PLANTAIN, ordered according to predicted MSE.

Because PLANTAIN uses `torch.compile`, the first couple compounds will take a while to run. But things should speed up after that.

## Training from scratch

In order to train PLANTAIN or run any of the model comparisons below, you'll first need to create your local configuration in `configs/local.yaml`.

```yaml

project: "Your wandb project (optional)"

platform:
  crossdocked_dir: "/path/to/crossdocked/folder" # see below
  # To speed up dataloading and other random things, many pickles are saved.
  # cache_dir specifies where they should be saved to
  cache_dir: "/path/to/cache/dir"
  num_workers: num_workers_for_your_dataloader

  # if you're running on older hardware, sometimes pytorch JIT compilation
  # doesn't work. If you're having trouble, uncomment this line. This will
  # disable all compilation -- it's slow, but at least it works
  # compile: false

  # the following options are only necessary if you're running
  # all the baseline comparisons
  vina_exec: "/path/to/vina/executable"
  gnina_exec: "/path/to/gnina/executable"
  obabel_exec: "/path/to/openbabel/executable"

  crossdocked_vina_dir: "/output/folder/for/vina"
  crossdocked_gnina_dir: "/output/folder/for/gnina"

  diffdock_dir: "/path/to/diffdock/repo"
  diffdock_env: "diffdock conda environment"

```

Additionally, you'll need to download the [preprocessed CrossDocked dataset](https://storage.googleapis.com/plantain_data/CrossDockedPreproc.tar.gz) to the `crossdocked_dir` folder specified above.

Once you have these, you're ready to train. Start by running:
```bash
python train.py icml
```
This will start training PLANTAIN using the configuration from `configs/icml.yaml`. If you want to mess around with the hyperparameters, I'd recommend creating a new config file. You can also override the configuration from the command line; for instance, running `python train.py icml learn_rate=1e-5` will train with a lower learn rate.

If you've specified a [wandb](wandb.ai/) project in the local config file, the training code will automatically log model weights and metrics to your project.

## Reproducing paper numbers

Running all the baselines and comparing results is a bit more involved. First, make sure you've installed [AutoDock Vina](https://vina.scripps.edu/), [GNINA](https://github.com/gnina/gnina), and [OpenBabel](https://openbabel.org/wiki/Main_Page) (necessary for preprocessing files for Vina). Additionally, clone [DiffDock](https://github.com/gcorso/DiffDock) and setup its prereqs in a new conda environment. Make sure `local.yaml` reflects where you've put everything.

Now save the results (and timings) from Vina,  GNINA, and DiffDock with the following commands:
```bash
python -m baselines.vina_gnina
python -m baselines.diffdock
```
Note that these commands will take a very long time! Each of Vina, GNINA, and DiffDock took several days to run on my computer. This is because at the moment there is no parallelization in the scripts, but it shouldn't be hard to add that in.

Now that you've saved the baseline results, run:
```bash
python -m analysis.model_comparison
```
This will print out all the the metrics used in the paper. It also saves more comprehensive metrics to `outputs/model_comparison_test.csv`.
