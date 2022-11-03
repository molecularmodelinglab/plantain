# BANANA

Implimentation od BANANA (BAsic NeurAl Network for bindding Affinity), as described in the BigBind paper (coming to a ChemRxiv near you...)

## Dependencies

In addition to pip installing the requirements, you'll need to install torch, dgl, and rdkit.

# Running with pretrained weights

Once you've installed all the dependancies, you're ready to run the pretrained model. First create a PDB file for the pocket of the protein you want to target, then create a smi file will the SMILES strings of the compounds you want to screen. Now simply run `python inference.py compounds.smi pocket.pdb--out_file out_file.txt`. This will write all the scores of the compounds to `out_file.txt`. It will automatically download the pretrained weights to `data/` if they are not already there.

## Training

If you want to train the model yourself, first make sure you've downloaded the [BigBind dataset](https://drive.google.com/file/d/15D6kQZM0FQ2pgpMGJK-5P9T12ZRjBjXS/view?usp=sharing).

Now create a file `configs/local.yaml`. This contains all the configuration that should differ per-computer. Add in this information:
```yaml

project: "Your wandb project (optional)"

platform:
  bigbind_dir: "/path/to/bigbind/folder"
  # lit_pcba_dir is optional, only needed if you want to test your model on LIT_PCBA
  lit_pcba_dir: "/path/to/lit_pcba/folder"
  # To speed up dataloading and other random things, many pickles are saved.
  # cache_dir specifies where they should be saved to
  cache_dir: "/path/to/cache/dir"
  num_workers: num_workers_for_your_dataloader

```

Now that you have this, train a model py running `python train.py config_name overrides`. The config name used in the BigBind paper is `short_thicc_op_gnn`. This config is found in `configs/classification.yaml`. Feel free to make your own config. Anything in the config file can be overriden with command line arguments. For instance, train with a batch size of 2 with `python train.py classification batch_size=2`.

Enjoy!
