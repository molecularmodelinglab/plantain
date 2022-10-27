# e2ebind (name in progress)

To use this first, first make sure you've downloaded the [BigBind dataset](https://drive.google.com/file/d/15D6kQZM0FQ2pgpMGJK-5P9T12ZRjBjXS/view?usp=sharing).

Now create a file `cfg/local.yaml`. This contains all the configuration that should differ per-computer. Add in this information:
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

Now that you have this, train a model py running `python train.py config_name overrides`. The config name used in the BigBind paper is `short_thicc_op_gnn`. This config is found in `cfg/short_thicc_op_gnn.yaml`. Feel free to make your own config. Anything in the config file can be overriden with command line arguments. For instance, train with a batch size of 2 with `python train.py short_thic_op_gnn batch_size=2`.

Enjoy!