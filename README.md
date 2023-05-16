# PLANTAIN (backroynm tdb)

To get this code running, create a config file `configs/local.yaml`.
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
  # infer_workers will not be necessary after we remove jax dependancy
  infer_workers: num_cpus_when_running_inference
```
