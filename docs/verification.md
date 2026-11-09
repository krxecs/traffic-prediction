# Current checks and historical verification

The [historical verification Gist](https://gist.github.com/krxecs/b8c0e4198cdb5952244ac1745aa0989a) contains the earlier pure PyTorch versus Lightning results and the scripts used to obtain them. Anyone with the link can read it. Those scripts require the original trainer and cannot run in the current Lightning-only checkout. The original tracked notebook remains in Git history.

From the project root with `datasets/metr-la` present:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 pixi run python -m unittest discover -s tests -p 'test_*.py' -q
pixi run main.py fit --config configs/ablations/A0.yaml --print_config
pixi run main.py fit --config configs/ablations/A5.yaml --trainer.max_epochs 1 \
  --trainer.limit_train_batches 1 --trainer.limit_val_batches 1 \
  --report_test=false
```

The tests cover YAML schema and the nine historical configs, with one metadata correction: A0 now records its actual RMSprop learning rate of 0.001 rather than the old unused 0.0003 default. They also exercise CLI parsing, a one-batch fit, learning-rate scheduling, Optuna pruning and a short winner retrain. They do not repeat the historical cross-framework comparison or establish full-budget GPU parity.
