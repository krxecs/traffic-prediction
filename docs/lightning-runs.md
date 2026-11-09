# Lightning experiments

The [historical verification archive](https://gist.github.com/krxecs/b8c0e4198cdb5952244ac1745aa0989a)
records the earlier reduced CPU parity tests of all nine ablations and a reduced
A5 GPU test. Full-budget GPU parity was not established. The original PyTorch
trainer and its parity runner have since been removed.

Install the locked environment with `pixi install`. The data files belong in
`datasets/metr-la/`. Inspect the extracted data contract with:

```bash
pixi run python -m src.experiment_data --data-dir datasets/metr-la
```

The training entrypoint uses [LightningCLI](https://lightning.ai/docs/pytorch/stable/reference/cli/lightning_cli.html)
with its standard `fit` subcommand for training. It does not expose `test` or
`predict`: this experiment selects a validation threshold and reports the
held-out test set after `fit`. Trainer and data options remain configurable.
`jsonargparse`, `docstring_parser`, and `typeshed-client` are in the Pixi
environment. The nine YAML presets use
LightningCLI's [`--config` support](https://lightning.ai/docs/pytorch/stable/reference/cli/lightning_cli_advanced.md):

```bash
pixi run main.py fit --config configs/ablations/A0.yaml
pixi run main.py fit --config configs/ablations/A1.yaml --print_config
pixi run main.py fit --config configs/ablations/A5.yaml --trainer.max_epochs 2
```

Each file in `configs/ablations/` specifies its model name, graph-only
`experiment` options, standard `optimizer`, `lr_scheduler` and `trainer`
settings. Trainer callbacks specify the three monitored checkpoints and, where
used, early stopping. `src/experiment_config.py` reads those same files for
Optuna; Python does not carry another set of ablation values. Always use a YAML
file to start a run. Standard flags such as `--optimizer.lr=0.0007` and
`--trainer.max_epochs=2` override the preset and make a custom run, not a
frozen baseline. `--print_config` shows parsed settings; `config.yaml` saves
replayable settings without generated checkpoint paths. For a fresh run, use
`pixi run main.py fit --config runs/<run>/config.yaml`; remove `ckpt_path` if
it is present in a resumed run's config.

The CLI seeds with 42 and uses one device by default. LightningCLI handles the
optimizer and scheduler class paths; the graph/data link and checkpoint-based
threshold and test reporting remain project-specific. `trainer.max_epochs` in
YAML supplies the default and `--trainer.max_epochs` is the only runtime epoch
override. The warmup-cosine schedule receives that effective Trainer budget.
Use `--data.data_dir` for the dataset and `--data.batch_size` for batch size.
Keep the `best` checkpoint callback in a custom config: reporting loads its
selected checkpoint. Run directories contain `config.yaml`, `history.csv`,
three best checkpoints, `last.ckpt`, `model.pt` inference weights, metadata,
the validation threshold and test metrics.
Resume with the same ablation, seed and epoch budget using `pixi run main.py
fit --config runs/<run>/config.yaml --ckpt_path runs/<run>/last.ckpt`. Only load checkpoints from trusted sources. Batch-limited
runs require `--report_test=false` and are for mechanics checks only.

Run a validation-only study with:

```bash
pixi run python -m src.tune --ablation A1 --study a1-loss --trials 10 \
  --storage sqlite:///traffic-study.db --output-dir tuning-runs
```

Use the same study name and contract to resume. Tuning uses single-device runs,
Optuna TPE, and the integration pruning callback. SQLite is for local,
non-concurrent use. DDP is not implemented; a future DDP path must use RDB
storage and call `check_pruned()` immediately after fit. The study objective
never evaluates test data. After trials complete, run the finalizer:

```bash
pixi run python -m src.tune --ablation A1 --study a1-loss --epochs 150 \
  --storage sqlite:///traffic-study.db --output-dir tuning-runs --finalize-only
```

It selects the best completed trial by the study's validation objective, starts
a fresh training run with its parameters, seed and epoch budget on full loaders,
selects a checkpoint and classification threshold on validation, then evaluates
test once. `--finalize` runs new trials and finalizes in one command. Trial
smoke runs may limit batches, but cannot be finalized or mixed with full-loader
trials in the same study. The study contract also fixes batch size and data path;
older studies without the complete contract require a new study. Each
finalization creates a unique directory with `selection.json`, checkpoints,
`history.csv`, `test_metrics.csv` and `threshold.json`. Finalizing the same study
a second time starts another training run. Keep tuned reports separate from the
frozen baseline results.

## Local checks

```bash
pixi run python -m compileall -q main.py src tests modal_app.py
pixi run python -c 'import lightning, optuna, optuna_integration, modal, modal_app'
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 pixi run python -m unittest discover -s tests -p 'test_*.py' -v
OMP_NUM_THREADS=2 pixi run main.py fit --config configs/ablations/A0.yaml \
  --trainer.max_epochs 1 --data.batch_size 1 \
  --trainer.limit_train_batches 1 --trainer.limit_val_batches 1 --report_test=false \
  --output_dir /tmp/traffic-smoke
OMP_NUM_THREADS=2 pixi run python -m src.tune --trials 1 --epochs 1 --batch-size 1 \
  --limit-train-batches 1 --limit-val-batches 1 --storage sqlite:////tmp/traffic-study.db \
  --output-dir /tmp/traffic-tune
pixi run python -m modal run modal_app.py --help
```

The Lightning smoke performs a real graph forward/backward pass and masked loss.
The unit tests compare LR sequences, force an integration-callback prune,
select the best completed trial and exercise a real one-epoch retrain/report
with 51 origins per split. Running the study command again adds a trial to the
same database. The parity reports describe past results; that runner is no
longer part of the project.

## Modal

A previous reduced A5 T4/BF16 run is documented in the verification archive.
The workspace's `2026.10` image builder required Modal SDK 1.6.3, now pinned
in Pixi. Full-budget A5 attempts did not finish. Check [pricing](https://modal.com/pricing)
and your workspace limits before another paid run. If the data Volume needs
repopulating, upload it with:

```bash
pixi run modal volume create traffic-metr-la
pixi run modal volume put traffic-metr-la datasets/metr-la/METR-LA.h5 /METR-LA.h5
pixi run modal volume put traffic-metr-la datasets/metr-la/adj_METR-LA.pkl /adj_METR-LA.pkl
# --ablation A5 --epochs 1 uses the full data and may exceed the 900-second timeout
# --ablation A1 --trials 3 uses full data for all trials and winner retraining
pixi run modal volume ls traffic-artifacts /
# Use the returned test_metrics path, dropping the /artifacts/ mount prefix:
pixi run modal volume get traffic-artifacts <run-subpath>/test_metrics.csv ./test_metrics.csv
```

The data Volume is read-only. On success the job commits the artifact Volume
before returning its path. A long-lived consumer must close files and call
`reload()` before reading new writes. A timeout can interrupt the function
before commit or leave incomplete artifacts; check the returned path and CSV.
