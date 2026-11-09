# A5 STGCN tuning on Modal

Audience: GPT-6.1 Terra implementation agent. Own the implementation, cloud execution, and result reporting. The parent agent only supervises. Use the repository's `modal` skill and current project commands in `docs/lightning-runs.md`.

## 1. Establish the run contract

Inspect `configs/ablations/A5.yaml`, `src/tune.py`, `src/lightning_train.py`, `modal_app.py`, the dataset files, and the existing Modal volumes. Keep the A5 architecture and data split fixed. Tune only `learning_rate`, `weight_decay`, and `max_grad_norm` in the existing Optuna search space; minimize `val_total_loss` with seed 42. Use the full training and validation loaders, the A5 150 epoch ceiling and existing early stopping. Run 10 trials, then one fresh winner retrain and held out test report. Record dataset identity, code revision, study contract, trial states, best validation value, winner parameters, and artifact paths. Completion: the exact commands and effective study contract are documented before a paid run.

## 2. Prepare the Modal execution path

Make `modal_app.py` run A5 tuning through `python -m src.tune` on exactly one `gpu='A100-40GB'` per job. Set a timeout that can finish a full A5 trial and winner retrain; retain one concurrent GPU container and persistent data/artifact volumes. Make the study resumable across remote invocations with a stable study name and SQLite path on the artifact volume. Keep each trial's checkpoint directory unique. Add any CLI parameters needed to submit this A5 study and finalize it separately without changing other ablations' behavior. Ensure incomplete or timed-out invocations can resume without overwriting completed trials. Completion: local CLI/import checks pass and the submitted command clearly names A5, A100-40GB, 10 trials, 150 epochs, study name, and volumes.

## 3. Execute and verify

Check Modal authentication and that both METR-LA data files exist in the data volume; upload missing files from `datasets/metr-la` if necessary. Run the A5 study on Modal, using the project utility CLI and Optuna. Resume interrupted work until 10 trials have terminal states and at least one is complete. Finalize the best completed trial once, retrieve `selection.json`, the study summary, and `test_metrics.csv`, and verify the final metrics file is nonempty and belongs to A5-tuned. Cloud execution is authorized by the user's request; report any account, quota, data, or runtime blocker with its exact error if encountered. Completion: provide the best validation value, parameters, trial state counts, final test metrics location, Modal artifact paths, and commands needed to reproduce or resume.

## 4. Report and hand back

Update `docs/lightning-runs.md` with the working A5-specific Modal commands and any implementation changes. Run targeted local tests for changed code and report their outcomes. Keep trial and final test artifacts distinct from frozen baseline reports. The final response must state whether the complete cloud study and winner evaluation finished, with evidence; never present a smoke run as the full study.
