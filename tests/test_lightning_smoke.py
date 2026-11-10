"""Local mechanics checks for Lightning training and pruning."""
import copy
import tempfile
import unittest

import optuna
import torch
from optuna_integration import PyTorchLightningPruningCallback

from src.experiment_data import EXPERIMENT_CONFIGS, prepare_metr_la
from src.lightning_train import WarmupCosine, fit


class Smoke(unittest.TestCase):
    def test_scheduler(self):
        c = EXPERIMENT_CONFIGS['A1']
        lr = c.optimizer['init_args']['lr']
        schedule = c.lr_scheduler['init_args']
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=lr)
        scheduler = WarmupCosine(optimizer, schedule['min_learning_rate'], schedule['warmup_epochs'], 7)
        expected = [lr * (epoch + 1) / schedule['warmup_epochs'] for epoch in range(5)]
        expected += [lr, (lr + schedule['min_learning_rate']) / 2, schedule['min_learning_rate']]
        for value in expected:
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], value)
            optimizer.step(); scheduler.step()

    def test_pruning(self):
        class AlwaysPrune(optuna.pruners.BasePruner):
            def prune(self, study, trial): return True
        study = optuna.create_study(pruner=AlwaysPrune())
        data = prepare_metr_la('datasets/metr-la')
        config = copy.deepcopy(EXPERIMENT_CONFIGS['A1']); config.trainer['max_epochs'] = 1
        def objective(trial):
            callback = PyTorchLightningPruningCallback(trial, 'val_total_loss')
            with tempfile.TemporaryDirectory() as path:
                fit(data, config, path, batch_size=1, limit_train_batches=1,
                    limit_val_batches=1, extra_callbacks=[callback])
            return 0.
        study.optimize(objective, n_trials=1)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.PRUNED)


if __name__ == '__main__': unittest.main()
