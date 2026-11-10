"""Winner selection must keep test evaluation outside the Optuna objective."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
from torch.utils.data import DataLoader

from src.experiment_data import make_loaders, prepare_metr_la
from src.tune import finalize_winner, main


def reduced_loaders(prepared):
    """A small real-data fixture for the Optuna integration test."""
    result = {}
    for split, (_, dataset) in make_loaders(prepared).items():
        chosen = copy.copy(dataset)
        chosen.origins = dataset.origins[:51].copy()
        result[split] = (DataLoader(chosen, batch_size=50, shuffle=False), chosen)
    return result


class FinalizeWinnerTests(unittest.TestCase):
    def test_batch_limited_study_cannot_be_finalized(self):
        with patch('sys.argv', ['tune', '--finalize', '--limit-train-batches', '1']):
            with self.assertRaises(SystemExit):
                main()

    def test_existing_study_rejects_changed_batch_limits(self):
        study = optuna.create_study(direction='minimize')
        study.set_user_attr('contract', {'ablation': 'A1', 'objective': 'val_total_loss',
            'epochs': 1, 'seed': 42, 'batch_size': 50, 'limit_train_batches': 1,
            'limit_val_batches': 1, 'data_dir': str(Path('datasets/metr-la').resolve())})
        with patch('sys.argv', ['tune', '--ablation', 'A1', '--epochs', '1']), \
                patch('src.tune.optuna.create_study', return_value=study):
            with self.assertRaisesRegex(ValueError, 'Study contract differs'):
                main()

    def test_only_completed_trial_retrains_on_full_loaders_and_reports_once(self):
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))

        def objective(trial):
            trial.suggest_float('learning_rate', 1e-4, 1e-3)
            trial.suggest_float('weight_decay', 1e-6, 1e-2)
            trial.suggest_float('max_grad_norm', 1., 5.)
            return 2. - trial.number

        study.optimize(objective, n_trials=2)
        with tempfile.TemporaryDirectory() as temporary, patch('src.tune.fit', return_value=('model', 'data', 'trainer', 'best.ckpt')) as train, patch('src.tune.report') as report:
            directory = finalize_winner(study, ablation='A1', objective='val_total_loss',
                epochs=2, seed=42, batch_size=50, output_dir=temporary, prepared={'data': 'reference'})
            selection = json.loads((directory / 'selection.json').read_text())
            self.assertEqual(selection['trial'], 1)
            self.assertEqual(selection['validation_value'], 1.)
            self.assertEqual(selection['parameters'], study.best_trial.params)
            self.assertEqual(selection['config']['trainer']['max_epochs'], 2)
            self.assertTrue(selection['full_train_and_validation_loaders'])
            _, config, output = train.call_args.args
            self.assertEqual(config.optimizer['init_args']['lr'], study.best_trial.params['learning_rate'])
            self.assertEqual(output, directory)
            self.assertEqual(train.call_args.kwargs, {'seed': 42, 'batch_size': 50})
            report.assert_called_once_with('model', 'data', 'best.ckpt', directory, 'A1-tuned')

    def test_real_short_retrain_writes_test_report(self):
        data = prepare_metr_la('datasets/metr-la')
        # Keep the real metrics and checkpoint path, but limit CPU work to 51 origins per split.
        small = reduced_loaders(data)
        with tempfile.TemporaryDirectory() as temporary:
            study = optuna.create_study(study_name='finalize-integration',
                storage='sqlite:///' + str(Path(temporary) / 'study.db'), direction='minimize')
            study.optimize(lambda trial: (
                trial.suggest_float('learning_rate', 3e-4, 3e-4),
                trial.suggest_float('weight_decay', 1e-4, 1e-4),
                trial.suggest_float('max_grad_norm', 2., 2.), 0.5)[-1], n_trials=1)
            with patch('src.lightning_train.make_loaders', return_value=small):
                output = finalize_winner(study, ablation='A1', objective='val_total_loss',
                    epochs=1, seed=42, batch_size=50, output_dir=temporary, prepared=data)
            self.assertEqual(len(list(output.glob('test_metrics.csv'))), 1)
            self.assertEqual(len(list(output.glob('threshold.json'))), 1)
            self.assertEqual(len(list(output.glob('best.ckpt'))), 1)
            self.assertEqual(len(list(output.glob('model.pt'))), 1)

    def test_no_complete_trial_never_evaluates_test(self):
        study = optuna.create_study()
        with tempfile.TemporaryDirectory() as temporary, patch('src.tune.fit') as train, patch('src.tune.report') as report:
            with self.assertRaisesRegex(ValueError, 'No completed trial'):
                finalize_winner(study, ablation='A1', objective='val_total_loss',
                    epochs=2, seed=42, batch_size=50, output_dir=temporary, prepared={})
            train.assert_not_called()
            report.assert_not_called()


if __name__ == '__main__':
    unittest.main()
