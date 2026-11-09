"""Exercise the LightningCLI entrypoint, not just the programmatic fit helper."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from main import main
from src.experiment_data import EXPERIMENT_CONFIGS, make_loaders

ABLATIONS = Path(__file__).resolve().parents[1] / 'configs' / 'ablations'


class LightningCLITests(unittest.TestCase):
    def test_ablation_trainer_and_data_arguments_reach_real_fit(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv', ['lightning_cli']):
                cli = main(['fit', '--config=' + str(ABLATIONS / 'A1.yaml'), '--trainer.max_epochs=1',
                    '--optimizer.lr=0.0007', '--data.batch_size=1', '--trainer.limit_train_batches=1',
                    '--trainer.limit_val_batches=1', '--report_test=false',
                    '--output_dir=' + directory])
            self.assertEqual(cli.trainer.max_epochs, 1)
            self.assertEqual(cli.model.ablation, 'A1')
            self.assertEqual(cli.trainer.lr_scheduler_configs[0].scheduler.maximum, 1)
            self.assertEqual(cli.trainer.optimizers[0].defaults['lr'], 0.0007)
            self.assertEqual(cli.model.optimizer_spec['init_args']['lr'], 0.0007)
            self.assertEqual(cli.datamodule.batch_size, 1)
            self.assertTrue((cli.output / 'best.ckpt').is_file())
            self.assertTrue((cli.output / 'history.csv').is_file())
            self.assertIn('max_epochs: 1', (cli.output / 'config.yaml').read_text())
            self.assertIn('lr: 0.0007', (cli.output / 'config.yaml').read_text())
            self.assertIn('ablation: A1', (cli.output / 'config.yaml').read_text())
            self.assertIn('NamedCheckpoint', (cli.output / 'config.yaml').read_text())
            self.assertNotIn(str(cli.output), (cli.output / 'config.yaml').read_text())
            self.assertNotIn('\n  epochs: 1\n', (cli.output / 'config.yaml').read_text())
            with patch('sys.argv', ['main.py']), redirect_stdout(io.StringIO()) as preview, self.assertRaises(SystemExit) as result:
                main(['fit', '--config', str(cli.output / 'config.yaml'), '--trainer.max_epochs=2', '--print_config'])
            self.assertEqual(result.exception.code, 0)
            self.assertIn('max_epochs: 2', preview.getvalue())
            self.assertEqual(json.loads((cli.output / 'metadata.json').read_text())['seed'], 42)
            self.assertFalse((cli.output / 'test_metrics.csv').exists())
            with patch('sys.argv', ['main.py']):
                resumed = main(['fit', '--config', str(cli.output / 'config.yaml'),
                                '--ckpt_path', str(cli.output / 'last.ckpt')])
            self.assertEqual(resumed.output, cli.output)
            self.assertEqual(resumed.model.history, cli.model.history)

    def test_yaml_epoch_budget_is_the_default_without_a_model_override(self):
        from main import TrafficCLI

        class StopBeforeTraining(Exception):
            pass

        budgets = []

        def stop(cli):
            budgets.append(cli.trainer.max_epochs)
            raise StopBeforeTraining

        with patch('sys.argv', ['main.py']), patch.object(TrafficCLI, 'before_fit', stop), self.assertRaises(StopBeforeTraining):
            main(['fit', '--config', str(ABLATIONS / 'A0.yaml')])
        self.assertEqual(budgets, [50])
        with (patch('sys.argv', ['main.py']), patch('main.prepare_metr_la') as prepare,
              redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error):
            main(['fit', '--model.epochs=2'])
        self.assertEqual(error.exception.code, 2)
        prepare.assert_not_called()

    def test_fit_reports_on_the_best_checkpoint(self):
        torch.set_num_threads(2)

        def small_loaders(prepared, batch_size, num_workers, pin_memory):
            result = {}
            for name, (_, dataset) in make_loaders(prepared, batch_size, num_workers, pin_memory).items():
                small = copy.copy(dataset)
                small.origins = dataset.origins[:51].copy()
                result[name] = (DataLoader(small, batch_size=batch_size, shuffle=name == 'train'), small)
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv', ['main.py']), patch('main.make_loaders', side_effect=small_loaders):
                cli = main(['fit', '--config', str(ABLATIONS / 'A5.yaml'),
                            '--trainer.max_epochs=1', '--output_dir=' + directory])
            self.assertTrue((cli.output / 'best.ckpt').is_file())
            self.assertTrue((cli.output / 'threshold.json').is_file())
            self.assertEqual(len(json.loads((cli.output / 'threshold.json').read_text())), 2)
            self.assertEqual(len((cli.output / 'test_metrics.csv').read_text().splitlines()), 4)
            self.assertTrue((cli.output / 'model.pt').is_file())

    def test_only_fit_subcommand_is_available(self):
        from main import TrafficCLI
        self.assertEqual(set(TrafficCLI.subcommands()), {'fit'})

    def test_yaml_presets_parse_without_loading_data(self):
        with patch('sys.argv', ['main.py']), patch('main.prepare_metr_la') as prepare:
            for name in EXPERIMENT_CONFIGS:
                with self.subTest(name=name), redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
                    main(['fit', '--config', str(ABLATIONS / f'{name}.yaml'), '--print_config'])
                self.assertEqual(error.exception.code, 0)
                self.assertIn(f'ablation: {name}\n', output.getvalue())
        prepare.assert_not_called()

    def test_standard_optimizer_flags_parse_without_loading_data(self):
        with patch('sys.argv', ['main.py']), patch('main.prepare_metr_la') as prepare:
            with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
                main(['fit', '--config', str(ABLATIONS / 'A1.yaml'), '--optimizer.lr=0.0007', '--print_config'])
            self.assertEqual(error.exception.code, 0)
            self.assertIn('lr: 0.0007', output.getvalue())
        prepare.assert_not_called()

    def test_ablation_name_cannot_be_relabelled_with_a_flag(self):
        with patch('main.prepare_metr_la') as prepare, self.assertRaisesRegex(ValueError, 'Set model.ablation in the YAML'):
            main(['fit', '--config', str(ABLATIONS / 'A0.yaml'), '--model.ablation=A1'])
        prepare.assert_not_called()

    def test_limited_batches_require_explicit_no_test(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv', ['lightning_cli']), self.assertRaisesRegex(ValueError, 'Batch-limited'):
                main(['fit', '--config', str(ABLATIONS / 'A1.yaml'), '--trainer.max_epochs=1',
                      '--trainer.limit_train_batches=1', '--output_dir=' + directory])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
