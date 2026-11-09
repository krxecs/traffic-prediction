"""The presets have the expected optimizer, scheduler, and Trainer settings."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from src.experiment_config import EXPERIMENT_CONFIGS, load_experiment, parse_experiment


class AblationConfigTests(unittest.TestCase):
    def test_yaml_presets_and_actual_a0_optimizer(self):
        self.assertEqual(set(EXPERIMENT_CONFIGS), {'A0', 'A1', 'A2', 'A3a', 'A3b', 'A4', 'A5', 'A5a', 'A6', 'A7'})
        baseline = EXPERIMENT_CONFIGS['A0']
        self.assertEqual(baseline.optimizer['class_path'], 'torch.optim.RMSprop')
        self.assertEqual(baseline.optimizer['init_args']['lr'], 0.001)
        self.assertEqual(baseline.lr_scheduler['init_args'], {'step_size': 5, 'gamma': 0.7})
        self.assertEqual(baseline.trainer['max_epochs'], 50)
        self.assertIsNone(baseline.trainer['gradient_clip_val'])
        for name, config in EXPERIMENT_CONFIGS.items():
            with self.subTest(ablation=name):
                self.assertEqual(config, load_experiment(name))
                self.assertEqual([callback['init_args']['filename']
                    for callback in config.trainer['callbacks'][:3]], ['best', 'best-mae', 'best-auprc'])
                if name != 'A0':
                    self.assertEqual(config.optimizer['class_path'], 'torch.optim.AdamW')
                    self.assertEqual(config.trainer['max_epochs'], 150)
                    self.assertEqual(config.trainer['gradient_clip_val'], 4.879639408647977 if name == 'A5a' else 2)
                    self.assertEqual(config.lr_scheduler['class_path'], 'src.lightning_train.WarmupCosine')
                    self.assertNotIn('max_epochs', config.lr_scheduler['init_args'])
                    self.assertEqual(config.trainer['callbacks'][3]['init_args']['patience'], 15)

    def test_a5a_preserves_a5_contract_with_tuned_parameters(self):
        original = asdict(EXPERIMENT_CONFIGS['A5'])
        tuned = asdict(EXPERIMENT_CONFIGS['A5a'])
        self.assertEqual(tuned['name'], 'A5a')
        self.assertEqual(tuned['optimizer']['init_args'], {
            'lr': 0.0005105903209394755, 'weight_decay': 1.2087541473056965e-06})
        tuned['name'] = 'A5'
        tuned['optimizer']['init_args'] = original['optimizer']['init_args']
        tuned['trainer']['gradient_clip_val'] = original['trainer']['gradient_clip_val']
        self.assertEqual(tuned, original)

    def test_only_graph_settings_in_experiment(self):
        values = asdict(EXPERIMENT_CONFIGS['A0'].experiment)
        with self.assertRaisesRegex(ValueError, 'missing'):
            parse_experiment({key: value for key, value in values.items() if key != 'graph_mode'})
        with self.assertRaisesRegex(ValueError, 'Invalid type for adaptive_top_k'):
            parse_experiment({**values, 'adaptive_top_k': '16'})
        with self.assertRaisesRegex(ValueError, 'extra'):
            parse_experiment({**values, 'unknown': 1})

    def test_yaml_name_must_match_file(self):
        with tempfile.TemporaryDirectory() as directory:
            text = (Path(__file__).resolve().parents[1] / 'configs/ablations/A0.yaml').read_text()
            Path(directory, 'A0.yaml').write_text(text.replace('ablation: A0', 'ablation: A1'))
            with patch('src.experiment_config.ABLATION_DIR', Path(directory)), self.assertRaisesRegex(ValueError, 'match the filename'):
                load_experiment('A0')


if __name__ == '__main__':
    unittest.main()
