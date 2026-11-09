"""Declarative training CLI; domain-specific checkpoint and metric rules stay fixed."""
import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.cli import LightningCLI

from src.experiment_config import GraphConfig, parse_experiment
from src.experiment_data import make_loaders, prepare_metr_la
from src.lightning_train import (
    NamedCheckpoint,
    TrafficDataModule,
    TrafficModule,
    report,
)
from src.multitask_stgcn import class_support


class TrafficCLIData(TrafficDataModule):
    def __init__(self, data_dir: str = 'datasets/metr-la', batch_size: int = 50,
                 num_workers: int = 0):
        prepared = prepare_metr_la(data_dir)
        super().__init__(prepared, batch_size, num_workers)
        self.data_dir = data_dir
        self.loaders = make_loaders(prepared, batch_size, num_workers, torch.cuda.is_available())
        support = class_support(self.loaders['train'][1], 40.)
        if not support['positive']:
            raise ValueError('No positive training targets')
        self.pos_weight = support['negative'] / support['positive']


def resolve_experiment(value):
    if isinstance(value, GraphConfig):
        return value
    return parse_experiment(value.as_dict() if hasattr(value, 'as_dict') else value)


class TrafficCLIModel(TrafficModule):
    def __init__(self, prepared: dict, pos_weight: float, ablation: str,
                 experiment: GraphConfig | None = None):
        if experiment is None:
            raise ValueError('A graph experiment configuration is required')
        super().__init__(prepared, resolve_experiment(experiment), pos_weight)
        self.ablation = ablation


class TrafficCLI(LightningCLI):
    @staticmethod
    def subcommands():
        # Reporting and threshold selection are defined only for a completed fit.
        return {'fit': LightningCLI.subcommands()['fit']}

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--output_dir', type=str, default='runs')
        parser.add_argument('--report_test', type=bool, default=True)
        parser.add_class_arguments(GraphConfig, 'experiment')
        parser.link_arguments('experiment', 'model.experiment', apply_on='instantiate')
        parser.link_arguments('data.prepared', 'model.prepared', apply_on='instantiate')
        parser.link_arguments('data.pos_weight', 'model.pos_weight', apply_on='instantiate')

    def before_instantiate_classes(self):
        config = self.config.fit
        replay_config = copy.deepcopy(config)
        if config.trainer.max_epochs is None or config.trainer.max_epochs < 1:
            raise ValueError('trainer.max_epochs must be positive')
        if config.optimizer is None or config.lr_scheduler is None:
            raise ValueError('The YAML must specify an optimizer and lr_scheduler')
        if config.report_test and any(
            not (value is None or (isinstance(value, float) and value == 1.0))
            for value in (config.trainer.limit_train_batches, config.trainer.limit_val_batches)
        ):
            raise ValueError('Batch-limited training requires --report_test=false')
        if config.lr_scheduler.class_path == 'src.lightning_train.WarmupCosine':
            config.lr_scheduler.init_args.max_epochs = config.trainer.max_epochs
        # Save the replayable YAML before setting output-specific checkpoint paths.
        self.saved_config = self._parser('fit').dump(replay_config)
        if config.ckpt_path and not Path(config.ckpt_path).is_file():
            raise ValueError('--ckpt_path must name an existing checkpoint file')
        seed = config.seed_everything
        name = config.model.ablation
        self.output = (Path(config.ckpt_path).resolve().parent if config.ckpt_path else
                       Path(config.output_dir) / f'{name}-{seed}-{uuid4().hex}')
        config.trainer.default_root_dir = str(self.output)
        if config.trainer.fast_dev_run or config.trainer.enable_checkpointing is False:
            raise ValueError('Training requires regular checkpointing; use batch limits for smoke tests')
        if config.trainer.devices != 1:
            raise ValueError('This experiment requires a single device for exact evaluation')
        for callback in config.trainer.callbacks or []:
            if callback.class_path == 'src.lightning_train.NamedCheckpoint':
                callback.init_args.dirpath = str(self.output)

    def after_instantiate_classes(self):
        self.model.optimizer_spec = self.config.fit.optimizer.as_dict()
        self.model.scheduler_spec = self.config.fit.lr_scheduler.as_dict()

    def before_fit(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'config.yaml').write_text(self.saved_config)

    def after_fit(self):
        self.trainer.save_checkpoint(self.output / 'last.ckpt')
        pd.DataFrame(self.model.history).to_csv(self.output / 'history.csv', index=False)
        best = next((cb.best_model_path for cb in self.trainer.callbacks
                     if isinstance(cb, NamedCheckpoint) and cb.filename == 'best'), None)
        if not best:
            raise RuntimeError('No best checkpoint was selected')
        def sha256(path):
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest()
        prepared = self.datamodule.prepared
        identity = {name: sha256(Path(self.datamodule.data_dir) / name)
                    for name in ('METR-LA.h5', 'adj_METR-LA.pkl')}
        (self.output / 'metadata.json').write_text(json.dumps({
            'seed': self.config.fit.seed_everything, 'config': asdict(self.model.config),
            'optimizer': self.model.optimizer_spec, 'lr_scheduler': self.model.scheduler_spec,
            'max_epochs': self.trainer.max_epochs,
            'data_sha256': identity,
            'split': [prepared['train_end'], prepared['val_end'], len(prepared['raw_mph'])],
            'torch': torch.__version__, 'lightning': L.__version__,
            'best_epoch': int(torch.load(best, map_location='cpu', weights_only=False)['epoch']) + 1}, indent=2))
        if self.config.fit.report_test:
            report(self.model, self.datamodule, best, self.output, self.model.ablation)
        print(self.output)


def main(args=None):
    supplied = sys.argv[1:] if args is None else args
    if any(value == '--model.ablation' or value.startswith('--model.ablation=') for value in supplied):
        raise ValueError('Set model.ablation in the YAML config, not as a CLI override')
    precision = ('bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed') if torch.cuda.is_available() else '32-true'
    cli = TrafficCLI(TrafficCLIModel, TrafficCLIData, args=args,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={'accelerator': 'auto', 'devices': 1, 'precision': precision,
                          'num_sanity_val_steps': 0, 'logger': False,
                          'enable_progress_bar': False, 'enable_model_summary': False})
    return cli


if __name__ == '__main__':
    main()
