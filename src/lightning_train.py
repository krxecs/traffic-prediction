"""Single-device Lightning training and checkpoint-based reporting.

Lightning checkpoints retain optimizer state; exported model weights retain the
MultiTaskSTGCN state-dict keys for inference.
"""
import copy
import json
from dataclasses import asdict
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import instantiate_class

from .experiment_data import make_loaders
from .multitask_stgcn import (
    MultiTaskSTGCN,
    _validation_summary,
    class_support,
    congestion_labels,
    evaluate_model,
    masked_huber,
    masked_weighted_bce,
    original_unit_metrics,
    select_validation_threshold,
)


class NamedCheckpoint(ModelCheckpoint):
    @property
    def state_key(self):
        return super().state_key + ':' + str(self.filename)


class TrafficDataModule(L.LightningDataModule):
    def __init__(self, prepared, batch_size=50, num_workers=0):
        super().__init__()
        self.prepared, self.batch_size, self.num_workers = prepared, batch_size, num_workers
        self.loaders = None

    def setup(self, stage=None):
        if self.loaders is None:
            self.loaders = make_loaders(self.prepared, self.batch_size, self.num_workers, torch.cuda.is_available())

    def train_dataloader(self): return self.loaders['train'][0]
    def val_dataloader(self): return self.loaders['val'][0]
    def test_dataloader(self): return self.loaders['test'][0]
    def predict_dataloader(self): return self.test_dataloader()


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    """Warm-up and cosine decay with one update per epoch."""
    def __init__(self, optimizer: torch.optim.Optimizer, min_learning_rate: float,
                 warmup_epochs: int, max_epochs: int | None = None):
        if max_epochs is None:
            raise ValueError('WarmupCosine requires the effective Trainer epoch budget')
        self.minimum, self.warmup, self.maximum = min_learning_rate, max(0, warmup_epochs), max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup:
            return [base * (epoch + 1) / self.warmup for base in self.base_lrs]
        if self.maximum <= self.warmup:
            return self.base_lrs
        progress = min(1., (epoch - self.warmup) / (self.maximum - self.warmup))
        return [self.minimum + .5 * (base - self.minimum) * (1 + np.cos(np.pi * progress)) for base in self.base_lrs]


class TrafficModule(L.LightningModule):
    def __init__(self, prepared, config, pos_weight, lambda_cls=1., optimizer_spec=None, scheduler_spec=None):
        super().__init__()
        self.config = config
        self.optimizer_spec, self.scheduler_spec = optimizer_spec, scheduler_spec
        self.model = MultiTaskSTGCN(prepared['edge_index'], prepared['edge_weight'], 12, experiment_config=config)
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))
        self.mean, self.std, self.lambda_cls = prepared['mean'], prepared['std'], lambda_cls
        self.history = []
        self.train_losses = []
        self.val_losses, self.val_outputs = [], []

    def forward(self, batch):
        return self.model(batch['recent'], batch['time_feat'], batch['periodic_feat'])

    def predict_step(self, batch, batch_idx):
        speed, logits = self(batch)
        return {'origin': batch['origin'].detach().cpu(),
                'speed_mph': (speed.float() * self.std + self.mean).detach().cpu(),
                'congestion_probability': torch.sigmoid(logits.float()).detach().cpu()}

    def losses(self, batch):
        pred, logits = self(batch)
        reg = masked_huber(pred, batch['target'], batch['mask'])
        cls = masked_weighted_bce(logits, congestion_labels(batch['raw_target'], 40.), batch['mask'], self.pos_weight)
        total = reg + self.lambda_cls * cls
        if not torch.isfinite(total): raise FloatingPointError('Non-finite loss')
        return pred, logits, reg, cls, total

    def on_train_epoch_start(self):
        self.train_losses = []
        self.epoch_lr = self.optimizers().param_groups[0]['lr']

    def training_step(self, batch, batch_idx):
        _, _, reg, cls, total = self.losses(batch)
        self.train_losses.append([x.detach().item() for x in (reg, cls, total)])
        return total

    def on_validation_epoch_start(self):
        self.val_losses, self.val_outputs = [], []

    def validation_step(self, batch, batch_idx):
        with torch.autocast(device_type=self.device.type, enabled=False):
            pred, logits, reg, cls, total = self.losses(batch)
        self.val_losses.append([x.item() for x in (reg, cls, total)])
        self.val_outputs.append(tuple(x.detach().float().cpu().numpy() for x in
            (pred, batch['target'], batch['raw_target'], logits, batch['mask'])))

    def on_validation_epoch_end(self):
        outputs = tuple(np.concatenate(items) for items in zip(*self.val_outputs))
        summary = _validation_summary(np.mean(self.val_losses, axis=0), outputs, 40., self.mean, self.std)
        for key, value in summary.items(): self.log(key, value, on_epoch=True, logger=False)
        train = np.mean(self.train_losses, axis=0)
        self.history.append(dict(epoch=self.current_epoch + 1, lr=self.epoch_lr,
            train_regression_loss=float(train[0]), train_classification_loss=float(train[1]),
            train_total_loss=float(train[2]), **summary))

    def on_save_checkpoint(self, checkpoint):
        checkpoint['traffic_history'] = self.history
        checkpoint['traffic_config'] = self.checkpoint_contract()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('traffic_config') != self.checkpoint_contract():
            raise ValueError('Resume requires the same experiment config and epoch budget')
        self.history = checkpoint.get('traffic_history', [])

    def checkpoint_contract(self):
        return {'graph': asdict(self.config), 'optimizer': self.optimizer_spec,
                'lr_scheduler': self.scheduler_spec, 'max_epochs': self.trainer.max_epochs}

    def configure_optimizers(self):
        # LightningCLI supplies the standard --optimizer and --lr_scheduler arguments.
        # The programmatic Optuna path uses the same YAML class specifications.
        if self.optimizer_spec is None or self.scheduler_spec is None:
            raise RuntimeError('An optimizer and scheduler must be configured')
        optimizer = instantiate_class(self.parameters(), self.optimizer_spec)
        scheduler = instantiate_class(optimizer, self.scheduler_spec)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


def fit(prepared, config, output, seed=42, batch_size=50, limit_train_batches=1.0,
        limit_val_batches=1.0, resume=None, extra_callbacks=()):
    L.seed_everything(seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dm = TrafficDataModule(prepared, batch_size); dm.setup()
    support = class_support(dm.loaders['train'][1], 40.)
    if not support['positive']: raise ValueError('No positive training targets')
    optimizer_spec = copy.deepcopy(config.optimizer)
    scheduler_spec = copy.deepcopy(config.lr_scheduler)
    if scheduler_spec['class_path'] == 'src.lightning_train.WarmupCosine':
        scheduler_spec['init_args']['max_epochs'] = config.trainer['max_epochs']
    module = TrafficModule(prepared, config.experiment, support['negative'] / support['positive'],
                           optimizer_spec=optimizer_spec, scheduler_spec=scheduler_spec)
    callbacks = []
    for item in config.trainer['callbacks']:
        spec = copy.deepcopy(item)
        if spec['class_path'] == 'src.lightning_train.NamedCheckpoint':
            spec['init_args']['dirpath'] = str(output)
        callbacks.append(instantiate_class((), spec))
    callbacks += list(extra_callbacks)
    checkpoints = [cb for cb in callbacks if isinstance(cb, NamedCheckpoint) and cb.filename == 'best']
    precision = ('bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed') if torch.cuda.is_available() else '32-true'
    trainer = L.Trainer(max_epochs=config.trainer['max_epochs'], accelerator='auto', devices=1, precision=precision,
        num_sanity_val_steps=0, callbacks=callbacks, default_root_dir=output, logger=False,
        gradient_clip_val=config.trainer.get('gradient_clip_val'),
        limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, datamodule=dm, ckpt_path=resume)
    for callback in extra_callbacks:
        if hasattr(callback, 'check_pruned'): callback.check_pruned()
    trainer.save_checkpoint(output / 'last.ckpt')
    pd.DataFrame(module.history).to_csv(output / 'history.csv', index=False)
    if not checkpoints[0].best_model_path: raise RuntimeError('No best checkpoint was selected')
    return module, dm, trainer, checkpoints[0].best_model_path


def report(module, dm, checkpoint, output, name, horizons=((2, 15), (5, 30), (8, 45))):
    state = torch.load(checkpoint, map_location=module.device, weights_only=False)
    module.load_state_dict(state['state_dict'])
    torch.save(module.model.state_dict(), Path(output) / 'model.pt')
    args = (module.device, False, module.pos_weight, module.lambda_cls, 40.)
    _, val = evaluate_model(module.model, dm.val_dataloader(), *args)
    valid = val[4].astype(bool)
    threshold, _ = select_validation_threshold(val[2][valid] < 40., 1 / (1 + np.exp(-val[3][valid])))
    _, test = evaluate_model(module.model, dm.test_dataloader(), *args)
    rows = []
    for index, horizon in horizons:
        arrays = [x[:, index, :] for x in test]
        if not arrays[4].any(): raise ValueError(f'No observed test targets at {horizon} minutes')
        rows.append(dict(Model='Joint multitask STGCN(Cheb)', Experiment=name, ExperimentConfig=str(asdict(module.config)),
            Horizon=f'{horizon} min', **original_unit_metrics(*arrays, module.mean, module.std, 40., threshold)))
    pd.DataFrame(rows).to_csv(Path(output) / 'test_metrics.csv', index=False)
    (Path(output) / 'threshold.json').write_text(json.dumps({'threshold': threshold, 'test_missing': int((~test[4].astype(bool)).sum())}, indent=2))
