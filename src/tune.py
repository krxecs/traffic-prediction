"""Tune on validation; evaluate test once after retraining the selected winner."""
import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import optuna
from optuna_integration import PyTorchLightningPruningCallback

from .experiment_data import EXPERIMENT_CONFIGS, prepare_metr_la
from .lightning_train import fit, report


def configured_settings(ablation, objective, epochs, params):
    """Apply trial values to the standard YAML optimizer, Trainer, and callbacks."""
    config = copy.deepcopy(EXPERIMENT_CONFIGS[ablation])
    config.trainer['max_epochs'] = epochs
    config.optimizer['init_args']['lr'] = params['learning_rate']
    config.optimizer['init_args']['weight_decay'] = params['weight_decay']
    config.trainer['gradient_clip_val'] = params['max_grad_norm']
    mode = 'max' if objective == 'val_auprc' else 'min'
    for callback in config.trainer['callbacks']:
        args = callback['init_args']
        if (callback['class_path'] == 'src.lightning_train.NamedCheckpoint'
                and args['filename'] == 'best') or callback['class_path'] == 'lightning.pytorch.callbacks.EarlyStopping':
            args['monitor'], args['mode'] = objective, mode
    return config


def finalize_winner(study, *, ablation, objective, epochs, seed, batch_size, output_dir,
                    prepared):
    """Retrain the best completed trial on full loaders, then test its best checkpoint."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError('No completed trial is available for final evaluation')
    winner = study.best_trial
    config = configured_settings(ablation, objective, epochs, winner.params)
    output = Path(output_dir) / study.study_name / f'winner-{winner.number}-{uuid4().hex}'
    output.mkdir(parents=True, exist_ok=False)
    selection = {'study': study.study_name, 'trial': winner.number,
                 'validation_objective': objective, 'validation_value': winner.value,
                 'parameters': winner.params, 'config': asdict(config), 'seed': seed,
                 'full_train_and_validation_loaders': True,
                 'selection': 'best completed trial; fresh full-data fit with same seed and epoch budget'}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    module, dm, _, checkpoint = fit(prepared, config, output, seed=seed, batch_size=batch_size)
    report(module, dm, checkpoint, output, f'{ablation}-tuned')
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='datasets/metr-la')
    p.add_argument('--output-dir', default='tuning-runs')
    p.add_argument('--storage', default='sqlite:///traffic-study.db')
    p.add_argument('--study', default='traffic')
    tunable = [name for name, config in EXPERIMENT_CONFIGS.items()
               if config.optimizer['class_path'] == 'torch.optim.AdamW']
    p.add_argument('--ablation', choices=tunable, default=tunable[0])
    p.add_argument('--objective', choices=['val_total_loss', 'val_auprc'], default='val_total_loss')
    p.add_argument('--trials', type=int, default=10); p.add_argument('--epochs', type=int)
    p.add_argument('--finalize', action='store_true', help='Retrain the best completed trial on full data and evaluate test once')
    p.add_argument('--finalize-only', action='store_true', help='Skip new trials and finalize an existing study')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--limit-train-batches', type=int); p.add_argument('--limit-val-batches', type=int)
    a = p.parse_args()
    if a.epochs is None:
        a.epochs = EXPERIMENT_CONFIGS[a.ablation].trainer['max_epochs']
    if not 1 <= a.trials <= 100: p.error('trials must be between 1 and 100')
    if a.epochs < 1: p.error('epochs must be positive')
    if any(x is not None and x < 1 for x in (a.limit_train_batches, a.limit_val_batches)):
        p.error('batch limits must be positive')
    if (a.finalize or a.finalize_only) and (a.limit_train_batches is not None or a.limit_val_batches is not None):
        p.error('Batch-limited smoke studies cannot be finalized')
    direction = 'maximize' if a.objective == 'val_auprc' else 'minimize'
    study = optuna.create_study(study_name=a.study, storage=a.storage, load_if_exists=True,
        direction=direction, sampler=optuna.samplers.TPESampler(seed=a.seed), pruner=optuna.pruners.MedianPruner())
    if study.direction.name.lower() != direction: raise ValueError('Study direction differs from requested objective')
    contract = {'ablation': a.ablation, 'objective': a.objective, 'epochs': a.epochs, 'seed': a.seed,
                'batch_size': a.batch_size, 'limit_train_batches': a.limit_train_batches,
                'limit_val_batches': a.limit_val_batches,
                'data_dir': str(Path(a.data_dir).resolve())}
    existing = study.user_attrs.get('contract')
    if (existing is None and study.trials) or (existing is not None and existing != contract):
        raise ValueError('Study contract differs from requested configuration; use a new study')
    study.set_user_attr('contract', contract)
    # A resumed invocation must not replay the initial random sampler stream.
    study.sampler = optuna.samplers.TPESampler(seed=a.seed + len(study.trials))
    prepared = prepare_metr_la(a.data_dir)
    def objective(trial):
        params = {'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                  'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
                  'max_grad_norm': trial.suggest_float('max_grad_norm', 1., 5.)}
        config = configured_settings(a.ablation, a.objective, a.epochs, params)
        output = Path(a.output_dir) / a.study / f'trial-{trial.number}-{uuid4().hex}'
        trial.set_user_attr('output_dir', str(output))
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps({'config': asdict(config), 'seed': a.seed,
            'objective': a.objective, 'optuna': optuna.__version__}, indent=2))
        callback = PyTorchLightningPruningCallback(trial, monitor=a.objective)
        module, _, _, _ = fit(prepared, config, output, a.seed, a.batch_size,
            a.limit_train_batches or 1.0, a.limit_val_batches or 1.0, extra_callbacks=[callback])
        return max(r[a.objective] for r in module.history) if direction == 'maximize' else min(r[a.objective] for r in module.history)
    if not a.finalize_only:
        study.optimize(objective, n_trials=a.trials)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        if a.finalize or a.finalize_only:
            raise ValueError('No completed trial is available for final evaluation')
        print('No completed trial yet; no winner to report')
        return
    print('Best trial:', study.best_trial.number, study.best_value, study.best_params)
    if a.finalize or a.finalize_only:
        output = finalize_winner(study, ablation=a.ablation, objective=a.objective,
            epochs=a.epochs, seed=a.seed, batch_size=a.batch_size,
            output_dir=a.output_dir, prepared=prepared)
        print('Final evaluation:', output)


if __name__ == '__main__': main()
