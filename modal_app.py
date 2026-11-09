"""Modal submission definitions. Remote training needs an account and incurs charges."""
import modal

from src.experiment_config import EXPERIMENT_CONFIGS

app = modal.App('traffic-stgcn')
data_volume = modal.Volume.from_name('traffic-metr-la', create_if_missing=True)
artifacts = modal.Volume.from_name('traffic-artifacts', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('torch==2.13.0', 'torch-geometric==2.8.0.post1', 'lightning==2.6.6',
                    'optuna==4.9.0', 'optuna-integration==4.9.0', 'pandas==3.0.6',
                    'numpy==2.5.3', 'scikit-learn==1.9.1', 'tables==3.11.1',
                    'jsonargparse==4.40.2', 'docstring-parser==0.18.0',
                    'typeshed-client==2.15.0', 'PyYAML==6.0.3')
    .add_local_python_source('src', 'main')
    .add_local_dir('configs/ablations', '/root/configs/ablations'))


@app.function(image=image, gpu='T4', timeout=900, max_containers=1,
              volumes={'/data': data_volume.with_mount_options(read_only=True), '/artifacts': artifacts})
def run_job(ablation: str, seed: int, epochs: int, trials: int):
    import subprocess
    from pathlib import Path
    from uuid import uuid4

    from src.experiment_config import EXPERIMENT_CONFIGS

    if ablation not in EXPERIMENT_CONFIGS:
        raise ValueError('Unknown ablation')
    if not 1 <= epochs <= 150 or not 0 <= trials <= 20:
        raise ValueError('epochs must be 1..150 and trials 0..20')
    root = '/artifacts/' + ablation + '-' + str(seed) + '-' + uuid4().hex
    if trials:
        if EXPERIMENT_CONFIGS[ablation].optimizer['class_path'] != 'torch.optim.AdamW':
            raise ValueError('Tuning requires an AdamW ablation')
        Path(root).mkdir(parents=True, exist_ok=True)
        cmd = ['python', '-m', 'src.tune', '--ablation', ablation, '--seed', str(seed),
               '--epochs', str(epochs), '--trials', str(trials), '--finalize',
               '--storage', 'sqlite:///' + root + '/study.db']
        cmd += ['--data-dir', '/data', '--output-dir', root]
    else:
        cmd = ['python', '-m', 'main', 'fit', '--config', f'/root/configs/ablations/{ablation}.yaml',
               '--seed_everything', str(seed), '--trainer.max_epochs', str(epochs),
               '--data.data_dir', '/data', '--output_dir', root]
    subprocess.run(cmd, check=True)
    metrics = sorted(str(path) for path in Path(root).rglob('test_metrics.csv'))
    if len(metrics) != 1:
        raise RuntimeError(f'Expected one final test metrics CSV under {root}, found {len(metrics)}')
    artifacts.commit()
    return {'run_dir': root, 'test_metrics': metrics[0]}


@app.local_entrypoint()
def main(ablation: str = 'A0', seed: int = 42, epochs: int = 1, trials: int = 0):
    names = tuple(EXPERIMENT_CONFIGS) if ablation == 'all' else (ablation,)
    if trials and ablation == 'all':
        raise ValueError('Use one ablation per study')
    for name in names:
        print(run_job.remote(name, seed, epochs, trials))


@app.function(image=image, gpu='A100-40GB', timeout=86400, max_containers=1,
              volumes={'/data': data_volume.with_mount_options(read_only=True), '/artifacts': artifacts})
def run_a5(finalize: bool = False):
    """Run or resume the fixed A5 study, or retrain its winner separately."""
    import subprocess
    from pathlib import Path

    import torch
    gpu = torch.cuda.get_device_properties(0)
    if 'A100' not in gpu.name or not 35 * 1024**3 <= gpu.total_memory <= 45 * 1024**3:
        raise RuntimeError(f'Expected an A100 40 GB; received {gpu.name} ({gpu.total_memory} bytes)')
    root = Path('/artifacts/a5-optuna-seed42')
    root.mkdir(parents=True, exist_ok=True)
    cmd = ['python', '-m', 'src.tune', '--ablation', 'A5', '--seed', '42',
           '--epochs', '150', '--trials', '10', '--study', 'a5-loss-seed42',
           '--objective', 'val_total_loss', '--storage', f'sqlite:///{root}/study.db',
           '--data-dir', '/data', '--output-dir', str(root), '--target-trials']
    if finalize:
        cmd.append('--finalize-only')
    try:
        subprocess.run(cmd, check=True)
    finally:
        artifacts.commit()
    return str(root)


@app.local_entrypoint(name='a5')
def a5(finalize: bool = False):
    print(run_a5.remote(finalize))


@app.function(image=image, gpu='A100-40GB', timeout=3600, volumes={'/data': data_volume.with_mount_options(read_only=True), '/artifacts': artifacts})
def evaluate_winner_60():
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from src.experiment_config import parse_experiment
    from src.experiment_data import prepare_metr_la, make_loaders
    from src.multitask_stgcn import MultiTaskSTGCN, evaluate_model, original_unit_metrics

    root = Path('/artifacts/a5-optuna-seed42/a5-loss-seed42/winner-4-24d5c70f1bc646db8b24999346efa51a')
    selection = json.loads((root / 'selection.json').read_text())
    threshold = json.loads((root / 'threshold.json').read_text())['threshold']
    prepared = prepare_metr_la('/data')
    config = parse_experiment(selection['config']['experiment'])
    device = torch.device('cuda')
    checkpoint = torch.load(root / 'best.ckpt', map_location=device, weights_only=False)
    model = MultiTaskSTGCN(prepared['edge_index'], prepared['edge_weight'], 12, experiment_config=config).to(device)
    model.load_state_dict({k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
    loader = make_loaders(prepared, 50, 0, True)['test'][0]
    _, test = evaluate_model(model, loader, device, False, checkpoint['state_dict']['pos_weight'], 1., 40.)
    rows = []
    for index, horizon in [(2, 15), (5, 30), (8, 45), (11, 60)]:
        arrays = [x[:, index, :] for x in test]
        rows.append({'Horizon': f'{horizon} min', **original_unit_metrics(*arrays, prepared['mean'], prepared['std'], 40., threshold)})
    previous = pd.read_csv(root / 'test_metrics.csv')
    differences = {key: float(np.max(np.abs(previous[key].to_numpy() - np.array([r[key] for r in rows[:3]])))) for key in rows[0] if key != 'Horizon'}
    result = {'gpu': torch.cuda.get_device_name(0), 'threshold': threshold, 'rows': rows, 'max_absolute_difference_from_existing': differences, 'observed_60': int(test[4][:, 11, :].sum()), 'total_60': int(test[4][:, 11, :].size)}
    path = root / 'test_metrics_60_evaluation.json'
    path.write_text(json.dumps(result, indent=2))
    artifacts.commit()
    return result


@app.local_entrypoint(name='evaluate-winner-60')
def evaluate_winner_60_entry():
    print(evaluate_winner_60.remote())


@app.function(image=image, gpu='A100-40GB', timeout=86400, max_containers=1,
              volumes={'/data': data_volume.with_mount_options(read_only=True), '/artifacts': artifacts})
def finalize_a5_trial10():
    import optuna
    import torch
    from pathlib import Path
    from src.experiment_data import prepare_metr_la
    from src.tune import finalize_winner

    gpu = torch.cuda.get_device_properties(0)
    if 'A100' not in gpu.name or not 35 * 1024**3 <= gpu.total_memory <= 45 * 1024**3:
        raise RuntimeError(f'Expected A100 40 GB; received {gpu.name} ({gpu.total_memory} bytes)')
    print(f'GPU: {gpu.name}, memory bytes: {gpu.total_memory}', flush=True)
    root = Path('/artifacts/a5-optuna-seed42')
    study = optuna.load_study(study_name='a5-loss-seed42', storage=f'sqlite:///{root}/study.db')
    expected = {'ablation': 'A5', 'objective': 'val_total_loss', 'epochs': 150, 'seed': 42,
                'batch_size': 50, 'limit_train_batches': None, 'limit_val_batches': None, 'data_dir': str(Path('/data').resolve())}
    if study.user_attrs.get('contract') != expected or study.direction.name != 'MINIMIZE':
        raise ValueError(f'Unexpected study contract: {study.user_attrs.get("contract")}')
    if any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials):
        raise RuntimeError('Study has running trials; refusing finalization')
    if study.best_trial.number != 10:
        raise RuntimeError(f'Expected best trial 10, got {study.best_trial.number}')
    if list((root / study.study_name).glob('winner-10-*')):
        raise RuntimeError('Trial 10 winner directory already exists; inspect before resuming')
    print(f'Winner: {study.best_trial.number}, value: {study.best_value}, parameters: {study.best_params}', flush=True)
    try:
        output = finalize_winner(study, ablation='A5', objective='val_total_loss', epochs=150,
            seed=42, batch_size=50, output_dir=root, prepared=prepare_metr_la('/data'), include_60=True)
        print(f'Final evaluation: {output}', flush=True)
        return str(output)
    finally:
        artifacts.commit()


@app.local_entrypoint(name='finalize-a5-trial10')
def finalize_a5_trial10_entry():
    print(finalize_a5_trial10.remote())
