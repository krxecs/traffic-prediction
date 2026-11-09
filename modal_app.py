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
