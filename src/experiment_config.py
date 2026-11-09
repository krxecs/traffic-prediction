"""Load ablations from standard LightningCLI YAML plus graph-specific options."""
import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class GraphConfig:
    graph_mode: str
    use_daily_lag: bool
    use_weekly_lag: bool
    temporal_mode: str
    adaptive_embed_dim: int
    adaptive_top_k: int
    physical_graph_alpha: float
    adaptive_edge_dropout: float

    def encoder_options(self):
        return {key: getattr(self, key) for key in (
            'graph_mode', 'temporal_mode', 'use_daily_lag', 'use_weekly_lag',
            'adaptive_embed_dim', 'adaptive_top_k', 'physical_graph_alpha',
            'adaptive_edge_dropout')}


@dataclass
class AblationSpec:
    name: str
    experiment: GraphConfig
    optimizer: dict
    lr_scheduler: dict
    trainer: dict


def parse_experiment(values):
    """Require complete, typed graph settings with no hidden defaults."""
    if not isinstance(values, dict):
        raise TypeError('experiment must be a YAML mapping')
    expected = {field.name: field.type for field in fields(GraphConfig)}
    if set(values) != set(expected):
        raise ValueError(f'Invalid experiment fields: missing={sorted(set(expected)-set(values))}, '
                         f'extra={sorted(set(values)-set(expected))}')
    checked = {}
    for key, kind in expected.items():
        value = values[key]
        if kind is float and type(value) in (int, float):
            value = float(value)
        elif type(value) is not kind:
            raise ValueError(f'Invalid type for {key}: expected {kind.__name__}')
        if kind is float and not math.isfinite(value):
            raise ValueError(f'Invalid non-finite value for {key}')
        checked[key] = value
    config = GraphConfig(**checked)
    if config.graph_mode not in ('physical', 'adaptive') or config.temporal_mode not in ('single', 'multiscale'):
        raise ValueError('Unknown graph or temporal mode')
    if config.adaptive_embed_dim < 1 or config.adaptive_top_k < 1:
        raise ValueError('Adaptive graph dimensions must be positive')
    if not 0 <= config.adaptive_edge_dropout < 1 or not 0 <= config.physical_graph_alpha <= 1:
        raise ValueError('Graph weights and dropout must be in range')
    return config


ABLATION_DIR = Path(__file__).resolve().parents[1] / 'configs' / 'ablations'


def load_experiment(name):
    path = ABLATION_DIR / f'{name}.yaml'
    if not path.is_file():
        raise ValueError(f'Unknown ablation: {name}')
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {'model', 'experiment', 'trainer', 'optimizer', 'lr_scheduler'}:
        raise ValueError(f'{path}: expected model, experiment, trainer, optimizer and lr_scheduler')
    model = payload['model']
    if not isinstance(model, dict) or set(model) != {'ablation'} or model['ablation'] != name:
        raise ValueError(f'{path}: model.ablation must match the filename')
    graph = parse_experiment(payload['experiment'])
    for key in ('optimizer', 'lr_scheduler'):
        item = payload[key]
        if not isinstance(item, dict) or set(item) != {'class_path', 'init_args'} or not isinstance(item['init_args'], dict):
            raise ValueError(f'{path}: invalid {key} class specification')
    trainer = payload['trainer']
    if not isinstance(trainer, dict) or not isinstance(trainer.get('max_epochs'), int) or trainer['max_epochs'] < 1:
        raise ValueError(f'{path}: trainer.max_epochs must be positive')
    if not isinstance(trainer.get('callbacks'), list) or len(trainer['callbacks']) < 3:
        raise ValueError(f'{path}: expected three checkpoint callbacks')
    return AblationSpec(name, graph, payload['optimizer'], payload['lr_scheduler'], trainer)


EXPERIMENT_CONFIGS = {path.stem: load_experiment(path.stem)
                      for path in sorted(ABLATION_DIR.glob('*.yaml'))}
if not EXPERIMENT_CONFIGS:
    raise RuntimeError(f'No ablation YAML files under {ABLATION_DIR}')
