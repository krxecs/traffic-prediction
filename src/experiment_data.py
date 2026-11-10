"""Reusable METR-LA preparation for the notebook's joint experiment.

All statistics and labels follow notebook cells 5, 9 and 13. No future
observation is used to fill an earlier missing speed.
"""

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .experiment_config import EXPERIMENT_CONFIGS
from .multitask_stgcn import class_support, make_joint_loader


def sensor_key(value):
    return str(value.decode() if isinstance(value, bytes) else value)


def read_metr_la_hdf(path):
    try:
        return pd.read_hdf(path)
    except TypeError as exc:
        try:
            import tables
            with tables.open_file(path) as handle:
                values = handle.get_node('/df/block0_values').read()
                columns = handle.get_node('/df/block0_items').read()
                timestamps = handle.get_node('/df/axis1').read()
            return pd.DataFrame(values, index=pd.to_datetime(timestamps),
                                columns=[sensor_key(column) for column in columns])
        except Exception as fallback_exc:  # noqa: BLE001 - HDF backends raise different exception types.
            raise RuntimeError(f"Unable to read METR-LA HDF5 at {path}: {fallback_exc}") from exc


def load_adjacency(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle, encoding='latin1')
    if isinstance(obj, (tuple, list)) and len(obj) >= 3:
        ids, mapping, adjacency = obj[:3]
    elif isinstance(obj, dict):
        ids = obj.get('sensor_ids') or obj.get('sensors')
        mapping = obj.get('sensor_id_to_ind') or obj.get('sensor_id_to_index')
        adjacency = obj.get('adj_mx') if 'adj_mx' in obj else obj.get('adjacency')
    else:
        raise ValueError('Unsupported METR-LA adjacency format')
    if ids is None or mapping is None or adjacency is None:
        raise ValueError('Adjacency must contain sensor IDs, mapping and matrix')
    return list(ids), mapping, np.asarray(adjacency, dtype=np.float32)


def causal_forward_fill(values):
    filled = values.copy()
    for t in range(1, len(filled)):
        missing = ~np.isfinite(filled[t])
        filled[t, missing] = filled[t - 1, missing]
    return filled


def pyg_graph_from_adjacency(adjacency):
    graph = (np.maximum(adjacency, adjacency.T)
             if not np.allclose(adjacency, adjacency.T, atol=1e-6)
             else adjacency.copy())
    np.fill_diagonal(graph, 0.0)
    rows, columns = np.nonzero(graph)
    return (torch.from_numpy(np.stack((rows, columns))).long(),
            torch.from_numpy(graph[rows, columns]).float())


def prepare_metr_la(data_dir):
    data_dir = Path(data_dir)
    frame = read_metr_la_hdf(data_dir / 'METR-LA.h5')
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(frame.index)
    frame = frame.sort_index()
    minute = (frame.index.hour * 60 + frame.index.minute).to_numpy(dtype=np.float32)
    day = frame.index.dayofweek.to_numpy(dtype=np.float32)
    time_features = np.column_stack((np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440),
                                     np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7))).astype(np.float32)
    ids, mapping, adjacency = load_adjacency(data_dir / 'adj_METR-LA.pkl')
    if frame.shape[1] != 207 or len(ids) != 207 or adjacency.shape != (207, 207):
        raise ValueError('Expected 207 METR-LA sensors and a 207 by 207 adjacency')
    mapping = {sensor_key(key): int(value) for key, value in mapping.items()}
    if set(mapping) != {sensor_key(item) for item in ids} or sorted(mapping.values()) != list(range(207)):
        raise ValueError('Invalid sensor-to-adjacency mapping')
    ordered_ids = [key for key, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    columns = {sensor_key(column): column for column in frame.columns}
    if any(key not in columns for key in ordered_ids):
        raise ValueError('A graph sensor is absent from the HDF5 columns')
    frame = frame.loc[:, [columns[key] for key in ordered_ids]].astype(np.float32)
    if not np.isfinite(adjacency).all() or not np.isfinite(time_features).all():
        raise ValueError('Non-finite adjacency or time features')
    raw = frame.to_numpy(dtype=np.float32).copy()
    observed = np.isfinite(raw) & (raw > 0.0)
    raw[~observed] = np.nan
    train_end, val_end = int(.7 * len(raw)), int(.8 * len(raw))
    if not 0 < train_end < val_end < len(raw):
        raise ValueError('Invalid chronological split')
    filled = causal_forward_fill(raw)
    values = filled[:train_end][observed[:train_end]]
    mean, std = float(np.nanmean(values)), float(np.nanstd(values))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid training normalization')
    edge_index, edge_weight = pyg_graph_from_adjacency(adjacency)
    return {"sensor_ids": ordered_ids, "timestamps": frame.index.to_numpy(dtype='datetime64[ns]'),
                "normalized": (filled-mean)/std, "raw_mph": raw,
                "observed": observed, "time_features": time_features, "train_end": train_end,
                "val_end": val_end, "mean": mean, "std": std, "edge_index": edge_index,
                "edge_weight": edge_weight}


def make_loaders(prepared, batch_size=50, num_workers=0, pin_memory=False):
    args = {key: prepared[key] for key in ('normalized', 'raw_mph', 'observed', 'time_features', 'train_end', 'val_end')}
    args.update(history=12, horizon=12, pin_memory=pin_memory, num_workers=num_workers,
                daily_offset_steps=288, weekly_offset_steps=2016)
    return {split: make_joint_loader(split, batch_size, split == 'train', **args)
            for split in ('train', 'val', 'test')}


def main():
    parser = argparse.ArgumentParser(description='Report the frozen notebook data contract')
    parser.add_argument('--data-dir', default='datasets/metr-la')
    args = parser.parse_args()
    prepared = prepare_metr_la(args.data_dir)
    loaders = make_loaders(prepared)
    def digest(tensor):
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    print(json.dumps({"sensor_ids": prepared['sensor_ids'],
        "split": [prepared['train_end'], prepared['val_end'], len(prepared['raw_mph'])],
        "windows": {k: len(v[1]) for k, v in loaders.items()},
        "support": {k: class_support(v[1], 40.0) for k, v in loaders.items()},
        "mean": prepared['mean'], "std": prepared['std'],
        "edge_count": prepared['edge_index'].shape[1],
        "edge_index_sha256": digest(prepared['edge_index']), "edge_weight_sha256": digest(prepared['edge_weight']),
        "configurations": {k: asdict(v) for k, v in EXPERIMENT_CONFIGS.items()}}, indent=2))


if __name__ == '__main__':
    main()
