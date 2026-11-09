"""Joint STGCN model, causal windows, losses, and evaluation."""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils import data

from .stgcn_model import STGCNEncoder


class MultiTaskSTGCN(nn.Module):
    """Shared encoder and existing [B, H, N] regression/classification heads."""

    def __init__(self, edge_index, edge_weight, horizon, num_time_features=4, dropout=0.1,
                 experiment_config=None):
        super().__init__()
        self.horizon = horizon
        if experiment_config is None:
            raise ValueError('An ablation experiment config is required')
        self.experiment_config = experiment_config
        self.encoder = STGCNEncoder(edge_index, edge_weight, num_time_features, dropout,
                                    **self.experiment_config.encoder_options())
        self.reg_head = nn.Conv2d(128, horizon, kernel_size=(2, 1))
        self.cls_head = nn.Conv2d(128, horizon, kernel_size=(2, 1))

    def forward(self, recent, time_feat, periodic_feat=None):
        shared = self.encoder(recent, time_feat, periodic_feat)
        return self.reg_head(shared).squeeze(2), self.cls_head(shared).squeeze(2)


class JointForecastDataset(data.Dataset):
    """One origin predicts the next horizon and returns exact causal lag features."""

    daily_offset_steps = 288
    weekly_offset_steps = 2016

    def __init__(self, normalized, raw_mph, observed, time_features, target_range, history, horizon,
                 daily_offset_steps=288, weekly_offset_steps=2016):
        self.normalized, self.raw_mph, self.observed = normalized, raw_mph, observed
        self.time_features, self.history, self.horizon = time_features, history, horizon
        self.daily_offset_steps, self.weekly_offset_steps = daily_offset_steps, weekly_offset_steps
        origins = []
        for origin in range(target_range.start - 1, target_range.stop - horizon):
            recent = slice(origin - history + 1, origin + 1)
            if recent.start >= 0 and np.isfinite(normalized[recent]).all():
                origins.append(origin)
        if not origins:
            raise ValueError("No valid joint STGCN windows were created.")
        self.origins = np.asarray(origins, dtype=np.int64)

    def _periodic_features(self, recent_indices):
        count, nodes = len(recent_indices), self.normalized.shape[1]
        out = np.zeros((count, nodes, 4), dtype=np.float32)
        for channel, offset in ((0, self.daily_offset_steps), (2, self.weekly_offset_steps)):
            lag_indices = recent_indices - offset
            available = lag_indices >= 0
            if available.any():
                values = np.nan_to_num(self.normalized[lag_indices[available]], nan=0.0).astype(np.float32)
                valid = self.observed[lag_indices[available]].astype(np.float32)
                out[available, :, channel] = values
                out[available, :, channel + 1] = valid
        return out

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        future = slice(origin + 1, origin + self.horizon + 1)
        recent = slice(origin - self.history + 1, origin + 1)
        recent_indices = np.arange(recent.start, recent.stop, dtype=np.int64)
        return {
            "recent": torch.from_numpy(self.normalized[recent].astype(np.float32)),
            "time_feat": torch.from_numpy(self.time_features[recent].astype(np.float32)),
            "periodic_feat": torch.from_numpy(self._periodic_features(recent_indices)),
            "target": torch.from_numpy(np.nan_to_num(self.normalized[future], nan=0.0).astype(np.float32)),
            "raw_target": torch.from_numpy(np.nan_to_num(self.raw_mph[future], nan=0.0).astype(np.float32)),
            "mask": torch.from_numpy(self.observed[future].astype(np.float32)),
            "origin": torch.tensor(origin, dtype=torch.long),
        }


def make_joint_loader(split, batch_size, shuffle, *, normalized, raw_mph, observed, time_features,
                      train_end, val_end, history, horizon, pin_memory, num_workers,
                      daily_offset_steps=288, weekly_offset_steps=2016):
    ranges = {"train": range(train_end), "val": range(train_end, val_end), "test": range(val_end, len(normalized))}
    dataset = JointForecastDataset(normalized, raw_mph, observed, time_features, ranges[split], history, horizon,
                                   daily_offset_steps, weekly_offset_steps)
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                  persistent_workers=num_workers > 0, pin_memory=pin_memory)
    try:
        loader = data.DataLoader(dataset, **kwargs)
    except (OSError, RuntimeError):
        kwargs.update(num_workers=0, persistent_workers=False)
        loader = data.DataLoader(dataset, **kwargs)
    return loader, dataset


def masked_huber(prediction, target, mask, beta=1.0):
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_weighted_bce(logits, target, mask, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    return (criterion(logits, target) * mask).sum() / mask.sum().clamp_min(1.0)


def congestion_labels(raw_target, threshold_mph):
    return (raw_target < threshold_mph).to(dtype=torch.float32)


def class_support(dataset, threshold_mph):
    future_indices = dataset.origins[:, None] + np.arange(1, dataset.horizon + 1)[None, :]
    mask, raw = dataset.observed[future_indices], dataset.raw_mph[future_indices]
    valid = mask.astype(bool)
    n_pos = int(((raw < threshold_mph) & valid).sum())
    n_valid = int(valid.sum())
    return {"positive": n_pos, "negative": n_valid - n_pos, "prevalence": n_pos / max(n_valid, 1)}


def select_validation_threshold(labels, probabilities):
    """Fit the operating threshold on validation predictions only."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        _, _, f1, _ = precision_recall_fscore_support(labels, probabilities >= threshold, average="binary", zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold, best_f1


def binary_metrics(labels, probabilities, threshold):
    prediction = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
    result = {"Precision": float(precision), "Recall": float(recall), "F1-score": float(f1),
              "AUPRC": float(average_precision_score(labels, probabilities)), "Accuracy": float(np.mean(labels == prediction))}
    result["AUROC"] = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan
    return result


def original_unit_metrics(prediction, target, raw_target, logits, mask, train_mean, train_std, threshold_mph, threshold):
    prediction_mph = prediction * train_std + train_mean
    valid = mask.astype(bool)
    error = prediction_mph[valid] - raw_target[valid]
    labels = raw_target[valid] < threshold_mph
    probabilities = 1.0 / (1.0 + np.exp(-logits[valid]))
    return {"MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAPE": float(np.mean(np.abs(error) / raw_target[valid]) * 100.0),
            **binary_metrics(labels, probabilities, threshold)}


def _model_forward(model, batch):
    return model(batch["recent"], batch["time_feat"], batch.get("periodic_feat"))


def evaluate_model(model, loader, device, pin_memory, pos_weight, lambda_cls, threshold_mph):
    model.eval()
    totals, predictions, targets, raw_targets, logits, masks = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            speed_pred, congestion_logits = _model_forward(model, batch)
            reg_loss = masked_huber(speed_pred, batch["target"], batch["mask"])
            cls_loss = masked_weighted_bce(congestion_logits, congestion_labels(batch["raw_target"], threshold_mph), batch["mask"], pos_weight)
            total = reg_loss + lambda_cls * cls_loss
            if not torch.isfinite(total):
                raise FloatingPointError("Joint STGCN produced a non-finite loss.")
            totals.append((reg_loss.item(), cls_loss.item(), total.item()))
            predictions.append(speed_pred.cpu().numpy()); targets.append(batch["target"].cpu().numpy())
            raw_targets.append(batch["raw_target"].cpu().numpy()); logits.append(congestion_logits.cpu().numpy()); masks.append(batch["mask"].cpu().numpy())
    return np.mean(totals, axis=0), tuple(np.concatenate(values) for values in (predictions, targets, raw_targets, logits, masks))


def _validation_summary(losses, outputs, threshold_mph, train_mean=None, train_std=None):
    prediction, target, raw, logits, mask = outputs
    valid = mask.astype(bool)
    labels, probabilities = raw[valid] < threshold_mph, 1.0 / (1.0 + np.exp(-logits[valid]))
    auprc = float(average_precision_score(labels, probabilities)) if labels.size else np.nan
    mae = float(np.mean(np.abs(prediction[valid] - target[valid]))) if valid.any() else np.nan
    if train_mean is not None and train_std is not None and valid.any():
        mae = float(np.mean(np.abs((prediction[valid] * train_std + train_mean) - raw[valid])))
    return {"val_regression_loss": float(losses[0]), "val_classification_loss": float(losses[1]),
            "val_total_loss": float(losses[2]), "val_mae": mae, "val_auprc": auprc}
