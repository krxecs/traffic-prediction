"""Joint multi-horizon STGCN data, objectives, training, and smoke checks."""

import copy
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils import data
from tqdm.auto import trange

from stgcn_model import STGCNEncoder


@dataclass
class ExperimentConfig:
    """Small explicit configuration for A0 through A4 ablations."""

    optimizer_mode: str = "legacy"  # legacy | adamw
    graph_mode: str = "physical"  # physical | adaptive
    use_daily_lag: bool = False
    use_weekly_lag: bool = False
    temporal_mode: str = "single"  # single | multiscale
    adaptive_embed_dim: int = 16
    adaptive_top_k: int = 16
    physical_graph_alpha: float = 0.8
    adaptive_edge_dropout: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_learning_rate: float = 1e-5
    max_epochs: int = 150
    early_stop_patience: int = 15
    max_grad_norm: float = 2.0
    checkpoint_metric: str = "val_total_loss"  # val_total_loss | val_mae | val_auprc

    def encoder_options(self):
        return {
            key: getattr(self, key) for key in (
                "graph_mode", "temporal_mode", "use_daily_lag", "use_weekly_lag",
                "adaptive_embed_dim", "adaptive_top_k", "physical_graph_alpha", "adaptive_edge_dropout",
            )
        }


# Keep the frozen multitask baseline independent from the longer modern
# optimizer runs.  A1 retains the 150-epoch budget and AdamW schedule.
A0_CONFIG = ExperimentConfig(
    optimizer_mode="legacy",
    graph_mode="physical",
    use_daily_lag=False,
    use_weekly_lag=False,
    temporal_mode="single",
    max_epochs=50,
)
A1_CONFIG = ExperimentConfig(
    optimizer_mode="adamw",
    graph_mode="physical",
    use_daily_lag=False,
    use_weekly_lag=False,
    temporal_mode="single",
    max_epochs=150,
)


class MultiTaskSTGCN(nn.Module):
    """Shared encoder and existing [B, H, N] regression/classification heads."""

    def __init__(self, edge_index, edge_weight, horizon, num_time_features=4, dropout=0.1,
                 experiment_config=None):
        super().__init__()
        self.horizon = horizon
        self.experiment_config = experiment_config or ExperimentConfig()
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


class WarmupCosineScheduler:
    """Epoch scheduler with a finite positive warm-up/cosine LR sequence."""

    def __init__(self, optimizer, base_lr, min_lr, warmup_epochs, max_epochs):
        self.optimizer, self.base_lr, self.min_lr = optimizer, base_lr, min_lr
        self.warmup_epochs, self.max_epochs, self.epoch = max(0, warmup_epochs), max_epochs, 0
        self._set_lr(base_lr / max(self.warmup_epochs, 1))

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        self.epoch += 1
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs
        elif self.max_epochs <= self.warmup_epochs:
            lr = self.base_lr
        else:
            progress = min(1.0, (self.epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + np.cos(np.pi * progress))
        self._set_lr(float(lr))
        return float(lr)


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


def smoke_test(model, loader, device, pin_memory, pos_weight, lambda_cls, threshold_mph, horizon,
               optimizer_config=None):
    """Checks A0 invariants plus lags, graph gradients, temporal modes, and AdamW."""
    batch = next(iter(loader))
    dataset = loader.dataset
    assert batch["periodic_feat"].shape[1:] == (dataset.history, dataset.normalized.shape[1], 4)
    origins = batch["origin"].numpy()
    recent = origins[:, None] - dataset.history + 1 + np.arange(dataset.history)[None, :]
    assert np.all(recent - dataset.daily_offset_steps < recent)
    assert np.all(recent - dataset.weekly_offset_steps < recent)
    unavailable = recent < dataset.daily_offset_steps
    assert np.all(batch["periodic_feat"].numpy()[unavailable, :, 0:2] == 0)
    periodic = batch["periodic_feat"].numpy()
    for channel, offset in ((0, dataset.daily_offset_steps), (2, dataset.weekly_offset_steps)):
        available = recent >= offset
        for row in range(len(origins)):
            lag_indices = recent[row, available[row]] - offset
            assert np.allclose(periodic[row, available[row], :, channel], np.nan_to_num(dataset.normalized[lag_indices], nan=0.0))
            assert np.array_equal(periodic[row, available[row], :, channel + 1], dataset.observed[lag_indices].astype(np.float32))
    batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
    model.train(); model.zero_grad(set_to_none=True)
    speed_pred, logits = _model_forward(model, batch)
    assert speed_pred.shape == logits.shape == batch["target"].shape == batch["mask"].shape
    assert speed_pred.shape[1] == horizon
    reg = masked_huber(speed_pred, batch["target"], batch["mask"])
    cls = masked_weighted_bce(logits, congestion_labels(batch["raw_target"], threshold_mph), batch["mask"], pos_weight)
    total = reg + lambda_cls * cls
    assert torch.isfinite(total)
    total.backward()
    assert model.reg_head.weight.grad is not None and model.cls_head.weight.grad is not None
    if model.encoder.adaptive_graph is not None:
        graph = model.encoder.adaptive_graph
        assert graph.node_src.grad is not None and graph.node_src.grad.norm() > 0
        assert graph.node_dst.grad is not None and graph.node_dst.grad.norm() > 0
        edge_index, edge_weight = graph.mixed_edges()
        assert edge_index.min() >= 0 and edge_index.max() < graph.num_nodes and torch.isfinite(edge_weight).all()
        assert graph.adaptive_edges()[0].shape[1] <= graph.num_nodes * 2 * graph.top_k
    toy_mask = torch.tensor([[1.0, 0.0]], device=device); toy_target = torch.tensor([[0.0, 0.0]], device=device)
    toy_prediction = torch.tensor([[0.0, 100.0]], device=device); toy_logits = torch.tensor([[0.0, 100.0]], device=device)
    assert torch.equal(congestion_labels(torch.tensor([[39.9, 40.0]], device=device), threshold_mph), torch.tensor([[1.0, 0.0]], device=device))
    assert torch.isclose(masked_huber(toy_prediction, toy_target, toy_mask), torch.tensor(0.0, device=device))
    assert torch.isclose(masked_weighted_bce(toy_logits, torch.tensor([[0.0, 1.0]], device=device), toy_mask, pos_weight), torch.nn.functional.softplus(torch.tensor(0.0, device=device)))
    labels = np.array([0, 1, 1, 0]); probs = np.array([0.1, 0.7, 0.9, 0.2]); frozen_threshold, _ = select_validation_threshold(labels, probs)
    assert np.isfinite(binary_metrics(labels, probs, frozen_threshold)["F1-score"])
    if optimizer_config is not None and optimizer_config.optimizer_mode == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.learning_rate, weight_decay=optimizer_config.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        fresh_pred, fresh_logits = _model_forward(model, batch)
        fresh_total = masked_huber(fresh_pred, batch["target"], batch["mask"]) + lambda_cls * masked_weighted_bce(fresh_logits, congestion_labels(batch["raw_target"], threshold_mph), batch["mask"], pos_weight)
        scaler.scale(fresh_total).backward(); scaler.unscale_(optimizer)
        clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config.max_grad_norm)
        assert torch.isfinite(clipped); scaler.step(optimizer); scaler.update()
        schedule = WarmupCosineScheduler(optimizer, optimizer_config.learning_rate, optimizer_config.min_learning_rate, optimizer_config.warmup_epochs, optimizer_config.max_epochs)
        assert np.isfinite(schedule.step()) and optimizer.param_groups[0]["lr"] > 0
    print("Smoke test passed: shapes, masks, lags, losses, heads, graph, and optimizer checks.")


def train_joint_stgcn(train_loader, val_loader, *, edge_index, edge_weight, num_time_features, horizon,
                      dropout, device, pin_memory, epochs=None, lambda_cls=1.0, threshold_mph=40.0,
                      pos_weight=None, experiment_config=None, train_mean=None, train_std=None):
    config = experiment_config or ExperimentConfig()
    if epochs is not None:
        config = copy.copy(config); config.max_epochs = epochs
    if config.optimizer_mode not in {"legacy", "adamw"}:
        raise ValueError("optimizer_mode must be 'legacy' or 'adamw'.")
    model = MultiTaskSTGCN(edge_index, edge_weight, horizon, num_time_features, dropout, config).to(device)
    if config.optimizer_mode == "legacy":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = WarmupCosineScheduler(optimizer, config.learning_rate, config.min_learning_rate, config.warmup_epochs, config.max_epochs)
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_curve, val_curve, history = [], [], []
    best_state, best_epoch = None, None
    best_by_metric = float("-inf") if config.checkpoint_metric == "val_auprc" else float("inf")
    best_mae_state, best_auprc_state, best_mae, best_auprc, stale_epochs = None, None, float("inf"), float("-inf"), 0
    started_at = time.perf_counter()
    for epoch in trange(1, config.max_epochs + 1, desc="Joint STGCN", unit="epoch"):
        model.train(); train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with context:
                speed_pred, logits = _model_forward(model, batch)
                reg = masked_huber(speed_pred, batch["target"], batch["mask"])
                cls = masked_weighted_bce(logits, congestion_labels(batch["raw_target"], threshold_mph), batch["mask"], pos_weight)
                total = reg + lambda_cls * cls
            if not torch.isfinite(total):
                raise FloatingPointError("Joint STGCN produced a non-finite training loss.")
            scaler.scale(total).backward()
            if config.optimizer_mode == "adamw":
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Joint STGCN produced non-finite gradients.")
            scaler.step(optimizer); scaler.update()
            train_losses.append((reg.item(), cls.item(), total.item()))
        losses, outputs = evaluate_model(model, val_loader, device, pin_memory, pos_weight, lambda_cls, threshold_mph)
        validation = _validation_summary(losses, outputs, threshold_mph, train_mean, train_std)
        train_mean_losses = np.mean(train_losses, axis=0)
        record = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]),
                  "train_regression_loss": float(train_mean_losses[0]), "train_classification_loss": float(train_mean_losses[1]),
                  "train_total_loss": float(train_mean_losses[2]), **validation}
        history.append(record); train_curve.append(record["train_total_loss"]); val_curve.append(record["val_total_loss"])
        if validation["val_mae"] < best_mae:
            best_mae, best_mae_state = validation["val_mae"], copy.deepcopy(model.state_dict())
        if validation["val_auprc"] > best_auprc:
            best_auprc, best_auprc_state = validation["val_auprc"], copy.deepcopy(model.state_dict())
        score, maximize = validation[config.checkpoint_metric], config.checkpoint_metric == "val_auprc"
        improved = score > best_by_metric if maximize else score < best_by_metric
        if improved:
            best_by_metric, best_state, best_epoch, stale_epochs = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale_epochs += 1
        scheduler.step()
        if config.optimizer_mode == "adamw" and stale_epochs >= config.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError("Joint STGCN training produced no validation state.")
    model.load_state_dict(best_state)
    model.training_history = history
    model.training_metadata = {"config": asdict(config), "best_epoch": best_epoch,
                               "best_checkpoint_metric": config.checkpoint_metric, "best_metric": best_by_metric,
                               "best_validation_mae": best_mae, "best_validation_auprc": best_auprc,
                               "best_mae_state": best_mae_state, "best_auprc_state": best_auprc_state,
                               "parameter_count": sum(p.numel() for p in model.parameters()),
                               "training_duration_seconds": time.perf_counter() - started_at}
    return model, train_curve, val_curve
