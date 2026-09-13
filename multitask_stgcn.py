"""Minimal joint multi-horizon congestion experiment for the existing STGCN notebook."""

import copy
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils import data
from tqdm.auto import trange

from stgcn_model import STGCNEncoder


class MultiTaskSTGCN(nn.Module):
    """Shared STGCN encoder with 12-step speed and congestion heads."""

    def __init__(self, edge_index, edge_weight, horizon, num_time_features=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.encoder = STGCNEncoder(edge_index, edge_weight, num_time_features, dropout)
        self.reg_head = nn.Conv2d(128, horizon, kernel_size=(2, 1))
        self.cls_head = nn.Conv2d(128, horizon, kernel_size=(2, 1))

    def forward(self, recent, time_feat):
        shared = self.encoder(recent, time_feat)
        return self.reg_head(shared).squeeze(2), self.cls_head(shared).squeeze(2)


class JointForecastDataset(data.Dataset):
    """One origin predicts its next `horizon` steps. Targets retain raw mph."""

    def __init__(self, normalized, raw_mph, observed, time_features, target_range, history, horizon):
        self.normalized = normalized
        self.raw_mph = raw_mph
        self.observed = observed
        self.time_features = time_features
        self.history = history
        self.horizon = horizon
        origins = []
        for origin in range(target_range.start - 1, target_range.stop - horizon):
            recent = slice(origin - history + 1, origin + 1)
            if recent.start >= 0 and np.isfinite(normalized[recent]).all():
                origins.append(origin)
        if not origins:
            raise ValueError("No valid joint STGCN windows were created.")
        self.origins = np.asarray(origins, dtype=np.int64)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        future = slice(origin + 1, origin + self.horizon + 1)
        recent = slice(origin - self.history + 1, origin + 1)
        return {
            "recent": torch.from_numpy(self.normalized[recent].astype(np.float32)),
            "time_feat": torch.from_numpy(self.time_features[recent].astype(np.float32)),
            "target": torch.from_numpy(np.nan_to_num(self.normalized[future], nan=0.0).astype(np.float32)),
            "raw_target": torch.from_numpy(np.nan_to_num(self.raw_mph[future], nan=0.0).astype(np.float32)),
            "mask": torch.from_numpy(self.observed[future].astype(np.float32)),
        }


def make_joint_loader(split, batch_size, shuffle, *, normalized, raw_mph, observed, time_features,
                      train_end, val_end, history, horizon, pin_memory, num_workers):
    ranges = {"train": range(train_end), "val": range(train_end, val_end), "test": range(val_end, len(normalized))}
    dataset = JointForecastDataset(normalized, raw_mph, observed, time_features, ranges[split], history, horizon)
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
    mask = dataset.observed[future_indices]
    raw = dataset.raw_mph[future_indices]
    valid = mask.astype(bool)
    positive = (raw < threshold_mph) & valid
    n_pos = int(positive.sum())
    n_valid = int(valid.sum())
    n_neg = n_valid - n_pos
    return {"positive": n_pos, "negative": n_neg, "prevalence": n_pos / max(n_valid, 1)}


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
    result = {"Precision": float(precision), "Recall": float(recall), "F1-score": float(f1), "AUPRC": float(average_precision_score(labels, probabilities)), "Accuracy": float(np.mean(labels == prediction))}
    result["AUROC"] = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan
    return result


def original_unit_metrics(prediction, target, raw_target, logits, mask, train_mean, train_std, threshold_mph, threshold):
    prediction_mph = prediction * train_std + train_mean
    valid = mask.astype(bool)
    error = prediction_mph[valid] - raw_target[valid]
    labels = raw_target[valid] < threshold_mph
    probabilities = 1.0 / (1.0 + np.exp(-logits[valid]))
    return {"MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))), "MAPE": float(np.mean(np.abs(error) / raw_target[valid]) * 100.0), **binary_metrics(labels, probabilities, threshold)}


def evaluate_model(model, loader, device, pin_memory, pos_weight, lambda_cls, threshold_mph):
    model.eval()
    totals, predictions, targets, raw_targets, logits, masks = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            speed_pred, congestion_logits = model(batch["recent"], batch["time_feat"])
            reg_loss = masked_huber(speed_pred, batch["target"], batch["mask"])
            cls_loss = masked_weighted_bce(congestion_logits, congestion_labels(batch["raw_target"], threshold_mph), batch["mask"], pos_weight)
            total = reg_loss + lambda_cls * cls_loss
            if not torch.isfinite(total):
                raise FloatingPointError("Joint STGCN produced a non-finite loss.")
            totals.append((reg_loss.item(), cls_loss.item(), total.item()))
            predictions.append(speed_pred.cpu().numpy()); targets.append(batch["target"].cpu().numpy())
            raw_targets.append(batch["raw_target"].cpu().numpy()); logits.append(congestion_logits.cpu().numpy()); masks.append(batch["mask"].cpu().numpy())
    return np.mean(totals, axis=0), tuple(np.concatenate(values) for values in (predictions, targets, raw_targets, logits, masks))


def smoke_test(model, loader, device, pin_memory, pos_weight, lambda_cls, threshold_mph, horizon):
    batch = next(iter(loader))
    batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
    model.train(); model.zero_grad(set_to_none=True)
    speed_pred, logits = model(batch["recent"], batch["time_feat"])
    assert speed_pred.shape == logits.shape == batch["target"].shape == batch["mask"].shape
    assert speed_pred.shape[1] == horizon
    reg = masked_huber(speed_pred, batch["target"], batch["mask"])
    cls = masked_weighted_bce(logits, congestion_labels(batch["raw_target"], threshold_mph), batch["mask"], pos_weight)
    total = reg + lambda_cls * cls
    assert torch.isfinite(total)
    total.backward()
    assert model.reg_head.weight.grad is not None and model.cls_head.weight.grad is not None
    # Invalid cells must not affect either masked objective.
    toy_mask = torch.tensor([[1.0, 0.0]], device=device)
    toy_target = torch.tensor([[0.0, 0.0]], device=device)
    toy_prediction = torch.tensor([[0.0, 100.0]], device=device)
    toy_logits = torch.tensor([[0.0, 100.0]], device=device)
    toy_labels = torch.tensor([[1.0, 0.0]], device=device)
    toy_bce_labels = torch.tensor([[0.0, 1.0]], device=device)
    assert torch.equal(congestion_labels(torch.tensor([[39.9, 40.0]], device=device), threshold_mph), toy_labels)
    assert torch.isclose(masked_huber(toy_prediction, toy_target, toy_mask), torch.tensor(0.0, device=device))
    assert torch.isclose(masked_weighted_bce(toy_logits, toy_bce_labels, toy_mask, pos_weight), torch.nn.functional.softplus(torch.tensor(0.0, device=device)))
    labels = np.array([0, 1, 1, 0]); probs = np.array([0.1, 0.7, 0.9, 0.2])
    frozen_threshold, _ = select_validation_threshold(labels, probs)
    test_metrics = binary_metrics(labels, probs, frozen_threshold)
    assert np.isfinite(test_metrics["F1-score"]) and set(test_metrics) >= {"Precision", "Recall", "AUPRC", "AUROC", "Accuracy"}
    print("Smoke test passed: shapes, masks, losses, both heads, and frozen-threshold metrics.")


def train_joint_stgcn(train_loader, val_loader, *, edge_index, edge_weight, num_time_features, horizon,
                      dropout, device, pin_memory, epochs, lambda_cls, threshold_mph, pos_weight):
    model = MultiTaskSTGCN(edge_index, edge_weight, horizon, num_time_features, dropout).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_curve, val_curve, best_state, best_val = [], [], None, float("inf")
    for _ in trange(1, epochs + 1, desc="Joint STGCN", unit="epoch"):
        model.train(); epoch_losses = []
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                speed_pred, logits = model(batch["recent"], batch["time_feat"])
                reg = masked_huber(speed_pred, batch["target"], batch["mask"])
                cls = masked_weighted_bce(logits, congestion_labels(batch["raw_target"], threshold_mph), batch["mask"], pos_weight)
                total = reg + lambda_cls * cls
            if not torch.isfinite(total): raise FloatingPointError("Joint STGCN produced a non-finite training loss.")
            if device.type == "cuda": scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
            else: total.backward(); optimizer.step()
            epoch_losses.append(total.item())
        val_loss, _ = evaluate_model(model, val_loader, device, pin_memory, pos_weight, lambda_cls, threshold_mph)
        train_curve.append(float(np.mean(epoch_losses))); val_curve.append(float(val_loss[2]))
        if val_curve[-1] < best_val: best_val, best_state = val_curve[-1], copy.deepcopy(model.state_dict())
        scheduler.step()
    model.load_state_dict(best_state)
    return model, train_curve, val_curve
