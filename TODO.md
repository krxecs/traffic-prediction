# Traffic prediction experiment TODO

## Preserved baseline protocol

- [x] Keep METR-LA sensor order, causal preprocessing, training-only normalization, and chronological 70/10/20 splits.
- [x] Keep 12 recent steps, joint next-12 targets, `[B, H, N]` predictions/logits, raw speed `< 40 mph` labels, masked Huber and training-only class-weighted BCE.
- [x] Keep validation-only threshold fitting and isolated 15/30/60-minute test reporting.

## Implemented ablation controls

- [x] A0 uses `optimizer_mode="legacy"`, `graph_mode="physical"`, no periodic lags, and `temporal_mode="single"`.
- [x] A1 selects `optimizer_mode="adamw"`: AdamW (`3e-4`, `1e-4` decay), five-epoch warm-up, cosine decay to `1e-5`, AMP-safe norm clipping at `2.0`, and 15-epoch validation early stopping. The trainer records LR, both task losses, total loss, validation MAE, validation AUPRC, parameter count, best epoch, and duration. `checkpoint_metric` explicitly controls main checkpoint selection while MAE and AUPRC states are retained.
- [x] A2 selects `graph_mode="adaptive"`. Trainable `[N, 16]` source/destination embeddings produce non-self top-16 sparse edges. The model symmetrizes them and mixes duplicate edges as `0.8 * A_physical + 0.2 * A_adaptive`. It recomputes the normalized Chebyshev operator in forward, so graph weights remain differentiable. `adaptive_embed_dim`, `adaptive_top_k`, `physical_graph_alpha`, and `adaptive_edge_dropout` control this path.
- [x] A3a enables `use_daily_lag`; A3b also enables `use_weekly_lag`. `JointForecastDataset` returns `[B, T, N, 4]`: normalized daily speed, daily observed mask, normalized weekly speed, weekly observed mask. The offsets are exactly 288 and 2016 steps. Missing history uses zero value and zero mask.
- [x] A4 selects `temporal_mode="multiscale"`. The two STConv blocks use causal GLU branches with `(kernel, dilation)` values `(3,1)`, `(3,2)`, and `(3,4)`, left-padded so every branch returns `T - 2`; a 1x1 fusion follows. The final output temporal layer remains the A0 layer.
- [x] The Colab badge and clone command explicitly target `new-refactor` and reject an existing checkout on a different branch.

## Verification

- [x] Static syntax and notebook JSON validation are part of the implementation check.
- [x] Smoke checks cover both heads, masks, frozen threshold metrics, lag shapes and offsets, graph indices/top-k/finite weights, adaptive embedding gradients, and AdamW clipping/scheduler behavior.
- [ ] Train A1, A2, A3a, A3b, A4, and component-removal ablations to report empirical metrics. Do not infer improvements from smoke tests.

## Future work, not implemented

- [ ] Dynamic or input-conditioned adjacency, attention/Transformers, ST-SSL, STEP pretraining, focal loss, regression/classification consistency loss, calibration, ensembling, PEMS-BAY, hyperparameter sweeps, and multi-seed studies.
