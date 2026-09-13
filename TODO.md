# Traffic prediction experiment TODO

## Goal 0 — Preserve baseline invariants

- [x] Preserve METR-LA chronological 70/10/20 splits.
- [x] Preserve leakage-safe preprocessing, causal history filling, and training-only normalization.
- [x] Preserve 12-step / 60-minute input history and the raw-speed congestion definition: `< 40 mph`.
- [x] Keep the existing STGCN encoder and current optimizer/scheduler unless an interface change requires otherwise.
- [x] Retain MAE, RMSE, and MAPE reporting.

## Goal 1 — Classification-aware evaluation

- [x] Report positive/negative support and prevalence separately for train, validation, and test.
- [x] Retain MAE, RMSE, and MAPE.
- [x] Report precision, recall, F1, AUPRC, AUROC, and accuracy.
- [x] Keep test information out of training, class weighting, threshold selection, and model selection.

## Goal 2 — Joint 12-step dataset/model

- [x] Return next-12 targets, observation masks, and raw-mph targets from each dataset example.
- [x] Use inspectable `[B, H, N]` target, mask, prediction, and logits tensors with `H = 12`.
- [x] Replace separate horizon models with one joint model and evaluate horizon indices 2, 5, and 11.

## Goal 3 — Dedicated congestion head

- [x] Reuse the shared STGCN encoder with only regression and congestion output heads.
- [x] Return `speed_pred, congestion_logits`.

## Goal 4 — Minimal multitask loss

- [x] Use masked Huber regression loss on observed targets.
- [x] Construct labels from raw future speed `< 40.0 mph`.
- [x] Use masked weighted `BCEWithLogitsLoss`, with training-only `pos_weight`.
- [x] Expose one `lambda_cls` configuration value.

## Goal 5 — Validation-only threshold protocol

- [x] Select F1 threshold from validation probabilities only.
- [x] Freeze that threshold for test evaluation.

## Goal 6 — Verification

- [x] Statically verify notebook JSON and Python syntax, tensor-shape conventions, horizon mapping, and threshold flow.
- [x] Run the runtime smoke test with Pixi on a real METR-LA batch: shapes, raw labels, masking, finite losses, gradients to both heads, and frozen-threshold evaluation.
- [x] Verify 15/30/60-minute metric output with the frozen validation threshold.

## Deferred / Not Now

- [ ] Focal loss and regression/classification consistency loss.
- [ ] Optimizer/scheduler changes, early-stopping redesign, adaptive/dynamic graphs, lag features, multi-scale convolutions, attention/Transformers, self-supervision, calibration, ensembling, PEMS-BAY, sweeps, and multi-seed studies.
