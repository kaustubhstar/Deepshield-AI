# Model checkpoints

The API loads whatever architecture `model_metadata.json` names, then checks that the tensors match that network. The filename is not the architecture.

## Installed checkpoint

`deepshield_efficientnet_b0.pt` contains the trained **artifact_cnn** (510,697 parameters, 128×128 input). An older training script always wrote that filename, even when `--architecture artifact_cnn` was used. Do not rename the file and call it EfficientNet-B0.

Metadata records:

- `architecture`: `artifact_cnn`
- `classes`: authentic index 0, manipulated index 1
- `input_size`: 128
- ImageNet mean and std, matching training and inference
- `score_type`: `model_score` (not calibrated)
- validation metrics from the training machine: accuracy 0.673, manipulated recall 0.448, ROC-AUC 0.761

Those validation numbers are not a held-out test result. The 16,000-image manifest points at another computer, so this Mac cannot recompute test metrics.

## New training

`training/train.py` now saves `deepshield_<architecture>.pt`. Set `CHECKPOINT_FILENAME` to that name and restart the API. A run counts only after training finishes and `training/evaluate.py` writes `test_metrics.json`.
