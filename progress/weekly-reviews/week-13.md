# Week 13 Reflection Report — End-to-End PyTorch Training, Validation & Diagnostics

## 1. Weekly Overview & Objectives

* **Focus Area:** Tabular Retail Demand Datasets, Tapered MLP Regression Architecture, Mini-Batch Training Loops, Held-Out Evaluation & Checkpointing, Single-Batch Debugging, and Regression Diagnostic Visualization
* **Total Time Spent:** 12.0 hours
* **Status:** Complete

### Key Deliverables Completed
* [x] **Day 85 — Tabular Dataset & DataLoader Pipeline:** Built `RetailDemandDataset(Dataset)` with fail-fast row alignment and `NaN` guards inside `__init__`, enforced `torch.float32` precision and `(N, 1)` target reshaping, and configured `build_dataloaders` (`800` train rows with `shuffle=True`, `200` test rows with `shuffle=False`)
* [x] **Day 86 — Tabular MLP Architecture & Regularization:** Designed `RetailDemandMLP` (`12 -> 64 -> 32 -> 1`, totaling `2,945` trainable parameters) with `nn.ReLU()` hidden activations, `nn.Dropout(0.1)` regularization, and an unbounded linear output head for continuous demand forecasting
* [x] **Day 87 — End-to-End PyTorch Training Loop:** Implemented `train_one_epoch` and `fit_model` executing the five-step mini-batch update cycle with `nn.MSELoss()`, `torch.optim.Adam(lr=1e-2, weight_decay=1e-4)`, and sample-weighted `MSE`, `RMSE`, and `MAE` aggregation
* [x] **Day 88 — Held-Out Validation & Model Checkpointing:** Built `evaluate` and `fit_with_evaluation` pairing `model.eval()` with `torch.no_grad()`, computing `MSE`, `RMSE`, `MAE`, and $R^2$ across held-out data, and conditionally saving best weights (`model.state_dict()`) to `checkpoints/retail_demand_mlp_best.pt`
* [x] **Day 89 — Neural Network Debugging & Sanity Checks:** Implemented `check_tensor_health`, `compute_total_grad_norm`, and `run_single_batch_overfit_test` (`dropout=0.0`), driving single-batch `MSE` from `24889.70` down to `0.2597` (`RMSE: 0.5096 units`) to verify gradient flow and loss wiring
* [x] **Day 90 — Regression Diagnostic Visualization:** Built `collect_predictions`, `plot_learning_curves`, and `plot_prediction_diagnostics` to export training curves (`artifacts/day_90_learning_curves.png`) and predicted-vs-actual/residual plots (`artifacts/day_90_diagnostics.png`) with `plt.close(fig)` memory management

---

## 2. Technical Mastery & Code Benchmarks

### Fail-Fast Data Hygiene & Tabular Architecture Design
* Validated feature/target row counts and `np.isnan()` inside `RetailDemandDataset.__init__` before training begins, while reshaping targets to `(-1, 1)` to eliminate unintended `(B, B)` loss broadcasting bugs.
* Structured `RetailDemandMLP` with compact hidden widths `(64, 32)` and light inverted dropout (`0.1`) to prevent memorization on 12-feature tabular splits, keeping the final `nn.Linear(32, 1)` layer unactivated so continuous predictions remain unbounded without dead-ReLU saturation.

### Sample-Weighted Training, Evaluation & Checkpointing
* Weighted per-batch `MSE` and `MAE` sums by `batch_size` before dividing by `total_samples`, eliminating partial trailing-batch distortion when `drop_last=False`.
* Combined `model.eval()` (disabling `nn.Dropout` randomness) with `with torch.no_grad():` (halting autograd graph allocation) to evaluate held-out generalization (`Val RMSE: 15.35`, `Val MAE: 12.52`, `Val R2: 0.7026` at peak Epoch 15) and persist only improving `model.state_dict()` checkpoints.

### Systematic Debugging & Visual Error Diagnostics
* Established a pre-training sanity protocol using `check_tensor_health` (`torch.isfinite`), global gradient L2 norm tracking, and single-batch memorization (`dropout=0.0`) to isolate model/optimizer bugs from dataset-wide noise in seconds.
* Replaced scalar-only log inspection with visual diagnostics (`RMSE` learning curves, $45^\circ$ Predicted vs. Actual scatter calibration, and zero-centered residual histograms showing `mean = 2.785 units`, `std = 15.118 units`), enforcing `plt.close(fig)` to prevent programmatic memory leaks.

---

## 3. Implementation Synthesis: End-to-End Retail Demand Neural Network Pipeline (Days 85–90)

* **Problem Domain:** Engineering a complete, reproducible PyTorch regression workflow—from raw tabular ingestion through model training, validation checkpointing, sanity debugging, and visual residual analysis—for Nigerian retail demand forecasting.
* **Vectorized Pipeline:**
  1. **Ingestion & Batch Collation (`day_85_dataset.py`):** Loaded 12 standardized continuous and one-hot features across `800` training (`25` batches) and `200` test (`7` batches) samples as `torch.float32` tensors of shape `(32, 12)` and `(32, 1)`.
  2. **Model Definition (`day_86_architecture.py`):** Instantiated `RetailDemandMLP(in_features=12, hidden_dims=(64, 32), dropout=0.1)` with `2,945` trainable parameters and verified `(32, 1)` output projections.
  3. **Training & Validation Loop (`day_87_training.py` & `day_88_validation.py`):** Executed 30 epochs of `Adam` (`lr=1e-2`, `weight_decay=1e-4`) optimization against `nn.MSELoss()`, reducing `Train RMSE` from `151.09` to `22.06` units and `Val RMSE` from `84.88` to `15.37` units (`Val R2: 0.7019`) while saving the best checkpoint.
  4. **Sanity Probes & Visual Artifacts (`day_89_debugging.py` & `day_90_visualization.py`):** Verified single-batch convergence (`RMSE: 0.5096 units`, `Grad Norm: 2.8219`) and exported publication-ready learning curve and residual diagnostic PNG artifacts.

---

## 4. Key Takeaways & Adjustments for Week 14

* **Focus Area:** Convolutional Neural Networks (`CNNs`), Spatial Feature Extraction, Pooling Layers & Image Classification Pipelines in PyTorch
* **Objectives:** Transition from tabular Multi-Layer Perceptrons to spatial computer vision architectures, applying this week's training loop, validation checkpointing, single-batch debugging, and diagnostic plotting patterns to image datasets.
* **Next Steps:** Build and inspect 2D convolution (`nn.Conv2d`) and pooling layers, verify tensor shape transformations across feature maps, and train a baseline CNN classifier in PyTorch.