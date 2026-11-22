# Week 12 Reflection Report — PyTorch Core Abstractions & Automatic Differentiation

## 1. Weekly Overview & Objectives

* **Focus Area:** PyTorch Tensors, Custom Datasets, Mini-Batch DataLoaders, `nn.Module` Architectures, Dynamic Autograd Computation Graphs, and Reverse-Mode Optimization Mechanics
* **Total Time Spent:** 12.0 hours
* **Status:** Complete

### Key Deliverables Completed
* [x] **Day 78 — PyTorch Tensors & Hardware Interoperability:** Contrasted `torch.Tensor` with `numpy.ndarray`, inspected tensor metadata (`shape`, `dtype`, `device`), executed batch linear algebra (`X @ W + b`) with broadcasting, and standardized safe NumPy conversion via `.detach().cpu().numpy()`
* [x] **Day 79 — PyTorch Datasets & Tabular Data Abstractions:** Decoupled sample indexing from training loops using `TensorDataset` and a custom `TabularDemandDataset(Dataset)` subclass, enforcing `torch.float32` precision and 2D target reshaping (`.reshape(-1, 1)`)
* [x] **Day 80 — DataLoaders & Mini-Batch Iteration:** Configured `DataLoader` pipelines separating randomized training batches (`shuffle=True`) from deterministic evaluation passes (`shuffle=False`), while establishing dynamic batch sizing (`xb.size(0)`) for partial trailing batches
* [x] **Day 81 — Neural Networks (`nn.Module`) & Forward Pass Architecture:** Built `DemandRegressorNet` using `nn.Sequential`, `nn.Linear`, and `nn.ReLU`, audited parameter shapes and `requires_grad` flags via `named_parameters()`, and verified hook execution via `model(x)`
* [x] **Day 82 — Autograd & Dynamic Computation Graphs:** Verified analytical chain-rule derivatives on leaf tensors (`requires_grad=True`) via `.backward()`, diagnosed un-zeroed gradient accumulation (`+=`), and isolated evaluation passes inside `torch.no_grad()`
* [x] **Day 83 — Optimization Mechanics, VJPs & Gradient Hygiene:** Formulated Vector-Jacobian Products (VJPs) to explain scalar loss requirements during reverse-mode autodiff and analyzed how stale gradients corrupt momentum and adaptive optimizer state statistics

---

## 2. Technical Mastery & Code Benchmarks

### Tensor Precision, Device Placement & Data Pipelines
* Enforced explicit `torch.float32` casting and `.to(device)` placement across CPU/CUDA workflows, eliminating runtime dtype mismatches and host-device transfer errors when converting tensors via `.detach().cpu().numpy()`.
* Separated single-record retrieval (`__len__`, `__getitem__`) in `torch.utils.data.Dataset` from mini-batch collation and shuffling in `DataLoader`, preventing off-by-one slicing errors and unweighted loss distortion on remainder batches.

### Modular Network Construction (`nn.Module`)
* Structured feedforward regression layers inside `nn.Module` subclasses (`12 -> 64 -> 32 -> 1`, totaling 2,945 trainable parameters) with `nn.ReLU` hidden activations and an unbounded linear output layer to prevent zero-gradient dead zones on continuous targets.
* Standardized invoking `model(x)` (`Module.__call__`) rather than `model.forward(x)` so registered forward pre-hooks, forward hooks, and backward autograd hooks execute reliably.

### Reverse-Mode Autograd & Gradient Hygiene
* Verified that `.backward()` computes Vector-Jacobian Products ($\mathbf{v}^T \mathbf{J}$) in reverse topological order across `grad_fn` nodes, requiring a scalar objective with an implicit upstream gradient of `1.0`.
* Demonstrated that un-zeroed `.grad` buffers accumulate across steps (`-16.0` to `-32.0`), corrupting gradient descent directions and adaptive moment estimators (Adam, AdamW, SGD with momentum) unless cleared each step, while disabling graph construction during evaluation via `with torch.no_grad():`.

---

## 3. Implementation Synthesis: PyTorch Tabular Regression Pipeline Primitives (Days 78–83)

* **Problem Domain:** Translating first-principles NumPy neural network concepts into modular, hardware-ready PyTorch components tailored for continuous retail demand forecasting.
* **Vectorized Pipeline:**
  1. **Tensor & Device Layer (Day 78):** Validated batch matrix multiplication `(3, 4) @ (4, 2) + (1, 2)` with bias broadcasting and implemented device-agnostic `.to(device)` and `.detach().cpu().numpy()` conversion routines.
  2. **Dataset & DataLoader Ingestion (Days 79–80):** Built `TabularDemandDataset` for `(100, 12)` synthetic retail features and `(100, 1)` targets, wrapping train (`batch_size=16, shuffle=True`, 7 batches) and test (`shuffle=False`, 2 batches) splits into `DataLoader` iterators.
  3. **Feedforward Regressor Architecture (Day 81):** Constructed `DemandRegressorNet(in_features=12, hidden_dim=64)`, verifying `(8, 12)` to `(8, 1)` batch forward passes and auditing all 6 parameter tensors (`2,945` weights and biases).
  4. **Autograd & Gradient Control Diagnostics (Days 82–83):** Benchmarked exact polynomial differentiation ($y = 3w^2 + 2w + 5 \implies \frac{dy}{dw} = 26.0$ at $w=4.0$), verified `w.grad.zero_()` reset mechanics, and confirmed `requires_grad=False` inference outputs under `torch.no_grad()`.

---

## 4. Key Takeaways & Adjustments for Week 13

* **Focus Area:** End-to-End PyTorch Training Loops, Loss Functions (`nn.MSELoss`), Optimizers (`Adam`, `SGD`), Learning Rate Schedulers & Model Checkpointing
* **Objectives:** Assemble `TabularDemandDataset`, `DataLoader`, `DemandRegressorNet`, `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` into a complete multi-epoch training and validation workflow.
* **Next Steps:** Train and evaluate `DemandRegressorNet` on the preprocessed Nigerian retail demand dataset, tracking training vs. validation loss curves and saving model weights via `.state_dict()`.