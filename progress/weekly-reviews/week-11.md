# Week 11 Reflection Report — Deep Learning & Neural Network Foundations

## 1. Weekly Overview & Objectives

* **Focus Area:** Artificial Neurons, Neural Network Layers, Activation Functions, Loss Functions, Gradient Descent Optimization, and Backpropagation Calculus
* **Total Time Spent:** 12.0 hours
* **Status:** Complete

### Key Deliverables Completed
* [x] **Day 71 — Artificial Neurons (The Perceptron):** Mapped biological neural structures to mathematical abstractions ($z = \mathbf{w}^T \mathbf{x} + b$), implemented a vectorized single neuron with a Heaviside step activation in NumPy, and analyzed the geometric roles of weights vs. biases
* [x] **Day 72 — Neural Network Layers & Dense Architecture:** Scaled single-neuron operations into multi-neuron dense layers via batch matrix multiplication ($\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$), verifying tensor dimensionality alignment and ReLU forward propagation
* [x] **Day 73 — Activation Functions & Non-Linearity:** Proved mathematically why stacked linear layers collapse into a single affine transformation, implemented Sigmoid, Tanh, and ReLU from scratch, and evaluated zero-centering and vanishing gradient dynamics
* [x] **Day 74 — Loss Functions & Optimization Objectives:** Contrasted continuous surrogate loss functions against discrete evaluation metrics, implementing Mean Squared Error (MSE) for regression and numerically stable Binary Cross-Entropy (BCE) with epsilon clipping
* [x] **Day 75 — Gradient Descent & Optimization Mechanics:** Formulated the iterative parameter update rule ($w_{\text{new}} = w_{\text{old}} - \alpha \frac{\partial L}{\partial w}$), diagnosed learning rate ($\alpha$) failure modes, and executed a quadratic optimization loop in NumPy
* [x] **Day 76 — Backpropagation & Chain Rule Calculus:** Derived multivariate chain rule gradient propagation across nested functions and built a complete forward-and-backward training loop from scratch in NumPy

---

## 2. Technical Mastery & Code Benchmarks

### Vectorized Forward Propagation & Non-Linearity
* Replaced iterative scalar loops with hardware-aligned NumPy dot products and matrix multiplications (`np.dot`), leveraging contiguous memory layouts and SIMD/BLAS execution efficiency.
* Demonstrated that non-linear activation functions ($\text{ReLU}(z) = \max(0, z)$, $\tanh(z)$, $\sigma(z)$) prevent associative matrix collapse across deep layers, preserving universal approximation capacity and mitigating vanishing gradients.

### Differentiable Objectives & Numerical Stability
* Formulated regression (MSE, MAE) and probabilistic classification (BCE) cost objectives, verifying why smooth, continuous derivatives are mandatory for gradient-based optimization compared to step-based accuracy metrics.
* Enforced floating-point stability in logarithmic cross-entropy calculations by bounding predicted probabilities via `np.clip(y_pred, 1e-15, 1 - 1e-15)` to eliminate $\log(0)$ singularities.

### Gradient Descent & Chain Rule Backpropagation
* Evaluated convergence trade-offs across Batch, Stochastic (SGD), and Mini-Batch ($B \in [32, 512]$) gradient descent paradigms, balancing GPU tensor core saturation against stochastic gradient variance.
* Implemented exact analytical backpropagation from output loss back to input weights ($\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$), confirming dynamic programming efficiency over numerical finite-difference approximations.

---

## 3. Implementation Synthesis: NumPy Deep Learning Primitives Suite (Days 71–76)

* **Problem Domain:** First-principles construction of core neural network computational building blocks in pure NumPy prior to transitioning into high-level autograd frameworks.
* **Vectorized Pipeline:**
  1. **Single-Neuron & Dense Layer Forward Passes (Days 71–72):** Built modular forward-pass functions mapping $(N \times D)$ input batches through $(D \times M)$ weight matrices and $(1 \times M)$ bias vectors with ReLU thresholding.
  2. **Activation & Loss Modules (Days 73–74):** Implemented vectorized Sigmoid, Tanh, ReLU, MSE, and clipped BCE functions, validating output ranges and numerical edge cases across sample logits and probability arrays.
  3. **Gradient Descent Optimization Loop (Day 75):** Executed a 15-epoch analytical gradient descent solver minimizing $L(w) = (w - 3.0)^2$ from $w = 0.0$ ($\alpha = 0.1$), converging smoothly to $w = 2.8944$.
  4. **End-to-End Backpropagation Cycle (Day 76):** Integrated forward pass caching, chain-rule partial derivative computation, and simultaneous weight/bias updates across 5 epochs ($x = 2.0, y_{\text{true}} = 10.0$), reducing MSE loss from $42.25$ to $0.17$.

---

## 4. Key Takeaways & Adjustments for Week 12

* **Focus Area:** Multi-Layer Perceptron (MLP) Assembly, Regularization, Optimizers & PyTorch Transition
* **Objectives:** Combine this week's NumPy primitives into a complete multi-layer neural network trained on non-linear tabular data before transitioning into PyTorch tensors, `autograd`, and `nn.Module` workflows.
* **Next Steps:** Implement a full multi-layer backpropagation network in NumPy to solve non-linear classification boundaries, then benchmark equivalent architectures in PyTorch.