# Week 2 Reflection Report — Numerical Computing with NumPy

## 1. Weekly Overview & Objectives

* **Focus Area:** High-Performance Vectorization & Linear Algebra Foundations
* **Total Time Spent:** 10.5 hours
* **Status:** Complete

### Key Deliverables Completed
* [x] Mastered Universal Functions (`ufuncs`), memory management using the `out` parameter, and axis-based aggregations[cite: 1]
* [x] Implemented synchronized multi-array sorting using `np.argsort()` and built structured arrays for heterogeneous data[cite: 1]
* [x] Solved square linear systems ($A\vec{x} = \vec{b}$) directly using `np.linalg.solve()` and extracted eigendecompositions (`np.linalg.eig`)[cite: 4]
* [x] Evaluated least-squares solutions (`np.linalg.lstsq()`) for non-square overdetermined matrices[cite: 4]
* [x] Built the **Sensor Telemetry Data Processor** mini-project to clean, filter, aggregate, and normalize multi-channel time-series data without explicit Python loops

---

## 2. Technical Mastery & Code Benchmarks

### Vectorization & Memory Efficiency
* Learned to minimize temporary array allocations by directing ufunc output directly into pre-allocated buffers using the `out` argument.
* Used implicit multidimensional expansion (broadcasting) to scale and normalize feature matrices across spatial (`axis=0`) and temporal (`axis=1`) dimensions without iterative looping.

### Linear Algebra & Stability
* Verified that direct system solving using `np.linalg.solve()` is numerically superior and computationally faster ($\mathcal{O}(n^3)$ via LU decomposition) than explicitly computing matrix inverses ($A^{-1}b$).
* Identified that non-square overdetermined systems ($M > N$) must be solved via least-squares minimization ($\Vert{}Ax - b\Vert{}^2$) using `np.linalg.lstsq()`.

---

## 3. Mini-Project Implementation: Sensor Telemetry Data Processor

* **Problem Domain:** Clean and process noise-corrupted telemetry channels from multi-sensor arrays over continuous time-series steps.
* **Vectorized Pipeline:**
  1. Identified missing markers (`-999.0`) and physical anomalies using compound boolean arrays `(raw > 0) & (raw < 100)`.
  2. Imputed anomalous elements in-place using valid subset medians.
  3. Applied column-wise $Z$-score feature standardization ($Z = \frac{X - \mu}{\sigma}$) leveraging 2D-to-1D broadcasting along `axis=0`.

---

## 4. Key Takeaways & Adjustments for Week 3

* **Axis Orientation:** `axis=n` collapses dimension $n$, reducing the rank of the array while preserving remaining feature axes.
* **Boolean Index Safety:** Always compute summary metrics (like means or medians) on boolean-masked sub-arrays *before* performing in-place replacement to avoid data corruption.
* **Next Steps:** Shift focus to **Pandas & Structured Data Analysis** for tabular data manipulation, missing value handling, and exploratory data analysis (EDA).