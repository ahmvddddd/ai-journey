# Week 4 Reflection Report — Probability, Statistics & Bivariate Analysis

## 1. Weekly Overview & Objectives

* **Focus Area:** Univariate Statistics, Probability Distributions, Bivariate Analysis, Correlation, and Model Validation
* **Total Time Spent:** 15.0 hours
* **Status:** Complete

### Key Deliverables Completed
* [x] Analyzed measures of central tendency (mean, median, mode), quantifying outlier sensitivity and establishing linear shift/scaling properties
* [x] Computed population and sample variance/standard deviation, applying Bessel's correction ($n-1$) and utilizing $Z$-score standardization for gradient descent stabilization
* [x] Evaluated non-parametric dispersion metrics, computing percentiles, quartiles, Interquartile Range ($\text{IQR}$), and Tukey outlier fences to monitor tail-latency ($p95, p99$)
* [x] Differentiated discrete Probability Mass Functions (PMF) from continuous Probability Density Functions (PDF) and analyzed the impact of right-skewed features on linear regression
* [x] Explored bivariate associations, contrasting covariance with Pearson ($r$) and Spearman ($\rho$) correlation coefficients, and analyzed multicollinearity ($r = 1.0$) in OLS matrix inversion
* [x] Completed the **Week 4 Statistics Assessment**, synthesizing conceptual problem-solving with vectorized NumPy/Pandas implementations

---

## 2. Technical Mastery & Code Benchmarks

### Central Tendency & Dispersion Mechanics
* Quantified how extreme outliers pull the arithmetic mean upward while leaving the median stable, making the median essential for right-skewed distributions like e-commerce revenue or server latency.
* Applied Bessel's correction ($n - 1$) in sample variance ($\text{ddof=1}$) to eliminate downward bias in population parameter estimation.
* Leveraged $\text{IQR}$-based Tukey outlier fences ($Q_3 + 1.5 \times \text{IQR}$) for robust anomaly detection immune to extreme numerical spikes.

### Probability Distributions & Bivariate Relationships
* Identified how unscaled right-skewed features create anisotropic loss surfaces during gradient descent and distort ordinary least squares (OLS) regression coefficients.
* Distinguished between Pearson $r$ (measuring linear clustering and vulnerable to leverage outliers) and Spearman $\rho$ (converting raw values to ranks to capture non-linear monotonic trends).
* Examined matrix singularity in $X^T X$ caused by perfect multicollinearity ($r = 1.0$), resolving it via feature dropping or L2 regularization.

---

## 3. Mini-Project / Assessment Implementation: Statistics Mastery

* **Problem Domain:** Evaluated skewed transaction distributions, calculated sample variance with degrees of freedom correction, detected anomalous records using Tukey fences, and generated multi-method correlation matrices.
* **Vectorized Pipeline:**
  1. Extracted descriptive statistics and computed sample variance and standard deviation using Bessel's correction.
  2. Determined quartile boundaries and isolated outliers exceeding upper Tukey fences.
  3. Computed and compared Pearson and Spearman correlation matrices across synthetic features.

---

## 4. Key Takeaways & Adjustments for Week 5

* **Data analysis project:** Build-Nigerian Retail Sales Analyzer
* **Objectives:** (Data ingestion, Cleaning, Exploratory analysis, Statistical analysis, Visualization, Write findings Phase test + rest afterwards)
* **Next Steps:** Build the Analyzer and test it.