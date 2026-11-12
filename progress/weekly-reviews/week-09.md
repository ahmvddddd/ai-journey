# Week 9 Reflection Report — Supervised Learning Algorithms & Model Evaluation

## 1. Weekly Overview & Objectives

* **Focus Area:** Linear Regression, Logistic Regression, Decision Trees, Random Forests, Classification Metrics, and Regression Evaluation with K-Fold Cross-Validation
* **Total Time Spent:** 13.0 hours
* **Status:** Complete

### Key Deliverables Completed
* [x] Implemented simple and multiple linear regression models to predict continuous target variables and interpreted feature coefficients ($w$) and intercepts ($b$)
* [x] Built binary classification pipelines using logistic regression, applying the sigmoid activation function ($\sigma(z)$) and log-loss optimization
* [x] Constructed non-parametric decision tree classifiers and tuned structural hyperparameters (`max_depth`, `min_samples_split`) to mitigate overfitting
* [x] Developed ensemble learning workflows using Random Forests (bagging) to reduce model variance and extracted feature importance rankings
* [x] Evaluated classification models beyond raw accuracy using confusion matrices, Precision, Recall, F1-Score, and ROC-AUC curves
* [x] Assessed regression model error using MAE, MSE, RMSE, and $R^2$, implementing K-Fold cross-validation for robust generalization estimates

---

## 2. Technical Mastery & Code Benchmarks

### Linear & Logistic Regression Models
* Modeled linear relationships via $\hat{y} = Xw + b$, optimizing parameters via Ordinary Least Squares (OLS) and gradient-based solvers while inspecting residual distributions.
* Transformed linear combinations into probability estimates ($P(y=1|X) \in [0, 1]$) via the logistic sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, adjusting decision thresholds to balance false positives and false negatives.

### Tree-Based Models & Ensemble Learning
* Trained decision trees using impurity metrics (Gini impurity and Entropy/Information Gain) to establish non-linear, axis-aligned decision boundaries.
* Aggregated multiple decorrelated decision trees via Random Forest ensembles, utilizing bootstrap aggregation (bagging) and random feature subsampling to stabilize predictions and prevent high-variance overfitting.

### Comprehensive Model Evaluation & Cross-Validation
* Diagnosed class imbalance pitfalls by replacing naive accuracy checks with Precision-Recall tradeoffs, F1-scores, and threshold-independent ROC-AUC analysis.
* Quantified continuous prediction errors across multiple scales (MAE for linear interpretability, RMSE for penalizing large outliers, and $R^2$ for variance explained).
* Executed $K$-Fold cross-validation pipelines (`cross_val_score`) to eliminate single-split evaluation bias and verify model stability across $k$ independent data partitions.

---

## 3. Mini-Project / Assessment Implementation: Supervised Model Benchmarking & Validation Suite

* **Problem Domain:** End-to-end comparative evaluation of parametric linear models and non-parametric tree ensembles across regression and classification tasks.
* **Vectorized Pipeline:**
  1. Preprocessed tabular feature matrices ($X$) and target vectors ($y$) using leak-free `Pipeline` transformations (`StandardScaler` paired with estimators).
  2. Trained baseline linear/logistic models alongside tuned Decision Tree and Random Forest estimators.
  3. Generated multi-metric diagnostic reports (Confusion Matrices, Classification Reports, ROC curves, and residual plots).
  4. Validated model generalization across $K=5$ and $K=10$ cross-validation folds to select optimal estimator architectures based on mean validation scores and standard deviation.

---

## 4. Key Takeaways & Adjustments for Week 10

* **Focus Area:** Unsupervised Learning & Dimensionality Reduction (or Advanced Ensembles / Hyperparameter Optimization)
* **Objectives:** Explore clustering algorithms (K-Means, DBSCAN), Principal Component Analysis (PCA), and automated hyperparameter search techniques (`GridSearchCV`, `RandomizedSearchCV`).
* **Next Steps:** Transition from supervised predictive modeling into unlabeled pattern discovery, feature space compression, and systematic pipeline tuning.