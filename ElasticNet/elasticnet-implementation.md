# 📈 ML Hackathon — ElasticNet Solution (Baseline Exploration)

## 🎯 Overview

This repository contains an **ElasticNet Regression** exploration for the ML Hackathon — Fugacity 2026, IIT Kharagpur.

The purpose of this model was to test if a highly interpretable, regularized linear approach combined with interaction terms (`PolynomialFeatures`) could outperform the tree-based models (XGBoost) and complex kernels (SVR) on our small 150-sample dataset.

---

## 🤖 ElasticNet & Polynomial Features

Because chemical yields involve kinetics that are rarely perfectly linear (e.g., Temperature × Concentration), we augmented the feature space with a Polynomial generator (`degree=2` and `degree=3`).

This massively expanded the feature count from 7 to up to 120 features. To prevent immediate overfitting, we applied **ElasticNet Regression**.

ElasticNet combines L1 (Lasso) and L2 (Ridge) penalties. This allows it to automatically perform feature selection—shrinking the coefficients of irrelevant or highly collinear interaction terms down to exactly zero, leaving only the most important features.

---

## ⚙️ Pipeline & Tuning

We used a `GridSearchCV` to optimize the polynomial degree and the ElasticNet hyperparameters:

```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('enet', ElasticNet(max_iter=10000, random_state=42))
])
```

**Grid Searched Parameters:**
- `poly__degree`: `[2, 3]`
- `enet__alpha`: `[0.01, 0.1, 1.0, 10.0, 100.0]`
- `enet__l1_ratio`: `[0.001, 0.1, 0.5, 0.9, 1.0]`

### Final Best Configuration
- `poly__degree`: **3**
- `enet__alpha`: **1.0**
- `enet__l1_ratio`: **1.0** (Pure Lasso regression)

*Insight: The model preferred a pure Lasso penalty (`l1_ratio=1.0`), aggressively dropping most of the 120 degree-3 polynomial features to avoid overfitting the 150 training samples.*

---

## 🧪 Validation & Results

A rigorous **5-fold cross-validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`) was used.
Physical clipping (`np.clip(y, 0, 100)`) was applied to the predictions.

### Performance Metrics
- **Raw RMSE:** `26.0761`
- **Clipped RMSE:** `23.0514`
- **Clipping Improvement:** `3.0247`

**Mean CV RMSE:** `23.0514` ± `3.0749`

---

## 📝 Conclusion & Model Comparison

- **XGBoost (Baseline):** `~18.64`
- **SVR (RBF):** `22.09`
- **ElasticNet (Poly 3):** `23.05`

Despite aggressively utilizing Lasso regularization to filter out noise, the linear foundation of ElasticNet—even when augmented with 3rd-degree polynomials—failed to capture the complex underlying physics of the reactor as effectively as XGBoost or SVR. 

The dataset's non-linearities are too intricate for standard interaction terms to map perfectly. Therefore, **ElasticNet will not be used as the final predictive model**, but it successfully served to validate our hypothesis that complex tree-based or RBF-kernel models are mathematically necessary for this specific chemical process.
