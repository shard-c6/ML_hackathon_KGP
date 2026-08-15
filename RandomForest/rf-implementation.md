# 🌲 ML Hackathon — RandomForest Solution

## 🎯 Overview

This repository contains a **RandomForest Regressor** exploration for the ML Hackathon — Fugacity 2026, IIT Kharagpur.

The purpose of this model was to test if a tree-based *bagging* ensemble (Random Forest) could outperform our reigning tree-based *boosting* champion (XGBoost) on this dataset.

---

## 🤖 Random Forest Architecture

Random Forests work by constructing a multitude of decision trees at training time, and outputting the mean prediction of the individual trees. This guards against the overfitting seen in individual decision trees.

Because Random Forests naturally handle non-linear interactions without needing explicit polynomial feature generation, we provided it the same 7 core features we fed to XGBoost and SVR.

---

## ⚙️ Pipeline & Tuning

We used a `GridSearchCV` to optimize the forest's architecture, including tree depth, split criteria, and total estimators.

```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
])
```

**Grid Searched Parameters:**
- `rf__n_estimators`: `[100, 200, 300]`
- `rf__max_depth`: `[None, 5, 10, 15]`
- `rf__min_samples_split`: `[2, 5, 10]`
- `rf__min_samples_leaf`: `[1, 2, 4]`

### Final Best Configuration
- `rf__n_estimators`: **300**
- `rf__max_depth`: **15**
- `rf__min_samples_split`: **2**
- `rf__min_samples_leaf`: **1**

*Insight: The grid search maxed out the tree counts (`300`) and allowed relatively deep, fine-grained splits (`max_depth=15`, `min_samples_leaf=1`). This implies the underlying relationships in the data are intricate and benefit from highly detailed tree constructions.*

---

## 🧪 Validation & Results

A rigorous **5-fold cross-validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`) was used.
Physical clipping (`np.clip(y, 0, 100)`) was applied to the predictions (though Random Forests natively respect the bounds of their training data, making clipping functionally a no-op).

### Performance Metrics
- **Raw RMSE:** `20.0607`
- **Clipped RMSE:** `20.0607`

**Mean CV RMSE:** `20.0607` ± `2.9543`

---

## 📝 Conclusion & Model Comparison

- **XGBoost (Baseline):** `~18.64`
- **Random Forest:** `20.06`
- **GPR (Matern):** `~20.48`
- **SVR (RBF):** `22.09`
- **ElasticNet (Poly 3):** `23.05`

The Random Forest performed exceptionally well, slotting in tightly as our second-best model (beating out GPR, SVR, and ElasticNet).

However, **XGBoost** still reigns supreme. Boosting algorithms (which sequentially correct the errors of previous trees) appear to be slightly more effective at navigating this specific chemical reactor data space than Bagging algorithms (which average parallel trees). 

XGBoost remains our final predictive model.
