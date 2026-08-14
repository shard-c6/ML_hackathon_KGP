# ML Hackathon — SVR Solution (Current Implementation)

## Overview

This repository contains our Support Vector Regression (SVR) solution for the ML Hackathon — Fugacity 2026, IIT Kharagpur.

The task is to predict `overall_yield` from industrial chemical-process data containing variables such as flow rate, concentration, temperature, and reactor length.

This document covers the current 7-feature experiment approach using physical engineering features alongside raw inputs.

---

## Dataset

The training dataset contains 150 observations.

### Features

| Feature                | Description                  |
| ---------------------- | ---------------------------- |
| `flow_rate_L_min`      | Flow rate in L/min           |
| `concentration_mol_L`  | Concentration in mol/L       |
| `inlet_temperature_K`  | Inlet temperature in Kelvin  |
| `length_m`             | Reactor length in meters     |
| `jacket_temperature_K` | Jacket temperature in Kelvin |
| `overall_yield`        | Target variable (%)          |

The test dataset contains 50 observations without the target variable.

---

## Support Vector Regression (SVR)

SVR is a variant of Support Vector Machines (SVM) used for regression tasks. It relies on finding a margin defined by support vectors, aiming to fit the errors within a threshold (epsilon). Using a Radial Basis Function (RBF) kernel allows the model to learn complex, non-linear relationships.

## Feature Engineering

Two engineered features were retained after experimentation.

### 1. Residence-Time Proxy

A proxy for residence time was created using:

`residence_proxy = length_m / flow_rate_L_min`

### 2. Mean Temperature

The mean temperature feature was calculated as:

`mean_T = (inlet_temperature_K + jacket_temperature_K) / 2`

**Final Model Features (7):**
- flow_rate_L_min
- concentration_mol_L
- inlet_temperature_K
- length_m
- jacket_temperature_K
- residence_proxy
- mean_T

## Hyperparameter Tuning

We ran a GridSearchCV across C, gamma, and epsilon with the RBF kernel. 

### Final Configuration

```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=300, gamma=0.1, epsilon=0.5))
])
```

## Validation

A rigorous 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`) was used to score the model against the `overall_yield`.

Because yield is physically bounded between 0% and 100%, we tested physical clipping:
`y_clipped = np.clip(y, 0, 100)`

### Clipping Result
- **Raw RMSE:** 23.1076
- **Clipped RMSE:** 22.0950
- **Clipping Helped:** True (improved by 1.0126)

### Fold RMSEs (Clipped)
1. 24.0814
2. 19.9064
3. 24.9621
4. 20.0496
5. 21.4757

**Mean CV RMSE:** 22.0950  
**Standard deviation:** 2.3197

---

## Output Shape

The final output is a target prediction array for the 50 hidden test observations, which was also clipped (0 to 100) and saved.

**Sample predictions path:** `SVR_predictions.csv`

## Model Comparison

Comparing this 7-feature model against our earlier 5-feature SVR Baseline (RMSE 21.7020):
- **Difference:** The new model is worse by 0.3930 absolute RMSE (1.81%).

## Conclusion

Although physically valid features (residence proxy, mean temperature) were added and clipping improved the RMSE considerably, the increased dimensionality (7 features vs 5) with a very small observation limit (150 rows) led to slight overfitting when mapped via the RBF kernel. The baseline 5-feature SVR model performed marginally better.
