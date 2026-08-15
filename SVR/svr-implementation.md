# 📈 ML Hackathon — SVR Solution (Final Implementation)

## 🎯 Overview

This repository contains our **Support Vector Regression (SVR)** solution for the ML Hackathon — Fugacity 2026, IIT Kharagpur.

The primary objective is to predict `overall_yield` from continuous flow reactor data containing variables such as flow rate, concentration, temperature, and reactor length.

This document covers the **7-feature experiment approach**, which integrates engineered physical features alongside raw inputs, and details the final notebook workflow.

---

## 📊 Dataset

The training dataset contains **150 observations**, while the test dataset contains **50 hidden observations**.

### Features Used

| Feature | Description |
| :--- | :--- |
| `flow_rate_L_min` | Flow rate in L/min |
| `concentration_mol_L` | Concentration in mol/L |
| `inlet_temperature_K` | Inlet temperature in Kelvin |
| `length_m` | Reactor length in meters |
| `jacket_temperature_K` | Jacket temperature in Kelvin |
| `overall_yield` | **Target variable (%)** |

---

## 🤖 Support Vector Regression (SVR)

SVR is a variant of Support Vector Machines (SVM) used for regression tasks. It relies on finding a margin defined by support vectors, aiming to fit the errors within a threshold ($\epsilon$). Using a **Radial Basis Function (RBF) kernel** allows the model to map the data into a higher-dimensional space and learn complex, non-linear relationships.

### Feature Engineering

Two domain-specific physical features were engineered to provide the model with thermodynamic and kinetic context:

1. **Residence-Time Proxy:**  
   `residence_proxy = length_m / flow_rate_L_min`
2. **Mean Temperature:**  
   `mean_T = (inlet_temperature_K + jacket_temperature_K) / 2`

**Final Model Features (7):**
`flow_rate_L_min`, `concentration_mol_L`, `inlet_temperature_K`, `length_m`, `jacket_temperature_K`, `residence_proxy`, `mean_T`.

*(Note: Experiments limiting to 3 features degraded RMSE to ~27.43, proving the necessity of the raw feature variance for the RBF kernel).*

---

## ⚙️ Pipeline & Tuning

We ran a `GridSearchCV` to optimize the SVR hyperparameters.

### Final Best Configuration

```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=300, gamma=0.1, epsilon=0.5))
])
```

---

## 🧪 Validation & Results

A rigorous **5-fold cross-validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`) was used to score the model against the `overall_yield`.

Because the chemical yield is physically bounded between 0% and 100%, we applied physical clipping:
`y_clipped = np.clip(y, 0, 100)`

### Performance Metrics
- **Raw RMSE:** `23.1076`
- **Clipped RMSE:** `22.0950`
- **Clipping Improvement:** `1.0126`

**Mean CV RMSE:** `22.0950` ± `2.3197`

---

## 🌐 Google Colab & Reproducibility

The final SVR notebook (`SVR/ML_Hackathon_SVR_Final.ipynb`) is completely **Google Colab-ready**. 
It includes a dynamic `RUNNING_IN_COLAB` flag at the top of the notebook that automatically switches data paths, ensuring the workflow runs seamlessly in both local environments and the cloud.

Additionally, a **Visualization Section** has been appended to plot the **Training Fit (True vs Predicted Yield)**, maintaining parity with the XGBoost and GPR workflows for the Phase 2 pitch.

---

## 📝 Conclusion

Although physically valid features (residence proxy, mean temperature) were added and clipping improved the RMSE considerably, the increased dimensionality (7 features vs 5) with a very small observation limit (150 rows) led to slight overfitting when mapped via the RBF kernel. 

The baseline **5-feature SVR model (RMSE 21.70)** performed marginally better by **~0.39 RMSE (1.81%)**.
