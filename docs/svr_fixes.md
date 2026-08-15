# SVR Implementation Fixes

This document outlines the fixes applied to the SVR implementation files (`SVR/run_svr.py` and `SVR/ML_Hackathon_SVR_Final.ipynb`) based on the code review feedback.

## 1. Feature Multicollinearity Fix

**Issue:** The newly engineered features (`residence_proxy` and `mean_T`) were mathematical combinations of four existing features (`length_m`, `flow_rate_L_min`, `inlet_temperature_K`, and `jacket_temperature_K`). Keeping the original features alongside the new ones distorted the Euclidean distance metric used by the RBF kernel in SVR, causing degraded performance (RMSE increased from 21.7 to 22.09).

**Fix:** Dropped the four original redundant features from the training pipeline. The model now only trains on:
- `concentration_mol_L`
- `residence_proxy`
- `mean_T`

*Files modified:*
- `SVR/run_svr.py`
- `SVR/ML_Hackathon_SVR_Final.ipynb`

## 2. Hardcoded Hyperparameters in Notebook

**Issue:** The Jupyter notebook `SVR/ML_Hackathon_SVR_Final.ipynb` claimed to perform `GridSearchCV` hyperparameter tuning but effectively bypassed it by hardcoding a single set of parameters into the `param_grid` (`[300]`, `[0.1]`, and `[0.5]`). Furthermore, the KFold loop was hardcoded to use those parameters rather than the best parameters discovered by the search.

**Fix:** 
- Restored the `param_grid` to use actual arrays for searching:
  - `svr__C`: `[0.1, 1, 10, 50, 100, 150, 200, 300, 500]`
  - `svr__gamma`: `['scale', 'auto', 0.001, 0.01, 0.1, 1.0]`
  - `svr__epsilon`: `[0.01, 0.1, 0.5, 1.0, 2.0]`
- Replaced the hardcoded `.set_params()` in the KFold evaluation loop with `pipeline.set_params(**grid_search.best_params_)` so that it uses the dynamically found optimal parameters.

*Files modified:*
- `SVR/ML_Hackathon_SVR_Final.ipynb`

## 3. Hardcoded Windows Paths (Previously Fixed)

**Issue:** Hardcoded absolute Windows paths in `run_svr.py` caused a `FileNotFoundError` on macOS or Linux environments.
**Fix:** This was proactively fixed in the script by replacing the absolute path with a relative `os.path.dirname` calculation, ensuring universal compatibility.
