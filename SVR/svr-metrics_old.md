# ML Hackathon — Previous SVR Benchmarks

## Overview

This document preserves the older results for the Support Vector Regression (SVR) experiments, including the 5-feature baseline and the 9-feature experimental models, as requested by the team.

---

## 5-Feature Baseline SVR

This benchmark used the original 5 process variables.

### Features
- `flow_rate_L_min`
- `concentration_mol_L`
- `inlet_temperature_K`
- `length_m`
- `jacket_temperature_K`

### Model Parameters
- **Kernel:** RBF (Support Vector Regression)
- **C:** 150
- **epsilon:** 1.0
- **gamma:** "scale"

### Validation Performance
Using a 5-fold KFold strategy:

**Fold RMSE:**
1. 21.0414
2. 21.5916
3. 24.9987
4. 18.5447
5. 22.3336

**Mean CV RMSE:** 21.702026860055522  
**Standard Deviation:** 2.0825054862815184

---

## 9-Feature Experimental SVR

This experiment included additional ratio and difference features to test if they improved performance.

### Additional Features Used
- `length_flow_ratio`
- `temperature_difference`
- `concentration_length`
- `concentration_flow_ratio`

### Validation Performance
**Mean CV RMSE:** 26.116052808378555

*(Note: Adding these 4 ratio features degraded the performance considerably from the 5-feature baseline, increasing error by roughly 4.4 overall.)*
