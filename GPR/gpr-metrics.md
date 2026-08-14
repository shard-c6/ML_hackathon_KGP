# GPR Model Metrics & Output

### 1. Model Evaluation

- **Repeated CV RMSE:** `20.9064 +/- 2.3391`

_(Note: While slightly higher than XGBoost's 18.64, GPR compensates by providing uncertainty bounds, which is crucial for chemical process reliability and phase 2 judging criteria)._

### 2. Final Kernel Optimized by the Model

The optimizer settled on these physical parameters after fitting the training data:

```python
1.05**2 * Matern(length_scale=2.14, nu=1.5) + WhiteKernel(noise_level=0.0505)
```

_This shows the model found a smooth physical trend (Matérn) while accounting for about 5% inherent noise (WhiteKernel) in the dataset._

### 3. Sample Predictions (with Uncertainty)

These are the first 5 test cases from the generated CSV (`GPR_predictions.csv`), including the $\pm$ standard deviation (uncertainty) that GPR provides:

- **Test case 1:** Yield = `54.47% ± 17.06%`
- **Test case 2:** Yield = `100.00% ± 16.76%` _(clipped to physical max)_
- **Test case 3:** Yield = `35.57% ± 27.24%`
- **Test case 4:** Yield = `100.00% ± 26.98%` _(clipped to physical max)_
- **Test case 5:** Yield = `0.00% ± 19.09%` _(clipped to physical min)_

### 4. Output Data Shape

- **Training features shape:** `(150, 7)`
- **Testing features shape:** `(50, 7)`
- **Predictions File:** `../predictions/GPR_predictions.csv` generated successfully with exactly 50 rows, meeting the hackathon submission format guidelines.
