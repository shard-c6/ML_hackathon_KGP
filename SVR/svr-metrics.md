# SVR Model Metrics & Output

### 1. Model Evaluation

- **Raw RMSE:** `23.1076`
- **Clipped RMSE:** `22.0950`
- **Clipping improvement:** `1.0126`

**Fold RMSEs after clipping:**
- **Fold 1:** `24.0814`
- **Fold 2:** `19.9064`
- **Fold 3:** `24.9621`
- **Fold 4:** `20.0496`
- **Fold 5:** `21.4757`

- **Mean CV RMSE:** `22.0950`
- **Standard deviation:** `2.3197`

### 2. Final Model Configuration

The final model was trained on the full dataset using the following tuned hyperparameters:

```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=300, gamma=0.1, epsilon=0.5))
])
```

### 3. Sample Predictions

These are the first 5 test cases from the newly generated CSV (`SVR_predictions.csv`), physically constrained between 0% and 100%:

- **Test case 1:** Yield = `42.99%`
- **Test case 2:** Yield = `100.00%`
- **Test case 3:** Yield = `24.91%`
- **Test case 4:** Yield = `100.00%`
- **Test case 5:** Yield = `1.46%`

### 4. Output Data Shape

- **Training features:** `(150, 7)`
- **Testing features:** `(50, 7)`
- **Predictions:** `50`
- **Prediction file:** `SVR_predictions.csv`

### 5. Model Comparison

Comparing this 7-feature model against the previous 5-feature SVR baseline:

- **5-feature RMSE:** `21.7020`
- **7-feature RMSE:** `22.0950`
- **Difference:** `0.3930` RMSE
- **Percentage degradation:** `1.81%`

The 5-feature SVR remains the better configuration, as this 7-feature iteration performed slightly worse in cross-validation due to the challenges of higher dimensionality with a small sample size (150 observations).
