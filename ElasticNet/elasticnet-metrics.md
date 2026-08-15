# ElasticNet Model Metrics & Output

### 1. Model Evaluation

- **Raw RMSE:** `26.0761`
- **Clipped RMSE:** `23.0514`
- **Clipping improvement:** `3.0247`

**Fold RMSEs after clipping:**
- **Fold 1:** `22.9337`
- **Fold 2:** `21.9368`
- **Fold 3:** `24.3780`
- **Fold 4:** `18.8330`
- **Fold 5:** `27.1755`

- **Mean CV RMSE:** `23.0514`
- **Standard deviation:** `3.0749`

### 2. Final Model Configuration

The final model was trained on the full dataset using the following tuned hyperparameters:

```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('enet', ElasticNet(alpha=1.0, l1_ratio=1.0, max_iter=10000, random_state=42))
])
```
*Note: `l1_ratio=1.0` means the model aggressively used pure Lasso regression to zero-out redundant polynomial features.*

### 3. Sample Predictions

These are the first 5 test cases from the newly generated CSV (`ElasticNet_predictions.csv`), physically constrained between 0% and 100%:

- **Test case 1:** Yield = `33.56%`
- **Test case 2:** Yield = `79.85%`
- **Test case 3:** Yield = `50.38%`
- **Test case 4:** Yield = `100.00%`
- **Test case 5:** Yield = `0.00%`

### 4. Output Data Shape

- **Training features (before poly):** `(150, 7)`
- **Testing features (before poly):** `(50, 7)`
- **Predictions:** `50`
- **Prediction file:** `ElasticNet_predictions.csv`

### 5. Model Comparison

Comparing this 3rd-degree Polynomial ElasticNet baseline against our other models:

- **XGBoost (Baseline):** `~18.64` RMSE
- **GPR (Matern):** `~20.48` RMSE
- **SVR (RBF):** `~22.09` RMSE
- **ElasticNet (Poly 3):** `23.05` RMSE

ElasticNet performed the worst of the models we tested. Despite generating up to 120 features with 3rd-degree polynomials and using Lasso to filter out noise, it failed to approximate the complex, continuous non-linearities of the chemical reactor physics as well as the tree-based or RBF-kernel approaches.
