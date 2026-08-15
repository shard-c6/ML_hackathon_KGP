# RandomForest Model Metrics & Output

### 1. Model Evaluation

- **Raw RMSE:** `20.0607`
- **Clipped RMSE:** `20.0607`
- **Clipping improvement:** `0.0000` (Random Forests natively do not extrapolate beyond training targets [0-100]).

**Fold RMSEs after clipping:**
- **Fold 1:** `16.8499`
- **Fold 2:** `21.4423`
- **Fold 3:** `23.1754`
- **Fold 4:** `16.9516`
- **Fold 5:** `21.8843`

- **Mean CV RMSE:** `20.0607`
- **Standard deviation:** `2.9543`

### 2. Final Model Configuration

The final model was trained on the full dataset using the following tuned hyperparameters:

```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42, 
        n_jobs=-1
    ))
])
```

### 3. Output Data Shape

- **Training features:** `(150, 7)`
- **Testing features:** `(50, 7)`
- **Predictions:** `50`
- **Prediction file:** `RandomForest_predictions.csv`

### 4. Model Comparison

Comparing this RandomForest baseline against our other models:

- **XGBoost (Baseline):** `~18.64` RMSE
- **RandomForest:** `20.06` RMSE
- **GPR (Matern):** `~20.48` RMSE
- **SVR (RBF):** `~22.09` RMSE
- **ElasticNet (Poly 3):** `23.05` RMSE

RandomForest secures the **2nd place** spot in our model lineup. It significantly outperformed standard kernels (SVR) and regularized linear polynomials (ElasticNet), proving that tree-based ensembles are highly suited for this dataset's complex physics. However, XGBoost's sequential boosting approach still edges out Random Forest's bagging approach.
