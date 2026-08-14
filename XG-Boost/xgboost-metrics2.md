# XGBoost Model Metrics & Output

### 1. Model Evaluation
*   **Repeated CV RMSE:** `19.0110 +/- 3.2937`

*(Note: XGBoost achieves a slightly lower cross-validation error compared to GPR's ~20.91, demonstrating its strong predictive accuracy on this dataset, though it lacks the uncertainty bounds that GPR provides).*

### 2. Final Model Configuration
The final model was trained on the full dataset using the following tuned hyperparameters:
```python
XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=5,
    random_state=42
)
```

### 3. Sample Predictions
These are the first 5 test cases from the newly generated CSV (`xgboost_pred2.csv`), physically constrained between 0% and 100%:
*   **Test case 1:** Yield = `21.46%`
*   **Test case 2:** Yield = `90.37%`
*   **Test case 3:** Yield = `1.95%`
*   **Test case 4:** Yield = `68.73%`
*   **Test case 5:** Yield = `13.37%`

### 4. Output Data Shape
*   **Training features shape:** `(150, 7)`
*   **Testing features shape:** `(50, 7)`
*   **Predictions File:** `../predictions/xgboost_pred2.csv` generated successfully with exactly 50 rows, meeting the hackathon submission format guidelines.
