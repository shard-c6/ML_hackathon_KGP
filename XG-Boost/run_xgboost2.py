import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train_df = pd.read_csv('../datasets/train_dataset.csv')
test_df = pd.read_csv('../datasets/test_dataset.csv')

def engineer_features(df):
    df = df.copy()
    # 1. Residence-Time Proxy
    df['residence_proxy'] = df['length_m'] / df['flow_rate_L_min']
    # 2. Mean Temperature
    df['mean_T'] = (df['inlet_temperature_K'] + df['jacket_temperature_K']) / 2
    return df

# Apply feature engineering
X_train_full = engineer_features(train_df.drop(columns=['overall_yield']))
y_train_full = train_df['overall_yield']
X_test = engineer_features(test_df)

print(f"Training features shape: {X_train_full.shape}")
print(f"Testing features shape: {X_test.shape}")

# Define the XGBoost model
xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=5,
    random_state=42
)

# Cross Validation
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
rmse_scores = []

for train_idx, val_idx in rkf.split(X_train_full):
    X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # Train
    xgb.fit(X_tr, y_tr)
    
    # Predict
    y_pred = xgb.predict(X_val)
    
    # Clip predictions to physical bounds [0, 100]
    y_pred_clipped = np.clip(y_pred, 0, 100)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_clipped))
    rmse_scores.append(rmse)

print(f"Repeated CV RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

# Train on full dataset
print("\nTraining final model on full dataset...")
xgb.fit(X_train_full, y_train_full)

# Predict on test set
y_test_pred = xgb.predict(X_test)

# Apply physical constraints
y_test_pred_clipped = np.clip(y_test_pred, 0, 100)

# Prepare submission
submission = pd.DataFrame({
    'overall_yield': np.round(y_test_pred_clipped, 3)
})

# Save predictions
os.makedirs('../predictions', exist_ok=True)
submission_path = '../predictions/xgboost_pred2.csv'
submission.to_csv(submission_path, index=False)
print(f"\nPredictions saved to {submission_path}")

# Display a few predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Test case {i+1}: Yield = {y_test_pred_clipped[i]:.2f}%")
