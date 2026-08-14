import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train_df = pd.read_csv('../datasets/train_dataset.csv')
test_df = pd.read_csv('../datasets/test_dataset.csv')

def engineer_features(df):
    df = df.copy()
    # 1. Residence-Time Proxy (L/Q)
    df['residence_proxy'] = df['length_m'] / df['flow_rate_L_min']
    # 2. Mean Temperature (inlet + jacket) / 2
    df['mean_T'] = (df['inlet_temperature_K'] + df['jacket_temperature_K']) / 2
    return df

# Apply feature engineering
X_train_full = engineer_features(train_df.drop(columns=['overall_yield']))
y_train_full = train_df['overall_yield']
X_test = engineer_features(test_df)

print(f"Training features shape: {X_train_full.shape}")
print(f"Testing features shape: {X_test.shape}")

# GPR is highly sensitive to feature scales (it relies on distance measurements).
# We strictly scale all features to mean=0, std=1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Define the kernel
# Matern kernel allows for smooth but slightly rougher functions than RBF, good for real physical processes.
# WhiteKernel explicitly models the inherent measurement noise in the data.
kernel = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)

gpr = GaussianProcessRegressor(
    kernel=kernel, 
    n_restarts_optimizer=10, 
    random_state=42,
    normalize_y=True # Important for targets not centered around 0
)

# Robust Cross Validation
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
rmse_scores = []

for train_idx, val_idx in rkf.split(X_train_scaled):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # Train
    gpr.fit(X_tr, y_tr)
    
    # Predict
    y_pred, _ = gpr.predict(X_val, return_std=True)
    
    # Clip predictions to physical bounds [0, 100]
    y_pred_clipped = np.clip(y_pred, 0, 100)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_clipped))
    rmse_scores.append(rmse)

print(f"Repeated CV RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

# Train on full dataset
print("\nTraining final model on full dataset...")
gpr.fit(X_train_scaled, y_train_full)

print("\nOptimized Kernel parameters after fitting:")
print(gpr.kernel_)

# Predict on test set
# Also extract standard deviation (uncertainty) to show judges
y_test_pred, y_test_std = gpr.predict(X_test_scaled, return_std=True)

# Apply physical constraints
y_test_pred_clipped = np.clip(y_test_pred, 0, 100)

# Prepare submission
submission = pd.DataFrame({
    'overall_yield': np.round(y_test_pred_clipped, 3)
})

# Save predictions
os.makedirs('../predictions', exist_ok=True)
submission_path = '../predictions/GPR_predictions.csv'
submission.to_csv(submission_path, index=False)
print(f"\nPredictions saved to {submission_path}")

# Display a few predictions with their uncertainty (great for Phase 2 pitch)
print("\nSample Predictions with Uncertainty:")
for i in range(5):
    print(f"Test case {i+1}: Yield = {y_test_pred_clipped[i]:.2f}% +/- {y_test_std[i]:.2f}%")
