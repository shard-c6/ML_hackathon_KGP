import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

# --- Path Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_dir, '..', 'datasets')

TRAIN_PATH = os.path.join(datasets_dir, 'train_dataset.csv')
TEST_PATH = os.path.join(datasets_dir, 'test_dataset.csv')
PREDS_PATH = os.path.join(base_dir, '..', 'predictions', 'RandomForest_predictions.csv')

# --- 1. Load Datasets ---
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# --- 2. Feature Engineering ---
train_df['residence_proxy'] = train_df['length_m'] / train_df['flow_rate_L_min']
test_df['residence_proxy'] = test_df['length_m'] / test_df['flow_rate_L_min']

train_df['mean_T'] = (train_df['inlet_temperature_K'] + train_df['jacket_temperature_K']) / 2
test_df['mean_T'] = (test_df['inlet_temperature_K'] + test_df['jacket_temperature_K']) / 2

features = [
    'flow_rate_L_min',
    'concentration_mol_L',
    'inlet_temperature_K',
    'length_m',
    'jacket_temperature_K',
    'residence_proxy',
    'mean_T'
]
target = 'overall_yield'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

# --- 3. Pipeline Setup ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# --- 4. Hyperparameter Tuning ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 5, 10, 15],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

print("Running GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV RMSE (raw): {-grid_search.best_score_:.4f}")

# --- 5. Custom Cross-Validation Evaluation (with Clipping) ---
fold_rmses = []
fold_rmses_clipped = []

for train_idx, val_idx in kf.split(X_train):
    X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    pipeline.set_params(**grid_search.best_params_)
    pipeline.fit(X_tr_f, y_tr_f)
    
    val_preds = pipeline.predict(X_val_f)
    val_preds_clipped = np.clip(val_preds, 0, 100)
    
    fold_rmses.append(np.sqrt(mean_squared_error(y_val_f, val_preds)))
    fold_rmses_clipped.append(np.sqrt(mean_squared_error(y_val_f, val_preds_clipped)))

raw_oof_rmse = np.mean(fold_rmses)
clipped_oof_rmse = np.mean(fold_rmses_clipped)
mean_fold_rmse = clipped_oof_rmse
std_fold_rmse = np.std(fold_rmses_clipped, ddof=1)

print(f"\nRaw RMSE = {raw_oof_rmse:.4f}")
print(f"Clipped RMSE = {clipped_oof_rmse:.4f}")
print(f"Clipping improvement = {raw_oof_rmse - clipped_oof_rmse:.4f}")

print("\nClipped fold RMSEs:")
for rmse in fold_rmses_clipped:
    print(f"{rmse:.4f}")

print(f"\nMean CV RMSE = {mean_fold_rmse:.4f}")
print(f"Standard deviation = {std_fold_rmse:.4f}")

# --- 6. Final Model Fitting and Predictions ---
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

preds = np.clip(best_model.predict(X_test), 0, 100)

os.makedirs(os.path.dirname(PREDS_PATH), exist_ok=True)
pd.DataFrame({'overall_yield': preds}).to_csv(PREDS_PATH, index=False)
print(f"\nPredictions saved to {PREDS_PATH}")
