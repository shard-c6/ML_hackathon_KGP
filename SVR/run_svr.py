import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
import math

def calculate_rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred)**2))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "datasets", "train_dataset.csv")
    test_path = os.path.join(base_dir, "datasets", "test_dataset.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Feature Engineering
    for df in [train_df, test_df]:
        df['residence_proxy'] = df['length_m'] / df['flow_rate_L_min']
        df['mean_T'] = (df['inlet_temperature_K'] + df['jacket_temperature_K']) / 2.0

    features = [
        "concentration_mol_L",
        "residence_proxy",
        "mean_T"
    ]

    target = "overall_yield"

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]

    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf'))
    ])

    # KFold (as exactly specified in requirements)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search logic for reasonably tuning over C, gamma, epsilon without leaking
    param_grid = {
        'svr__C': [0.1, 1, 10, 50, 100, 150, 200, 300, 500],
        'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'svr__epsilon': [0.01, 0.1, 0.5, 1.0, 2.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=kf,
        n_jobs=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Calculate actual Fold RMSEs (Raw vs Clipped) on OOF validation sets
    fold_rmses = []
    fold_rmses_clipped = []
    
    kf_manual = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf_manual.split(X_train):
        X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Clone pipeline with best params
        fold_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', 
                        C=best_params['svr__C'], 
                        gamma=best_params['svr__gamma'], 
                        epsilon=best_params['svr__epsilon']))
        ])
        fold_pipeline.fit(X_tr_f, y_tr_f)
        
        val_preds = fold_pipeline.predict(X_val_f)
        val_preds_clipped = np.clip(val_preds, 0, 100)
        
        fold_rmses.append(calculate_rmse(y_val_f, val_preds))
        fold_rmses_clipped.append(calculate_rmse(y_val_f, val_preds_clipped))
    
    mean_raw_rmse = np.mean(fold_rmses)
    std_raw_rmse = np.std(fold_rmses, ddof=1)
    
    mean_clipped_rmse = np.mean(fold_rmses_clipped)
    std_clipped_rmse = np.std(fold_rmses_clipped, ddof=1)
    
    clipping_helped = mean_clipped_rmse < mean_raw_rmse
    clipping_diff = mean_raw_rmse - mean_clipped_rmse
    
    final_cv_rmse = mean_clipped_rmse if clipping_helped else mean_raw_rmse
    final_cv_std = std_clipped_rmse if clipping_helped else std_raw_rmse
    
    benchmark_rmse = 21.7020
    is_better = final_cv_rmse < benchmark_rmse
    better_worse_str = "better" if is_better else "worse"
    magnitude_diff = abs(final_cv_rmse - benchmark_rmse)
    percentage_diff = (magnitude_diff / benchmark_rmse) * 100
    
    # Predict on test set
    # Best model is already fit on all X_train by GridSearchCV
    final_preds = best_model.predict(X_test)
    if clipping_helped:
        final_preds = np.clip(final_preds, 0, 100)
        
    predictions_path = os.path.join(base_dir, "SVR_predictions.csv")
    preds_df = pd.DataFrame({'overall_yield': final_preds})
    preds_df.to_csv(predictions_path, index=False)
    
    # Output Final Report
    print("--- Final Report ---")
    print("\nFiles created/modified (full paths):")
    print(f"- {os.path.join(base_dir, 'models', 'svr_model.py')}")
    print(f"- {predictions_path}")
    print("\nBest SVR parameters (C, gamma, epsilon):")
    print(f"C: {best_params['svr__C']}, gamma: {best_params['svr__gamma']}, epsilon: {best_params['svr__epsilon']}")
    print("\nFold RMSEs (all 5, individually):")
    if clipping_helped:
        for i, rmse_val in enumerate(fold_rmses_clipped):
            print(f"Fold {i+1}: {rmse_val:.4f} (Clipped)")
    else:
        for i, rmse_val in enumerate(fold_rmses):
            print(f"Fold {i+1}: {rmse_val:.4f} (Raw)")
            
    print(f"\nMean CV RMSE: {final_cv_rmse:.4f}")
    print(f"Standard deviation of CV RMSE: {final_cv_std:.4f}")
    
    print("\nClipping result (raw RMSE vs. clipped RMSE, and whether clipping helped):")
    print(f"Raw RMSE: {mean_raw_rmse:.4f}")
    print(f"Clipped RMSE: {mean_clipped_rmse:.4f}")
    print(f"Clipping helped: {clipping_helped} (improved by {clipping_diff:.4f})")
    
    print(f"\nComparison with benchmark {benchmark_rmse} (better/worse, magnitude):")
    print(f"The new model is {better_worse_str} by {magnitude_diff:.4f} absolute RMSE ({percentage_diff:.2f}%).")
    
    print(f"\nLocation of SVR_predictions.csv:\n{predictions_path}")
    
if __name__ == "__main__":
    main()
