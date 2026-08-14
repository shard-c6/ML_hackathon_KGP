import json

def add_cell(notebook_path, code_string):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Process lines to keep newlines correctly for Jupyter
    lines = [line + "\n" for line in code_string.split("\n")]
    if lines and lines[-1] == "\n":
        lines = lines[:-1]
    
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Visualizations\n",
            "The following plots help visualize the model's behavior and are highly recommended for the Phase 2 pitch."
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }
    
    nb["cells"].append(markdown_cell)
    nb["cells"].append(code_cell)
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

# Add to XGBoost Notebook
xgb_code = """# Visualization: Feature Importance and True vs Predicted
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Feature Importance
importance = xgb.feature_importances_
features = X_train_full.columns
sns.barplot(x=importance, y=features, ax=axes[0], palette='viridis')
axes[0].set_title('XGBoost Feature Importance', fontsize=14, pad=15)
axes[0].set_xlabel('Importance Score')

# 2. Training Fit (True vs Predicted)
y_train_pred = np.clip(xgb.predict(X_train_full), 0, 100)
axes[1].scatter(y_train_full, y_train_pred, alpha=0.6, color='b', edgecolor='w')
axes[1].plot([0, 100], [0, 100], 'r--', lw=2) # Perfect prediction line
axes[1].set_title('Training Data: True vs Predicted Yield', fontsize=14, pad=15)
axes[1].set_xlabel('True Yield (%)')
axes[1].set_ylabel('Predicted Yield (%)')

plt.tight_layout()
plt.show()"""

add_cell('/Users/shard/projects/ML_hackathon_KGP/XG-Boost/xgboost_nbk2.ipynb', xgb_code)

# Add to GPR Notebook
gpr_code = """# Visualization: Prediction Uncertainty
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 6))

# Sort test predictions for better visualization of the curve
sort_idx = np.argsort(y_test_pred_clipped)
y_sorted = y_test_pred_clipped[sort_idx]
std_sorted = y_test_std[sort_idx]
x_axis = np.arange(len(y_sorted))

plt.plot(x_axis, y_sorted, 'b-', label='Predicted Yield', linewidth=2)

# Plot 95% Confidence Interval (1.96 * standard deviation)
lower_bound = np.clip(y_sorted - 1.96 * std_sorted, 0, 100)
upper_bound = np.clip(y_sorted + 1.96 * std_sorted, 0, 100)
plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2, color='blue', label='95% Confidence Interval')

plt.title('GPR Test Predictions with Uncertainty Bounds (Physical Reality)', fontsize=14, pad=15)
plt.xlabel('Test Instance (Sorted by Predicted Yield)')
plt.ylabel('Overall Yield (%)')
plt.legend()
plt.tight_layout()
plt.show()"""

add_cell('/Users/shard/projects/ML_hackathon_KGP/GPR/ML_Hackathon_GPR_Final.ipynb', gpr_code)
print("Successfully added visualization cells to both notebooks!")
