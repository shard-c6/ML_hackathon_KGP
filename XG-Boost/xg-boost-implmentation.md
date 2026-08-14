# ML Hackathon — XGBoost Solution

## Overview

This repository contains our XGBoost solution for the ML Hackathon — Fugacity 2026, IIT Kharagpur.

The task is to predict `overall_yield` from industrial chemical-process data containing variables such as flow rate, concentration, temperature, and reactor length.

Our team is comparing multiple regression approaches:

- Ridge / ElasticNet
- Support Vector Regression (SVR)
- XGBoost / Gradient Boosting

This repository contains the XGBoost implementation, final notebook, theory, and submission file.

---

## Dataset

The training dataset contains 150 observations.

### Features

| Feature | Description |
|---|---|
| `flow_rate_L_min` | Flow rate in L/min |
| `concentration_mol_L` | Concentration in mol/L |
| `inlet_temperature_K` | Inlet temperature in Kelvin |
| `length_m` | Reactor length in meters |
| `jacket_temperature_K` | Jacket temperature in Kelvin |
| `overall_yield` | Target variable (%) |

The test dataset contains 50 observations without the target variable.

---

## XGBoost / Gradient Boosting

XGBoost (Extreme Gradient Boosting) is an ensemble machine learning method based on decision trees.

Gradient boosting builds trees sequentially. Each new tree attempts to correct the errors made by the previous trees.

Conceptually:

```text
Initial prediction
       ↓
Calculate residuals/errors
       ↓
Train a decision tree
       ↓
Update prediction
       ↓
Calculate remaining errors
       ↓
Train another tree
       ↓
Repeat
       ↓
Final prediction

The general boosting update can be represented as:

F_m(x) = F_(m-1)(x) + η h_m(x)

where:

F_m(x) = prediction after tree m
F_(m-1)(x) = prediction from previous trees
h_m(x) = new decision tree
η = learning rate
Feature Engineering

Two engineered features were retained after experimentation.

1. Residence-Time Proxy

A proxy for residence time was created using:

τ_proxy = L / Q

where:

L = reactor length
Q = flow rate

In the dataset:

residence_proxy = length_m / flow_rate_L_min

This feature was useful because the amount of time material spends inside the reactor can influence the extent of reaction and therefore the final yield.

2. Mean Temperature

The mean temperature feature was calculated as:

T_mean = (T_inlet + T_jacket) / 2

In the implementation:

mean_T = (
    inlet_temperature_K + jacket_temperature_K
) / 2

This provides a simple representation of the overall thermal environment.

Feature Selection Experiments

Different feature combinations were tested using cross-validation.

Feature configuration	Result
Original features	RMSE ≈ 22.71
+ residence_proxy	RMSE ≈ 21.56
+ mean_T	Repeated CV RMSE ≈ 19.40
+ delta_T	Repeated CV RMSE ≈ 19.56
+ tau_meanT	RMSE ≈ 21.46

Based on these experiments, the final feature set contains:

The original 5 process variables
residence_proxy
mean_T

The following experimental features were removed:

delta_T
tau_meanT
Hyperparameter Tuning

The main XGBoost hyperparameters were experimentally evaluated.

Number of Estimators

n_estimators controls the number of boosting trees.

Tested values:

50
100
200
300
500

The performance improved as more trees were added, with diminishing improvements at higher values.

Selected:

n_estimators = 500
Learning Rate

The learning rate controls how strongly each new tree contributes to the ensemble.

The update can be represented as:

F_m(x) = F_(m-1)(x) + η h_m(x)

where η is the learning rate.

Tested values:

0.01
0.03
0.05
0.10

Selected:

learning_rate = 0.05
Maximum Tree Depth

max_depth controls the maximum depth of each decision tree.

Tested values:

1
2
3
4
5

Depth 3 provided the best validation performance.

Selected:

max_depth = 3
Minimum Child Weight

min_child_weight controls how much training weight/data is required for a node to be split further.

Tested values:

1
3
5
10

The best result was obtained with:

min_child_weight = 5
Final XGBoost Configuration
XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=5,
    random_state=42
)
Validation

Repeated 5-fold cross-validation was used to make the evaluation more robust.

Five different random seeds were used, resulting in:

5 × 5 = 25 validation evaluations.

The selected model achieved approximately:

18.79 RMSE

before prediction clipping.

Since overall_yield is physically bounded between 0 and 100%, prediction clipping was also tested:

y_clipped = min(100, max(0, y))

Clipping improved the repeated cross-validation RMSE to approximately:

18.64 RMSE

Therefore, final predictions were clipped to:

0 <= y <= 100

Final Pipeline
Raw Dataset
     ↓
Data Inspection
     ↓
Feature Engineering
     ├── residence_proxy = length / flow_rate
     └── mean_T = (inlet_temperature + jacket_temperature) / 2
     ↓
XGBoost Regressor
     ↓
500 Trees
     ↓
Learning Rate = 0.05
     ↓
Maximum Depth = 3
     ↓
Minimum Child Weight = 5
     ↓
Prediction
     ↓
Clip predictions to [0, 100]
     ↓
Submission CSV
Repository Structure
ML-Hackathon/
│
├── README.md
│
├── notebook/
│   └── ML_Hackathon_XGBoost_Final.ipynb
│
├── theory/
│   ├── gradient_boosting.md
│   ├── xgboost.md
│   ├── feature_engineering.md
│   └── evaluation_metrics.md
│
├── data/
│   ├── train_dataset.csv
│   └── test_dataset.csv
│
└── submission/
    └── XGBoost_submission.csv

Note: Dataset files should only be committed if permitted by the hackathon rules.

Theory and Mathematical Relations

The theory/ directory contains the mathematical concepts and relations used in the model.

Important topics include:

Gradient Boosting
Decision Trees
Residuals
Learning Rate
Gradient Descent
XGBoost Objective Function
RMSE
Cross-Validation
Bias-Variance Tradeoff
Regularization
Feature Engineering
Residence-Time Proxy
Temperature Relationships
Team Model Comparison

Our team is comparing:

Team Member	Model
Teammate 1	Ridge / ElasticNet
Teammate 2	Support Vector Regression
Teammate 3	XGBoost / Gradient Boosting

The final model will be selected by comparing validation performance and competition performance.

Final XGBoost Result

Model: XGBoost Regressor

Training samples: 150

Test samples: 50

Number of features: 7

Final Hyperparameters
n_estimators     = 500
learning_rate    = 0.05
max_depth        = 3
min_child_weight = 5
Validation Performance
Repeated CV RMSE before clipping: ≈ 18.79
Repeated CV RMSE after clipping:  ≈ 18.64

The final model was trained using all available training observations and was used to generate predictions for the 50 test observations.

XGBoost Contribution

The XGBoost contribution includes:

Dataset inspection
Exploratory analysis
Physics-inspired feature engineering
Gradient boosting implementation
Feature selection
Hyperparameter tuning
Repeated cross-validation
Error analysis
Final model training
Test-set prediction
Prediction validation
Submission file generation