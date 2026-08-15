# Process Mathematical Relations & Formulas

This document outlines the key mathematical relationships and formulas used throughout the predictive modeling process for predicting the overall yield of Product B.

## A. Evaluation Metric (RMSE)

The primary quantitative metric for evaluating model performance is the Root Mean Squared Error, calculated as:

```math
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
```

_Where $y_i$ is the actual yield, $\hat{y}_i$ is the predicted yield, and $n$ is the number of test observations._

## B. Feature Engineering Relations

**Residence-Time Proxy ($\tau_{\text{proxy}}$):**

```math
\tau_{\text{proxy}} = \frac{L}{Q}
```

_Where $L$ is the reactor length (`length_m`) and $Q$ is the volumetric flow rate (`flow_rate_L_min`)._

**Mean Temperature ($T_{\text{mean}}$):**

```math
T_{\text{mean}} = \frac{T_{\text{inlet}} + T_{\text{jacket}}}{2}
```

_Where $T_{\text{inlet}}$ is `inlet_temperature_K` and $T_{\text{jacket}}$ is `jacket_temperature_K`._

## C. Model Training (Gradient Boosting Update)

At each step $m$, the model is updated by adding a new decision tree $h_m(x)$ scaled by the learning rate $\eta$:

```math
F_m(x) = F_{m-1}(x) + \eta h_m(x)
```

_Where $F_m(x)$ is the new ensemble prediction and $F_{m-1}(x)$ is the previous prediction._

## D. Physical Constraints (Prediction Clipping)

To ensure the model respects the physical reality of chemical yields (0% to 100%), predictions are clipped:

```math
y_{\text{clipped}} = \min(100, \max(0, y_{\text{predicted}}))
```

