# Preprocessing and Feature Engineering

This document details the feature engineering strategies and the final input features used by the predictive model. The focus is on demonstrating process insight and creating features grounded in physical and chemical principles.

## 1. Feature Engineering Strategy

The predictive model uses "Engineering Intuition" to transform raw inputs into physically meaningful features:

- **Residence-Time Proxy (`length_m / flow_rate_L_min`):** This is a physics-inspired feature. In continuous flow reactors, the extent of a reaction heavily depends on the residence time ($\tau$). By deriving $\tau$, the model captures the underlying chemical kinetics.
- **Mean Temperature:** Chemical reactions are highly sensitive to temperature. Creating a thermal proxy that averages the inlet and jacket temperatures models the overall thermal environment of the reactor.
- **Prediction Clipping:** Chemical yields are physically constrained between 0% and 100%. Applying clipping `min(100, max(0, y))` ensures the predictions adhere to physical reality, which also improves cross-validation RMSE.

## 2. Final Feature Set

To successfully run the model and generate the final predictions, the following features are extracted or engineered:

### Input Features (7 Total)
1. **`flow_rate_L_min`**: Volumetric flow rate of the reactant mixture (L/min).
2. **`concentration_mol_L`**: Inlet concentration of Reactant A (mol/L).
3. **`inlet_temperature_K`**: Temperature of the feed entering the reactor (K).
4. **`length_m`**: The specific length of the reactor (m).
5. **`jacket_temperature_K`**: The temperature of the external heating jacket (K).
6. **`residence_proxy`**: Engineered feature ($L/Q$), representing residence time.
7. **`mean_T`**: Engineered feature ($(T_{\text{inlet}} + T_{\text{jacket}})/2$), representing the mean thermal environment.

### Target Output
*   **`overall_yield`**: The final yield percentage of Product B at the reactor exit. The final output is constrained between 0 and 100.
