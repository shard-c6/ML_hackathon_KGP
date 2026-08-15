<div align="center">
  <h1>🚀 Continuous Flow Reactor Yield Prediction</h1>
  <h3>IITKGP ML Hackathon — Team QuarterCodes</h3>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-172b4d?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

## 📖 Overview

Welcome to **Team QuarterCodes'** repository for the IITKGP ML Hackathon! This project focuses on predicting the **overall yield** of a continuous flow reactor using machine learning models.

Our approach involves rigorous feature engineering and physical constraint mapping (Yield $\in [0, 100]\%$) to build surrogate models that not only predict accurately but also respect the underlying chemical kinetics. Over the course of the hackathon, we implemented, tuned, and evaluated **five distinct mathematical architectures** to find the optimal solution.

---

## 🏆 Comparative Model Study

We rigorously tested five different modeling paradigms. All models utilized the same base feature engineering (`residence_proxy` and `mean_T`) and underwent 5-fold cross-validation with physical prediction clipping to ensure realistic outputs.

| Rank | Model Architecture | Paradigm | Clipped RMSE | Conclusion |
|:---:|:---|:---|:---:|:---|
| 🥇 | **[XGBoost](XG-Boost/)** | Tree Ensembles (Boosting) | `~18.64` | **Champion.** Sequentially correcting errors proved the most effective way to navigate the highly complex, non-linear chemical dataset space. |
| 🥈 | **[Random Forest](RandomForest/)** | Tree Ensembles (Bagging) | `20.06` | Extremely strong performance. Validated that tree-based architectures are fundamentally the correct choice for this data. |
| 🥉 | **[Gaussian Process (GPR)](GPR/)** | Probabilistic (Matern Kernel) | `~20.48` | Highly mathematically rigorous. Modeled the continuous spatial nature of the reactor excellently but struggled slightly on edge cases. |
| 4 | **[Support Vector (SVR)](SVR/)** | Margin-based (RBF Kernel) | `22.09` | The RBF kernel handled infinite-dimensional mapping well, but was overly sensitive to the small dataset size (150 samples). |
| 5 | **[ElasticNet (Poly 3)](ElasticNet/)** | Regularized Linear Polynomials | `23.05` | Despite expanding to 120 polynomial interaction terms and using Lasso to prevent overfitting, linear combinations simply cannot capture the reactor's physical complexity. |

---

## 🏗️ System Architecture

Our solution is built on a structured pipeline, transforming raw reactor parameters into highly accurate, physically constrained predictions.

```mermaid
graph TD
    A[Raw Datasets] --> B(Feature Engineering)
    B -->|residence_proxy = L/Q| C[Pre-processed Data]
    B -->|mean_T = (T_in + T_jacket) / 2| C

    C --> D{Model Training & Tuning}

    D --> E[XGBoost]
    D --> F[Random Forest]
    D --> G[Gaussian Process]
    D --> J[Support Vector]
    D --> K[ElasticNet Poly]

    E --> H(Physical Constraint Clipping: 0-100%)
    F --> H
    G --> H
    J --> H
    K --> H

    H --> I((Final Yield Predictions))
```

For an in-depth breakdown of our pipeline design choices, see our [Architecture Documentation](docs/architecture.md).

---

## 📂 Repository Structure

Our repository is organized cleanly by model, with dedicated mathematical documentation for each approach.

### 📚 Core Documentation
- **[`/docs/achieve.md`](docs/achieve.md)**: Mathematical formulations and target objectives.
- **[`/docs/architecture.md`](docs/architecture.md)**: Detailed pipeline architecture and feature engineering choices.

### 🧠 Model Implementations
- **[`/XG-Boost`](XG-Boost/) (Baseline & Champion)**
  - [`xgb-implementation.md`](XG-Boost/xgb-implementation.md): Implementation details.
  - [`xgb-metrics.md`](XG-Boost/xgb-metrics.md): Performance breakdown.
- **[`/RandomForest`](RandomForest/)**
  - [`rf-implementation.md`](RandomForest/rf-implementation.md): Tuning logic.
  - [`rf-metrics.md`](RandomForest/rf-metrics.md): Performance breakdown.
- **[`/GPR`](GPR/)**
  - [`gpr-implementation.md`](GPR/gpr-implementation.md): Matern kernel design.
  - [`gpr-metrics.md`](GPR/gpr-metrics.md): Performance breakdown.
- **[`/SVR`](SVR/)**
  - [`svr-implementation.md`](SVR/svr-implementation.md): Radial Basis Function setup.
  - [`svr-metrics.md`](SVR/svr-metrics.md): Performance breakdown.
- **[`/ElasticNet`](ElasticNet/)**
  - [`elasticnet-implementation.md`](ElasticNet/elasticnet-implementation.md): Polynomial feature expansion.
  - [`elasticnet-metrics.md`](ElasticNet/elasticnet-metrics.md): Performance breakdown.

### 💾 Data & Outputs
- **[`/datasets`](datasets/)**: The raw training and testing CSV files.
- **[`/predictions`](predictions/)**: The final generated CSV predictions ready for submission, separated by model.

---

## 🚀 Getting Started

We have configured our models to run seamlessly locally or via **Google Colab**. Every model folder contains an automated Python script (`run_model.py`) and an equivalent Colab-ready Jupyter Notebook.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shard-c6/ML_hackathon_KGP.git
    cd ML_hackathon_KGP
    ```
2.  **Run Models in Colab:**
    Open any of the notebooks (e.g., `ML_Hackathon_XGBoost_Final.ipynb`) in Google Colab. The paths are dynamically configured using a `RUNNING_IN_COLAB` flag; just upload the dataset CSVs to your `/content/` directory and run all cells!

---

> **Built with ❤️ by Team QuarterCodes**
