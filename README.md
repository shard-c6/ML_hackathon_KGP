<div align="center">
  <h1>🚀 Continuous Flow Reactor Yield Prediction</h1>
  <h3>IITKGP ML Hackathon — Team QuarterCodes</h3>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-172b4d?style=for-the-badge&logo=xgboost&logoColor=white" />
</p>

## 📖 Overview

Welcome to **Team QuarterCodes'** repository for the IITKGP ML Hackathon! This project focuses on predicting the **overall yield** of a continuous flow reactor using machine learning models.

Our approach involves rigorous feature engineering and physical constraint mapping to build surrogate models that not only predict accurately but also respect the underlying chemical kinetics.

---

## 🏗️ System Architecture

Our solution is built on a structured pipeline, transforming raw reactor parameters into highly accurate, physically constrained predictions.

```mermaid
graph TD
    A[Raw Datasets] --> B(Feature Engineering)
    B -->|residence_proxy = L/Q| C[Pre-processed Data]
    B -->|mean_T = (T_in + T_jacket) / 2| C

    C --> D[Model Training]

    D --> E[XGBoost]
    D --> F[Gaussian Process Regression]
    D --> G[Support Vector Regression]

    E --> H(Physical Constraint Clipping: 0-100%)
    F --> H
    G --> H

    H --> I[Final Yield Prediction]
```

For an in-depth breakdown of our pipeline, see our [Architecture Documentation](docs/architecture.md).

---

## 📂 Repository Structure

- **[`/docs`](./docs)**: Project documentation.
  - [`achieve.md`](./docs/achieve.md): Mathematical formulations and target objectives.
  - [`architecture.md`](./docs/architecture.md): Detailed pipeline architecture and feature choices.
- **[`/datasets`](./datasets)**: The raw training and testing CSV files.
- **[`/XG-Boost`](./XG-Boost)**: Our XGBoost implementation (Highest pure accuracy).
- **[`/GPR`](./GPR)**: Our Gaussian Process Regression model (Uncertainty bounds & physical reality).
- **[`/SVR`](./SVR)**: Our Support Vector Regression model.
- **[`/predictions`](./predictions)**: The final generated CSV predictions ready for submission.

---

## 🚀 Getting Started

We have configured our models to run seamlessly locally or via **Google Colab**.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shard-c6/ML_hackathon_KGP.git
    cd ML_hackathon_KGP
    ```
2.  **Run Models in Colab:**
    Open the notebooks inside `/XG-Boost` or `/GPR` in Google Colab. The paths are dynamically configured; just upload your datasets and run the cells!

---

> **Built with ❤️ by Team QuarterCodes**
