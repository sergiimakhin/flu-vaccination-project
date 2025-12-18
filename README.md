# Flu Vaccine Prediction Project

This project focuses on predicting the uptake of H1N1 and seasonal flu vaccines using machine learning techniques. It implements a complete data science workflow, ranging from exploratory data analysis (EDA) and robust data preprocessing to model selection, hyperparameter tuning, and interpretable AI analysis.

## 📋 Project Overview

The goal of this project is to build predictive models that can identify individuals most likely to receive H1N1 and seasonal flu vaccines. Understanding these drivers can help public health officials target vaccination campaigns more effectively.

**Key Features:**
*   **Comprehensive EDA:** detailed analysis of feature distributions and correlations.
*   **Robust Preprocessing:** automated pipelines for missing value imputation, ordinal mapping, and target/one-hot encoding.
*   **Model Comparison:** evaluation of Logistic Regression, Random Forest, and LightGBM models.
*   **Advanced Modeling:** Fine-tuned LightGBM classifiers with cross-validation to prevent data leakage.
*   **Explainability:** SHAP (SHapley Additive exPlanations) analysis to interpret model decisions and feature importance.
*   **Visualization:** custom-generated plots for confusion matrices, ROC curves, and feature importance.

## 📂 Repository Structure

```text
├── artifacts/          # Generated models (.pkl), metrics (.json), and plots (.png)
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for analysis and experimentation
│   ├── EDA.ipynb                 # Initial exploratory data analysis
│   ├── EDA_extended.ipynb        # Deeper dive into data patterns
│   ├── preprocessing.ipynb       # Data cleaning and encoding workflow
│   ├── feature_engineering.ipynb # Feature extraction and selection
│   ├── model_selection.ipynb     # Baseline model comparison
│   ├── modeling_lgbm.ipynb       # LightGBM training and fine-tuning
│   ├── models_reduced.ipynb      # Training on reduced feature sets
│   └── SHAP.ipynb                # Feature importance analysis
├── src/                # Modular Python source code
│   ├── preprocessing.py          # Data cleaning and encoding functions
│   ├── modeling.py               # Model training, evaluation, and saving
│   ├── mappings.py               # Categorical variable mappings
│   └── visualization/            # Plotting utilities (e.g., SHAP plots)
├── tests/              # Unit tests and scripts for reproducibility
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   The dependencies listed in `requirements.txt`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/sergiimakhin/flu-vaccination-project.git
    cd flu-vaccination-project
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage Workflow

1.  **Data Preprocessing**:
    Run `notebooks/preprocessing.ipynb` to clean the raw data and generate processed datasets in the `data/` folder.

2.  **Model Training**:
    Use `notebooks/model_selection.ipynb` to compare baselines or `notebooks/modeling_lgbm.ipynb` to train and tune the final LightGBM model.
    Alternatively, use the source script directly:
    ```python
    from src.modeling import train_final_model
    # see src/modeling.py for detailed usage
    ```

3.  **Evaluation & Visualization**:
    Check the `artifacts/` directory for generated confusion matrices (`_cm.png`) and metrics files (`_metrics.json`).
    Run `notebooks/SHAP.ipynb` to generate feature importance plots.

## 📊 Methodology

### Data Pipeline
The `src/preprocessing.py` module handles:
*   **Imputation**: Median values for numerical columns; 'Missing' placeholder for categoricals.
*   **Encoding**:
    *   *Ordinal*: Education, Age Group, Income Poverty.
    *   *One-Hot*: Nominal categories like Race, Sex, Marital Status.
    *   *Target Encoding*: High-cardinality features like Employment Industry/Occupation.

### Modeling Strategy
We utilize **LightGBM** as our primary model due to its efficiency and high performance on structured data. The training process includes:
*   Stratified Train-Test Splits
*   RandomizedSearchCV for hyperparameter tuning
*   Recall, Precision, F1-score, and ROC-AUC metric tracking

## 📈 Results

The project generates several key artifacts to visualize performance:
*   **Confusion Matrices**: Normalized heatmaps showing true positive/negative rates.
*   **SHAP Plots**: Bar charts highlighting the top features driving vaccination predictions (e.g., Doctor Recommendation, Age Group).

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
