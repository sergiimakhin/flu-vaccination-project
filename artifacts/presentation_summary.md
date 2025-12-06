# Model Selection: LightGBM

## 1. Candidate Models & Performance
We evaluated three distinct architectures to identify the optimal model:
*   **Logistic Regression**: Linear baseline.
*   **Random Forest**: Non-linear bagging ensemble.
*   **LightGBM**: Gradient boosting framework.

**Result**: LightGBM outperformed baselines on the validation set for both targets using the **ROC-AUC** metric.

| Target | Model | ROC-AUC | Lift vs LogReg |
| :--- | :--- | :--- | :--- |
| **H1N1 Vaccine** | **LightGBM** | **0.8705** | +0.83% |
| **Seasonal Flu** | **LightGBM** | **0.8636** | +0.72% |

## 2. Why ROC-AUC & LightGBM?

### The Challenge: Imbalance
*   **H1N1** is effectively imbalanced (**21%** positive / 79% negative).
*   **Seasonal Flu** is balanced (**47%** positive / 53% negative).
*   *Selection Metric*: **ROC-AUC** was chosen over Accuracy/F1 because it is threshold-independent and robust to imbalance, ensuring the model prioritizes ranking actual positive cases higher rather than just maximizing majority class accuracy.

### The Solution: LightGBM
*   **Boosting Mechanism**: By training sequentially on residuals, LightGBM naturally focuses learning on "hard" examples (often the minority positive class in H1N1), improving discrimination where others fail.
*   **Interpretability (SHAP)**: LightGBM supports **TreeSHAP**, enabling the computation of *exact* feature contributions (Shapley values). This is critical for explaining *why* specific demographics are predicted as high-affinity for the vaccine, ensuring actionable and trustworthy insights.
