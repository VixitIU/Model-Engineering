# Breast Cancer Malignancy Classifier
### IU International University of Applied Sciences — DLBDSME01
**Student:** Tatiana Lvova  
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) — 569 samples, 30 features  
**Task:** Binary classification of tumour diagnosis (Malignant / Benign)  
**Methodology:** CRISP-DM pipeline

---

## Project Overview

This project builds an interpretable binary classifier to distinguish malignant from benign breast tumours using the WDBC dataset. The primary optimisation target is **minimising false negatives** (missed malignancies); F1 score on the malignant class is the reported headline metric, with a project target of F1 > 0.95.

The final selected model is **Logistic Regression** (`C=1.0`, `solver=lbfgs`, `class_weight="balanced"`, `threshold=0.45`) trained on a **21-feature set** that drops the area, perimeter, and concavity feature families while retaining radius, texture, smoothness, compactness, symmetry, concave ponts and fractal dimension across all three measurement types (_mean, _se, _worst).

---

## Repository Structure

```
.
├── Data for Task 1.csv                      # WDBC source data
├── breast_cancer_Modelling.ipynb            # modelling → evaluation
├── Iterative_VIF_Feature_Elimination.ipynb  # VIF-only feature elimination experiment
├── Combined_Feature_Elimination.ipynb       # Domain-guided + VIF elimination experiment
├── Cross-Validation.ipynb                   # Seed sweep + repeated stratified K-fold stability check
├── LR_threshold_optimization.ipynb          # Threshold sweep (0.25–0.50) via 10×10 repeated CV
├── feature_importance_explainer.ipynb       # standardised-value breakdown
├── Visualization.ipynb                      # EDA plots and report-ready figures
└── README.md
```

---

## Notebooks — Purpose and Order

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `Visualization.ipynb` | Produces all report figures: feature histograms, box plots, correlation heatmap, ROC curves. |
| 2 | `breast_cancer_Modelling.ipynb` | Main pipeline. EDA, feature selection, model comparison (LR, Decision Tree, SVM, EBM), and final model evaluation. |
| 3 | `Iterative_VIF_Feature_Elimination.ipynb` | Baseline: removes features with VIF > 10 one at a time. Retained as a methodological foil: shows VIF elimination discards high-signal features. |
| 4 | `Combined_Feature_Elimination.ipynb` | Domain drops first (area, perimeter, concavity), then VIF elimination on remainder. |
| 5 | `Cross-Validation.ipynb` | Compares LR vs Linear SVM under 30-seed split sweep and 10×10 repeated K-fold. Confirms LR stability advantage. |
| 6 | `LR_threshold_optimization.ipynb` | Sweeps classification thresholds 0.25–0.50 under 10×10 repeated stratified K-fold. Identifies threshold 0.45 as the operating point. |
| 7 | `feature_importance_explainer.ipynb` | Per-patient explanation: coefficient × standardised feature value breakdown. Produces contribution bar charts for individual predictions. |

---

## Final Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| C | 1.0 |
| Solver | lbfgs |
| class_weight | balanced |
| max_iter | 10000 |
| Classification threshold | 0.35 |
| Feature set | 21 features (see below) |
| Scaler | StandardScaler |

