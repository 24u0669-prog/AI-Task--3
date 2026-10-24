# AI Task 3: Model Fairness, Bias & Explainability Analysis

Comprehensive fairness audit, interpretability analysis (SHAP / LIME), and practical algorithmic bias mitigation on customer churn prediction models.

---

## 🌟 Quick Start

### 1. View Interactive Web Dashboard
Double-click `run_dashboard.bat` or open `index.html` directly in any web browser:
```cmd
start index.html
```

### 2. Run the Full Python Pipeline
Execute the end-to-end audit, plot generation, and mitigation benchmark:
```powershell
.\.venv\Scripts\python.exe run_analysis.py
```

### 3. Open the Jupyter Notebook
Open `bias_fairness_explainability.ipynb` in VS Code or JupyterLab:
```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

---

## 📁 Repository Structure

```text
AI Task -3/
├── index.html                           <- Interactive web dashboard & simulator
├── run_dashboard.bat                    <- One-click web dashboard launcher
├── run_analysis.py                      <- Complete end-to-end pipeline script
├── build_notebook.py                    <- Notebook generator script
├── bias_fairness_explainability.ipynb   <- Deliverable Jupyter Notebook
├── FAIRNESS_AND_BIAS_MITIGATION_REPORT.md <- Deliverable technical report
├── README.md                            <- Project guide & documentation
├── data/
│   └── Telco-Customer-Churn.csv         <- Cleaned dataset (7,043 samples)
├── plots/                               <- 12 High-Resolution Analysis Plots
│   ├── 01_global_feature_importance.png
│   ├── 02_shap_beeswarm.png
│   ├── 03_shap_dependence_tenure.png
│   ├── 04_shap_waterfall_persona_A.png
│   ├── 05_shap_waterfall_persona_B.png
│   ├── 06_shap_waterfall_persona_C.png
│   ├── 07_lime_explanation_persona_A.png
│   ├── 08_lime_explanation_comparison.png
│   ├── 09_fairness_metrics_by_group.png
│   ├── 10_confusion_matrices_by_group.png
│   ├── 11_mitigation_fairness_vs_performance.png
│   └── 12_mitigation_group_metrics_comparison.png
└── outputs/                             <- Exported CSV Scorecards & JSON Summaries
    ├── analysis_results_summary.json
    ├── baseline_model_performance.csv
    ├── feature_importance_comparison.csv
    ├── fairness_audit_detailed_groups.csv
    ├── fairness_disparities_summary.csv
    ├── mitigation_comparison_scorecard.csv
    └── representative_personas.csv
```

---

## 🔬 Key Results Overview

### 1. Explainability (SHAP & LIME)
- **Top Global Drivers:** `Contract` (Two-year vs Month-to-month), `tenure`, `InternetService_Fiber optic`, `MonthlyCharges`, and `TechSupport`.
- **Local Consistency:** Both TreeSHAP and LIME tabular surrogates consistently identify short tenure and month-to-month contracts as the primary factors elevating individual customer churn risk.

### 2. Fairness & Bias Audit (Random Forest Baseline)
- **Senior Citizens (Age 65+):** Heavy disparate impact. Selection rate is **64.9%** for seniors vs **35.7%** for non-seniors ($\Delta_{DP} = 0.2914$, Disparate Impact Ratio = $0.5507$, violating the EEOC 80% Rule). Senior False Positive Rate is **45.2%** vs **23.5%** for non-seniors.
- **Gender:** High parity ($\Delta_{DP} = 0.0084$, Disparate Impact Ratio = $0.9795$).

### 3. Practical Mitigation Benchmark
- **Pre-processing (Sample Reweighting):** **73.4% reduction** in Equalized Odds disparity ($0.2167 \to 0.0576$), with an overall Accuracy increase to **80.1%**.
- **In-processing (Exponentiated Gradient):** Slashes Demographic Parity difference to **0.0127** (near-perfect statistical parity).
- **Post-processing (ThresholdOptimizer):** Reduces Equal Opportunity difference to **0.0456** without model retraining.
 ### Author 
 **Soundarya Umesh Barigidad**
 **Information Science Engineering Student**