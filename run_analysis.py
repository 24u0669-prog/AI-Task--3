"""
Task 3: Model Fairness, Bias, and Explainability Pipeline
- Dataset: IBM Telco Customer Churn
- Models: Logistic Regression and Random Forest Classifier
- Explainability: Global feature importance, TreeSHAP, and LIME tabular local explanations
- Fairness & Bias Audit: Sensitive groups (SeniorCitizen, gender, Dependents)
- Practical Mitigation: Pre-processing (Reweighting), In-processing (Exponentiated Gradient), Post-processing (ThresholdOptimizer)
"""

import os
import sys
import json
import warnings
from pathlib import Path

# Ensure writable matplotlib config
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/Temp"), "mpl_task3")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import shap
from lime import lime_tabular
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds as ReductionEqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

warnings.filterwarnings("ignore")

# Define directories
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
PLOTS_DIR = BASE_DIR / "plots"
OUTPUTS_DIR = BASE_DIR / "outputs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 200
})

print("=" * 70)
print("TASK 3: MODEL FAIRNESS, BIAS AUDIT & EXPLAINABILITY (SHAP / LIME)")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# ---------------------------------------------------------
print("\n[Step 1/6] Loading and cleaning dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns.")

# Clean TotalCharges: whitespace to NaN, impute with median
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
median_total_charges = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID
data = df.drop(columns=["customerID"])

# Identify features
sensitive_cols = ["SeniorCitizen", "gender", "Dependents", "Partner"]
target_col = "Churn"
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in data.columns if c not in num_cols + [target_col]]

print(f"Numerical features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Sensitive features audited: {sensitive_cols}")

# Train/Test Split (80/20 Stratified)
X = data.drop(columns=[target_col])
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
print(f"Train churn prevalence: {y_train.mean():.1%} | Test churn prevalence: {y_test.mean():.1%}")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
    ]
)

preprocessor.fit(X_train)
cat_encoder = preprocessor.named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

X_train_df = pd.DataFrame(X_train_trans, columns=all_feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_trans, columns=all_feature_names, index=X_test.index)

# ---------------------------------------------------------
# 2. MODEL TRAINING & BASELINE EVALUATION
# ---------------------------------------------------------
print("\n[Step 2/6] Training baseline models (Random Forest & Logistic Regression)...")

rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
rf.fit(X_train_df, y_train)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_df, y_train)

def eval_model(model, X_te, y_te, name="Model"):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "ROC_AUC": roc_auc_score(y_te, y_prob)
    }

metrics_rf = eval_model(rf, X_test_df, y_test, "Random Forest")
metrics_lr = eval_model(lr, X_test_df, y_test, "Logistic Regression")

baseline_metrics_df = pd.DataFrame([metrics_rf, metrics_lr])
baseline_metrics_df.to_csv(OUTPUTS_DIR / "baseline_model_performance.csv", index=False)
print("Baseline Performance:")
print(baseline_metrics_df.to_string(index=False))

# ---------------------------------------------------------
# 3. GLOBAL & LOCAL EXPLAINABILITY (SHAP & LIME)
# ---------------------------------------------------------
print("\n[Step 3/6] Computing SHAP explanations and feature importances...")

# 3A. Random Forest MDI & SHAP
rf_importances = rf.feature_importances_
tree_explainer = shap.TreeExplainer(rf)
shap_values = tree_explainer.shap_values(X_test_df)

# Check shape: for binary classifier, TreeExplainer might return list of [N, F] or [N, F, 2]
if isinstance(shap_values, list):
    shap_vals_churn = shap_values[1]  # positive class: Churn=1
elif len(shap_values.shape) == 3:
    shap_vals_churn = shap_values[:, :, 1]
else:
    shap_vals_churn = shap_values

mean_abs_shap = np.abs(shap_vals_churn).mean(axis=0)

feat_imp_df = pd.DataFrame({
    "Feature": all_feature_names,
    "RF_MDI_Importance": rf_importances,
    "SHAP_Mean_Abs": mean_abs_shap
}).sort_values(by="SHAP_Mean_Abs", ascending=False)

feat_imp_df.to_csv(OUTPUTS_DIR / "feature_importance_comparison.csv", index=False)

# Plot 1: Global Feature Importance Comparison (Top 15)
top_15 = feat_imp_df.head(15).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
axes[0].barh(top_15["Feature"], top_15["RF_MDI_Importance"], color="#3182bd")
axes[0].set_title("Random Forest MDI Importance (Gini)")
axes[0].set_xlabel("Relative Importance")

axes[1].barh(top_15["Feature"], top_15["SHAP_Mean_Abs"], color="#e6550d")
axes[1].set_title("Mean |SHAP Value| (Impact on Churn Prob)")
axes[1].set_xlabel("Mean |SHAP value|")

plt.suptitle("Global Feature Importance Comparison (Top 15 Features)", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_global_feature_importance.png", bbox_inches="tight")
plt.close()
print("Saved: plots/01_global_feature_importance.png")

# Plot 2: SHAP Beeswarm Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals_churn, X_test_df, feature_names=all_feature_names, max_display=15, show=False)
plt.title("SHAP Beeswarm Summary Plot (Class: Churn = Yes)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_shap_beeswarm.png", bbox_inches="tight")
plt.close()
print("Saved: plots/02_shap_beeswarm.png")

# Plot 3: SHAP Dependence Plot for Tenure
fig, ax = plt.subplots(figsize=(8, 6))
shap.dependence_plot(
    "tenure", shap_vals_churn, X_test_df,
    feature_names=all_feature_names,
    interaction_index="MonthlyCharges",
    show=False, ax=ax
)
ax.set_title("SHAP Dependence Plot: Tenure vs MonthlyCharges Interaction", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_shap_dependence_tenure.png", bbox_inches="tight")
plt.close()
print("Saved: plots/03_shap_dependence_tenure.png")

# 3B. Persona Selection for Local Explanations
# Find representative personas in the test set
test_preds = rf.predict(X_test_df)
test_probs = rf.predict_proba(X_test_df)[:, 1]

# Persona A: High-risk customer (Actual churn = 1, High predicted churn probability)
idx_high_risk = np.where((y_test.values == 1) & (test_probs > 0.80))[0][0]

# Persona B: Low-risk customer (Actual churn = 0, Low predicted churn probability)
idx_low_risk = np.where((y_test.values == 0) & (test_probs < 0.15))[0][0]

# Persona C: Borderline customer (Predicted churn probability close to 0.50)
idx_borderline = np.argmin(np.abs(test_probs - 0.50))

persona_indices = {
    "Persona_A_High_Risk": idx_high_risk,
    "Persona_B_Low_Risk": idx_low_risk,
    "Persona_C_Borderline": idx_borderline
}

# Base expected value for SHAP
if isinstance(tree_explainer.expected_value, (list, np.ndarray)):
    base_val = tree_explainer.expected_value[1]
else:
    base_val = tree_explainer.expected_value

# Helper to plot SHAP waterfall-style horizontal bar chart
def plot_local_shap_waterfall(sample_idx, persona_name, save_filename):
    sample_shap = shap_vals_churn[sample_idx]
    sample_feats = X_test_df.iloc[sample_idx]
    pred_prob = test_probs[sample_idx]
    actual_label = y_test.iloc[sample_idx]
    
    # Sort top 10 contributing features by absolute SHAP value
    top_indices = np.argsort(np.abs(sample_shap))[-10:]
    top_features = [all_feature_names[i] for i in top_indices]
    top_values = sample_shap[top_indices]
    feature_display = [f"{all_feature_names[i]} = {sample_feats.iloc[i]:.2f}" for i in top_indices]
    
    colors = ["#d95f02" if val > 0 else "#2ca02c" for val in top_values]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feature_display, top_values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
    
    ax.set_title(
        f"Local SHAP Explanation: {persona_name}\n"
        f"Predicted Churn Prob: {pred_prob:.1%} | Ground Truth: {'Churned (1)' if actual_label==1 else 'Retained (0)'} | Base Value: {base_val:.2f}",
        fontsize=12, pad=12
    )
    ax.set_xlabel("SHAP Value (Contribution to Churn Log-Odds)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / save_filename, bbox_inches="tight")
    plt.close()

plot_local_shap_waterfall(idx_high_risk, "High Churn Risk Customer (Persona A)", "04_shap_waterfall_persona_A.png")
plot_local_shap_waterfall(idx_low_risk, "Low Churn Risk Customer (Persona B)", "05_shap_waterfall_persona_B.png")
plot_local_shap_waterfall(idx_borderline, "Borderline Ambiguous Customer (Persona C)", "06_shap_waterfall_persona_C.png")
print("Saved: plots/04, 05, 06 SHAP local explanation waterfall plots.")

# 3C. LIME Local Explanations
print("Computing LIME local explanations...")
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_df.values,
    feature_names=all_feature_names,
    class_names=["Retained", "Churn"],
    mode="classification",
    random_state=42
)

# Explain Persona A with LIME
lime_exp_A = lime_explainer.explain_instance(
    data_row=X_test_df.iloc[idx_high_risk].values,
    predict_fn=rf.predict_proba,
    num_features=10
)

# Plot LIME explanation for Persona A
lime_list_A = lime_exp_A.as_list()
lime_feats, lime_vals = zip(*reversed(lime_list_A))
lime_colors = ["#d95f02" if v > 0 else "#2ca02c" for v in lime_vals]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(lime_feats, lime_vals, color=lime_colors, edgecolor="black", linewidth=0.6)
ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
ax.set_title(
    f"LIME Local Explanation: Persona A (High Churn Risk)\nPredicted Churn Probability: {test_probs[idx_high_risk]:.1%}",
    fontsize=13, pad=12
)
ax.set_xlabel("LIME Feature Weight (Contribution to Churn)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_lime_explanation_persona_A.png", bbox_inches="tight")
plt.close()
print("Saved: plots/07_lime_explanation_persona_A.png")

# Plot 8: Side-by-side LIME vs SHAP for Persona A
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# LIME plot
axes[0].barh(lime_feats, lime_vals, color=lime_colors, edgecolor="black", linewidth=0.6)
axes[0].axvline(0, color="grey", linestyle="--", alpha=0.7)
axes[0].set_title("LIME Feature Attributions (Local Surrogate Linear Model)")
axes[0].set_xlabel("LIME Weight")

# SHAP plot
top_sh_idx = np.argsort(np.abs(shap_vals_churn[idx_high_risk]))[-10:]
sh_names = [all_feature_names[i] for i in top_sh_idx]
sh_vals = shap_vals_churn[idx_high_risk][top_sh_idx]
sh_colors = ["#d95f02" if v > 0 else "#2ca02c" for v in sh_vals]

axes[1].barh(sh_names, sh_vals, color=sh_colors, edgecolor="black", linewidth=0.6)
axes[1].axvline(0, color="grey", linestyle="--", alpha=0.7)
axes[1].set_title("TreeSHAP Feature Attributions (Exact Shapley Values)")
axes[1].set_xlabel("SHAP Value")

plt.suptitle("Explainability Consistency: LIME vs TreeSHAP for High-Risk Customer", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_lime_explanation_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: plots/08_lime_explanation_comparison.png")

# Save Persona profile details for report
persona_records = []
for p_name, p_idx in persona_indices.items():
    raw_row = X_test.iloc[p_idx].to_dict()
    raw_row["Persona"] = p_name
    raw_row["Actual_Churn"] = int(y_test.iloc[p_idx])
    raw_row["RF_Pred_Churn"] = int(test_preds[p_idx])
    raw_row["RF_Churn_Prob"] = round(float(test_probs[p_idx]), 3)
    persona_records.append(raw_row)

pd.DataFrame(persona_records).to_csv(OUTPUTS_DIR / "representative_personas.csv", index=False)

# ---------------------------------------------------------
# 4. FAIRNESS & BIAS AUDIT ACROSS SENSITIVE GROUPS
# ---------------------------------------------------------
print("\n[Step 4/6] Conducting systematic Fairness & Bias Audit...")

def compute_group_fairness(y_true, y_pred, sensitive_series, group_col_name):
    groups = np.unique(sensitive_series)
    records = []
    
    for g in groups:
        mask = (sensitive_series == g)
        y_t_g = y_true[mask]
        y_p_g = y_pred[mask]
        
        n_total = len(y_t_g)
        n_pos_pred = np.sum(y_p_g == 1)
        selection_rate = n_pos_pred / n_total if n_total > 0 else 0
        
        base_rate = np.mean(y_t_g)
        acc = accuracy_score(y_t_g, y_p_g)
        
        cm = confusion_matrix(y_t_g, y_p_g, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Equal Opportunity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Fallout
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_t_g, y_p_g, zero_division=0)
        
        records.append({
            "Sensitive_Attribute": group_col_name,
            "Group": str(g),
            "Sample_Count": n_total,
            "Base_Rate": base_rate,
            "Selection_Rate": selection_rate,
            "Accuracy": acc,
            "Precision": prec,
            "TPR_Recall": tpr,
            "FPR": fpr,
            "FNR": fnr,
            "F1_Score": f1
        })
    
    df_res = pd.DataFrame(records)
    # Calculate disparities (Group max - Group min)
    dp_diff = df_res["Selection_Rate"].max() - df_res["Selection_Rate"].min()
    eo_diff = df_res["TPR_Recall"].max() - df_res["TPR_Recall"].min()
    fpr_diff = df_res["FPR"].max() - df_res["FPR"].min()
    eq_odds_diff = max(eo_diff, fpr_diff)
    disp_impact = df_res["Selection_Rate"].min() / df_res["Selection_Rate"].max() if df_res["Selection_Rate"].max() > 0 else 1.0
    
    summary = {
        "Attribute": group_col_name,
        "Demographic_Parity_Difference": dp_diff,
        "Disparate_Impact_Ratio": disp_impact,
        "Equal_Opportunity_Difference": eo_diff,
        "Equalized_Odds_Difference": eq_odds_diff
    }
    return df_res, summary

# Audit across SeniorCitizen, gender, and Dependents
audit_dfs = []
audit_summaries = []

for s_col in ["SeniorCitizen", "gender", "Dependents"]:
    g_df, g_sum = compute_group_fairness(
        y_test.values,
        test_preds,
        X_test[s_col].values,
        s_col
    )
    audit_dfs.append(g_df)
    audit_summaries.append(g_sum)

full_audit_df = pd.concat(audit_dfs, ignore_index=True)
full_audit_df.to_csv(OUTPUTS_DIR / "fairness_audit_detailed_groups.csv", index=False)

disparities_df = pd.DataFrame(audit_summaries)
disparities_df.to_csv(OUTPUTS_DIR / "fairness_disparities_summary.csv", index=False)

print("\n--- FAIRNESS DISPARITY SUMMARY (Random Forest Baseline) ---")
print(disparities_df.to_string(index=False))

print("\nDetailed Group Metrics:")
print(full_audit_df[["Sensitive_Attribute", "Group", "Sample_Count", "Base_Rate", "Selection_Rate", "Accuracy", "TPR_Recall", "FPR", "F1_Score"]].to_string(index=False))

# Plot 9: Group Fairness Metrics Bar Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Selection Rate (Demographic Parity)
sns.barplot(data=full_audit_df, x="Sensitive_Attribute", y="Selection_Rate", hue="Group", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Selection Rate by Protected Group (Demographic Parity Check)")
axes[0, 0].set_ylabel("Predicted Churn Rate (P(Y_hat=1))")

# Subplot 2: TPR / Recall (Equal Opportunity)
sns.barplot(data=full_audit_df, x="Sensitive_Attribute", y="TPR_Recall", hue="Group", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("True Positive Rate (Equal Opportunity Check)")
axes[0, 1].set_ylabel("TPR (Recall)")

# Subplot 3: FPR (Fallout / Predictive Equality)
sns.barplot(data=full_audit_df, x="Sensitive_Attribute", y="FPR", hue="Group", ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("False Positive Rate (Predictive Equality Check)")
axes[1, 0].set_ylabel("False Positive Rate")

# Subplot 4: Accuracy Parity
sns.barplot(data=full_audit_df, x="Sensitive_Attribute", y="Accuracy", hue="Group", ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Overall Accuracy Across Groups")
axes[1, 1].set_ylabel("Accuracy")

plt.suptitle("Bias & Fairness Audit Across Sensitive Attributes (SeniorCitizen, Gender, Dependents)", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_fairness_metrics_by_group.png", bbox_inches="tight")
plt.close()
print("Saved: plots/09_fairness_metrics_by_group.png")

# Plot 10: Confusion Matrices by Sensitive Group (SeniorCitizen: 0 vs 1)
cm_non_senior = confusion_matrix(y_test[X_test["SeniorCitizen"] == 0], test_preds[X_test["SeniorCitizen"] == 0])
cm_senior = confusion_matrix(y_test[X_test["SeniorCitizen"] == 1], test_preds[X_test["SeniorCitizen"] == 1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_non_senior, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=["Pred: Retain", "Pred: Churn"], yticklabels=["True: Retain", "True: Churn"])
axes[0].set_title(f"Non-Senior Citizens (SeniorCitizen=0)\nN = {np.sum(cm_non_senior)}")

sns.heatmap(cm_senior, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[1],
            xticklabels=["Pred: Retain", "Pred: Churn"], yticklabels=["True: Retain", "True: Churn"])
axes[1].set_title(f"Senior Citizens (SeniorCitizen=1)\nN = {np.sum(cm_senior)}")

plt.suptitle("Confusion Matrix Disparity: Non-Seniors vs Senior Citizens", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "10_confusion_matrices_by_group.png", bbox_inches="tight")
plt.close()
print("Saved: plots/10_confusion_matrices_by_group.png")

# ---------------------------------------------------------
# 5. BIAS MITIGATION STRATEGIES (PRE, IN, POST PROCESSING)
# ---------------------------------------------------------
print("\n[Step 5/6] Implementing Practical Bias Mitigation Strategies...")

# Primary sensitive attribute for mitigation: SeniorCitizen
A_train = X_train["SeniorCitizen"].values
A_test = X_test["SeniorCitizen"].values

# Strategy 1: Pre-processing Mitigation - Sample Reweighting
# Calculate Fairlearn demographic sample weights to equalize joint distribution
N_train = len(y_train)
p_y = y_train.value_counts(normalize=True).to_dict()
p_a = pd.Series(A_train).value_counts(normalize=True).to_dict()

sample_weights = np.zeros(N_train)
for i in range(N_train):
    yi = y_train.iloc[i]
    ai = A_train[i]
    joint_p = np.mean((y_train.values == yi) & (A_train == ai))
    expected_p = p_y[yi] * p_a[ai]
    sample_weights[i] = expected_p / joint_p if joint_p > 0 else 1.0

rf_reweighted = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf_reweighted.fit(X_train_df, y_train, sample_weight=sample_weights)
preds_reweighted = rf_reweighted.predict(X_test_df)
probs_reweighted = rf_reweighted.predict_proba(X_test_df)[:, 1]

# Strategy 2: In-processing Mitigation - Exponentiated Gradient Reduction (Equalized Odds)
print("Fitting In-processing Exponentiated Gradient...")
base_estimator = LogisticRegression(max_iter=1000, random_state=42)
exp_grad = ExponentiatedGradient(
    estimator=base_estimator,
    constraints=ReductionEqualizedOdds(),
    eps=0.015,
    max_iter=50
)
exp_grad.fit(X_train_df, y_train, sensitive_features=A_train)
preds_exp_grad = exp_grad.predict(X_test_df)

# Strategy 3: Post-processing Mitigation - ThresholdOptimizer (Equalized Odds)
print("Fitting Post-processing ThresholdOptimizer...")
thresh_opt = ThresholdOptimizer(
    estimator=rf,
    constraints="equalized_odds",
    objective="balanced_accuracy_score",
    prefit=True,
    predict_method="predict_proba"
)
thresh_opt.fit(X_train_df, y_train, sensitive_features=A_train)
preds_thresh_opt = thresh_opt.predict(X_test_df, sensitive_features=A_test)

# Compare all mitigation methods
mitigation_models = {
    "Baseline (Random Forest)": (test_preds, test_probs),
    "Pre-processing (Reweighting)": (preds_reweighted, probs_reweighted),
    "In-processing (Exponentiated Gradient)": (preds_exp_grad, None),
    "Post-processing (ThresholdOptimizer)": (preds_thresh_opt, None)
}

comparison_rows = []
for m_name, (preds, probs) in mitigation_models.items():
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if probs is not None else np.nan
    
    # Fairness for SeniorCitizen
    mask_0 = (A_test == 0)
    mask_1 = (A_test == 1)
    
    sr_0 = np.mean(preds[mask_0])
    sr_1 = np.mean(preds[mask_1])
    dp_diff = abs(sr_1 - sr_0)
    
    cm_0 = confusion_matrix(y_test[mask_0], preds[mask_0], labels=[0, 1])
    cm_1 = confusion_matrix(y_test[mask_1], preds[mask_1], labels=[0, 1])
    
    tpr_0 = cm_0[1, 1] / (cm_0[1, 1] + cm_0[1, 0])
    tpr_1 = cm_1[1, 1] / (cm_1[1, 1] + cm_1[1, 0])
    eo_diff = abs(tpr_1 - tpr_0)
    
    fpr_0 = cm_0[0, 1] / (cm_0[0, 1] + cm_0[0, 0])
    fpr_1 = cm_1[0, 1] / (cm_1[0, 1] + cm_1[0, 0])
    fpr_diff = abs(fpr_1 - fpr_0)
    eq_odds_diff = max(eo_diff, fpr_diff)
    
    comparison_rows.append({
        "Model_Configuration": m_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Demographic_Parity_Diff": dp_diff,
        "Equal_Opportunity_Diff": eo_diff,
        "Equalized_Odds_Diff": eq_odds_diff
    })

mitigation_comparison_df = pd.DataFrame(comparison_rows)
mitigation_comparison_df.to_csv(OUTPUTS_DIR / "mitigation_comparison_scorecard.csv", index=False)

print("\n--- MITIGATION STRATEGIES SCORECARD ---")
print(mitigation_comparison_df.to_string(index=False))

# Plot 11: Mitigation Fairness vs Performance Pareto Trade-off
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(comparison_rows))
width = 0.35

rects1 = ax1.bar(x - width/2, mitigation_comparison_df["Accuracy"], width, label="Accuracy", color="#2b5c8f")
rects2 = ax1.bar(x + width/2, mitigation_comparison_df["F1_Score"], width, label="F1 Score", color="#41b6c4")

ax1.set_ylabel("Performance Metrics")
ax1.set_title("Performance vs Fairness Across Mitigation Strategies", fontsize=14, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(mitigation_comparison_df["Model_Configuration"], rotation=15, ha="right")
ax1.set_ylim(0, 1.0)

ax2 = ax1.twinx()
ax2.plot(x, mitigation_comparison_df["Equalized_Odds_Diff"], color="#e41a1c", marker="o", linewidth=2.5, label="Equalized Odds Diff (Fairness Gap)")
ax2.plot(x, mitigation_comparison_df["Demographic_Parity_Diff"], color="#ff7f00", marker="s", linestyle="--", linewidth=2.0, label="Demographic Parity Diff")
ax2.set_ylabel("Fairness Gap (Lower is Fairer)", color="#e41a1c")
ax2.tick_params(axis="y", labelcolor="#e41a1c")
ax2.set_ylim(0, 0.45)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "11_mitigation_fairness_vs_performance.png", bbox_inches="tight")
plt.close()
print("Saved: plots/11_mitigation_fairness_vs_performance.png")

# Plot 12: Group TPR and Selection Rate Before vs After Mitigation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models_plot = ["Baseline (Random Forest)", "Pre-processing (Reweighting)", "In-processing (Exponentiated Gradient)", "Post-processing (ThresholdOptimizer)"]
tpr_non_seniors = []
tpr_seniors = []
sr_non_seniors = []
sr_seniors = []

for m_name in models_plot:
    preds = mitigation_models[m_name][0]
    cm_0 = confusion_matrix(y_test[mask_0], preds[mask_0], labels=[0, 1])
    cm_1 = confusion_matrix(y_test[mask_1], preds[mask_1], labels=[0, 1])
    
    tpr_0 = cm_0[1, 1] / (cm_0[1, 1] + cm_0[1, 0])
    tpr_1 = cm_1[1, 1] / (cm_1[1, 1] + cm_1[1, 0])
    tpr_non_seniors.append(tpr_0)
    tpr_seniors.append(tpr_1)
    
    sr_non_seniors.append(np.mean(preds[mask_0]))
    sr_seniors.append(np.mean(preds[mask_1]))

x_idx = np.arange(len(models_plot))
w = 0.35

# TPR parity
axes[0].bar(x_idx - w/2, tpr_non_seniors, w, label="Non-Seniors (A=0)", color="#74add1")
axes[0].bar(x_idx + w/2, tpr_seniors, w, label="Seniors (A=1)", color="#fdae61")
axes[0].set_title("True Positive Rate (Recall) Parity")
axes[0].set_ylabel("TPR")
axes[0].set_xticks(x_idx)
axes[0].set_xticklabels(["Baseline", "Reweighting", "In-Process (ExpGrad)", "Post-Process (ThreshOpt)"], rotation=15, ha="right")
axes[0].legend()
axes[0].set_ylim(0, 1.0)

# Selection Rate parity
axes[1].bar(x_idx - w/2, sr_non_seniors, w, label="Non-Seniors (A=0)", color="#74add1")
axes[1].bar(x_idx + w/2, sr_seniors, w, label="Seniors (A=1)", color="#fdae61")
axes[1].set_title("Selection Rate (Predicted Churn Rate) Parity")
axes[1].set_ylabel("Selection Rate")
axes[1].set_xticks(x_idx)
axes[1].set_xticklabels(["Baseline", "Reweighting", "In-Process (ExpGrad)", "Post-Process (ThreshOpt)"], rotation=15, ha="right")
axes[1].legend()
axes[1].set_ylim(0, 1.0)

plt.suptitle("Equal Opportunity & Selection Rate Convergence Across Mitigation Techniques", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "12_mitigation_group_metrics_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: plots/12_mitigation_group_metrics_comparison.png")

# ---------------------------------------------------------
# 6. SAVE COMPLETE RESULTS METADATA JSON
# ---------------------------------------------------------
summary_json = {
    "project_name": "Task 3: Model Fairness, Bias & Explainability Analysis",
    "dataset": "IBM Telco Customer Churn",
    "total_samples": int(df.shape[0]),
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "baseline_models": baseline_metrics_df.to_dict(orient="records"),
    "top_5_shap_features": feat_imp_df.head(5)["Feature"].tolist(),
    "sensitive_attributes_audited": sensitive_cols,
    "disparity_summary": disparities_df.to_dict(orient="records"),
    "mitigation_scorecard": mitigation_comparison_df.to_dict(orient="records")
}

with open(OUTPUTS_DIR / "analysis_results_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

print("\n[Step 6/6] Pipeline execution completed successfully!")
print(f"All 12 plots exported to: {PLOTS_DIR}")
print(f"All metrics tables exported to: {OUTPUTS_DIR}")
print("=" * 70)
