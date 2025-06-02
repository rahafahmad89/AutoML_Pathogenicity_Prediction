# ✅  H2O Phase 2 Script with Statistical + Correlation-based Feature Selection

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import h2o
import pandas as pd  

print("✅ pd module is:", pd)
print("✅ pd type is:", type(pd)) 

from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, roc_curve, precision_recall_curve, confusion_matrix
)
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance



# Init H2O
h2o.init()
shap.initjs()

# Paths
BASE_DIR = "/AutoML"
STAT_FEATURES_PATH = os.path.join("/statistical_tests_all_datasets.csv")
DATASET_PATHS = {
    "Dataset-1": os.path.join(BASE_DIR, "Filename"),
    "Dataset-2": os.path.join(BASE_DIR, "Filename"),
#Add datasets if you need
}
BEST_SEED_FILE = os.path.join(BASE_DIR, "h2o", "results", "best_seeds.json")
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "h2o", "results", "final")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# === Feature Selection ===
def select_features_from_stats_and_correlation(df, dataset_name, stats_path, p_thresh=0.05, corr_thresh=0.9):
    stats_df = pd.read_csv(stats_path)
    p_col = 'Kruskal-Wallis p-value' if 'Kruskal-Wallis p-value' in stats_df.columns else 'ANOVA p-value'
    selected = stats_df[stats_df[p_col] < p_thresh]
    selected_features = selected['Feature'].tolist()
    df_selected = df[[col for col in selected_features if col in df.columns]].copy()
    corr_matrix = df_selected.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_thresh)]
    df_selected.drop(columns=to_drop, inplace=True, errors='ignore')

    if not df_selected.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_selected.corr(), cmap='coolwarm')
        plt.title(f"{dataset_name} Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"{dataset_name}_correlation_heatmap.png"), dpi=300)
        plt.close()

    return df_selected

# === Bootstrap CI ===
def compute_metric_confidence_intervals(y_true, y_prob, n_bootstrap=1000):
    metrics = {"AUC": [], "Precision": [], "Recall": [], "F1": [], "Kappa": [], "MCC": []}
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        yp_bin = (yp > 0.5).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        metrics["AUC"].append(roc_auc_score(yt, yp))
        metrics["Precision"].append(precision_score(yt, yp_bin))
        metrics["Recall"].append(recall_score(yt, yp_bin))
        metrics["F1"].append(f1_score(yt, yp_bin))
        metrics["Kappa"].append(cohen_kappa_score(yt, yp_bin))
        metrics["MCC"].append(matthews_corrcoef(yt, yp_bin))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in metrics.items()}

# === Evaluation Function ===
def evaluate_dataset(name, path, seed):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['CLIN_SIG'] = le.fit_transform(df['CLIN_SIG'])
    drop_cols = [c for c in df.select_dtypes('object').columns if c != '#Uploaded_variation'] + [
        'cDNA_position', 'Location', 'Protein_position', 'CDS_position', '#Uploaded_variation']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    y = df['CLIN_SIG']
    X = df.drop(columns='CLIN_SIG')
    X = select_features_from_stats_and_correlation(X, name, STAT_FEATURES_PATH)
    X['CLIN_SIG'] = y
    y = X['CLIN_SIG']
    X = X.drop(columns='CLIN_SIG')
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)

    train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
    train_h2o['CLIN_SIG'] = train_h2o['CLIN_SIG'].asfactor()

    aml = H2OAutoML(max_runtime_secs=300, seed=seed)
    aml.train(y='CLIN_SIG', training_frame=train_h2o)

    model = aml.leader
    pred = model.predict(test_h2o).as_data_frame()['p1'].values
    pred_bin = (pred > 0.5).astype(int)

    ci = compute_metric_confidence_intervals(y_test.values, pred)
    fpr, tpr, _ = roc_curve(y_test, pred)
    precision, recall, _ = precision_recall_curve(y_test, pred)

    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, pred):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC - {name}"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"ROC_{name}.png")); plt.close()

    plt.figure(); plt.plot(recall, precision)
    plt.title(f"PRC - {name}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"PRC_{name}.png")); plt.close()

    # Save leaderboard
    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard.to_csv(os.path.join(EVAL_OUTPUT_DIR, f"leaderboard_{name}.csv"), index=False)

    # SHAP with surrogate
    surrogate = GradientBoostingClassifier().fit(X_train, y_train)
    explainer_shap = shap.Explainer(surrogate.predict, X_train)
    shap_values = explainer_shap(X_train)
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"shap_beeswarm_{name}.png"), bbox_inches='tight'); plt.close()
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"shap_bar_{name}.png"), bbox_inches='tight'); plt.close()

    # Permutation Importance (with surrogate)
    result = permutation_importance(surrogate, X_test, y_test, n_repeats=30, random_state=seed, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1][:20]
    plt.figure(figsize=(12, 6))
    plt.barh(np.array(X_test.columns)[sorted_idx][::-1], result.importances_mean[sorted_idx][::-1])
    plt.xlabel("Permutation Importance")
    plt.title(f"Permutation Feature Importance - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"perm_importance_{name}.png"), dpi=300)
    plt.close()

    # LIME explanations for TP/TN/FP/FN
    explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=['Benign', 'Pathogenic'], discretize_continuous=True)
    pred_all = pred_bin.tolist()
    y_all = y_test.tolist()
    for (gt, pred_flag), label in zip([(1, 1), (0, 0), (1, 0), (0, 1)], ["TP", "TN", "FP", "FN"]):

        for i, (yi, pi) in enumerate(zip(y_all, pred_all)):
            if yi == gt and pi == pred_flag:

                exp = explainer.explain_instance(X_test.values[i], surrogate.predict_proba, num_features=10)
                exp.save_to_file(os.path.join(EVAL_OUTPUT_DIR, f"lime_{name}_{label}.html"))
                fig = exp.as_pyplot_figure(); fig.tight_layout()
                fig.savefig(os.path.join(EVAL_OUTPUT_DIR, f"lime_{name}_{label}.png"), bbox_inches='tight'); plt.close(fig)
                break

    return {
        "Dataset": name, "Seed": seed,
        "AUC": roc_auc_score(y_test, pred), "AUC CI Lower": ci['AUC'][0], "AUC CI Upper": ci['AUC'][1],
        "Precision": precision_score(y_test, pred_bin), "Precision CI Lower": ci['Precision'][0], "Precision CI Upper": ci['Precision'][1],
        "Recall": recall_score(y_test, pred_bin), "Recall CI Lower": ci['Recall'][0], "Recall CI Upper": ci['Recall'][1],
        "F1": f1_score(y_test, pred_bin), "F1 CI Lower": ci['F1'][0], "F1 CI Upper": ci['F1'][1],
        "Cohen_Kappa": cohen_kappa_score(y_test, pred_bin), "Kappa CI Lower": ci['Kappa'][0], "Kappa CI Upper": ci['Kappa'][1],
        "MCC": matthews_corrcoef(y_test, pred_bin), "MCC CI Lower": ci['MCC'][0], "MCC CI Upper": ci['MCC'][1]
    }, fpr, tpr, ci['AUC']

# === Main Runner ===
with open(BEST_SEED_FILE) as f:
    best_seeds = json.load(f)

summary = []; all_roc = []
shap_summary_all = []; perm_summary_all = []
shap_links = []; lime_links = []
all_roc_pr_data = []
for name, path in DATASET_PATHS.items():
    best_seed = best_seeds.get(name, {}).get("best_seed")
    if best_seed is None:
        print(f"⚠️ No best seed found for {name}, skipping...")
        continue
    print(f"\n🚀 Evaluating {name} with seed {best_seed}...")
    result = evaluate_dataset(name, path, best_seed)
    if len(result) == 4:
        metrics, fpr, tpr, auc_ci = result
        shap_df = pd.DataFrame()
        perm_df = pd.DataFrame()
        model_name = "Unknown"
    else:
        metrics, fpr, tpr, auc_ci, shap_df, perm_df, model_name = result
    summary.append({**metrics, "Model": model_name})
    all_roc.append((name, fpr, tpr, metrics['AUC'], auc_ci))

    shap_summary_all.append(shap_df)
    perm_summary_all.append(perm_df)

    for label in ["TP", "TN", "FP", "FN"]:
        lime_links.append({"Dataset": name, "Type": label, "File": f"lime_{name}_{label}.html"})
        lime_links.append({"Dataset": name, "Type": label, "File": f"lime_{name}_{label}.png"})

    shap_links.append({"Dataset": name, "Type": "SHAP", "File": f"shap_beeswarm_{name}.png"})
    shap_links.append({"Dataset": name, "Type": "SHAP", "File": f"shap_bar_{name}.png"})

summary_df = pd.DataFrame(summary)
summary_df.to_excel(os.path.join(EVAL_OUTPUT_DIR, "summary_metrics.xlsx"), index=False)
pd.DataFrame(lime_links + shap_links).to_csv(os.path.join(EVAL_OUTPUT_DIR, "explanation_links.csv"), index=False)
pd.concat(shap_summary_all).to_csv(os.path.join(EVAL_OUTPUT_DIR, "shap_summary_all.csv"), index=False)
pd.concat(perm_summary_all).to_csv(os.path.join(EVAL_OUTPUT_DIR, "perm_importance_all.csv"), index=False)

plt.figure()
for name, fpr, tpr, auc, auc_ci in all_roc:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f} [{auc_ci[0]:.2f}, {auc_ci[1]:.2f}])")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curve with AUC and CI")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_OUTPUT_DIR, "ROC_Combined.png"))
plt.close()

print(f"✅ Evaluation complete. Results saved in: {EVAL_OUTPUT_DIR}")


# ✅  MLJAR Phase 2 Script

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, roc_curve,
    precision_recall_curve, confusion_matrix
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import shap
from supervised.automl import AutoML

# Paths
BASE_DIR = "/AutoML"
DATASET_PATHS = {
    "Dataset-1": os.path.join(BASE_DIR, "Filename"),
    "Dataset-2": os.path.join(BASE_DIR, "Filename"),
#Add datasets if you need
}
STAT_FEATURES_PATH = os.path.join("/statistical_tests_all_datasets.csv")
BEST_SEED_FILE = os.path.join(BASE_DIR, "mljar", "results", "best_seeds.json")
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "mljar", "results", "final")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# Feature selection
def select_features(df, dataset_name, stats_path, p_thresh=0.05, corr_thresh=0.9):
    stats_df = pd.read_csv(stats_path)
    p_col = 'Kruskal-Wallis p-value' if 'Kruskal-Wallis p-value' in stats_df.columns else 'ANOVA p-value'
    selected = stats_df[stats_df[p_col] < p_thresh]
    selected_features = selected['Feature'].tolist()
    df_selected = df[[col for col in selected_features if col in df.columns]].copy()
    corr_matrix = df_selected.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_thresh)]
    df_selected.drop(columns=to_drop, inplace=True, errors='ignore')
    return df_selected

# Bootstrap CI
def compute_ci(y_true, y_prob, n_bootstrap=1000):
    metrics = {"AUC": [], "Precision": [], "Recall": [], "F1": [], "Kappa": [], "MCC": []}
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        yp_bin = (yp > 0.5).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        metrics["AUC"].append(roc_auc_score(yt, yp))
        metrics["Precision"].append(precision_score(yt, yp_bin))
        metrics["Recall"].append(recall_score(yt, yp_bin))
        metrics["F1"].append(f1_score(yt, yp_bin))
        metrics["Kappa"].append(cohen_kappa_score(yt, yp_bin))
        metrics["MCC"].append(matthews_corrcoef(yt, yp_bin))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in metrics.items()}

# Evaluation function
def evaluate_dataset(name, path, seed):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['CLIN_SIG'] = le.fit_transform(df['CLIN_SIG'])
    drop_cols = [c for c in df.select_dtypes('object').columns if c != '#Uploaded_variation'] + [
        'cDNA_position', 'Location', 'Protein_position', 'CDS_position', '#Uploaded_variation']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    y = df['CLIN_SIG']
    X = df.drop(columns='CLIN_SIG')
    X = select_features(X, name, STAT_FEATURES_PATH)
    X['CLIN_SIG'] = y
    y = X['CLIN_SIG']; X = X.drop(columns='CLIN_SIG')
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)

    automl = AutoML(
        algorithms=["Linear", "Decision Tree"],
        mode="Compete",
        eval_metric="auc",
        total_time_limit=300,
        random_state=seed,
        train_ensemble=True,
        explain_level=1,
        n_jobs=1,
        results_path=f"/content/mljar_final_{name}_{seed}"
    )
    automl.fit(X_train, y_train)
    preds = automl.predict_proba(X_test)[:, 1]
    pred_bin = (preds > 0.5).astype(int)
    ci = compute_ci(y_test.values, preds)

    fpr, tpr, _ = roc_curve(y_test, preds)
    precision, recall, _ = precision_recall_curve(y_test, preds)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, preds):.2f}"); plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC - {name}"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"ROC_{name}.png")); plt.close()

    plt.figure(); plt.plot(recall, precision)
    plt.title(f"PRC - {name}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"PRC_{name}.png")); plt.close()

    surrogate = GradientBoostingClassifier().fit(X_train, y_train)
    explainer = shap.Explainer(surrogate.predict, X_train)
    shap_values = explainer(X_train)
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"shap_beeswarm_{name}.png"), bbox_inches='tight'); plt.close()
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"shap_bar_{name}.png"), bbox_inches='tight'); plt.close()

    result = permutation_importance(surrogate, X_test, y_test, n_repeats=30, random_state=seed, n_jobs=1)
    sorted_idx = result.importances_mean.argsort()[::-1][:20]
    plt.figure(figsize=(12, 6))
    plt.barh(np.array(X_test.columns)[sorted_idx][::-1], result.importances_mean[sorted_idx][::-1])
    plt.xlabel("Permutation Importance"); plt.title(f"PMI - {name}"); plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT_DIR, f"perm_importance_{name}.png"), dpi=300); plt.close()

    lime_exp = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=['Benign', 'Pathogenic'], discretize_continuous=True)
    pred_all = pred_bin.tolist(); y_all = y_test.tolist()
    for (gt, pred_flag), label in zip([(1, 1), (0, 0), (1, 0), (0, 1)], ["TP", "TN", "FP", "FN"]):
        for i, (yt, yp) in enumerate(zip(y_all, pred_all)):
            if yt == gt and yp == pred_flag:
                exp = lime_exp.explain_instance(X_test.values[i], surrogate.predict_proba, num_features=10)
                fig = exp.as_pyplot_figure(); fig.tight_layout()
                fig.savefig(os.path.join(EVAL_OUTPUT_DIR, f"lime_{name}_{label}.png"), bbox_inches='tight'); plt.close(fig)
                break

    return {
        "Dataset": name, "Seed": seed,
        "AUC": roc_auc_score(y_test, preds), "AUC CI Lower": ci['AUC'][0], "AUC CI Upper": ci['AUC'][1],
        "Precision": precision_score(y_test, pred_bin), "Precision CI Lower": ci['Precision'][0], "Precision CI Upper": ci['Precision'][1],
        "Recall": recall_score(y_test, pred_bin), "Recall CI Lower": ci['Recall'][0], "Recall CI Upper": ci['Recall'][1],
        "F1": f1_score(y_test, pred_bin), "F1 CI Lower": ci['F1'][0], "F1 CI Upper": ci['F1'][1],
        "Cohen_Kappa": cohen_kappa_score(y_test, pred_bin), "Kappa CI Lower": ci['Kappa'][0], "Kappa CI Upper": ci['Kappa'][1],
        "MCC": matthews_corrcoef(y_test, pred_bin), "MCC CI Lower": ci['MCC'][0], "MCC CI Upper": ci['MCC'][1]
    }

# Run Phase 2
with open(BEST_SEED_FILE) as f:
    best_seeds = json.load(f)

summary = []
for name, path in DATASET_PATHS.items():
    best_seed = best_seeds.get(name, {}).get("best_seed")
    if best_seed is None:
        print(f"⚠️ No best seed found for {name}, skipping...")
        continue
    print(f"\n🚀 Evaluating {name} with seed {best_seed}...")
    metrics = evaluate_dataset(name, path, best_seed)
    summary.append(metrics)

summary_df = pd.DataFrame(summary)
summary_df.to_excel(os.path.join(EVAL_OUTPUT_DIR, "summary_metrics.xlsx"), index=False)
# === Combined ROC Plot with AUC and CI ===
plt.figure()
for result in summary:
    name = result["Dataset"]
    auc = result["AUC"]
    ci_low = result["AUC CI Lower"]
    ci_high = result["AUC CI Upper"]

    # Reload predictions for ROC
    df = pd.read_csv(DATASET_PATHS[name])
    le = LabelEncoder()
    df['CLIN_SIG'] = le.fit_transform(df['CLIN_SIG'])
    drop_cols = [c for c in df.select_dtypes('object').columns if c != '#Uploaded_variation'] + [
        'cDNA_position', 'Location', 'Protein_position', 'CDS_position', '#Uploaded_variation']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    y = df['CLIN_SIG']
    X = df.drop(columns='CLIN_SIG')
    X = select_features(X, name, STAT_FEATURES_PATH)
    X['CLIN_SIG'] = y
    y = X['CLIN_SIG']; X = X.drop(columns='CLIN_SIG')

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=result["Seed"])

    surrogate = GradientBoostingClassifier().fit(X_train, y_train)
    preds = surrogate.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, preds)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f} [{ci_low:.2f}, {ci_high:.2f}])")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curve with AUC and 95% CI")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_OUTPUT_DIR, "ROC_Combined.png"), dpi=300)
plt.close()

print(f"\n✅ Evaluation complete. Results saved to {EVAL_OUTPUT_DIR}")


# ✅  TPOT Phase 2 Script

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.inspection import permutation_importance
import shap
from lime.lime_tabular import LimeTabularExplainer
from tpot import TPOTClassifier
import seaborn as sns
import joblib

# ✅ Set paths
BASE_DIR = "/AutoML"
STAT_FEATURES_PATH = "/statistical_tests_all_datasets.csv"
DATASET_PATHS = {
    "Dataset-1": os.path.join(BASE_DIR, "Filename"),
    "Dataset-2": os.path.join(BASE_DIR, "Filename"),
#Add datasets if you need
}
BEST_SEED_FILE = os.path.join(BASE_DIR, "tpot", "results", "best_seeds.json")
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "tpot", "results", "final")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


# ✅ Feature selection + heatmap

def select_features_from_stats_and_correlation(df, dataset_name, stats_path, p_thresh=0.05, corr_thresh=0.9, output_dir="./"):
    stats_df = pd.read_csv(stats_path)

    if 'Feature' not in stats_df.columns:
        raise ValueError("'Feature' column is missing in the stats file.")

    # Pick best p-value column available
    p_col = 'Kruskal-Wallis p-value' if 'Kruskal-Wallis p-value' in stats_df.columns else 'ANOVA p-value'

    # Select features globally
    selected = stats_df[stats_df[p_col] < p_thresh]
    selected_features = selected['Feature'].tolist()

    df_selected = df[[col for col in selected_features if col in df.columns]].copy()

    # Drop highly correlated ones
    corr_matrix = df_selected.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_thresh)]
    df_selected.drop(columns=to_drop, inplace=True, errors='ignore')

    if not df_selected.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_selected.corr(), cmap='coolwarm', annot=False)
        heatmap_path = os.path.join(output_dir, f"{dataset_name}_correlation_heatmap.png")
        plt.title(f"{dataset_name} Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"✅ Correlation heatmap saved: {heatmap_path}")
    else:
        print(f"⚠️ No selected features found after filtering for {dataset_name}.")

    return df_selected

# Confidence intervals

def compute_metric_confidence_intervals(y_true, y_prob, n_bootstrap=1000):
    metrics = {"AUC": [], "Precision": [], "Recall": [], "F1": [], "Kappa": [], "MCC": []}
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        yp_bin = (yp > 0.5).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        metrics["AUC"].append(roc_auc_score(yt, yp))
        metrics["Precision"].append(precision_score(yt, yp_bin))
        metrics["Recall"].append(recall_score(yt, yp_bin))
        metrics["F1"].append(f1_score(yt, yp_bin))
        metrics["Kappa"].append(cohen_kappa_score(yt, yp_bin))
        metrics["MCC"].append(matthews_corrcoef(yt, yp_bin))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in metrics.items()}

# Curves

def plot_curves(y_test, y_prob, name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC - {name}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right"); plt.savefig(f"{EVAL_OUTPUT_DIR}/ROC_{name}.png"); plt.close()

    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"PRC - {name}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(f"{EVAL_OUTPUT_DIR}/PRC_{name}.png"); plt.close()
    return fpr, tpr, auc

# Permutation importance

def plot_permutation_importance(model, X_test, y_test, name):
    r = permutation_importance(model, X_test, y_test, n_repeats=30)
    df = pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(f"{EVAL_OUTPUT_DIR}/perm_importance_{name}.csv", index=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=df.head(15))
    plt.title(f"Top 15 Permutation Importances - {name}")
    plt.tight_layout()
    plt.savefig(f"{EVAL_OUTPUT_DIR}/perm_importance_{name}.png",  bbox_inches='tight')
    plt.close()

# SHAP

def plot_shap(model, X_train, name):
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)

    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.savefig(f"{EVAL_OUTPUT_DIR}/shap_beeswarm_{name}.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.savefig(f"{EVAL_OUTPUT_DIR}/shap_bar_{name}.png", bbox_inches='tight')
    plt.close()

    force_plot = shap.plots.force(shap_values[0], matplotlib=False)
    shap.save_html(f"{EVAL_OUTPUT_DIR}/shap_force_{name}.html", force_plot)

# LIME

def plot_lime(model, X_train, X_test, y_test, name):
    pred = model.predict(X_test)
    categories = [(1, 1), (0, 0), (1, 0), (0, 1)]
    names = ["TP", "TN", "FP", "FN"]
    explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=['Benign', 'Pathogenic'], discretize_continuous=True)
    for (label, pred_val), desc in zip(categories, names):
        for i in range(len(y_test)):
            if y_test.iloc[i] == label and pred[i] == pred_val:
                exp = explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=10)
                exp.save_to_file(f"{EVAL_OUTPUT_DIR}/lime_{name}_{desc}.html")
                try:
                    fig = exp.as_pyplot_figure(); fig.tight_layout()
                    fig.savefig(f"{EVAL_OUTPUT_DIR}/lime_{name}_{desc}.png", bbox_inches='tight')
                    plt.close(fig)
                except: continue
                break

# Evaluation

def evaluate_dataset(dataset_name, path, seed):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['CLIN_SIG'] = le.fit_transform(df['CLIN_SIG'])

    drop_cols = [col for col in df.select_dtypes('object').columns if col != '#Uploaded_variation'] + [
        'cDNA_position', 'Location', 'Protein_position', 'CDS_position', '#Uploaded_variation']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    y = df['CLIN_SIG']
    X = df.drop(columns='CLIN_SIG')
    X = select_features_from_stats_and_correlation(X, dataset_name, STAT_FEATURES_PATH, output_dir=EVAL_OUTPUT_DIR)
    X['CLIN_SIG'] = y

    y = X['CLIN_SIG']
    X = X.drop(columns='CLIN_SIG')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    model = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=seed, max_time_mins=30, max_eval_time_mins=5)  # ⏱️ optional cap per pipeline cv=5

    model.fit(X_train, y_train)

    pipeline_log_path = os.path.join(EVAL_OUTPUT_DIR, f"pipelines_{dataset_name}.txt")
    with open(pipeline_log_path, "w") as f:
        for pipeline in model.evaluated_individuals_.items():
            f.write(str(pipeline) + "\n")
    print(f"📄 Saved all evaluated pipelines: {pipeline_log_path}")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    ci = compute_metric_confidence_intervals(y_test.values, y_prob)
    fpr, tpr, auc = plot_curves(y_test, y_prob, dataset_name)
    plot_permutation_importance(model.fitted_pipeline_, X_test, y_test, dataset_name)
    plot_shap(model.fitted_pipeline_, X_train, dataset_name)
    plot_lime(model.fitted_pipeline_, X_train, X_test, y_test, dataset_name)

    metrics = {
        "Dataset": dataset_name, "Seed": seed,
        "AUC": auc, "AUC CI Lower": ci['AUC'][0], "AUC CI Upper": ci['AUC'][1],
        "Precision": precision_score(y_test, y_pred), "Precision CI Lower": ci['Precision'][0], "Precision CI Upper": ci['Precision'][1],
        "Recall": recall_score(y_test, y_pred), "Recall CI Lower": ci['Recall'][0], "Recall CI Upper": ci['Recall'][1],
        "F1": f1_score(y_test, y_pred), "F1 CI Lower": ci['F1'][0], "F1 CI Upper": ci['F1'][1],
        "Cohen_Kappa": cohen_kappa_score(y_test, y_pred), "Kappa CI Lower": ci['Kappa'][0], "Kappa CI Upper": ci['Kappa'][1],
        "MCC": matthews_corrcoef(y_test, y_pred), "MCC CI Lower": ci['MCC'][0], "MCC CI Upper": ci['MCC'][1]
    }
    return metrics, fpr, tpr, ci['AUC']

# Run all
with open(BEST_SEED_FILE) as f:
    best_seeds = json.load(f)

summary = []
all_roc = {}

for name, path in DATASET_PATHS.items():
    best_seed = best_seeds.get(name, {}).get("best_seed")
    if best_seed is None:
        print(f"⚠️ No best seed found for {name}, skipping...")
        continue
    print(f"\n🚀 Evaluating {name} with seed {best_seed}...")
    metrics, fpr, tpr, auc_ci = evaluate_dataset(name, path, best_seed)
    summary.append(metrics)
    all_roc[name] = (fpr, tpr, metrics['AUC'], auc_ci)

# Save summary
pd.DataFrame(summary).to_excel(os.path.join(EVAL_OUTPUT_DIR, "summary_metrics.xlsx"), index=False)

# Combined ROC
plt.figure()
for name, (fpr, tpr, auc, auc_ci) in all_roc.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f} [{auc_ci[0]:.2f}, {auc_ci[1]:.2f}])")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curve with AUC and CI")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(EVAL_OUTPUT_DIR, "ROC_Combined.png"))
plt.close()

print(f"✅ Evaluation complete. Results saved in: {EVAL_OUTPUT_DIR}")
