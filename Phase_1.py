# ✅ H2o Phase 1: Find Best Seed per Dataset (H2O)
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 📁 Paths
BASE_DIR = "/AutoML"
DATASET_PATHS = {
    "Dataset-1": os.path.join(BASE_DIR, "Filename"),
    "Dataset-2": os.path.join(BASE_DIR, "Filename"),
#Add datasets if you need
}
BEST_SEED_OUTPUT = os.path.join(BASE_DIR, "h2o", "results", "best_seeds.json")
PLOT_OUTPUT_DIR = os.path.join(BASE_DIR, "h2o", "results", "seed_auc_plots")
os.makedirs(os.path.dirname(BEST_SEED_OUTPUT), exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
SEEDS = [42, 101, 202, 303, 404]
h2o.init()

# Load and preprocess

def preprocess_data(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['CLIN_SIG'] = le.fit_transform(df['CLIN_SIG'])
    obj_cols = df.select_dtypes('object').columns.tolist()
    drop_cols = [c for c in obj_cols if c != '#Uploaded_variation'] + [
        'cDNA_position', 'Location', 'Protein_position', 'CDS_position', '#Uploaded_variation']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)
    return df

# Plot per dataset

def plot_seed_auc(dataset_name, auc_log):
    df = pd.DataFrame(auc_log)
    plt.plot(df['seed'], df['auc'], marker='o')
    plt.title(f"Seed-wise AUC - {dataset_name}")
    plt.xlabel("Seed"); plt.ylabel("AUC")
    plt.grid(True); plt.tight_layout()
    path = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_seed_auc.png")
    plt.savefig(path, dpi=300); plt.close()

# Evaluate best seed per dataset

def evaluate_best_seed(dataset_name, file_path, seeds):
    df = preprocess_data(file_path)
    X = df.drop(columns='CLIN_SIG')
    y = df['CLIN_SIG']
    
    best_seed = None; best_auc = -1; auc_log = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)
        train_df = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
        test_df = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
        train_df['CLIN_SIG'] = train_df['CLIN_SIG'].asfactor()

        aml = H2OAutoML(max_runtime_secs=300, seed=seed)
        aml.train(y='CLIN_SIG', training_frame=train_df)
        preds = aml.leader.predict(test_df).as_data_frame()['p1']
        auc = roc_auc_score(y_test, preds)
        auc_log.append({"seed": seed, "auc": auc})
        if auc > best_auc:
            best_auc, best_seed = auc, seed
        print(f"✅ {dataset_name} | Seed {seed} | AUC: {auc:.4f}")

    plot_seed_auc(dataset_name, auc_log)
    return dataset_name, {"best_seed": best_seed, "best_auc": best_auc, "all_seeds": auc_log}

# Run all
best_seeds_dict = {}; all_seed_logs = []
for name, path in DATASET_PATHS.items():
    print(f"\n🔍 Evaluating: {name}")
    ds_name, result = evaluate_best_seed(name, path, SEEDS)
    best_seeds_dict[ds_name] = result
    for entry in result["all_seeds"]:
        entry["Dataset"] = ds_name
        all_seed_logs.append(entry)

# Combined plot
plot_df = pd.DataFrame(all_seed_logs)
plt.figure(figsize=(10, 6))
for ds in plot_df["Dataset"].unique():
    sub = plot_df[plot_df["Dataset"] == ds]
    plt.plot(sub["seed"], sub["auc"], marker='o', label=ds)
plt.title("H2O AUC per Seed Across Datasets")
plt.xlabel("Seed"); plt.ylabel("AUC")
plt.grid(True); plt.legend()
plt.tight_layout()
combined_path = os.path.join(PLOT_OUTPUT_DIR, "combined_auc_per_seed.png")
plt.savefig(combined_path, dpi=300)
plt.close()

# Save JSON
with open(BEST_SEED_OUTPUT, "w") as f:
    json.dump(best_seeds_dict, f, indent=4)
print(f"\n✅ Best seeds saved to: {BEST_SEED_OUTPUT}")


# ✅ MLJAR Phase 1: Find Best Seed per Dataset (MLJAR)

import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from supervised.automl import AutoML

# === PATHS ===
BASE_DIR = "/AutoML"
DATASET_PATHS = {
    "Dataset-1": os.path.join(BASE_DIR, "Filename"),
    "Dataset-2": os.path.join(BASE_DIR, "Filename"),
#Add datasets if you need
}
BEST_SEED_OUTPUT = os.path.join(BASE_DIR, "mljar", "results", "best_seeds.json")
PLOT_OUTPUT_DIR = os.path.join(BASE_DIR, "mljar", "results", "seed_auc_plots")
os.makedirs(os.path.dirname(BEST_SEED_OUTPUT), exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
SEEDS = [42, 101, 202, 303, 404]

# === PREPROCESSING ===
def preprocess_data(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['CLIN_SIG'] = le.fit_transform(df['CLIN_SIG'])
    obj_cols = df.select_dtypes('object').columns.tolist()
    drop_cols = [c for c in obj_cols if c != '#Uploaded_variation'] + [
        'cDNA_position', 'Location', 'Protein_position', 'CDS_position', '#Uploaded_variation']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)
    return df

# === PLOT AUC ===
def plot_seed_auc(dataset_name, auc_log):
    df = pd.DataFrame(auc_log)
    plt.plot(df['seed'], df['auc'], marker='o')
    plt.title(f"Seed-wise AUC - {dataset_name}")
    plt.xlabel("Seed"); plt.ylabel("AUC")
    plt.grid(True); plt.tight_layout()
    path = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_seed_auc.png")
    plt.savefig(path, dpi=300); plt.close()

# === EVALUATE ===
def evaluate_best_seed(dataset_name, file_path, seeds):
    df = preprocess_data(file_path)
    X = df.drop(columns='CLIN_SIG')
    y = df['CLIN_SIG']

    best_seed = None; best_auc = -1; auc_log = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)

        # 📁 Unique results path per seed
        result_path = f"/content/mljar_temp_{dataset_name}_{seed}"
        if os.path.exists(result_path):
            os.system(f"rm -rf {result_path}")  # 🚫 Delete cache

        automl = AutoML(
            mode="Compete",
            eval_metric="auc",
            total_time_limit=300,
            random_state=seed,
            algorithms=["Linear", "Decision Tree"],
            train_ensemble=True,
            explain_level=0,
            n_jobs=1,
            results_path=result_path
        )
        automl.fit(X_train, y_train)
        preds = automl.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        auc_log.append({"seed": seed, "auc": auc})
        if auc > best_auc:
            best_auc, best_seed = auc, seed
        print(f"✅ {dataset_name} | Seed {seed} | AUC: {auc:.4f}")

    plot_seed_auc(dataset_name, auc_log)
    return dataset_name, {"best_seed": best_seed, "best_auc": best_auc, "all_seeds": auc_log}

# === MAIN RUN ===
best_seeds_dict = {}; all_seed_logs = []
for name, path in DATASET_PATHS.items():
    print(f"\n🔍 Evaluating: {name}")
    ds_name, result = evaluate_best_seed(name, path, SEEDS)
    best_seeds_dict[ds_name] = result
    for entry in result["all_seeds"]:
        entry["Dataset"] = ds_name
        all_seed_logs.append(entry)

# === COMBINED PLOT ===
plot_df = pd.DataFrame(all_seed_logs)
plt.figure(figsize=(10, 6))
for ds in plot_df["Dataset"].unique():
    sub = plot_df[plot_df["Dataset"] == ds]
    plt.plot(sub["seed"], sub["auc"], marker='o', label=ds)
plt.title("MLJAR AUC per Seed Across Datasets")
plt.xlabel("Seed"); plt.ylabel("AUC")
plt.grid(True); plt.legend()
plt.tight_layout()
combined_path = os.path.join(PLOT_OUTPUT_DIR, "combined_auc_per_seed.png")
plt.savefig(combined_path, dpi=300); plt.close()

# === SAVE BEST SEEDS ===
with open(BEST_SEED_OUTPUT, "w") as f:
    json.dump(best_seeds_dict, f, indent=4)
print(f"\n✅ Best seeds saved to: {BEST_SEED_OUTPUT}")


# ✅ TPOT Phase 1: Find Best Seed per Dataset (TPOT)

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier

# 📁 Paths
BASE_DIR = "/content/drive/MyDrive/AutoML"
BASE_DIR = "/AutoML"
DATASET_PATHS = {
    "Dataset-1": os.path.join(BASE_DIR, "Filename"),
    "Dataset-2": os.path.join(BASE_DIR, "Filename"),
#Add datasets if you need
}
BEST_SEED_OUTPUT = os.path.join(BASE_DIR, "tpot", "results", "best_seeds.json")
PLOT_OUTPUT_DIR = os.path.join(BASE_DIR, "tpot", "results", "seed_auc_plots")
os.makedirs(os.path.dirname(BEST_SEED_OUTPUT), exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# 🔁 Random seeds to try
SEEDS = [42, 101, 202, 303, 404]

# 🔄 Load and preprocess function
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if 'CLIN_SIG' not in data.columns:
        raise ValueError("❌ 'CLIN_SIG' column not found in dataset.")

    le = LabelEncoder()
    data['CLIN_SIG'] = le.fit_transform(data['CLIN_SIG'])

    non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()
    drop_cols = [col for col in non_numeric_cols if col != '#Uploaded_variation'] + [
        'cDNA_position', 'Location', 'Protein_position', 'CDS_position', '#Uploaded_variation']
    data.drop(columns=drop_cols, inplace=True, errors='ignore')

    return data

# 📊 Plot seed vs AUC

def plot_seed_auc_log(dataset_name, seed_auc_log):
    df = pd.DataFrame(seed_auc_log)
    plt.figure(figsize=(8, 4))
    plt.plot(df['seed'], df['auc'], marker='o')
    plt.title(f"Seed-wise AUC for {dataset_name}")
    plt.xlabel("Seed")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_seed_auc.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 Plot saved: {plot_path}")

# 🔍 Evaluate each dataset across multiple seeds
def evaluate_best_seed(dataset_name, file_path, seeds):
    data = load_and_preprocess_data(file_path)
    X = data.drop(columns=["CLIN_SIG"])
    y = data["CLIN_SIG"]

    best_seed = None
    best_auc = -1
    seed_auc_log = []
    all_seed_logs = []

    for seed in seeds:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed
            )

            tpot = TPOTClassifier(
                generations=5, population_size=20,
                verbosity=0, random_state=seed, max_time_mins=5, n_jobs=-1
            )
            tpot.fit(X_train, y_train)

            y_prob = tpot.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)

            seed_auc_log.append({"seed": seed, "auc": auc})
            print(f"✅ {dataset_name} | Seed {seed} | AUC: {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_seed = seed

        except Exception as e:
            print(f⚠️ {dataset_name} | Seed {seed} | Error: {e}")

    plot_seed_auc_log(dataset_name, seed_auc_log)

    return dataset_name, {
        "best_seed": best_seed,
        "best_auc": best_auc,
        "all_seeds": seed_auc_log
    }

# 🚀 Run evaluation
best_seeds_dict = {}
all_seed_logs = []  

for dataset_name, path in DATASET_PATHS.items():
    print(f"\n🔍 Evaluating: {dataset_name}")
    name, result = evaluate_best_seed(dataset_name, path, SEEDS)
    best_seeds_dict[name] = result
    for seed_entry in result["all_seeds"]:
        seed_entry["Dataset"] = dataset_name
        all_seed_logs.append(seed_entry)

# 📊 Combined plot for all datasets
plot_df = pd.DataFrame(all_seed_logs)
plt.figure(figsize=(10, 6))
for dataset in plot_df["Dataset"].unique():
    subset = plot_df[plot_df["Dataset"] == dataset]
    plt.plot(subset["seed"], subset["auc"], marker='o', label=dataset)

plt.title("TPOT AUC per Seed Across Datasets")
plt.xlabel("Seed")
plt.ylabel("AUC")
plt.grid(True)
plt.legend()
plt.tight_layout()
combined_plot_path = os.path.join(PLOT_OUTPUT_DIR, "combined_auc_per_seed.png")
plt.savefig(combined_plot_path, dpi=300)
plt.close()
print(f"📊 Combined AUC plot saved to: {combined_plot_path}")

# 💾 Save the best seeds
with open(BEST_SEED_OUTPUT, "w") as f:
    json.dump(best_seeds_dict, f, indent=4)

print(f"\n✅ Best seeds saved to: {BEST_SEED_OUTPUT}")
