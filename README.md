#🔬 AutoML-Based Pathogenicity Prediction of Breast Cancer Variants
This project benchmarks multiple breast cancer and pan-cancer datasets for variant pathogenicity prediction using three state-of-the-art AutoML frameworks: TPOT, H2O AutoML, and MLJAR-supervised.

It includes:

📊 Statistical & correlation-based feature selection

🔁 Seed-wise performance evaluation (Phase 1)

🧠 Full model evaluation with interpretability (Phase 2)

📉 ROC, PRC, SHAP, LIME, and Permutation Importance

📁 Structured results saved by dataset and framework

📁 Project Structure
text
Copy
Edit
├── phase1.py                    # Phase 1: Best seed identification per dataset
├── phase2.py                    # Phase 2: Final evaluation using best seed
├── requirements.txt             # Python dependencies
├── /results/
│   ├── /tpot/
│   ├── /h2o/
│   └── /mljar/
├── /datasets/
│   ├── Balanced_BC_HGMD.csv
│   ├── Balanced_BC_Alldatabases.csv
│   └── ... (more datasets)
🚀 Quick Start
Install requirements

bash
Copy
Edit
pip install -r requirements.txt
Run Phase 1 (best seed selection)

bash
Copy
Edit
python phase1.py
Run Phase 2 (model evaluation + plots)

bash
Copy
Edit
python phase2.py
📈 AutoML Frameworks Used
TPOT

H2O AutoML

MLJAR-supervised (MLJAR Cite Info)

📊 Evaluation & Interpretability
Each classifier is evaluated with:

AUC + 95% CI

Precision, Recall, F1, Kappa, MCC

SHAP bar & beeswarm plots

Permutation Feature Importance

LIME visual explanations for TP/TN/FP/FN
