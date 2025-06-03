# 🔬 Leveraging AutoML to Optimize Dataset Selection for Improved Breast Cancer Variant Pathogenicity Prediction

This project benchmarks breast cancer and pan-cancer variant datasets to identify the most effective data composition and AutoML framework for predicting variant pathogenicity. Using **TPOT**, **H2O AutoML**, and **MLJAR-supervised**, the pipeline performs:

- 📌 Dataset-level seed tuning (Phase 1)
- 📈 Full evaluation on best seed (Phase 2)
- ✅ Confidence interval bootstrapping
- 🔍 Model interpretability with **SHAP**, **LIME**, and **Permutation Importance**

---

## 📁 Project Structure

```
.
├── phase1.py                    # Phase 1: Find best random seed per dataset
├── phase2.py                    # Phase 2: Full model evaluation using best seed
├── requirements.txt             # Required Python packages
├── LICENSE                      # MIT License
├── .gitignore                   # Files and folders to ignore in Git
├── CITATION.cff                 # GitHub citation metadata
├── /datasets/                   # Input datasets (not included here)
├── /results/                    # Model outputs and plots
│   ├── /tpot/
│   ├── /h2o/
│   └── /mljar/
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AutoML_Pathogenicity_Prediction.git
cd AutoML_Pathogenicity_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your datasets
Place your CSV files in the `datasets/` folder (Dataset-1 sample is provided as a guide for data structure and for testing the workflow only):
- `Dataset-1.csv`
- `Dataset-2.csv`
- `Dataset-3.csv`
- `Dataset-4.csv`

### 4. Run Phase 1: Find best seed for each dataset
```bash
python phase1.py
```

### 5. Run Phase 2: Final evaluation with interpretability
```bash
python phase2.py
```

---

## 📊 Outputs

Each AutoML framework saves:
- **AUC vs. Seed plots** (Phase 1)
- **ROC & PR curves** with AUC + CI
- **Permutation Importance** bar plots
- **SHAP** bar and beeswarm plots
- **LIME** explanations for TP, TN, FP, FN
- **Summary metrics** in Excel files

All outputs are saved under:
```
/results/tpot/
/results/h2o/
/results/mljar/
```

---

## 🧠 AutoML Frameworks

| Framework       | Description                                                                 | Link                                                                 |
|-----------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
| TPOT            | Genetic programming-based AutoML pipeline                                   | https://github.com/EpistasisLab/tpot                                 |
| H2O AutoML      | High-performance machine learning platform with leaderboard stacking        | https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html          |
| MLJAR-Supervised| Fast, explainable AutoML with human-readable pipelines                      | https://github.com/mljar/mljar-supervised                           |

---

## 📈 Metrics & Evaluation

| Metric        | Description                                         |
|---------------|-----------------------------------------------------|
| AUC           | Area Under the ROC Curve                            |
| Precision     | Positive Predictive Value                           |
| Recall        | Sensitivity or True Positive Rate                   |
| F1-Score      | Harmonic mean of Precision and Recall               |
| MCC           | Matthews Correlation Coefficient                    |
| Cohen’s Kappa | Agreement beyond chance                             |
| 95% CI        | Confidence intervals via 1000x bootstrapping        |

---

## 🧪 Model Interpretability

Each model is explained using:
- **SHAP**: Global and local importance with beeswarm & bar plots
- **LIME**: Local explanations for TP, TN, FP, FN samples
- **Permutation Importance**: Feature impact on prediction stability

---

## 📚 Citation

If you use this repository, please cite as (will be updated once published):

```bibtex

  authors = {Rahaf M. Ahmad, Mohd Saberi Mohamad, Bassam R. Ali*},
  title = {Leveraging AutoML to Optimize Dataset Selection for Improved Breast Cancer Variant Pathogenicity Prediction},
  year = {2025},
  Link to GiHub repository annd publication doi = {\\url{https://github.com/rahafahmad89/AutoML_Pathogenicity_Prediction}},
  note = {MIT License}
}
```

---

## 🛡 License

This project is licensed under the [MIT License](./LICENSE).

---

## 👩‍💻 Author

**Rahaf M. Ahmad**  
Ph.D. Candidate | Genetics & Machine Learning  
United Arab Emirates University  
ORCID: [0000-0002-7531-5264](https://orcid.org/0000-0002-7531-5264)

---

## 🤝 Acknowledgements

- Inspired by the need for robust and interpretable predictions in precision oncology.
- Developed using open-source AutoML frameworks and Colab Pro.

---

## 🧠 Future Work
 
- Web-based clinical decision tool  
- Real-time explainability dashboards
