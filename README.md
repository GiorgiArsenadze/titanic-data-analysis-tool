# 🚢 Titanic Survival Analysis — End-to-End Data Science Project

A complete data science pipeline built in Python that explores, cleans, visualises, and models the Titanic passenger dataset to predict survival outcomes.

---

## 📊 Project Overview

This project demonstrates the five core skills required in any data science role:

| Step | Tool / Technique | What I did |
|------|-----------------|------------|
| 01 · EDA | pandas, descriptive stats | Explored 891 passenger records, identified missing values, uncovered survival patterns |
| 02 · Preprocessing | scikit-learn, LabelEncoder, StandardScaler | Cleaned nulls, encoded categoricals, scaled features, split train/test |
| 03 · Visualisation | matplotlib, seaborn | Built histograms, grouped bar charts, and a full correlation heatmap |
| 04 · Machine Learning | Random Forest Classifier | Trained & evaluated a model achieving 78.3% accuracy and 0.866 AUC |
| 05 · Feature Importance | feature_importances_ (Gini) | Ranked which passenger traits drove survival predictions |

---

## 🔍 Key Findings

- **Sex was the #1 predictor of survival** (36.9% importance) — women survived at 74% vs men at 19%, consistent with the "women and children first" evacuation policy
- **Fare paid ranked 2nd** (19%) — a proxy for wealth and cabin location, both of which affected lifeboat access
- **Passenger class mattered significantly** — 1st class survival rate (63%) was nearly 3× that of 3rd class (24%)
- **Embarkation port was nearly irrelevant** (2.9%) — which port a passenger boarded from had almost no bearing on survival once other factors were accounted for
- The trained **Random Forest model achieved 78.3% accuracy** and an **AUC of 0.866** on unseen test data

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data loading, exploration, and manipulation
- **numpy** — numerical operations
- **matplotlib & seaborn** — data visualisation
- **scikit-learn** — preprocessing, model training, and evaluation
- **base64 / io** — embedding charts directly into the HTML output

---

## 📁 Files

```
titanic-data-analysis/
├── ds_analysis.py                  # Full Python pipeline (run this)
├── titanic_ds_dashboard.html       # Self-contained HTML dashboard (open in browser)
└── README.md                       # This file
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/GiorgiArsenadze/titanic-data-analysis.git
cd titanic-data-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Run the analysis**
```bash
python ds_analysis.py
```

**4. Open the dashboard**
```bash
# Mac
open titanic_ds_dashboard.html

# Windows
start titanic_ds_dashboard.html

# Linux
xdg-open titanic_ds_dashboard.html
```

The script will regenerate the full HTML dashboard with all charts embedded. No API keys, databases, or internet connection required — everything runs locally.

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 78.3% |
| ROC-AUC | 0.866 |
| Precision (Survived) | 0.76 |
| Recall (Survived) | 0.72 |
| F1-Score (Survived) | 0.74 |

---

## 💡 What I Learned

- Real-world data is messy — nearly 80% of the "deck" column was missing, requiring careful decisions about what to drop vs. impute
- Feature engineering decisions (like using fare as a wealth proxy) can be validated after the fact through feature importance scores
- A well-tuned Random Forest is highly competitive on tabular data with relatively few features
- Visualising the data *before* modelling is not optional — the correlation heatmap correctly predicted which features would matter most, before I trained anything

---

*Dataset: Titanic passenger data via seaborn's built-in datasets*
*Model: Random Forest Classifier (n_estimators=200, max_depth=6, random_state=42)*
