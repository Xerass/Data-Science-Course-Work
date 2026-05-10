# 🏚️ Predicting Earthquake Damage in Kavrepalanchok, Nepal

> A machine learning classification project that predicts whether buildings sustained severe damage during the 2015 Nepal earthquake, using structural and geographic features from the Kavrepalanchok district.

---

## 📌 Project Overview

The 2015 Gorkha earthquake devastated Nepal, damaging hundreds of thousands of buildings across multiple districts. This project tackles a real-world humanitarian and engineering question: **can we predict which buildings are at risk of severe damage based on their structural characteristics?**

Using a SQLite database of post-earthquake survey data, I built and tuned a classification pipeline that predicts severe building damage in the Kavrepalanchok district, then analyzed which structural features matter most — insights that could inform retrofitting priorities and future construction codes.

---

## 🎯 Problem Statement

**Goal:** Build a binary classifier that predicts whether a building suffered severe damage (damage grade > 3) given pre-earthquake structural features.

**Why it matters:** Identifying high-risk building profiles helps governments and NGOs prioritize seismic retrofitting, allocate disaster-relief resources, and shape building codes that save lives in future earthquakes.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data** | SQLite, SQL, pandas |
| **ML** | scikit-learn, category_encoders |
| **Visualization** | matplotlib, seaborn |
| **Environment** | Jupyter Notebook |

---

## 🔄 Project Workflow

### 1. Data Acquisition (SQL)
- Connected to a multi-table SQLite database (`id_map`, `building_structure`, `building_damage`)
- Wrote multi-table `JOIN` queries with `DISTINCT` to deduplicate records and filter to Kavrepalanchok (`district_id = 3`)
- Used `LIMIT` clauses defensively to avoid memory issues during exploration

### 2. Data Wrangling
Built a reusable `wrangle()` function that:
- Pulls and joins data from three SQL tables in a single query
- **Removes leakage** by dropping all `post_eq` columns (information unavailable at prediction time)
- Engineers a binary target `severe_damage` from the multi-class `damage_grade`
- Drops multicollinear features (e.g., `count_floors_pre_eq`) and high-cardinality identifiers

### 3. Exploratory Data Analysis
- Assessed **class balance** to set a meaningful baseline
- Used boxplots to compare plinth area distributions across damage classes
- Built **pivot tables** to investigate the relationship between roof type and severe-damage rate — uncovering a strong signal that informed later feature-importance analysis

### 4. Modeling
- Established a **majority-class baseline** for honest benchmarking
- Trained a **Logistic Regression** pipeline with `OneHotEncoder` as the linear baseline
- Trained a **Decision Tree Classifier** with `OrdinalEncoder`
- Tuned `max_depth` from 1–15 and plotted **validation curves** to diagnose under/overfitting and select the optimal depth
- Refit the final model on the **full dataset** for production-style deployment

### 5. Evaluation & Communication
- Generated predictions on a held-out test set
- Extracted **Gini feature importances** from the trained decision tree
- Built horizontal bar plots to communicate which structural features drive damage risk — connecting back to the EDA findings on roof type

---

## 📊 Key Results

- **Baseline accuracy** established from class distribution
- **Logistic Regression** validated as a linear benchmark
- **Decision Tree** tuned via validation curve, selecting depth that balances bias and variance
- **Feature importance** analysis confirmed EDA hypotheses — particularly that roof type is a strong predictor of severe damage, consistent with the higher vulnerability of certain traditional construction methods

---

## 💡 Skills Demonstrated

### Data Engineering
- Writing efficient multi-table SQL `JOIN`s against a relational database
- Designing reusable, modular data-cleaning pipelines
- **Identifying and preventing data leakage** — a critical real-world ML skill

### Machine Learning
- End-to-end supervised classification workflow
- Pipeline construction with `scikit-learn` (`make_pipeline`)
- **Hyperparameter tuning** using validation curves
- Comparing linear (Logistic Regression) vs. tree-based (Decision Tree) models
- Encoding categorical features (Ordinal vs. One-Hot) appropriately for each model family

### Statistical & Analytical Thinking
- Establishing baselines before declaring model success
- Diagnosing overfitting via training-vs-validation gap analysis
- Connecting EDA insights to model-based feature importances
- Translating model output into actionable real-world recommendations

### Software Practices
- Clean, modular code with reusable functions
- Object-oriented matplotlib API (`fig, ax = plt.subplots()`) for production-quality plots
- Defensive querying against large databases

---

## 📁 Project Structure

```
.
├── notebook.ipynb              # Main analysis notebook
├── data/
│   └── kavrepalanchok-test-features.csv
├── nepal.sqlite                # Source database
└── README.md
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders jupyter

# Launch the notebook
jupyter notebook notebook.ipynb
```

Make sure `nepal.sqlite` is accessible at the path specified in the `wrangle()` function.

---

## 🔭 Future Improvements

- Try **ensemble methods** (Random Forest, Gradient Boosting) for likely accuracy gains
- Apply **SMOTE or class weighting** if class imbalance proves problematic
- Add **cross-validation** instead of a single train/val split for more robust estimates
- Incorporate **geospatial features** (latitude/longitude clusters) if available
- Deploy as a **REST API** or simple web dashboard for field use by relief workers

---

## 📚 Acknowledgments

- Dataset: Nepal Earthquake Open Data Portal
- Project completed as part of the **WorldQuant University Applied Data Science Lab**

---

## 👤 Author

Built with curiosity and a focus on real-world impact. Connect with me on [LinkedIn](#) or check out my [other projects](#).
