# 🇹🇼 Taiwan Bankruptcy Prediction — Random Forest Classifier

A machine learning project to predict corporate bankruptcy using financial features from Taiwanese companies. Built as part of a structured ML curriculum focused on real-world imbalanced classification challenges.

---

## 📌 Project Overview

- **Dataset:** 6,137 Taiwanese companies, 95 financial features
- **Target:** Binary classification — `bankrupt` (True/False)
- **Challenge:** Severe class imbalance (~220:1 ratio, only ~80 bankrupt companies in training)
- **Final Test Accuracy:** 97.7% — with meaningful minority class detection

---

## 🧠 Key Learnings

### 1. Handling Severely Imbalanced Data
With only ~80 positive cases against ~8,000 negative ones, raw accuracy is meaningless. A model that predicts "not bankrupt" every time would score 99% — and be completely useless. This project reinforced why **precision, recall, and F1-score on the minority class** are the metrics that actually matter.

```
True (bankrupt):  precision 0.71 | recall 0.41 | f1 0.52
False (not):      precision 0.98 | recall 0.99 | f1 0.99
```

### 2. Random Oversampling Before Splitting = Data Leakage
The correct pipeline order matters enormously with imbalanced datasets:
1. **Split first** (train/test)
2. **Oversample only the training set**
3. **Evaluate on the original, untouched test set**

Oversampling before splitting causes synthetic minority samples to leak into both sets, inflating results artificially.

### 3. GridSearchCV + Nested Cross-Validation
Passing a `GridSearchCV` object directly into `cross_val_score` creates nested CV — the grid search re-fits internally for every outer fold. While not incorrect, it's computationally expensive and usually unintentional. Using `model.best_estimator_` after fitting avoids this.

```python
# Inefficient — nested CV
cross_val_score(model, X_train_over, y_train_over, cv=5)

# Intended — evaluate the best model only
cross_val_score(model.best_estimator_, X_train_over, y_train_over, cv=5)
```

### 4. Hyperparameter Tuning with GridSearchCV
Tuned two key Random Forest parameters:

| Parameter | Values Tried | Best |
|-----------|-------------|------|
| `max_depth` | 30, 40 | **40** |
| `n_estimators` | 25, 50 | **50** |

Extracted cleanly with `model.best_params_` and inspected all combinations via `pd.DataFrame(model.cv_results_)`.

### 5. Feature Importance Visualization
Used Gini impurity-based feature importances from the best estimator to identify which financial ratios drove predictions the most. `feat_91` emerged as the most predictive feature by a significant margin.

```python
importances = model.best_estimator_.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()
feat_imp.tail(10).plot(kind="barh")
```

### 6. Evaluation Beyond Accuracy
Used a full suite of evaluation tools:
- `classification_report` — per-class precision, recall, F1
- `ConfusionMatrixDisplay` — visual breakdown of predictions
- Train vs. test accuracy comparison to detect overfitting (train: 100%, test: 97.7% — slight overfit worth monitoring)

---

## 🛠 Tech Stack

- **Python** — pandas, numpy, matplotlib
- **scikit-learn** — RandomForestClassifier, GridSearchCV, cross_val_score, pipeline tools
- **imbalanced-learn** — RandomOverSampler
- **Google Colab** — runtime environment

---

## 📁 Project Structure

```
project/
├── data/
│   ├── taiwan-bankruptcy-data.json.gz
│   └── taiwan-bankruptcy-data-test-features.json.gz
├── model-5-5.pkl
├── my_predictor_assignment.py
└── 055-assignment.ipynb
```

---

## 🔮 What I'd Improve Next

- Replace `RandomOverSampler` with **SMOTE** for more robust synthetic samples
- Use `scoring='f1'` in GridSearchCV instead of default accuracy, so hyperparameter selection is actually optimized for the minority class
- Expand the parameter grid with `min_samples_leaf` and `class_weight='balanced'`
- Try **Gradient Boosting** (XGBoost/LightGBM) which tend to handle imbalance better natively
