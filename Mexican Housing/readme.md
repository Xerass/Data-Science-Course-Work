# 🇲🇽 Predicting Apartment Prices in Mexico City

> A machine learning project from the **WorldQuant University Applied Data Science Lab** where I built a regression model to predict apartment prices in Mexico City using real estate listing data.

---

## 📖 About This Project

This was the capstone assignment of Module 2, where I had to apply everything I'd learned about the data science workflow **independently** — from raw CSVs to a deployed model with interpretable feature importances. Unlike earlier guided lessons, this project pushed me to make my own decisions about libraries, cleaning logic, and modeling choices.

The goal was straightforward in concept but rich in execution: **given an apartment's size, location, and borough, predict its price in USD.**

---

## 🎯 What I Set Out to Learn

- How to write a reusable, production-style data cleaning function
- How to combine multiple raw data files into a single clean training set
- How to handle real-world data issues: outliers, missing values, leakage, and multicollinearity
- How to build an end-to-end ML pipeline using scikit-learn
- How to interpret a linear model's coefficients to tell a story about the data

---

## 🗂️ Project Workflow

### 1. Data Wrangling — Building a Reusable `wrangle()` Function

The most important lesson here was working **iteratively**. Instead of trying to write the perfect cleaning function on the first attempt, I started with one filter, tested it, and added complexity step by step.

My final `wrangle()` function performs these tasks on each CSV:

- Subsets to apartments in **Distrito Federal** under **$100,000 USD**
- Removes outliers in `surface_covered_in_m2` (keeps the 10th–90th percentile)
- Splits the `lat-lon` column into separate numeric `lat` and `lon` features
- Extracts the **borough** from the `place_with_parent_names` column
- Drops columns with more than 50% missing values
- Drops columns with very low (<3) or very high (>100) cardinality to avoid useless or leaky features
- Keeps only the features relevant to the target: `surface_covered_in_m2`, `lat`, `lon`, `borough`

### 2. Combining Multiple Files

Using `glob` and a list comprehension, I applied `wrangle()` to every Mexico City CSV in the `data/` directory and concatenated them into one clean DataFrame:

```python
files = glob("data/mexico-city-real-estate-*.csv")
df = pd.concat([wrangle(frame) for frame in files], axis=0, ignore_index=True)
```

This taught me how powerful a well-written wrangle function becomes — one function, many files, zero hassle.

### 3. Exploratory Data Analysis

I used the **object-oriented matplotlib API** (`fig, ax = plt.subplots()`) instead of the global `plt.plot()` style. This is considered best practice and gave me much finer control over my visualizations.

Key visualizations I built:

- **Histogram** of apartment prices to understand the target distribution
- **Scatter plot** of price vs. surface area to check for a linear relationship
- **Mapbox scatter plot** with Plotly Express to visualize geographic price patterns across the city

### 4. Splitting Features and Target

```python
y_train = df["price_aprox_usd"]
X_train = df.drop(columns=["price_aprox_usd"])
```

### 5. Building the Baseline

Before training any model, I established a baseline by predicting the mean apartment price for every observation. The baseline MAE became my benchmark — any real model had to beat it.

### 6. Building the Model — A scikit-learn Pipeline

I used `make_pipeline` to chain three transformations into a single, clean model:

```python
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)
model.fit(X_train, y_train)
```

This pipeline handles:

1. **OneHotEncoder** — converts the `borough` categorical feature into numeric columns
2. **SimpleImputer** — fills any remaining missing values with column means
3. **Ridge Regression** — a regularized linear model that helps prevent overfitting

### 7. Communicating Results — Feature Importances

After training, I extracted the model's coefficients and paired them with feature names to see which variables actually drove price predictions. I then plotted the top 10 most influential features as a horizontal bar chart, sorted by absolute importance.

This step is where the modeling becomes a **story**: which boroughs add value? Does latitude or longitude matter more? Is size still king?

---

## 🧠 Key Takeaways as a Student

| Concept | What I Actually Learned |
|---------|------------------------|
| **Iterative development** | Don't try to write perfect code in one shot. Build, test, refine. |
| **Data leakage** | Some columns "cheat" by encoding the target. Drop them ruthlessly. |
| **Multicollinearity** | Highly correlated features confuse linear models. Pick one. |
| **Cardinality matters** | Categorical columns with too few or too many unique values rarely help. |
| **Pipelines > manual steps** | A pipeline ensures the same transformations apply to train and test data. |
| **OOP matplotlib** | `fig, ax = plt.subplots()` gives you full control and is the professional standard. |
| **Baselines first** | Always know what "doing nothing smart" looks like before celebrating a model. |

---

## 🛠️ Tech Stack

- **Python 3** — core language
- **pandas** — data wrangling and analysis
- **scikit-learn** — `Ridge`, `SimpleImputer`, `make_pipeline`, `mean_absolute_error`
- **category_encoders** — `OneHotEncoder` with named categories
- **matplotlib & seaborn** — static visualizations using the OOP API
- **Plotly Express** — interactive Mapbox geographic plots
- **glob** — batch file loading

---

## 📁 Project Structure

```
.
├── data/
│   ├── mexico-city-real-estate-1.csv
│   ├── mexico-city-real-estate-2.csv
│   ├── ...
│   └── mexico-city-test-features.csv
├── notebook.ipynb
└── README.md
```

---

## 🚀 What I'd Do Next

If I were to extend this project, I would:

- Try **gradient-boosted trees** (XGBoost, LightGBM) and compare against Ridge
- Engineer interaction features between `borough` and `surface_covered_in_m2`
- Use **cross-validation** to get a more honest estimate of generalization error
- Deploy the model as a small **Streamlit** or **Flask** app where users could enter apartment specs and get a predicted price
- Pull more recent listings to see if the model still holds up

---

## 🎓 Acknowledgment

This project is part of the **Applied Data Science Lab** offered by **WorldQuant University**. The course content is licensed for personal use — this README documents *my own learning journey* through the assignment, not a reproduction of course materials.

---

*Built with curiosity, debugging, and far too many `df.info()` calls.* 
