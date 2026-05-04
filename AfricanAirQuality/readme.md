# 🇹🇿 Air Quality Forecasting — Dar es Salaam

A time series machine learning project using MongoDB, AutoRegressive models, and walk-forward validation to predict PM2.5 particulate matter levels in Dar es Salaam, Tanzania.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![MongoDB](https://img.shields.io/badge/Database-MongoDB-green) ![statsmodels](https://img.shields.io/badge/Model-AutoReg-purple)

---

## 📌 Overview

This project queries real PM2.5 sensor readings from a MongoDB database, engineers a clean hourly time series, and builds an AutoRegressive model to forecast air quality — achieving a test MAE well below the naive mean baseline.

---

## 📊 Key Metrics

| Metric | Value |
|---|---|
| Training observations | 1,944 |
| Test observations | 216 |
| Baseline MAE (mean prediction) | 4.05 |
| **Test MAE (walk-forward validation)** | **3.97** |
| Optimal lag order (p) | 26 hours |

---

## 🔧 Project Pipeline

### 1. Connect & Query MongoDB
Queried Site 11 PM2.5 readings from a NoSQL document store using PyMongo with filters and projections to avoid fetching unnecessary data.

```python
client = MongoClient(host=host, port=27017)
db = client['air-quality']
dar = db['dar-es-salaam']
```

### 2. Wrangle & Clean the Time Series
- Localized UTC timestamps to `Africa/Dar_es_Salaam`
- Removed outliers above 100 PM2.5
- Resampled to 1-hour intervals with forward-fill for missing readings

```python
df.index = df.index.tz_localize("UTC").tz_convert("Africa/Dar_es_Salaam")
df = df[df["P2"] < 100]
y = df["P2"].resample("1H").mean().fillna(method='ffill')
```

### 3. Explore Autocorrelation
Used ACF and PACF plots to identify significant lags and confirm hourly seasonality patterns before selecting a model.

### 4. Hyperparameter Tuning
Iterated over lag values 1–30, fitting an `AutoReg` model for each and selecting the `p` with the minimum training MAE.

```python
for p in range(1, 31):
    model = AutoReg(y_train, lags=p).fit()
    mae = mean_absolute_error(y_train.iloc[p:], model.predict().dropna())
    maes.append(mae)

best_p = pd.Series(maes, index=range(1, 31)).idxmin()  # → 26
```

### 5. Walk-Forward Validation
Simulated real deployment by predicting one step ahead, appending the true observation to history, and repeating for all 216 test points — avoiding data leakage.

```python
for i in range(len(y_test)):
    model = AutoReg(history, lags=26).fit()
    next_pred = model.forecast().iloc[0]
    y_pred_wfv_list.append(next_pred)
    history = pd.concat([history, y_test.iloc[[i]]])
```

---

## 💡 Key Learnings

**NoSQL data extraction** — Querying MongoDB with filters and projections (rather than fetching all records) is essential for large sensor datasets where only a specific site and measurement type is needed.

**Time zone handling matters** — Sensor data stored in UTC must be converted to local time before analysis so that daily and weekly patterns (morning peaks, weekday cycles) are correctly aligned in the index.

**ACF / PACF interpretation** — These plots reveal how many past hours are statistically predictive of the current reading, guiding lag order selection rather than arbitrary guessing.

**Walk-forward validation prevents leakage** — Unlike a simple train/test split, walk-forward validation retrains on all available history at each step, mirroring how a deployed model would behave in production.

**Residual diagnostics** — Inspecting training residuals with a histogram and ACF plot confirms whether the model has captured the signal or left exploitable patterns behind.

**Resampling fills sensor gaps** — Real IoT sensors miss readings. Resampling to hourly frequency and forward-filling ensures a regular, gap-free index that time series models require.

---

## 🛠️ Tools & Libraries

| Library | Purpose |
|---|---|
| `pymongo` | NoSQL querying and aggregation |
| `pandas` | Time series manipulation and resampling |
| `statsmodels` | AutoReg model, ACF/PACF plots |
| `matplotlib` | OOP-style static visualizations |
| `plotly` | Interactive forecast charts |
| `sklearn` | MAE evaluation metric |

---

## 📁 Data Source

- **Database:** MongoDB (`air-quality` → `dar-es-salaam` collection)
- **Site:** Site 11 (173,242 total readings)
- **Measurement:** PM2.5 particulate matter (`P2`)
- **Date range:** January 2018 – March 2018

---

*WorldQuant University · Applied Data Science · Module 3.5 — Time Series Forecasting*
