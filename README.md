# WorldQuant University — Applied Data Science Lab

A portfolio of eight end-to-end data science projects completed through the [WorldQuant University Applied Data Science Lab](https://www.wqu.edu/) — a project-based program where each unit is a full pipeline: messy real-world data in, a working model or deployed app out.

Each project below is its own folder. They progress from data wrangling through supervised and unsupervised learning, time series, experimentation, and finally model deployment behind an API.

---

## The projects

| # | Project | Dataset / Region | Core task | Headline techniques |
|---|---------|------------------|-----------|---------------------|
| 1 | **Brazilian Housing** | Brazil property listings | Data wrangling & EDA | pandas cleaning, geospatial scatter/choropleth, correlation analysis |
| 2 | **Mexican Housing** | Mexico City apartments | Price regression | Ridge regression, scikit-learn `Pipeline`, `OneHotEncoder`, baseline vs. model |
| 3 | **AfricanAirQuality** | Dar es Salaam PM2.5 | Time-series forecasting | MongoDB ingestion, AutoRegressive model, walk-forward validation, ACF/PACF |
| 4 | **NepalEarthquakes** | Kavrepalanchok damage | Binary classification | SQL joins, logistic regression & decision trees, feature importance |
| 5 | **TaiwanBankruptcy** | Company financials | Imbalanced classification | Resampling for class imbalance, ensemble models, precision/recall trade-offs |
| 6 | **AmericanFinances** | US Survey of Consumer Finances | Customer segmentation | K-Means clustering, PCA, feature standardization, interactive Dash app |
| 7 | **ABTest** | WQU applicant funnel | Experiment design | Chi-square test, power analysis, MongoDB repository, Dash dashboard |
| 8 | **IndianStockMarketForecasting** | Equity returns (GARCH) | Volatility forecasting + deployment | `arch` GARCH model, SQLite repository, Pydantic, FastAPI `/fit` & `/predict` |

---

## Skills demonstrated across the course

Rather than list libraries, here's what each track actually built up:

**Data wrangling & EDA.** Cleaning genuinely messy, inconsistent real-world data; merging across sources; engineering features; and reading a dataset visually before modeling it. This is the foundation every later project leans on.

**Supervised learning.** Both halves of it — regression (predicting apartment prices) and classification (earthquake damage, corporate bankruptcy). The recurring lesson was *establish a baseline first*, then justify every increment of model complexity against it.

**Handling imbalanced data.** The bankruptcy project forced the point that accuracy is a trap on skewed classes — the real work is in resampling strategy and choosing the metric (precision, recall, their trade-off) that matches the cost of each error type.

**Unsupervised learning.** K-Means segmentation plus PCA for dimensionality reduction and visualization — finding structure without labels, then making it interpretable to a non-technical audience through a dashboard.

**Time series.** Stationarity, autocorrelation diagnostics (ACF/PACF), autoregressive modeling, and — critically — **walk-forward validation**, because a random train/test split leaks the future into the past and makes a temporal model meaningless.

**Statistics & experimentation.** Designing an A/B test properly: framing hypotheses, running a power analysis *before* collecting data, and using a chi-square test to decide significance afterward.

**Databases.** Hands-on with three paradigms — **SQL** (joins, aggregation for the earthquake data), **MongoDB** (document storage for air-quality and A/B data), and **SQLite** (local persistence behind the volatility app) — usually behind a clean repository abstraction so the modeling code stays storage-agnostic.

**MLOps & deployment.** The capstone took a model out of the notebook entirely: wrapped in a class, fronted by a **FastAPI** service with typed Pydantic request schemas, and made callable over HTTP. The shift from "runs on my machine" to "is a service" is the whole point.

**Communication.** Several projects ship as interactive **Dash / Plotly** apps — the deliverable isn't a notebook, it's something a stakeholder can click through.

---

## Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `statsmodels` · `arch` · `Plotly` / `Dash` · `Matplotlib` · `SQL` · `MongoDB (pymongo)` · `SQLite` · `FastAPI` · `pydantic` · `requests`

---

## What I took away from it

The through-line across all eight: a model is only as good as the pipeline around it. Most of the actual difficulty lived in the data (cleaning, leakage, imbalance, temporal ordering) and in making the result *usable* (a validated API, a dashboard) — not in the algorithm itself. The course is structured so that by Project 8 you're not just fitting a model, you're shipping one.

---

*Completed as part of the WorldQuant University Applied Data Science Lab. Project code is my own implementation; course materials remain the property of WQU.*
