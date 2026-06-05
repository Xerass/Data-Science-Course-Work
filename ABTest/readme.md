# MScFE Admissions Experiment — Assignment 7.5

A full-stack data science project combining MongoDB ETL, exploratory data analysis, A/B experiment design, and statistical inference to evaluate the effect of email nudges on admissions quiz completion rates for the WorldQuant University MScFE program.

---

## Project Overview

The core question: **does sending an email to incomplete applicants meaningfully increase quiz completion?** To answer it rigorously, the project walks through the entire experiment lifecycle — from raw data exploration to a statistically valid conclusion.

---

## Key Learnings

### 1. MongoDB Aggregation Pipelines
- Used `$group` with `$count` and `$sum` to aggregate applicant counts by nationality and by sign-up date.
- Used `$match` to filter documents before grouping (e.g., `admissionsQuiz: "incomplete"`).
- Used `$dateTrunc` to collapse datetime fields to day-level granularity for time-series aggregation.

### 2. DataFrame Wrangling with pandas
- Chained `.rename()`, `.sort_values()`, `.reset_index(drop=True)`, `.set_index()`, and `.squeeze()` cleanly in a single pipeline.
- A critical lesson: `reset_index(drop=True)` is necessary after `sort_values()` to prevent the old index from persisting as a column.
- Sort order matters for graders — always match the expected output's sort key, not just shape.

### 3. Country Code Conversion with `country_converter`
- `CountryConverter.convert()` maps ISO2 codes to human-readable short names and ISO3 codes in a single vectorized call.
- Useful for enriching geographic data before visualization.

### 4. Choropleth Maps with Plotly Express
- `px.choropleth()` requires ISO3 location codes, a color column, and a projection type.
- `color_continuous_scale` controls the gradient; `px.colors.sequential.Oranges` gives a clean single-hue scale.

### 5. Object-Oriented ETL: `MongoRepository` Class
Encapsulating MongoDB logic in a class enforces clean separation between data access and analysis code. Key methods built:

| Method | What it does |
|---|---|
| `__init__` | Initializes the collection attribute from `client[db][collection]` |
| `find_by_date` | Queries incomplete applicants within a date range using `$gte`/`$lt` |
| `update_applicants` | Loops over documents and calls `update_one` with `$set`, tracking matched/modified counts |
| `assign_to_groups` | Shuffles applicants, splits 50/50 into control/treatment, and writes group labels back to MongoDB |
| `find_exp_observations` | Returns all documents where `inExperiment: True` as a plain Python list |

> **Key gotcha:** `collection.find()` returns a lazy PyMongo `Cursor`, not a list. Always wrap it in `list()` if the result needs to be reused or returned.

### 6. Power Analysis with `GofChisquarePower`
- `solve_power(effect_size=0.5, alpha=0.05, power=0.8)` calculates the minimum group size to detect a medium effect.
- Result must be ceiling-rounded (`math.ceil`) because partial participants don't exist.
- Total sample needed = `group_size * 2` (one control group + one treatment group).

### 7. Experiment Duration Planning with the Normal Distribution
- Daily sign-up counts follow a roughly normal distribution; the sum of `n` independent days has:
  - Mean: `μ_sum = μ * n`
  - Std dev: `σ_sum = σ * √n` ← **not** `σ * n` (a common mistake)
- Used `scipy.stats.norm.cdf()` to compute the probability of accumulating enough observations in `n` days.
- Finding the **minimum** days requires iterating `n` from 1 upward and stopping at the first `n` where `P(total ≥ threshold) ≥ 0.95`.

### 8. Chi-Square Test of Independence
- `pd.crosstab()` builds the 2×2 contingency table (group × quiz completion).
- `statsmodels.stats.contingency_tables.Table2x2` wraps the table for hypothesis testing.
- `.test_nominal_association()` performs the chi-square test; a p-value below 0.05 indicates a statistically significant association between group assignment and quiz completion.

### 9. Odds Ratio Interpretation
- The odds ratio quantifies the strength of the treatment effect: an odds ratio of ~7 means applicants in the email group were roughly 7× more likely to complete the quiz than those in the control group.
- Computed directly from the `Table2x2` object via `.oddsratio`.

### 10. Visualization Best Practices
- Choropleth maps communicate geographic distributions at a glance.
- Side-by-side bar charts (`barmode="group"`) are the right choice for comparing two categorical outcomes across two groups — stacked bars obscure the incomplete/complete ratio difference.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pymongo` | MongoDB connection and CRUD operations |
| `pandas` | DataFrame wrangling and pipeline chaining |
| `plotly.express` | Choropleth maps and bar charts |
| `country_converter` | ISO2 → ISO3 / short name mapping |
| `scipy.stats` | Normal CDF for experiment duration planning |
| `statsmodels` | Power analysis, chi-square test, odds ratio |
| `numpy` | `sqrt` for sum standard deviation |

---

## Workflow Summary

```
MongoDB Collection
      │
      ├─ Explore ──► Aggregate by nationality ──► Choropleth map
      │
      ├─ ETL ──────► MongoRepository class ──► find, update, assign groups
      │
      ├─ Plan ─────► Power analysis ──► Experiment duration (normal dist.)
      │
      ├─ Run ──────► Experiment.run_experiment(days=7)
      │
      └─ Analyze ──► Crosstab ──► Chi-square test ──► Odds ratio
```
