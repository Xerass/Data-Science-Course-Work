# Small Business Owners in the US — Clustering Analysis

## Project Overview

This project analyzes the 2019 Survey of Consumer Finances (SCF) to segment small business owners in the United States into meaningful subgroups using unsupervised learning. The goal is to uncover financial patterns among small business owners (income under $500K) by applying K-Means clustering on high-variance features, then visualizing the results with PCA.

---

## Key Learnings

### 1. Exploratory Data Analysis Shapes Every Decision Downstream

Before building any model, understanding the data was essential. A few things stood out early:

- **Business owners are a minority in the dataset.** Filtering for them meant working with a smaller, more focused subset — a reminder that the population you care about isn't always the majority of your data.
- **Income distribution differs sharply between business owners and non-business owners.** Business owners skew toward higher income brackets. This confirmed that treating them as a distinct group for analysis was warranted.
- **The home value vs. debt scatterplot revealed different financial behavior.** Business owners showed wider variance in both debt and home value, hinting at diverse financial profiles within the group itself — exactly the kind of heterogeneity clustering can help untangle.

### 2. Variance Is Your Feature Selection Compass (But Outliers Can Mislead It)

Selecting features for clustering required identifying which variables carried the most information — measured here by variance.

- **Raw variance was dominated by outliers.** A handful of extreme values in financial columns inflated variance estimates and would have skewed feature selection.
- **Trimmed variance (removing the top and bottom 10%) gave a much more stable picture.** This is a practical, underused technique: it keeps feature selection grounded in the bulk of the data rather than a few edge cases.
- **The top five high-variance features became the feature matrix.** Rather than using all columns (which would add noise and curse-of-dimensionality problems), this focused the model on the dimensions where business owners actually differ from one another.

### 3. Choosing the Right Number of Clusters Is a Judgment Call, Not a Formula

Two diagnostic tools guided the choice of *k*:

- **The inertia (elbow) plot** showed diminishing returns in within-cluster variance reduction. The "elbow" suggested that going beyond 3 clusters added complexity without proportional improvement.
- **The silhouette score plot** measured how well-separated the clusters were. It reinforced the same conclusion — 3 clusters struck the best balance between cohesion and separation.
- **The takeaway:** Neither metric gives a single "correct" answer. They're guides. Domain knowledge and interpretability matter just as much. Three clusters were chosen because they were both statistically supported and produced groups that made intuitive financial sense.

### 4. Standardization Is Non-Negotiable for K-Means

K-Means uses Euclidean distance, which means features on larger scales (e.g., total debt in dollars) will dominate features on smaller scales. The pipeline included `StandardScaler` before `KMeans` to put all features on equal footing. Without this step, the clustering would have been driven almost entirely by whichever column had the largest absolute values — not by meaningful differences in financial behavior.

### 5. PCA Makes High-Dimensional Clusters Visible

With five features, the clusters can't be plotted directly. PCA reduced the data to two principal components while preserving as much variance as possible.

- **The 2D scatter plot showed clear visual separation between the three clusters**, confirming that the model found real structure — not just statistical artifacts.
- **PCA doesn't change the clustering.** It's purely a visualization tool here. The clusters were built in the original 5D feature space; PCA just projected them into 2D so humans could see them.

### 6. Cluster Profiles Tell the Real Story

The grouped bar chart of cluster means revealed distinct financial profiles among small business owners — for example, differences in debt levels, asset holdings, and net worth. This is where clustering becomes actionable: instead of treating all small business owners as one group, lenders, policymakers, or researchers can now reason about subgroups with different risk profiles and financial needs.

---

## Tools and Techniques Used

| Category | Tools |
|---|---|
| Data manipulation | pandas |
| Visualization | matplotlib, seaborn, plotly |
| Preprocessing | StandardScaler, trimmed variance |
| Modeling | K-Means (via sklearn Pipeline) |
| Evaluation | Inertia (elbow method), Silhouette Score |
| Dimensionality reduction | PCA |

---

## Reproducing the Results

The analysis uses `random_state=42` throughout for reproducibility. The data source is the 2019 Survey of Consumer Finances (`SCFP2019.csv.gz`). Small business owners are defined as respondents where `HBUS == 1` and `INCOME < 500,000`.
