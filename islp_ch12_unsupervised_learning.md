# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 12: Unsupervised Learning (PCA, Clustering, Anomaly Detection, Practical Segmentation)

### Core Concept
Unsupervised learning finds structure in data without labeled outcomes—useful for **company segmentation**, **peer group discovery**, **outlier detection** (quality-of-earnings red flags), and **dimensionality reduction** when building forecasting and comparable-selection pipelines. In valuation, it helps define more defensible comp sets, detect abnormal financial profiles, and reduce noisy feature sets before forecasting or risk modeling.

See also: Chapter 6 (regularization), Chapter 8 (trees), ISLP Chapter 2 (KNN), Damodaran relative valuation (peer selection logic).

---

### Formula/Methodology

#### 1) Standardisation (required before distance-based methods)
```text
x_standardized = (x − mean(x)) / std(x)
```
Why:
- Clustering and PCA depend on scale; unscaled features can dominate distances.

#### 2) Principal Component Analysis (PCA)
Goal: find orthogonal directions (principal components) that capture maximum variance.

Data matrix X (n × p), centered.
First principal component direction v1 solves:
```text
v1 = argmax_v  Var(X v)  subject to ||v|| = 1
```
Subsequent components orthogonal to earlier ones.

Scores (reduced representation):
```text
Z = X_centered × V_k
```
Where:
- V_k = matrix of first k loading vectors (p × k)
- Z = principal component scores (n × k)

Explained variance ratio:
```text
EVR_k = variance explained by component k / total variance
```
Practical valuation use:
- Reduce many KPIs into a few composite factors (e.g., “scale”, “profitability”, “capital intensity”).
- Use PCA scores for clustering and for stable regressions.

#### 3) K-means clustering
Objective:
```text
Minimise:  Σ_{i=1..n} ||x_i − μ_{c_i}||^2
```
Where:
- c_i is assigned cluster for observation i
- μ_k is centroid of cluster k

Algorithm (iterative):
- Assign each point to nearest centroid.
- Recompute centroids as mean of assigned points.

Choosing number of clusters K:
- Elbow method: plot within-cluster SSE vs K; pick point of diminishing returns.
- Stability checks: do clusters persist across resamples?

#### 4) Hierarchical clustering (agglomerative)
Start with each point as a cluster; iteratively merge closest clusters using linkage:
- Single linkage: min distance
- Complete linkage: max distance
- Average linkage: average distance
Produces dendrogram; choose cut level.

Practical note:
- Hierarchical clustering is useful for interpretability and smaller datasets (peer mapping).

#### 5) Anomaly detection (simple, practical)
Z-score outlier for metric m:
```text
z = (m − mean(m)) / std(m)
Flag if |z| > z_threshold (e.g., 3)
```
Robust alternative using median absolute deviation (MAD):
```text
z_robust = 0.6745 × (m − median(m)) / MAD
MAD = median(|m − median(m)|)
```
Use cases:
- Outlier margins, working capital swings, unusual accruals → QoE flags.

---

### Python Implementation
```python
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array.")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN/Inf).")
    return X


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize columns of X with safe handling for zero-variance columns.

    Returns:
        Xs, mean, std_safe
    """
    X = _as_2d(X)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd_safe = np.where(sd == 0.0, 1.0, sd)
    Xs = (X - mu) / sd_safe
    return Xs, mu, sd_safe


def pca_fit_transform(X: np.ndarray, n_components: int = 2) -> Dict[str, np.ndarray]:
    """
    Fit PCA on standardized X and return scores + loadings + explained variance.

    Args:
        X: (n, p)
        n_components: number of components

    Returns:
        dict with keys:
          scores (n,k), loadings (p,k), explained_variance_ratio (k,)

    Raises:
        ValueError: invalid inputs.
    """
    X = _as_2d(X)
    if not isinstance(n_components, int) or n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be integer in [1, p].")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(Xs)
    loadings = pca.components_.T  # p x k
    evr = pca.explained_variance_ratio_
    return {"scores": scores, "loadings": loadings, "explained_variance_ratio": evr}


def kmeans_fit(X: np.ndarray, k: int, seed: int = 0, n_init: int = 20) -> KMeans:
    """
    Fit KMeans with defensive checks.

    Args:
        X: (n, p)
        k: number of clusters

    Returns:
        fitted KMeans
    """
    X = _as_2d(X)
    if not isinstance(k, int) or k < 2 or k > X.shape[0]:
        raise ValueError("k must be an integer in [2, n].")
    if n_init <= 0:
        raise ValueError("n_init must be > 0.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    km.fit(Xs)
    km.scaler_ = scaler  # attach for later transforms (simple pattern)
    return km


def choose_k_by_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 8, seed: int = 0) -> Dict[str, float]:
    """
    Choose k via silhouette score (higher is better).
    """
    X = _as_2d(X)
    if k_min < 2 or k_max < k_min:
        raise ValueError("Invalid k_min/k_max.")
    k_max = min(k_max, X.shape[0] - 1)
    if k_max < 2:
        raise ValueError("Not enough rows to cluster.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k, best_score = None, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(Xs)
        # silhouette requires >1 cluster and <n clusters
        score = float(silhouette_score(Xs, labels))
        if score > best_score:
            best_k, best_score = k, score
    return {"best_k": float(best_k), "best_silhouette": float(best_score)}


def robust_z_scores(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using median and MAD.

    z_robust = 0.6745 * (x - median) / MAD

    Returns:
        np.ndarray of robust z scores (same length as x)

    Raises:
        ValueError: invalid input or MAD==0 for all values.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must not be empty.")
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values.")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0.0:
        # fallback: standard deviation
        sd = np.std(x, ddof=0)
        if sd == 0.0:
            return np.zeros_like(x)
        return (x - np.mean(x)) / sd
    return 0.6745 * (x - med) / mad


def flag_outliers(x: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """
    Flag outliers using robust z-scores.

    Returns:
        boolean array where True indicates outlier
    """
    if not np.isfinite(z_thresh) or z_thresh <= 0:
        raise ValueError("z_thresh must be finite and > 0.")
    z = robust_z_scores(x)
    return np.abs(z) > z_thresh


# Example usage (valuation context): segment companies into peer groups by fundamentals
# Features: [revenue_growth, ebitda_margin, roic, net_debt_to_ebitda, capex_to_sales]
X = np.array([
    [0.25, 0.18, 0.22, 1.0, 0.04],  # high growth, high ROIC (software-like)
    [0.18, 0.16, 0.18, 1.5, 0.05],
    [0.05, 0.10, 0.08, 2.8, 0.08],  # mature, levered
    [0.03, 0.09, 0.06, 3.2, 0.10],
    [0.12, 0.14, 0.12, 1.8, 0.06],  # balanced
    [0.10, 0.13, 0.11, 2.0, 0.07],
    [0.30, 0.08, 0.10, 0.5, 0.12],  # growth but low margin, high capex
    [0.28, 0.07, 0.09, 0.6, 0.11],
], dtype=float)

feature_names = ["growth", "margin", "roic", "nd_ebitda", "capex_sales"]

# PCA to 2 factors for plotting/inspection (scores used for clustering if needed)
pca_out = pca_fit_transform(X, n_components=2)
scores = pca_out["scores"]
print("Explained variance ratio:", [round(v, 3) for v in pca_out["explained_variance_ratio"].tolist()])

# Choose k by silhouette
sel = choose_k_by_silhouette(X, k_min=2, k_max=4, seed=1)
print("Best k:", sel)

# Fit kmeans
k = int(sel["best_k"])
km = kmeans_fit(X, k=k, seed=1)
labels = km.labels_
print("Cluster labels:", labels.tolist())

# QoE-style outlier detection example: margin outliers
margins = X[:, 1]
out = flag_outliers(margins, z_thresh=2.5)
print("Margin outliers:", out.tolist())
```

---

### Valuation Impact
Why this matters:
- **Comparable selection:** Clustering on fundamentals can form more defensible peer sets than “industry only”, improving relative valuation integrity.
- **Multiple explanation:** PCA factors can summarise “growth” and “quality” dimensions, linking observed multiples to fundamental clusters.
- **QoE diagnostics:** Unsupervised outlier flags can identify companies with unusual accruals, margins, working capital swings, or leverage—prompting normalisation before applying multiples or projecting cash flows.

Impact on multiples:
- Use cluster-based peer sets to compute more relevant median EV/EBITDA or EV/Revenue multiples.
- Remove outliers that reflect non-recurring accounting or business model differences (capital intensity, revenue recognition).

Impact on DCF inputs:
- PCA can reduce noisy driver sets before forecasting.
- Cluster membership can guide margin and reinvestment assumptions by peer “archetype” (mature vs growth vs capital intensive).

Practical adjustments:
```python
import numpy as np

def winsorize(x: np.ndarray, p_low: float = 0.05, p_high: float = 0.95) -> np.ndarray:
    """
    Winsorize a series to reduce outlier influence (useful for peer multiple inputs).

    Args:
        x: numeric array
        p_low/p_high: percentile bounds in [0,1], p_low < p_high

    Returns:
        winsorized array
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must not be empty.")
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values.")
    if not (0.0 <= p_low < p_high <= 1.0):
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 1.")
    lo = float(np.quantile(x, p_low))
    hi = float(np.quantile(x, p_high))
    return np.clip(x, lo, hi)
```

---

### Quality of Earnings Flags
⚠️ **Red flag: cluster driven by accounting policy differences** (lease capitalization, revenue recognition) rather than underlying economics—normalise first.  
⚠️ **Red flag: outliers ignored** — extreme margins or working capital may indicate one-offs, aggressive capitalisation, or recognition issues.  
⚠️ **Red flag: using market multiples as clustering features** — circularity (clusters reflect valuation, not fundamentals).  
✅ **Green flag: stable clusters** across time windows and robust to small data changes; clusters align with business model logic.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS | High-dimensional KPIs | PCA to reduce; clustering for archetypes (PLG vs enterprise, high vs low retention) |
| Banking | Balance-sheet comparability | Cluster on ROE, CET1, NIM, cost/income; ensure consistent definitions |
| Industrials | Capital intensity drives economics | Include capex/sales, working capital, asset turns; avoid mixing business models |
| Real Estate | Leverage and cap rates dominate | Segment by asset type and leverage; outlier detection for valuation anomalies |

---

### Real-World Example
Scenario: Build peer groups for multiple selection using fundamentals (growth, margin, ROIC, leverage, capex intensity), then flag margin outliers for QoE review.

```python
# The example code above:
# - runs PCA and reports explained variance
# - chooses k using silhouette score
# - fits k-means and assigns cluster labels
# - flags margin outliers with robust z-scores
```

Interpretation:
- Use cluster medians for multiples within each peer archetype.
- Investigate outliers before including them in peer multiple ranges or using them to set forecast guardrails.

See also: Chapter 6 (regularization), Chapter 8 (trees for non-linear segmentation), Chapter 4 (classification when labels exist)
