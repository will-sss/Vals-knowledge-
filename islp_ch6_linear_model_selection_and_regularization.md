# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 6: Linear Model Selection and Regularization (Feature Selection, Ridge, Lasso, Elastic Net, Cross-Validation)

### Core Concept
Linear model selection and regularization control model complexity to improve **out-of-sample forecasting** and reduce coefficient instability from multicollinearity. In valuation, these methods are practical when you have many potential drivers (macro, operational KPIs, pricing, cohorts) and need forecasts that are **stable, interpretable, and resistant to overfitting**—especially when forecasting margins, growth, working capital, and credit losses.

See also: Chapter 3 (OLS), Chapter 5 (cross-validation), Chapter 2 (bias–variance), Chapter 8 (tree-based methods).

---

### Formula/Methodology

#### 1) Why OLS fails in high-dimensional or correlated drivers
- When predictors are highly correlated, OLS coefficients become unstable (small data changes → big beta changes).
- When p is large relative to n, OLS can overfit (low training error, high test error).

#### 2) Ridge regression (L2 regularization)
Objective function:
```text
Minimise:  SSE(β) + λ × Σ_{j=1..p} β_j^2
SSE(β) = Σ (y_i − β0 − Σ β_j x_{ij})^2
```
Where:
- λ ≥ 0 is the penalty strength (higher λ ⇒ more shrinkage)
- β0 (intercept) is typically not penalised
- Ridge shrinks coefficients toward 0 but rarely sets them exactly to 0

Practical effects:
- Reduces variance, improves stability under multicollinearity
- Keeps all features, useful when many small effects exist

#### 3) Lasso (L1 regularization)
Objective function:
```text
Minimise:  SSE(β) + λ × Σ_{j=1..p} |β_j|
```
Key feature:
- Can set some β_j exactly to 0 ⇒ implicit feature selection

Practical effects:
- Produces sparse models (easier to explain)
- Can be unstable when predictors are highly correlated (may pick one and drop others)

#### 4) Elastic Net (L1 + L2)
Objective function:
```text
Minimise:  SSE(β) + λ × [ α × Σ |β_j| + (1 − α) × Σ β_j^2 ]
```
Where:
- α in [0,1] blends lasso (α=1) and ridge (α=0)

Practical use:
- Better than pure lasso when features are correlated (groups of drivers)

#### 5) Model selection by cross-validation (CV)
General rule:
- Choose λ (and α) that minimises **CV error**.

K-fold CV process:
```text
Split data into K folds.
For each fold:
  fit on K−1 folds, evaluate on held-out fold.
CV error = average error across folds.
Select λ* = argmin CV error.
```

#### 6) Standardisation (critical for fair penalties)
Penalties depend on coefficient scale, so standardise predictors:
```text
x_j_standardised = (x_j − mean(x_j)) / sd(x_j)
```
Practical rule:
- Always standardise X for ridge/lasso/elastic net (except binary dummies if already comparable scale).

---

### Python Implementation
```python
from typing import Dict, Tuple, List, Optional
import numpy as np


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array.")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN/Inf).")
    return X


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")
    if np.any(~np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN/Inf).")
    return y


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize columns of X: (X - mean) / std, with safe handling for zero-std columns.

    Returns:
        Xs, mean, std
    """
    X = _as_2d(X)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    # Prevent divide-by-zero: keep zero-variance columns unscaled (std=1)
    sd_safe = np.where(sd == 0.0, 1.0, sd)
    Xs = (X - mu) / sd_safe
    return Xs, mu, sd_safe


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float, add_const: bool = True) -> Dict[str, np.ndarray]:
    """
    Closed-form ridge regression (L2) using linear algebra.

    Objective:
        minimise ||y - Xb||^2 + lam * ||b||^2

    Notes:
        - intercept is not penalised if add_const=True

    Args:
        X: predictors (n, p)
        y: response (n,)
        lam: penalty λ >= 0
        add_const: add intercept column

    Returns:
        dict: beta (p+1,), yhat (n,), resid (n,)

    Raises:
        ValueError: invalid lam or inputs.
    """
    X = _as_2d(X)
    y = _as_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be >= 0.")

    if add_const:
        Xd = np.column_stack([np.ones(X.shape[0], dtype=float), X])
        p = Xd.shape[1]
        P = np.eye(p)
        P[0, 0] = 0.0  # do not penalise intercept
    else:
        Xd = X
        p = Xd.shape[1]
        P = np.eye(p)

    # Solve (X'X + lam*P) beta = X'y
    A = Xd.T @ Xd + lam * P
    b = Xd.T @ y
    beta = np.linalg.solve(A, b)
    yhat = Xd @ beta
    resid = y - yhat
    return {"beta": beta, "yhat": yhat, "resid": resid}


def soft_threshold(z: np.ndarray, gamma: float) -> np.ndarray:
    """Soft-thresholding operator used in lasso coordinate descent."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def lasso_fit_cd(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 2000) -> Dict[str, np.ndarray]:
    """
    Lasso regression via coordinate descent on standardized X (no intercept in penalization step).

    Objective:
        minimise (1/2n) * ||y - (b0 + Xb)||^2 + lam * ||b||_1

    Args:
        X: predictors (n, p)
        y: response (n,)
        lam: penalty >= 0 (on standardized scale)
        iters: iterations

    Returns:
        dict: intercept, beta (p,), yhat (n,)

    Raises:
        ValueError: invalid inputs or lam.
    """
    X = _as_2d(X)
    y = _as_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be >= 0.")
    if iters <= 0:
        raise ValueError("iters must be > 0.")

    # Standardize X and center y
    Xs, mu, sd = standardize(X)
    y_mean = float(np.mean(y))
    yc = y - y_mean

    n, p = Xs.shape
    beta = np.zeros(p, dtype=float)

    # Precompute column norms
    col_norm = np.sum(Xs * Xs, axis=0) / n
    col_norm = np.where(col_norm == 0.0, 1.0, col_norm)

    for _ in range(iters):
        # Coordinate updates
        for j in range(p):
            # partial residual excluding feature j
            r = yc - (Xs @ beta) + Xs[:, j] * beta[j]
            rho = np.dot(Xs[:, j], r) / n
            beta[j] = soft_threshold(rho, lam) / col_norm[j]

    # Convert back to original scale
    beta_unscaled = beta / sd
    intercept = y_mean - float(np.dot(mu, beta_unscaled))
    yhat = intercept + X @ beta_unscaled
    return {"intercept": np.array([intercept], dtype=float), "beta": beta_unscaled, "yhat": yhat}


def kfold_indices(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    """Generate k folds of indices (time-agnostic)."""
    if not isinstance(n, int) or n <= 1:
        raise ValueError("n must be integer > 1.")
    if not isinstance(k, int) or k < 2 or k > n:
        raise ValueError("k must be in [2, n].")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [f for f in folds]


def cv_select_lambda_ridge(X: np.ndarray, y: np.ndarray, lambdas: List[float], k: int = 5, seed: int = 0) -> Dict[str, float]:
    """
    Select ridge lambda via K-fold CV using MSE.

    Returns:
        dict: best_lambda, best_cv_mse
    """
    X = _as_2d(X)
    y = _as_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if not isinstance(lambdas, list) or len(lambdas) == 0:
        raise ValueError("lambdas must be a non-empty list.")
    for lam in lambdas:
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("All lambdas must be finite and >= 0.")

    folds = kfold_indices(len(y), k=k, seed=seed)
    best_lam, best_mse = None, float("inf")

    for lam in lambdas:
        mses = []
        for fold in folds:
            test_idx = fold
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)
            fit = ridge_fit(X[train_idx], y[train_idx], lam=lam, add_const=True)
            beta = fit["beta"]
            # Predict
            Xd_te = np.column_stack([np.ones(test_idx.size, dtype=float), X[test_idx]])
            yhat_te = Xd_te @ beta
            err = y[test_idx] - yhat_te
            mses.append(float(np.mean(err * err)))
        cv_mse = float(np.mean(mses))
        if cv_mse < best_mse:
            best_mse, best_lam = cv_mse, float(lam)

    return {"best_lambda": best_lam, "best_cv_mse": best_mse}


# Example usage (valuation context): forecast revenue growth using many correlated drivers
# Features: [GDP, CPI, FX, marketing_spend_growth, churn_change, competitor_index]
X = np.array([
    [0.02, 0.03, -0.01, 0.10, -0.02, 0.05],
    [0.01, 0.04,  0.00, 0.08, -0.01, 0.06],
    [0.00, 0.05,  0.02, 0.05,  0.01, 0.08],
    [-0.01, 0.04, 0.01, 0.02,  0.03, 0.09],
    [0.02, 0.03, -0.02, 0.12, -0.03, 0.04],
    [0.03, 0.02, -0.01, 0.15, -0.04, 0.03],
], dtype=float)

y = np.array([0.09, 0.07, 0.05, 0.02, 0.10, 0.12], dtype=float)  # revenue growth

# Select ridge lambda by CV
lams = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
sel = cv_select_lambda_ridge(X, y, lambdas=lams, k=3, seed=1)
print("Ridge best lambda:", sel["best_lambda"], "CV MSE:", round(sel["best_cv_mse"], 6))

# Fit ridge at best lambda
fit_r = ridge_fit(X, y, lam=sel["best_lambda"], add_const=True)
beta_r = fit_r["beta"]
print("Ridge beta (incl intercept):", [round(b, 4) for b in beta_r.tolist()])

# Fit lasso (standardized coordinate descent)
fit_l = lasso_fit_cd(X, y, lam=0.02, iters=3000)
print("Lasso intercept:", round(float(fit_l["intercept"][0]), 4))
print("Lasso beta:", [round(b, 4) for b in fit_l["beta"].tolist()])
```

---

### Valuation Impact
Why this matters:
- **Forecast stability:** Regularization reduces “whipsaw” forecasts driven by noise, stabilising revenue and margin projections in DCF models.
- **Driver selection:** Lasso/elastic net can identify which KPIs matter most (e.g., churn change vs marketing spend), improving narrative defensibility in valuation reports.
- **Comparability:** Consistent forecasting methodology across companies reduces model risk when comparing multiples or forecasting peer performance.

Impact on multiples:
- When explaining multiples using fundamentals (growth, ROIC, margins), regularization reduces spurious driver significance and improves out-of-sample explanatory power.
- Helps avoid “overfitted multiple models” used to rationalise a desired valuation.

Impact on DCF inputs:
- Regularized models provide more robust estimates for growth and margin ramps; use CV-selected λ and stress test outputs.
- Elastic net is often preferred when drivers are correlated (common in macro + operating KPI sets).

Practical adjustments (turning model outputs into valuation inputs):
```python
import numpy as np

def cap_forecast(forecast: float, low: float, high: float) -> float:
    """
    Cap forecast to a reasonable valuation range (guardrail).

    Args:
        forecast: numeric forecast (e.g., growth)
        low/high: bounds

    Returns:
        float: capped forecast

    Raises:
        ValueError: invalid bounds or non-finite values.
    """
    for name, v in {"forecast": forecast, "low": low, "high": high}.items():
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite.")
    if low > high:
        raise ValueError("low must be <= high.")
    return float(np.clip(forecast, low, high))
```

---

### Quality of Earnings Flags
⚠️ **Red flag: using revised/restated KPIs** (not available at forecast date) increases apparent predictive power (leakage).  
⚠️ **Red flag: penalty chosen on training error** (instead of CV) → overfitting remains.  
⚠️ **Red flag: no standardisation** → penalties unfairly target small-scale variables, distorting driver selection.  
⚠️ **Red flag: correlated drivers with lasso** → unstable variable selection; consider elastic net or ridge.  
✅ **Green flag: time-based validation** (when forecasting) and stable coefficient paths across λ choices.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS | Many KPIs, strong correlation (CAC, growth, retention) | Elastic net; time-split CV; keep interpretability for investment committee |
| Banking | Macro + balance-sheet drivers correlated | Ridge for stability; model drift monitoring across rate regimes |
| Consumer | Promotional variables highly collinear | Ridge/elastic net; feature grouping and guardrails |
| Industrials | Cyclical macro drivers | Ridge + scenario overlays; avoid single-model dependence |

---

### Real-World Example
Scenario: You have many correlated drivers for revenue growth. OLS produces unstable betas. Use ridge with cross-validation to stabilise growth forecasts and improve out-of-sample MSE.

```python
# The example code above:
# - selects ridge lambda via K-fold CV
# - fits ridge and prints stable coefficients
# - fits lasso (sparse) as a driver-selection alternative
```

Interpretation:
- Use ridge when you believe many drivers matter and want stability.
- Use lasso when you need a short driver list for explanation, but validate stability (especially with correlated inputs).

See also: Chapter 5 (cross-validation), Chapter 8 (trees for non-linearities), Chapter 12 (unsupervised learning for feature discovery)
