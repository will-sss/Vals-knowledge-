# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 3: Linear Regression (OLS, Interpretation, Diagnostics, Prediction Intervals, Feature Engineering)

### Core Concept
Linear regression models the relationship between a numeric outcome and one or more predictors using a linear approximation. In valuation, it is most useful for **driver-based forecasting** (revenue vs volume/price, margins vs utilisation), **normalisation** (cycle-adjusted margins, working capital vs sales), and **relative valuation support** (explaining multiples using fundamentals). The practical challenge is ensuring assumptions are not violated (outliers, non-linearity, multicollinearity) and avoiding data leakage.

See also: Chapter 6 (regularization), Chapter 5 (resampling/cross-validation), Chapter 4 (classification for binary outcomes).

---

### Formula/Methodology

#### 1) Simple linear regression
```text
Y = β0 + β1 X + ε
```
Where:
- Y = response (e.g., EBITDA margin, revenue growth)
- X = predictor (e.g., GDP growth, volume)
- β0 = intercept
- β1 = slope (marginal effect of X on Y)
- ε = error term

Prediction:
```text
Ŷ = β0 + β1 X
```

#### 2) Multiple linear regression
```text
Y = β0 + β1 X1 + β2 X2 + ... + βp Xp + ε
```

Matrix form:
```text
y = Xβ + ε
```
OLS solution:
```text
β̂ = (X'X)^(-1) X'y
```
Conditions:
- X'X must be invertible (or use pseudo-inverse / regularization)

#### 3) Residuals and fit metrics
Residual:
```text
e_i = y_i − ŷ_i
```
R-squared:
```text
R^2 = 1 − SSE / SST
SSE = Σ (y_i − ŷ_i)^2
SST = Σ (y_i − ȳ)^2
```
Standard error of regression:
```text
σ̂ = sqrt( SSE / (n − p − 1) )
```

#### 4) Prediction intervals (practical forecasting range)
For a new observation x0 in multiple regression:
```text
Prediction Interval: ŷ0 ± t_{α/2, n−p−1} × σ̂ × sqrt(1 + x0' (X'X)^(-1) x0)
```
Interpretation:
- Wider than confidence interval because it includes irreducible error (the “+1” term).

#### 5) Practical assumptions (what breaks models)
Key checks:
- Non-linearity: residual patterns vs fitted values
- Heteroskedasticity: residual variance increases with fitted values
- Outliers / high leverage points: can dominate β̂
- Multicollinearity: unstable coefficients and noisy forecasts
- Time-series leakage: random splitting across time is invalid for forecasting

---

### Python Implementation
```python
from typing import Dict, Tuple, Optional
import numpy as np


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have non-zero shape.")
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


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add an intercept column of 1s."""
    X = _as_2d(X)
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


def ols_fit(X: np.ndarray, y: np.ndarray, add_const: bool = True) -> Dict[str, np.ndarray]:
    """
    Fit OLS using pseudo-inverse (robust to non-invertible X'X).

    Args:
        X: predictors, shape (n, p)
        y: response, shape (n,)
        add_const: add intercept if True

    Returns:
        dict with keys:
          beta: coefficients (p+1,) if add_const else (p,)
          yhat: fitted values (n,)
          resid: residuals (n,)
          sse: SSE scalar
          r2: R-squared scalar
          sigma_hat: residual std error estimate
          XtX_inv: (X'X)^(-1) using pseudo-inverse (for intervals)

    Raises:
        ValueError: input shape issues, n too small.
    """
    X = _as_2d(X)
    y = _as_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if add_const:
        Xd = add_intercept(X)
    else:
        Xd = X.copy()

    n, p = Xd.shape
    if n <= p:
        raise ValueError("Need n > number of parameters for OLS diagnostics.")

    # Pseudo-inverse for stability
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ y)
    yhat = Xd @ beta
    resid = y - yhat

    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    sigma_hat = float(np.sqrt(sse / (n - p)))

    return {
        "beta": beta,
        "yhat": yhat,
        "resid": resid,
        "sse": np.array([sse], dtype=float),
        "r2": np.array([r2], dtype=float),
        "sigma_hat": np.array([sigma_hat], dtype=float),
        "XtX_inv": XtX_inv
    }


def predict(X_new: np.ndarray, beta: np.ndarray, add_const: bool = True) -> np.ndarray:
    """Predict using fitted beta."""
    X_new = _as_2d(X_new)
    beta = np.asarray(beta, dtype=float).ravel()
    if add_const:
        Xn = add_intercept(X_new)
    else:
        Xn = X_new
    if Xn.shape[1] != beta.shape[0]:
        raise ValueError("X_new columns (after intercept) must match beta length.")
    return Xn @ beta


def prediction_interval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_new: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate prediction interval for OLS using normal critical value (z).
    (In production, use t critical values; here we keep it dependency-light.)

    Args:
        X_train, y_train: training data
        X_new: new points (m, p)
        alpha: significance (0.05 => 95% PI)

    Returns:
        (yhat, lower, upper) arrays of shape (m,)

    Raises:
        ValueError: if alpha invalid or fit fails.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    fit = ols_fit(X_train, y_train, add_const=True)
    beta = fit["beta"]
    sigma = float(fit["sigma_hat"][0])
    XtX_inv = fit["XtX_inv"]

    X_new = _as_2d(X_new)
    Xn = add_intercept(X_new)
    yhat = Xn @ beta

    # z critical (approx): 1.96 for 95%, 2.58 for 99%
    # Use inverse error function approximation for general alpha.
    def z_crit(a: float) -> float:
        # Approximate z_{1-a/2} via scipy-free approximation.
        # Peter J. Acklam approximation is longer; keep simple:
        # Use numpy's quantile of standard normal via erfinv.
        return float(np.sqrt(2.0) * np.erfinv(1.0 - a))

    z = z_crit(alpha)
    # Standard error for prediction: sigma * sqrt(1 + x0'(X'X)^{-1}x0)
    se_pred = np.sqrt(1.0 + np.sum((Xn @ XtX_inv) * Xn, axis=1)) * sigma
    lower = yhat - z * se_pred
    upper = yhat + z * se_pred
    return yhat, lower, upper


def vif(X: np.ndarray) -> np.ndarray:
    """
    Variance Inflation Factor for each feature (no intercept).
    VIF_j = 1 / (1 - R_j^2) where R_j^2 from regressing X_j on other Xs.

    Returns:
        np.ndarray shape (p,) VIFs

    Raises:
        ValueError: if p < 2 or insufficient rows.
    """
    X = _as_2d(X)
    n, p = X.shape
    if p < 2:
        raise ValueError("Need at least 2 predictors to compute VIF.")
    vifs = np.zeros(p, dtype=float)
    for j in range(p):
        yj = X[:, j]
        X_others = np.delete(X, j, axis=1)
        fit_j = ols_fit(X_others, yj, add_const=True)
        r2_j = float(fit_j["r2"][0])
        if not np.isfinite(r2_j) or r2_j >= 1.0:
            vifs[j] = float("inf")
        else:
            vifs[j] = 1.0 / (1.0 - r2_j)
    return vifs


# Example usage (valuation context): forecast EBITDA margin from utilisation and input cost inflation
X = np.array([
    [0.70, 0.03],
    [0.75, 0.02],
    [0.65, 0.05],
    [0.80, 0.01],
    [0.60, 0.06],
    [0.85, 0.00],
], dtype=float)  # [capacity_utilisation, input_inflation]
y = np.array([0.14, 0.15, 0.11, 0.16, 0.10, 0.17], dtype=float)  # EBITDA margin

fit = ols_fit(X, y, add_const=True)
beta = fit["beta"]
print("Beta (intercept, util, infl):", [round(x, 4) for x in beta.tolist()])
print("R^2:", float(fit["r2"][0]))

# Predict next-year margin under a new operating environment
X_new = np.array([[0.78, 0.025]], dtype=float)
yhat, lo, hi = prediction_interval(X, y, X_new, alpha=0.05)
print(f"Predicted margin: {float(yhat[0]):.2%} (PI: {float(lo[0]):.2%} to {float(hi[0]):.2%})")

# Multicollinearity check
print("VIFs:", [round(v, 2) if np.isfinite(v) else v for v in vif(X).tolist()])
```

---

### Valuation Impact
Why this matters:
- **Forecasting (Chapter 9 alignment):** Linear regression is the workhorse for linking operational drivers to forecasted revenue, costs, and margins. It provides a transparent way to justify assumptions in a DCF.
- **Normalisation:** Regression helps estimate “normal” margins/working capital conditional on macro conditions, aiding cycle-adjusted valuation and comparability across periods.
- **Relative valuation support:** You can explain why a company deserves a multiple premium/discount by regressing multiples on fundamentals (growth, ROIC, risk proxies).

Impact on multiples:
- Multiples regressions are sensitive to outliers and accounting noise. Use robust diagnostics and avoid “fitting” a multiple to justify a preconceived answer.
- If explanatory variables overlap (multicollinearity), coefficient signs can flip. Use regularization (Chapter 6) or simplify.

Impact on DCF inputs:
- Regression outputs can inform base-case growth/margins and feed uncertainty bands (prediction intervals), supporting scenario and Monte Carlo valuation.

Practical adjustments (making regression outputs valuation-ready):
```python
import numpy as np

def winsorize(x: np.ndarray, p: float = 0.05) -> np.ndarray:
    """
    Winsorize array to reduce outlier impact.

    Args:
        x: numeric array
        p: tail proportion clipped at both ends (e.g., 0.05)

    Returns:
        winsorized array
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must not be empty.")
    if np.any(~np.isfinite(x)):
        raise ValueError("x must be finite.")
    if not (0.0 <= p < 0.5):
        raise ValueError("p must be in [0, 0.5).")
    lo = np.quantile(x, p)
    hi = np.quantile(x, 1.0 - p)
    return np.clip(x, lo, hi)
```

---

### Quality of Earnings Flags
⚠️ **Red flag: one-off periods drive coefficients** — restructuring year, exceptional pricing, or temporary subsidies contaminate “normal” relationships.  
⚠️ **Red flag: non-stationarity** — business model shifts (new product, new geography) make historical coefficients unreliable for forward forecasts.  
⚠️ **Red flag: multicollinearity** — overlapping drivers (e.g., volume and utilisation) produce unstable coefficients; forecast becomes fragile.  
⚠️ **Red flag: heteroskedasticity** — variability increases at high growth; prediction intervals must widen.  
✅ **Green flag: stable coefficients across time splits** — similar betas and errors across pre/post periods; drivers align with operational logic.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS | Unit economics and cohort effects | Use cohort/time-based splits; include retention/CAC drivers; avoid pooled regression leakage |
| Banking | Loss provisions and macro links | Model charge-offs vs unemployment/interest rates; watch regime changes |
| Industrials | Capacity constraints | Include utilisation and backlog; consider non-linear terms (piecewise) |
| Energy/commodities | Price dominates | Use scenarios by price regime; regression for operating leverage but not as sole forecast tool |

---

### Real-World Example
Scenario: You need a defendable EBITDA margin forecast for a cyclical industrial. You regress historical margin on utilisation and input inflation, then produce a prediction interval for next year.

```python
# The example code above prints:
# - estimated coefficients
# - R^2
# - predicted margin and prediction interval
# - VIFs for multicollinearity check
```

Interpretation:
- If VIFs are high (e.g., > 5–10), simplify features or use regularization (Chapter 6).
- Use the prediction interval to set base/bear/bull cases and avoid overstating confidence in the base-case margin.

See also: Chapter 6 (regularization), Chapter 5 (cross-validation), Chapter 8 (tree methods for non-linearities)
