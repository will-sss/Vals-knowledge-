# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 29: Matrices (linear algebra for portfolio risk, regressions, and valuation automation)

### Core Concept
Matrix methods let you express large financial calculations compactly and reliably (portfolio variance, factor models, regression betas, scenario transforms). In valuation and analytics work, matrices reduce spreadsheet complexity, improve auditability, and unlock scalable implementations in Python-in-Excel using NumPy/pandas rather than fragile cell-by-cell formulas.

---

### Formula/Methodology

#### 1) Portfolio variance and volatility (core valuation/asset pricing use)
For weights vector w and covariance matrix Σ:
```text
Portfolio variance = wᵀ Σ w
Portfolio volatility = sqrt(wᵀ Σ w)
```

Where:
- w = vector of portfolio weights (n×1)
- Σ = covariance matrix of asset returns (n×n)
- wᵀ = transpose of w

Key constraints:
```text
Sum of weights = 1
(Optional) weights >= 0  (long-only)
```

#### 2) Covariance matrix construction
If R is a matrix of returns (T observations × n assets), mean-centered:
```text
Σ = (1/(T-1)) · (R_centeredᵀ · R_centered)
```

#### 3) Ordinary Least Squares (OLS) in matrix form (beta estimation)
Given y (T×1), X (T×k) with intercept:
```text
β_hat = (Xᵀ X)^(-1) Xᵀ y
```

Where:
- β_hat = coefficients (k×1)
- (Xᵀ X)^(-1) is the inverse (requires full rank)

Predicted values and residuals:
```text
y_hat = X β_hat
e = y - y_hat
```

#### 4) Factor model covariance (useful for cost of capital / risk decomposition)
If asset returns are driven by factors:
```text
R = B F + ε
```

Then:
```text
Σ_assets = B Σ_factors Bᵀ + Σ_idio
```

Where:
- B = factor loadings (n×k)
- Σ_factors = factor covariance (k×k)
- Σ_idio = diagonal matrix of idiosyncratic variances (n×n)

#### 5) Scenario mapping / stress testing (linear transform)
If you have a vector of shocks z and mapping matrix A:
```text
Shock to outputs = A z
```

Example: map macro shocks (rates, FX, commodity) to revenue/cost impacts across business units.

---

### Practical Application (How to apply in valuation work)

#### A) Replace fragile Excel blocks with matrix operations
Common “sheet bloat” patterns:
- SUMPRODUCT repeated across many rows/cols for portfolio variance
- multi-column regression done with separate helper columns
- manual covariance calculations

Matrix approach:
- Keep data in tidy tables (returns by date, asset columns).
- Use one or two matrix expressions (wᵀΣw, (XᵀX)^(-1)Xᵀy).

#### B) Portfolio risk inputs for valuation assumptions
Matrices directly support:
- estimating beta (levered/unlevered) via OLS
- computing covariance for multi-asset portfolios (if valuing funds/asset managers)
- stress testing valuation drivers with correlated shocks

#### C) Model governance checklist
| Risk | Symptom | Control |
|---|---|---|
| Dimension mismatch | silent wrong result | assert shapes (n, k, T) |
| Singular matrices | inversion fails | regularisation / pseudo-inverse |
| Bad data | NaN/outliers | cleaning + robust checks |
| Look-ahead bias | beta too stable | align dates, lag factors |

---

### Python Implementation
```python
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd

def _as_2d(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D array-like.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr

def portfolio_variance(weights: Any, cov: Any) -> float:
    """Compute portfolio variance w' Σ w with validation.

    Args:
        weights: length-n vector of weights (sums to ~1 recommended)
        cov: (n x n) covariance matrix

    Returns:
        float: portfolio variance

    Raises:
        ValueError: on invalid shapes or non-finite data
    """
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    S = _as_2d(cov, "cov")
    if w.size < 1:
        raise ValueError("weights must be non-empty.")
    n = w.shape[0]
    if S.shape != (n, n):
        raise ValueError(f"cov must be shape ({n},{n}). Got {S.shape}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    var = float((w.T @ S @ w).item())
    if var < -1e-12:
        raise ValueError("Computed variance is negative; check covariance matrix.")
    return max(var, 0.0)

def portfolio_volatility(weights: Any, cov: Any) -> float:
    """Portfolio volatility as sqrt(w' Σ w)."""
    return float(np.sqrt(portfolio_variance(weights, cov)))

def covariance_matrix_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute sample covariance matrix from returns DataFrame.

    Args:
        returns: DataFrame (T rows x n assets) of periodic returns.

    Returns:
        DataFrame: covariance matrix (n x n)

    Raises:
        ValueError: if insufficient rows or non-numeric/NaN
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame.")
    if returns.shape[0] < 2:
        raise ValueError("returns must have at least 2 rows to compute covariance.")
    if returns.isna().any().any():
        raise ValueError("returns contains NaN; clean data before covariance.")
    cov = returns.astype(float).cov()
    if cov.isna().any().any():
        raise ValueError("covariance computation resulted in NaN; check data.")
    return cov

def ols_beta(X: Any, y: Any, add_intercept: bool = True, ridge: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS regression via matrix algebra with optional ridge regularisation.

    Args:
        X: design matrix (T x k) or vector (T,)
        y: target (T,) or (T x 1)
        add_intercept: if True, prepend a column of ones
        ridge: non-negative ridge penalty λ to stabilise inversion

    Returns:
        beta: (k(+1) x 1) coefficient vector
        y_hat: (T x 1) fitted values
        resid: (T x 1) residuals

    Raises:
        ValueError: invalid shapes or singular matrix (if ridge=0 and not invertible)
    """
    Xmat = _as_2d(X, "X")
    yvec = _as_2d(y, "y")
    if yvec.shape[1] != 1:
        raise ValueError("y must be a vector (T x 1).")
    if Xmat.shape[0] != yvec.shape[0]:
        raise ValueError("X and y must have the same number of rows (T).")
    if ridge < 0:
        raise ValueError("ridge must be >= 0.")

    if add_intercept:
        ones = np.ones((Xmat.shape[0], 1), dtype=float)
        Xmat = np.hstack([ones, Xmat])

    XtX = Xmat.T @ Xmat
    if ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0], dtype=float)

    try:
        beta = np.linalg.solve(XtX, Xmat.T @ yvec)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed (singular). Consider ridge>0 or check collinearity.")

    y_hat = Xmat @ beta
    resid = yvec - y_hat
    return beta, y_hat, resid

def beta_to_unlevered(levered_beta: float, tax_rate: float, net_debt: float, equity_value: float) -> float:
    """Unlever beta using a standard Hamada-style adjustment.

    Formula:
        beta_u = beta_l / (1 + (1 - Tc) * D/E)

    Raises:
        ValueError on invalid inputs.
    """
    bl = float(levered_beta)
    Tc = float(tax_rate)
    D = float(net_debt)
    E = float(equity_value)
    if not (np.isfinite(bl) and np.isfinite(Tc) and np.isfinite(D) and np.isfinite(E)):
        raise ValueError("Inputs must be finite.")
    if E <= 0:
        raise ValueError("equity_value must be > 0.")
    if Tc < 0 or Tc > 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    de = D / E
    denom = 1.0 + (1.0 - Tc) * de
    if denom <= 0:
        raise ValueError("Unlevering denominator <= 0; check inputs.")
    return float(bl / denom)

# Example usage: portfolio risk and beta estimation
# Portfolio variance example
cov = np.array([[0.04, 0.01],
                [0.01, 0.09]], dtype=float)
w = np.array([0.6, 0.4], dtype=float)
print(f"Portfolio vol: {portfolio_volatility(w, cov):.2%}")

# OLS beta example: stock returns ~ alpha + beta * market returns
market = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
stock  = np.array([0.012, -0.030, 0.020, 0.006, -0.012])
beta, y_hat, resid = ols_beta(market, stock, add_intercept=True, ridge=1e-8)
alpha = float(beta[0])
b = float(beta[1])
print(f"Alpha: {alpha:.4%}, Beta: {b:.3f}")
```

---

### Valuation Impact
Why this matters:
- Beta estimation and risk decomposition are matrix problems. Reliable betas support defensible cost of equity inputs (CAPM) and thus WACC, directly affecting DCF value.
- Portfolio covariance and factor exposures matter when valuing funds, insurers, asset managers, or any business where risk capital allocation and hedging economics drive value.

Impact on multiples:
- Risk and leverage differences explain why “similar” companies trade on different multiples; matrix-based factor analysis helps adjust comparability.
- High idiosyncratic risk (large residual variance) can justify valuation haircuts or wider discount rate ranges.

Impact on DCF inputs:
- Beta → cost of equity → WACC → DCF value sensitivity.
- Factor models support multi-scenario forecasts (rates/FX/commodity) with consistent correlation structure.

Comparability issues across companies:
- Betas computed on different frequencies, windows, or indices are not comparable. Standardise methods and document matrix inputs.

Practical adjustments:
```python
def cost_of_equity_capm(rf: float, beta: float, erp: float) -> float:
    """CAPM cost of equity: Re = Rf + beta * ERP."""
    rf = float(rf); beta = float(beta); erp = float(erp)
    if not (np.isfinite(rf) and np.isfinite(beta) and np.isfinite(erp)):
        raise ValueError("Inputs must be finite.")
    return float(rf + beta * erp)
```

---

### Quality of Earnings Flags
⚠️ Beta/volatility computed on noisy, illiquid prices without cleaning (stale pricing, outliers), producing unstable discount rates.  
⚠️ Regression uses mismatched time periods (market series not aligned), creating spurious betas.  
⚠️ Collinearity in multi-factor models ignored, leading to unstable coefficients and false precision.  
✅ Documented estimation window, frequency, cleaning steps; stability checks (rolling beta) and sanity bounds.

---

### Sector-Specific Considerations

| Sector | Matrix use | Key issue | Typical treatment |
|---|---|---|---|
| Asset managers | covariance + efficient frontier | estimation error | shrinkage / robust cov |
| Banks | factor/rate shocks mapping | non-linear exposures | scenario matrices + stress |
| Energy/commodities | correlation across drivers | regime shifts | rolling cov + stress regimes |
| Tech growth | beta instability | changing business mix | longer windows + peer unlevering |

---

### Real-World Example
Scenario: Estimate levered beta from returns, unlever to compare peers, then compute cost of equity.

```python
rf = 0.03
erp = 0.05

market = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005])
stock  = np.array([0.012, -0.030, 0.020, 0.006, -0.012, 0.026, -0.006])

beta_vec, _, _ = ols_beta(market, stock, add_intercept=True, ridge=1e-8)
beta_l = float(beta_vec[1])

beta_u = beta_to_unlevered(beta_l, tax_rate=0.25, net_debt=200.0, equity_value=800.0)
re = cost_of_equity_capm(rf, beta_l, erp)

print(f"Levered beta: {beta_l:.3f}")
print(f"Unlevered beta: {beta_u:.3f}")
print(f"Cost of equity: {re:.2%}")
```

Interpretation: Unlevered betas improve comparability across capital structures. Re-lever the peer median to the target’s debt/equity to estimate a defensible beta for WACC.

See also: Chapter 12 (variance-covariance), Chapter 13 (betas & SML), Chapter 25 (VaR uses covariance), Chapter 3 (WACC via beta/cost of equity).
