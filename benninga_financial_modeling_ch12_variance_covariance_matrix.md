# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 12: Calculating the Variance-Covariance Matrix (covariance, correlation, stability, shrinkage basics)

### Core Concept
The variance-covariance matrix Σ captures how asset returns move together and is the central input to portfolio risk, beta estimation, and mean-variance optimization. In valuation, Σ underpins the measurement of systematic risk (covariance with the market), informs cost of capital (via beta and risk premia), and supports scenario-driven risk narratives (e.g., correlation spikes during downturns). Getting Σ wrong (bad data alignment, unstable estimates, outliers) can materially distort discount rates and valuation ranges.

### Formula/Methodology

#### 1) Sample covariance (two assets i and j)
```text
Cov(i,j) = (1/(T-1)) × Σ_{t=1..T} (R_{i,t} - mean(R_i)) × (R_{j,t} - mean(R_j))
```

Where:
- R_{i,t} = return of asset i at time t
- T = number of observations

#### 2) Variance (special case)
```text
Var(i) = Cov(i,i)
σ_i = sqrt(Var(i))
```

#### 3) Correlation
```text
ρ(i,j) = Cov(i,j) / (σ_i × σ_j)
```

#### 4) Portfolio variance (matrix form)
```text
Var(R_p) = w' Σ w
σ_p = sqrt(w' Σ w)
```

#### 5) Annualization (k periods per year)
If covariance is estimated using periodic returns:
```text
Σ_annual ≈ Σ_period × k
```

---

### Practical Application (How to compute Σ correctly in real models)

#### Step 1: Define the return frequency and horizon (align with valuation purpose)
- **Beta / cost of equity**: common choices are daily or weekly returns over 2–5 years; use consistent frequency for stock and market.
- **Portfolio risk (pension/treasury)**: often monthly returns over 3–10 years to reduce noise; may better align with strategic allocation horizons.

Rule of thumb:
- More frequent returns → more observations but more microstructure noise.
- Less frequent returns → fewer observations but smoother estimates.

#### Step 2: Clean and align the data (most common source of errors)
- Use a consistent calendar and **inner join** on dates across assets before covariance.
- Avoid forward-filling prices to create “fake” returns.
- Remove or winsorize extreme outliers if they are data errors (e.g., bad prints).

#### Step 3: Handle missing values explicitly
Preferred:
- Drop rows with any missing return to keep Σ coherent across pairs.
Acceptable (advanced):
- Pairwise covariance (different T for each pair) — can yield non–positive-semidefinite Σ and break optimizers.

#### Step 4: Check stability and conditioning
Diagnostics:
- Look at eigenvalues: negative/near-zero eigenvalues indicate estimation issues or collinearity.
- Condition number: very large values indicate instability.

If Σ is unstable:
- Reduce asset universe (highly collinear assets).
- Use shrinkage (blend sample Σ with a structured target).

#### Step 5: Practical shrinkage (conceptual)
A simple shrinkage estimator:
```text
Σ_shrunk = (1 - λ) × Σ_sample + λ × Σ_target
```
Where:
- 0 ≤ λ ≤ 1 is the shrinkage intensity
- Σ_target could be a diagonal matrix (no correlations) or constant-correlation target

Shrinkage improves robustness when T is not large relative to N.

---

### Python Implementation
```python
from typing import Any, Tuple
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def returns_from_prices(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute returns from a price DataFrame.

    Args:
        prices: DataFrame indexed by date; columns are assets; values are prices (>0)
        method: 'simple' or 'log'

    Returns:
        DataFrame of returns with NaNs dropped (first row removed)

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame.")
    if prices.empty:
        raise ValueError("prices must not be empty.")
    if (prices <= 0).any().any():
        raise ValueError("All prices must be strictly positive.")
    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'.")

    if method == "simple":
        r = prices.pct_change()
    else:
        r = np.log(prices / prices.shift(1))
    return r.dropna(how="all")

def align_returns(returns: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """Align returns across assets by handling missing values.

    Args:
        returns: DataFrame of returns
        how: 'inner' (drop rows with any NaN) or 'pairwise' (leave NaNs; not recommended for optimization)

    Returns:
        aligned returns

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if how not in {"inner", "pairwise"}:
        raise ValueError("how must be 'inner' or 'pairwise'.")
    if how == "inner":
        r = returns.dropna(how="any")
        if r.shape[0] < 3:
            raise ValueError("Not enough observations after dropping NaNs (need >= 3).")
        return r
    return returns

def sample_covariance_matrix(returns: pd.DataFrame, annualize: bool = False, periods_per_year: int = 252) -> pd.DataFrame:
    """Compute sample covariance matrix (optionally annualized).

    Args:
        returns: DataFrame of aligned returns (no NaNs recommended)
        annualize: if True, multiply by periods_per_year
        periods_per_year: e.g., 252 daily, 52 weekly, 12 monthly

    Returns:
        covariance matrix DataFrame

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets (2 columns).")
    if annualize:
        k = int(periods_per_year)
        if k <= 0:
            raise ValueError("periods_per_year must be > 0.")
    cov = returns.cov()
    if annualize:
        cov = cov * periods_per_year
    return cov

def correlation_matrix_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix."""
    if not isinstance(cov, pd.DataFrame) or cov.empty:
        raise ValueError("cov must be a non-empty DataFrame.")
    S = cov.values.astype(float)
    if S.shape[0] != S.shape[1]:
        raise ValueError("cov must be square.")
    std = np.sqrt(np.diag(S))
    if (std <= 0).any():
        raise ValueError("All variances must be > 0 to compute correlation.")
    denom = np.outer(std, std)
    corr = S / denom
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

def is_positive_semidefinite(cov: pd.DataFrame, tol: float = 1e-10) -> bool:
    """Check PSD via eigenvalues."""
    if not isinstance(cov, pd.DataFrame) or cov.empty:
        raise ValueError("cov must be a non-empty DataFrame.")
    S = cov.values.astype(float)
    if S.shape[0] != S.shape[1]:
        raise ValueError("cov must be square.")
    eigvals = np.linalg.eigvalsh(S)
    return bool(eigvals.min() >= -abs(tol))

def shrink_covariance(cov: pd.DataFrame, lam: float = 0.1, target: str = "diagonal") -> pd.DataFrame:
    """Simple covariance shrinkage: (1-λ)Σ + λΣ_target."""
    if not isinstance(cov, pd.DataFrame) or cov.empty:
        raise ValueError("cov must be a non-empty DataFrame.")
    lam = _num(lam, "lam")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be between 0 and 1.")
    if target not in {"diagonal", "constant_correlation"}:
        raise ValueError("target must be 'diagonal' or 'constant_correlation'.")

    S = cov.values.astype(float)
    n = S.shape[0]
    if n != S.shape[1]:
        raise ValueError("cov must be square.")

    if target == "diagonal":
        S_t = np.diag(np.diag(S))
    else:
        corr = correlation_matrix_from_cov(cov).values
        mask = ~np.eye(n, dtype=bool)
        avg_corr = float(np.mean(corr[mask]))
        std = np.sqrt(np.diag(S))
        S_t = np.outer(std, std) * avg_corr
        np.fill_diagonal(S_t, np.diag(S))

    S_sh = (1.0 - lam) * S + lam * S_t
    return pd.DataFrame(S_sh, index=cov.index, columns=cov.columns)

# Example usage (synthetic; replace with real returns)
np.random.seed(11)
dates = pd.date_range("2022-01-01", periods=260, freq="B")
f = np.random.normal(0.0, 0.01, size=len(dates))
rA = 0.0002 + 1.0*f + np.random.normal(0.0, 0.006, size=len(dates))
rB = 0.0001 + 0.8*f + np.random.normal(0.0, 0.007, size=len(dates))
rC = 0.00015 + 0.2*f + np.random.normal(0.0, 0.010, size=len(dates))
rets = pd.DataFrame({"A": rA, "B": rB, "C": rC}, index=dates)

aligned = align_returns(rets, how="inner")
cov = sample_covariance_matrix(aligned, annualize=True, periods_per_year=252)
corr = correlation_matrix_from_cov(cov)

print("Annualized covariance:")
print(cov.round(6))
print("\nCorrelation:")
print(corr.round(3))
print("\nPSD check:", is_positive_semidefinite(cov))

cov_sh = shrink_covariance(cov, lam=0.2, target="diagonal")
print("\nShrunk covariance PSD check:", is_positive_semidefinite(cov_sh))
```

---

### Valuation Impact
Why this matters:
- Beta estimation is covariance-based: inaccurate Σ can misstate beta and therefore cost of equity, WACC, and DCF value.
- Risk narratives and valuation ranges depend on volatility and correlation regimes; Σ is where those assumptions live.

Impact on multiples:
- Higher implied systematic risk → higher discount rates → lower EV/EBITDA and P/E.
- Correlation regime shifts (risk-off) can compress multiples across sectors; showing Σ stress cases supports defensible valuation ranges.

Impact on DCF inputs:
- Cost of equity (CAPM) relies on beta, which relies on covariance with the market.
- Scenario analysis: increasing correlations/volatility can justify higher discount rates or higher distress probabilities.

Comparability issues across companies:
- Betas and risk measures can differ solely due to methodology (frequency/window, total vs price returns, currency).
- Cross-company comparability requires consistent Σ construction.

Practical adjustments:
```python
def beta_from_cov(stock_market_cov: float, market_var: float) -> float:
    """beta = Cov(stock, market) / Var(market)."""
    c = _num(stock_market_cov, "stock_market_cov")
    v = _num(market_var, "market_var")
    if v <= 0:
        raise ValueError("market_var must be > 0.")
    return c / v
```

---

### Quality of Earnings Flags
⚠️ Beta and WACC inputs presented without stating the return window/frequency (non-reproducible).  
⚠️ Covariance computed on misaligned dates (different calendars) or with pairwise deletion producing non-PSD Σ (optimization breaks).  
⚠️ Significant outliers from data errors (splits, bad prints) not cleaned, inflating volatility and beta.  
✅ Robust workflow: clean data, aligned returns, PSD checks, sensitivity to λ shrinkage and horizon.

---

### Sector-Specific Considerations

| Sector | Σ estimation issue | Typical handling |
|---|---|---|
| Small caps | illiquidity and stale prices | use weekly returns; consider shrinkage; treat beta as noisy |
| Emerging markets | currency and market regime shifts | estimate in local currency and/or USD consistently; include regime scenarios |
| Financials | leverage amplifies covariance | use unlevered beta comparisons; adjust for balance sheet leverage |
| Real estate | appraisal smoothing in private markets | use public proxies; apply higher correlation stress cases |

---

### Real-World Example
Scenario: You have returns for a listed peer group and want a stable covariance matrix for portfolio/beta work. Show PSD check and shrinkage if needed.

```python
# Suppose you computed a sample covariance that is not PSD due to missing data/pairwise deletion.
# Use shrinkage to stabilize it.

# cov_sample = cov  # computed above
# if not is_positive_semidefinite(cov_sample):
#     cov_stable = shrink_covariance(cov_sample, lam=0.3, target="diagonal")
# else:
#     cov_stable = cov_sample

# print("Using covariance matrix PSD:", is_positive_semidefinite(cov_stable))
```

Interpretation: A stable PSD covariance matrix is required for optimization and produces more reliable risk estimates. In valuation, this improves defensibility of beta and discount rate assumptions.

See also: Chapter 10 (portfolio basics), Chapter 11 (efficient frontier), Chapter 13 (beta/SML) for how Σ feeds into CAPM-based cost of equity.
