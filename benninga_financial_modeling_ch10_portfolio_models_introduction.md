# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 10: Portfolio Models—Introduction (returns, risk, diversification, efficient frontier setup)

### Core Concept
Portfolio models formalize how combining assets changes risk and return through diversification. For valuation and corporate finance, these tools underpin (i) estimating equity risk (beta) and cost of equity via CAPM, (ii) stress-testing portfolio exposures (treasury, pension assets), and (iii) translating market data (returns, covariances) into discount rate inputs and risk narratives.

### Formula/Methodology

#### 1) Simple and log returns
```text
Simple return:
R_t = (P_t + D_t - P_{t-1}) / P_{t-1}

Log return:
r_t = ln(P_t / P_{t-1})
```

Where:
- P_t = price at time t
- D_t = cash distributions during period t (dividends, coupons)

#### 2) Expected return (sample mean)
```text
Expected return (sample):
E[R] ≈ (1/T) × Σ_{t=1..T} R_t
```

#### 3) Variance and volatility
```text
Variance:
Var(R) = (1/(T-1)) × Σ (R_t - mean(R))^2

Volatility:
σ = sqrt(Var(R))
```

Annualization (assuming i.i.d. returns, k periods per year):
```text
Mean_annual ≈ Mean_period × k
Vol_annual ≈ Vol_period × sqrt(k)
```

#### 4) Portfolio expected return (weights sum to 1)
```text
E[R_p] = Σ_{i=1..N} w_i × E[R_i]

Constraints:
Σ w_i = 1
```

#### 5) Portfolio variance (covariance matrix form)
```text
Var(R_p) = w' Σ w

Where:
w = vector of portfolio weights (N×1)
Σ = covariance matrix of asset returns (N×N)
```

Expanded (two assets):
```text
Var(R_p) = w1^2 σ1^2 + w2^2 σ2^2 + 2 w1 w2 Cov(1,2)
Cov(1,2) = ρ12 σ1 σ2
```

#### 6) Correlation (scale-free dependence)
```text
ρ12 = Cov(1,2) / (σ1 σ2)
```

#### 7) Sharpe ratio (excess return per unit risk)
```text
Sharpe = (E[R_p] - R_f) / σ_p
```

Where:
- R_f = risk-free rate over the same period (decimal)

---

### Practical Application (How to apply in valuation work)

#### Step 1: Build return series from prices (cleaning matters)
- Align dates across assets; handle missing values (forward-fill is usually wrong for returns; prefer inner join on common dates).
- Use total return (include dividends) when estimating equity risk; if only prices are available, note the limitation.

#### Step 2: Estimate covariance matrix for risk inputs
- For beta estimation and CAPM, you need covariance of stock returns with market returns.
- For treasury/pension portfolios, you need Σ across asset classes (equities, bonds, alternatives).

#### Step 3: Use diversification to interpret “risk”
- Idiosyncratic risk diversifies away; systematic risk (market-related) does not.
- This is the bridge to cost of equity: CAPM prices systematic risk via beta.

#### Step 4: Sensitivity and scenario narratives
- Stress correlations (they often rise in crises): recalc portfolio σ_p under higher ρ assumptions.
- For valuation, a rising correlation regime increases portfolio risk and can justify higher discount rates or wider valuation ranges in volatile markets.

#### Step 5: Portfolio logic in capital structure decisions
- A company with concentrated cash/investment holdings has “asset-side” risk that can offset operating risk.
- Conversely, pension asset risk can add equity-like volatility to enterprise risk; include in risk narrative and, where material, scenario analysis.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def returns_from_prices(prices: pd.Series, method: str = "simple") -> pd.Series:
    """Compute returns from a price series.

    Args:
        prices: pandas Series indexed by date, values are prices (must be positive)
        method: 'simple' or 'log'

    Returns:
        Series of returns aligned to periods (first return is NaN, dropped)

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a pandas Series.")
    if prices.empty:
        raise ValueError("prices must not be empty.")
    if (prices <= 0).any():
        raise ValueError("prices must be strictly positive.")

    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'.")

    if method == "simple":
        r = prices.pct_change()
    else:
        r = np.log(prices / prices.shift(1))
    return r.dropna()

def annualize_mean_vol(mean_period: float, vol_period: float, periods_per_year: int) -> Tuple[float, float]:
    """Annualize mean and volatility from periodic figures.

    Args:
        mean_period: mean return per period (decimal)
        vol_period: volatility per period (decimal)
        periods_per_year: e.g., 252 for daily, 12 for monthly

    Returns:
        (mean_annual, vol_annual)
    """
    m = _num(mean_period, "mean_period")
    v = _num(vol_period, "vol_period")
    k = int(periods_per_year)
    if k <= 0:
        raise ValueError("periods_per_year must be > 0.")
    if v < 0:
        raise ValueError("vol_period must be >= 0.")
    return m * k, v * np.sqrt(k)

def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute sample covariance matrix from returns.

    Args:
        returns: DataFrame of returns with columns = assets

    Returns:
        covariance matrix DataFrame

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if returns.shape[1] < 2:
        raise ValueError("returns must have at least 2 assets (2 columns).")
    # drop rows with any missing returns to avoid inconsistent pairwise covariances
    r = returns.dropna(how="any")
    if r.shape[0] < 3:
        raise ValueError("Not enough observations after dropping NaNs (need >= 3).")
    return r.cov()

def portfolio_return(mean_returns: pd.Series, weights: np.ndarray) -> float:
    """Portfolio expected return given mean returns and weights.

    Constraints not enforced automatically: weights should sum to 1.

    Args:
        mean_returns: Series indexed by asset
        weights: numpy array of weights (N,)

    Returns:
        expected portfolio return (decimal)

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(mean_returns, pd.Series):
        raise ValueError("mean_returns must be a pandas Series.")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    if len(w) != len(mean_returns):
        raise ValueError("weights length must match mean_returns length.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    return float(np.dot(w, mean_returns.values))

def portfolio_variance(cov: pd.DataFrame, weights: np.ndarray) -> float:
    """Portfolio variance Var = w' Σ w.

    Args:
        cov: covariance matrix DataFrame (N×N)
        weights: numpy array of weights (N,)

    Returns:
        portfolio variance (decimal^2)

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a pandas DataFrame.")
    S = cov.values
    w = np.asarray(weights, dtype=float)
    if S.shape[0] != S.shape[1]:
        raise ValueError("cov must be square.")
    if len(w) != S.shape[0]:
        raise ValueError("weights length must match covariance size.")
    if not np.all(np.isfinite(S)):
        raise ValueError("covariance matrix must be finite.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    return float(w.T @ S @ w)

def sharpe_ratio(mean_portfolio: float, vol_portfolio: float, risk_free: float) -> float:
    """Sharpe ratio using consistent periodic inputs."""
    mu = _num(mean_portfolio, "mean_portfolio")
    sig = _num(vol_portfolio, "vol_portfolio")
    rf = _num(risk_free, "risk_free")
    if sig <= 0:
        raise ValueError("vol_portfolio must be > 0.")
    return (mu - rf) / sig

# Example usage (synthetic returns to illustrate; replace with real data in practice)
np.random.seed(7)
dates = pd.date_range("2023-01-01", periods=252, freq="B")
# Create two correlated return series
r1 = np.random.normal(0.0004, 0.012, size=len(dates))
r2 = 0.3 * r1 + np.random.normal(0.0002, 0.010, size=len(dates))
rets = pd.DataFrame({"Asset_A": r1, "Asset_B": r2}, index=dates)

mean_rets = rets.mean()
cov = covariance_matrix(rets)

weights = np.array([0.6, 0.4])
mu_p = portfolio_return(mean_rets, weights)
var_p = portfolio_variance(cov, weights)
sig_p = np.sqrt(var_p)

rf_daily = 0.0001  # illustrative
sr = sharpe_ratio(mu_p, sig_p, rf_daily)

mu_a, sig_a = annualize_mean_vol(mu_p, sig_p, periods_per_year=252)

print(f"Daily E[R_p]: {mu_p:.5f}, Daily σ_p: {sig_p:.5f}, Daily Sharpe: {sr:.2f}")
print(f"Annualized E[R_p]: {mu_a:.2%}, Annualized σ_p: {sig_a:.2%}")
```

---

### Valuation Impact
Why this matters:
- Portfolio math is the foundation for beta estimation and CAPM-based cost of equity, which feeds directly into WACC and DCF valuation.
- Covariance and correlation structure drives risk in corporate pension assets, treasury portfolios, and hedging programs, affecting equity risk and valuation uncertainty.

Impact on multiples:
- Higher perceived systematic risk (higher beta / higher covariance with market) raises discount rates and typically compresses valuation multiples.
- Stress correlations can explain multiple compression during crises (risk-on/risk-off regimes).

Impact on DCF inputs:
- Cost of equity depends on beta, which is estimated from returns and covariances.
- Scenario analysis: increased volatility and correlation can justify higher discount rates or wider valuation ranges.

Comparability issues across companies:
- Different geographic/sector exposures and business mixes alter covariance with the market; comparing multiples without risk normalization can mislead.

Practical adjustments:
```python
def normalize_beta_for_leverage(beta_unlevered: float, debt_to_equity: float, tax_rate: float) -> float:
    """Relever beta using a standard Hamada-style adjustment.

    beta_L = beta_U × [1 + (1 - Tc) × (D/E)]
    """
    bu = _num(beta_unlevered, "beta_unlevered")
    de = _num(debt_to_equity, "debt_to_equity")
    tc = _num(tax_rate, "tax_rate")
    if de < 0:
        raise ValueError("debt_to_equity must be >= 0.")
    if not (0.0 <= tc <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    return bu * (1.0 + (1.0 - tc) * de)
```

---

### Quality of Earnings Flags
⚠️ Management highlights “low risk” due to diversification but exposures are actually highly correlated in downturns (hidden concentration).  
⚠️ Beta or volatility computed on short windows or cherry-picked dates (unstable, biased).  
⚠️ Using price returns instead of total returns for dividend-paying stocks (understates expected return and can distort beta).  
✅ Clear methodology: frequency, window length, total-return basis, robust handling of missing data, and sensitivity checks.

---

### Sector-Specific Considerations

| Sector | Portfolio-model relevance | Typical handling |
|---|---|---|
| Asset managers | portfolio risk is the product | covariance matrix and factor exposures are core KPIs |
| Insurers / pensions | ALM and solvency depend on asset risk | stress correlations; scenario-based capital impacts |
| Banks | trading books & hedges | VaR and stress models build on covariance/correlation |
| Non-financial corporates | pension/treasury can affect equity risk | incorporate pension asset volatility into risk narrative |

---

### Real-World Example
Scenario: A company has a pension plan invested 60% equities and 40% bonds. Rising equity-bond correlation increases pension asset volatility, potentially increasing equity risk perception and valuation uncertainty.

```python
# Illustrative: compute portfolio vol under two correlation scenarios
sigma_eq = 0.16
sigma_bd = 0.06
w_eq, w_bd = 0.6, 0.4

def portfolio_vol_2_assets(w1, w2, s1, s2, rho):
    w1 = _num(w1, "w1"); w2 = _num(w2, "w2")
    s1 = _num(s1, "s1"); s2 = _num(s2, "s2")
    rho = _num(rho, "rho")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1 and 1.")
    var = (w1**2)*(s1**2) + (w2**2)*(s2**2) + 2*w1*w2*rho*s1*s2
    return float(np.sqrt(max(var, 0.0)))

vol_low_corr = portfolio_vol_2_assets(w_eq, w_bd, sigma_eq, sigma_bd, rho=-0.2)
vol_high_corr = portfolio_vol_2_assets(w_eq, w_bd, sigma_eq, sigma_bd, rho=+0.4)

print(f"Vol (rho=-0.2): {vol_low_corr:.2%}")
print(f"Vol (rho=+0.4): {vol_high_corr:.2%}")
```

Interpretation: When correlations rise, diversification benefits fall; asset-side volatility can increase, affecting funding risk, credit metrics, and potentially the discount rate narrative in valuation.

See also: Chapter 12 (variance-covariance matrix), Chapter 13 (betas and the Security Market Line), and Chapter 25 (VaR) for risk measurement extensions.
