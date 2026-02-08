# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 13: Estimating Betas and the Security Market Line (beta, CAPM, SML, regressions)

### Core Concept
Beta measures the sensitivity of an asset’s returns to market returns and is the workhorse input for CAPM-based cost of equity and WACC. In valuation, small changes in beta can materially change discount rates and therefore DCF values and implied multiples. The practical challenge is that beta is an estimate (not an observable truth), so methodology (return frequency, horizon, leverage adjustment, outliers) matters.

### Formula/Methodology

#### 1) CAPM cost of equity
```text
Re = Rf + βe × ERP
```

Where:
- Re = cost of equity
- Rf = risk-free rate (matched to currency and horizon)
- βe = equity beta (levered beta)
- ERP = equity risk premium (market risk premium)

#### 2) Beta from covariance (definition)
```text
βe = Cov(Ri, Rm) / Var(Rm)
```

Where:
- Ri = asset (equity) return series
- Rm = market return series

#### 3) Regression form (estimation)
```text
Ri,t - Rf,t = α + βe × (Rm,t - Rf,t) + εt
```

Where:
- α = intercept (abnormal return; usually not relied on for valuation)
- εt = residual

#### 4) Unlevering and relevering beta (capital structure adjustment)
A common practical approach is the Hamada relationship:

Unlever:
```text
βu = βe / (1 + (1 - Tc) × (D/E))
```

Relever:
```text
βe,new = βu × (1 + (1 - Tc) × (D/E)new)
```

Where:
- βu = unlevered (asset) beta
- Tc = marginal tax rate
- D/E = market-value debt-to-equity ratio

#### 5) Security Market Line (SML)
The SML is the relationship between expected return and beta:
```text
E[R] = Rf + β × (E[Rm] - Rf)
```

---

### Practical Application (How to estimate beta in a valuation workflow)

#### Step 1: Choose a consistent return window and frequency
Common choices (use one consistently across peers):
- **Weekly returns, 2–5 years**: reduces noise and microstructure effects in less-liquid stocks.
- **Daily returns, 2–3 years**: more observations, but can be noisy; watch for stale pricing in small caps.

#### Step 2: Use total returns where possible
For equity beta, use returns that include dividends (or ensure your price series is adjusted) to avoid bias.

#### Step 3: Align market and stock data precisely
- Same currency and same dates.
- Prefer “inner join” dates (drop non-overlapping days) rather than filling.

#### Step 4: Work with excess returns when risk-free varies
If risk-free is not constant (e.g., longer windows), compute excess returns:
- ri_excess = ri - rf
- rm_excess = rm - rf

#### Step 5: Diagnose and stabilize the estimate
Minimum checks:
- **Number of observations**: too few points → unstable beta.
- **Outliers**: one-day shocks (data errors) can dominate.
- **R-squared**: very low R² is common for single stocks; interpret beta cautiously.

Stabilizers:
- Use weekly returns.
- Winsorize extreme returns (if clearly data errors).
- Consider industry beta (peer median) if company beta is obviously noisy.

#### Step 6: Convert beta into discount rate inputs
- Use βe for cost of equity.
- If valuing the enterprise with WACC, ensure the beta input is consistent with your capital structure assumptions (relever βu to target D/E).

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def align_series(stock: pd.Series, market: pd.Series, rf: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """Align stock, market (and optional risk-free) series on common index.

    Args:
        stock: stock return series indexed by date
        market: market return series indexed by date
        rf: optional risk-free series indexed by date (same units as returns)

    Returns:
        (stock_aligned, market_aligned, rf_aligned_or_None)

    Raises:
        ValueError: on invalid inputs.
    """
    if not isinstance(stock, pd.Series) or not isinstance(market, pd.Series):
        raise ValueError("stock and market must be pandas Series.")
    if stock.empty or market.empty:
        raise ValueError("stock and market must not be empty.")

    df = pd.concat({"stock": stock, "market": market}, axis=1).dropna(how="any")
    if df.shape[0] < 30:
        raise ValueError("Need at least 30 aligned observations for a defensible beta estimate.")
    if rf is None:
        return df["stock"], df["market"], None

    if not isinstance(rf, pd.Series) or rf.empty:
        raise ValueError("rf must be a non-empty pandas Series when provided.")
    df = pd.concat({"stock": stock, "market": market, "rf": rf}, axis=1).dropna(how="any")
    if df.shape[0] < 30:
        raise ValueError("Need at least 30 aligned observations after including rf.")
    return df["stock"], df["market"], df["rf"]

def estimate_beta_ols(stock: pd.Series, market: pd.Series, rf: Optional[pd.Series] = None) -> Dict[str, float]:
    """Estimate CAPM beta using OLS on (excess) returns.

    Model:
        stock_excess = alpha + beta * market_excess + error

    Args:
        stock: stock returns (decimal; e.g., 0.01 = 1%)
        market: market returns (decimal)
        rf: optional risk-free returns (decimal). If provided, regression uses excess returns.

    Returns:
        dict with beta, alpha, r2, n

    Raises:
        ValueError: on invalid inputs.
    """
    s, m, r = align_series(stock, market, rf)

    if r is None:
        y = s.values.astype(float)
        x = m.values.astype(float)
    else:
        y = (s - r).values.astype(float)
        x = (m - r).values.astype(float)

    if np.allclose(np.var(x), 0.0):
        raise ValueError("market return variance is ~0; cannot estimate beta.")

    # OLS with intercept using least squares
    X = np.column_stack([np.ones_like(x), x])
    coeff, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coeff[0]), float(coeff[1])

    y_hat = X @ coeff
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

    return {"beta": beta, "alpha": alpha, "r2": r2, "n": float(len(y))}

def beta_from_cov(stock: pd.Series, market: pd.Series) -> float:
    """Compute beta as Cov(stock, market) / Var(market)."""
    s, m, _ = align_series(stock, market, None)
    cov = float(np.cov(s.values.astype(float), m.values.astype(float), ddof=1)[0, 1])
    var_m = float(np.var(m.values.astype(float), ddof=1))
    if var_m <= 0:
        raise ValueError("market variance must be > 0.")
    return cov / var_m

def capm_cost_of_equity(beta: float, rf: float, erp: float) -> float:
    """Compute cost of equity using CAPM.

    Args:
        beta: equity beta
        rf: risk-free rate (decimal; annual)
        erp: equity risk premium (decimal; annual)

    Returns:
        annual cost of equity (decimal)
    """
    b = _num(beta, "beta")
    r = _num(rf, "rf")
    p = _num(erp, "erp")
    return r + b * p

def unlever_beta(beta_equity: float, debt_to_equity: float, tax_rate: float) -> float:
    """Unlever equity beta into asset beta using Hamada relation."""
    be = _num(beta_equity, "beta_equity")
    de = _num(debt_to_equity, "debt_to_equity")
    tc = _num(tax_rate, "tax_rate")
    if de < 0:
        raise ValueError("debt_to_equity must be >= 0.")
    if not (0.0 <= tc <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    denom = 1.0 + (1.0 - tc) * de
    if denom <= 0:
        raise ValueError("invalid denominator in unlevering beta.")
    return be / denom

def relever_beta(beta_unlevered: float, debt_to_equity_target: float, tax_rate: float) -> float:
    """Relever asset beta to a target capital structure using Hamada relation."""
    bu = _num(beta_unlevered, "beta_unlevered")
    de = _num(debt_to_equity_target, "debt_to_equity_target")
    tc = _num(tax_rate, "tax_rate")
    if de < 0:
        raise ValueError("debt_to_equity_target must be >= 0.")
    if not (0.0 <= tc <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    return bu * (1.0 + (1.0 - tc) * de)

# Example usage (synthetic; replace with real returns)
np.random.seed(7)
dates = pd.date_range("2021-01-01", periods=156, freq="W-FRI")
rm = pd.Series(np.random.normal(0.001, 0.03, size=len(dates)), index=dates, name="market")
# stock with beta ~ 1.2 plus noise
ri = pd.Series(0.0005 + 1.2 * rm.values + np.random.normal(0.0, 0.04, size=len(dates)), index=dates, name="stock")

# Assume constant weekly risk-free return (for illustration)
rf_weekly = pd.Series(0.0002, index=dates, name="rf")

ols = estimate_beta_ols(ri, rm, rf=rf_weekly)
beta_cov = beta_from_cov(ri, rm)

print(f"OLS beta: {ols['beta']:.2f}, alpha: {ols['alpha']:.4f}, R^2: {ols['r2']:.2f}, n: {int(ols['n'])}")
print(f"Cov beta: {beta_cov:.2f}")

# Convert to annual cost of equity example
rf_annual = 0.04
erp_annual = 0.05
re = capm_cost_of_equity(ols["beta"], rf_annual, erp_annual)
print(f"Cost of equity (CAPM): {re:.1%}")

# Unlever/relever example
tax = 0.25
de_current = 0.50
beta_u = unlever_beta(ols["beta"], de_current, tax_rate=tax)
beta_relevered = relever_beta(beta_u, debt_to_equity_target=0.80, tax_rate=tax)
print(f"Unlevered beta: {beta_u:.2f} | Relevered beta (target D/E=0.8): {beta_relevered:.2f}")
```

---

### Valuation Impact
Why this matters:
- Beta is a primary driver of cost of equity; cost of equity feeds WACC; WACC drives DCF value.
- Beta selection also affects implied continuing value and valuation ranges when discount rates move.

Impact on multiples:
- Higher discount rates generally compress EV/EBITDA and P/E for a given growth/ROIC profile.
- Peer multiple comparability can be improved by normalizing for differences in risk (e.g., comparing P/E after adjusting for beta/WACC differences conceptually).

Impact on DCF inputs:
- Re impacts WACC via the equity component.
- Using unlevered betas enables consistent WACC when you model a target capital structure.

Comparability issues across companies:
- Different return horizons/frequencies produce different betas.
- Company-specific events (mergers, business model shifts) mean historical beta may not reflect forward-looking risk.

Practical adjustments:
```python
def normalized_beta(peer_betas: list, use_median: bool = True) -> float:
    """Simple rule: use median peer beta for stability; fall back to mean if needed."""
    if peer_betas is None or len(peer_betas) == 0:
        raise ValueError("peer_betas must be non-empty.")
    vals = [float(b) for b in peer_betas if b is not None and np.isfinite(float(b))]
    if len(vals) == 0:
        raise ValueError("No valid betas in peer_betas.")
    return float(np.median(vals)) if use_median else float(np.mean(vals))
```

---

### Quality of Earnings Flags
⚠️ Beta estimated on misaligned data (different date sets) or using price returns that are not adjusted for splits/dividends.  
⚠️ “One-number beta” presented without methodology (window, frequency, market index, currency).  
⚠️ Leverage adjustments done using book D/E or inconsistent tax rate assumptions (non-comparable).  
✅ Stable approach: peer-based unlevered beta, relevered to target capital structure, with sensitivity to horizon/frequency.

---

### Sector-Specific Considerations

| Sector | Key beta issue | Typical treatment |
|---|---|---|
| Financials | leverage is core to operations; debt-like liabilities complicate D/E | use bank-specific approaches; interpret beta cautiously; consider equity risk via stress tests |
| Utilities | regulated returns and stable cash flows | betas often low; ensure market index and frequency are consistent |
| Cyclicals | beta varies by cycle | consider mid-cycle beta or scenario betas (downturn vs expansion) |
| Small caps | illiquidity/stale prices | use weekly returns; consider peer beta + size risk premium separately |

---

### Real-World Example
Scenario: You are valuing a mid-cap industrial business and need a defendable beta.

1) Estimate betas for 6–10 listed peers using the same method (weekly, 5-year; same market index).  
2) Unlever each peer beta using each peer’s market D/E and tax rate.  
3) Take the median unlevered beta.  
4) Relever to the target D/E for your valuation case.  
5) Run WACC sensitivity (e.g., ±0.2 beta) to quantify impact on enterprise value.

```python
peer_equity_betas = [1.05, 0.92, 1.20, 1.10, 0.85, 1.35]
peer_de = [0.40, 0.30, 0.70, 0.55, 0.20, 0.80]
tax = 0.25

peer_unlevered = [unlever_beta(b, de, tax) for b, de in zip(peer_equity_betas, peer_de)]
beta_u_med = float(np.median(peer_unlevered))

target_de = 0.60
beta_target = relever_beta(beta_u_med, target_de, tax)

rf = 0.04
erp = 0.05
re_target = capm_cost_of_equity(beta_target, rf, erp)

print(f"Median unlevered beta: {beta_u_med:.2f}")
print(f"Target relevered beta: {beta_target:.2f}")
print(f"Cost of equity: {re_target:.1%}")
```

Interpretation: Peer-unlevered/relevered beta reduces noise versus a single-company regression beta and aligns risk with your assumed capital structure—both of which improve WACC defensibility.

See also: Chapter 12 (variance-covariance matrix), Chapter 11 (efficient frontier) for how covariance relates to beta; Chapter 3 (WACC) for discount rate application.
