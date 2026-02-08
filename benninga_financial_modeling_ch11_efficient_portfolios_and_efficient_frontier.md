# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 11: Efficient Portfolios and the Efficient Frontier (mean-variance optimization, constraints, capital allocation line)

### Core Concept
Efficient portfolios minimize risk for a given expected return (or maximize expected return for a given risk) under the mean-variance framework. The efficient frontier and capital allocation line (CAL) provide a disciplined way to translate return forecasts and covariance estimates into optimal allocations and to quantify the risk trade-offs behind “best” portfolios. In valuation work, this supports robust beta/risk estimation, scenario-driven cost of equity narratives, and treasury/pension investment decision support.

### Formula/Methodology

#### 1) Portfolio expected return and variance (matrix form)
```text
E[R_p] = w' μ
Var(R_p) = w' Σ w
σ_p = sqrt(w' Σ w)

Where:
w = weights vector (N×1), Σ w_i = 1 (typical)
μ = expected returns vector (N×1)
Σ = covariance matrix (N×N)
```

#### 2) Efficient frontier problem statements
Minimum variance for target return R*:
```text
min_w   w' Σ w
s.t.    w' μ = R*
        Σ w_i = 1
        (optional) w_i ≥ 0  (long-only)
```

Maximum return for target risk σ* (equivalent framing):
```text
max_w   w' μ
s.t.    w' Σ w ≤ (σ*)^2
        Σ w_i = 1
        (optional) w_i ≥ 0
```

#### 3) Global minimum variance (GMV) portfolio (no return constraint)
```text
min_w   w' Σ w
s.t.    Σ w_i = 1
```

Closed-form (unconstrained, allowing shorting):
```text
w_GMV = (Σ^{-1} 1) / (1' Σ^{-1} 1)
```

Where:
- 1 = vector of ones (N×1)

#### 4) Tangency (maximum Sharpe) portfolio with risk-free asset (unconstrained)
If a risk-free rate R_f exists, the tangency portfolio (risky assets only) is:
```text
w_T ∝ Σ^{-1} (μ - R_f 1)
Normalize:
w_T = Σ^{-1} (μ - R_f 1) / [1' Σ^{-1} (μ - R_f 1)]
```

#### 5) Capital Allocation Line (CAL)
For a mix of risk-free asset and tangency portfolio:
```text
E[R_c] = R_f + y × (E[R_T] - R_f)
σ_c = y × σ_T

Where:
y = weight allocated to tangency portfolio (can exceed 1 with borrowing)
```

#### 6) Two-fund separation (practical implication)
Under standard assumptions, any efficient portfolio can be built from:
- the risk-free asset, and
- a single tangency risky portfolio

---

### Practical Application (How to apply in valuation work)

#### Step 1: Build inputs that are valuation-consistent
- Use realistic expected returns μ (avoid extrapolating short-term outperformance).
- Estimate Σ from clean return data; test stability across windows.
- Align the risk-free rate R_f to the currency and horizon used for discount rates.

#### Step 2: Choose constraints that reflect reality
- Long-only constraints (w_i ≥ 0) match most corporate policies and many investor mandates.
- Sector/issuer caps reduce concentration.
- Minimum liquidity thresholds can be encoded via universe selection.

#### Step 3: Use efficient frontier outputs to support risk narratives, not as “the answer”
- Efficient frontier is sensitive to μ estimates; treat it as a scenario tool.
- Use it to quantify how much additional risk is taken for incremental expected return.

#### Step 4: Translate to valuation inputs
- Tangency portfolio conceptually links to the “market portfolio” in CAPM; beta estimation and cost of equity reasoning builds on this.
- For treasury and pension assets, frontier analysis supports explaining volatility and funding risk that can affect credit metrics and equity risk perception.

#### Step 5: Stress testing and robustness checks (mandatory in practice)
- Recompute frontier under:
  - higher correlations (crisis regime),
  - higher vol (volatility regime shift),
  - lower expected returns,
  - transaction-cost or turnover penalties (if optimizing dynamically).

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

def _as_vector(x: Any, name: str) -> np.ndarray:
    if isinstance(x, (pd.Series, list, tuple, np.ndarray)):
        v = np.asarray(x, dtype=float).reshape(-1, 1)
        if v.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be finite.")
        return v
    raise ValueError(f"{name} must be array-like.")

def global_minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Compute GMV weights w = inv(S)1 / (1' inv(S) 1).

    Args:
        cov: covariance matrix (N×N)

    Returns:
        weights (N,) summing to 1

    Raises:
        ValueError: invalid inputs or singular covariance.
    """
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("cov must be a square matrix.")
    if S.shape[0] < 2:
        raise ValueError("cov must be at least 2×2.")
    if not np.all(np.isfinite(S)):
        raise ValueError("cov must be finite.")
    n = S.shape[0]
    ones = np.ones((n, 1))
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"covariance matrix is singular or ill-conditioned: {e}")
    denom = float(ones.T @ invS @ ones)
    if denom == 0:
        raise ValueError("Invalid covariance matrix: denominator is zero.")
    w = (invS @ ones) / denom
    return w.flatten()

def tangency_portfolio_weights(mu: Any, cov: Any, risk_free: float) -> np.ndarray:
    """Compute tangency portfolio weights for risky assets (unconstrained, shorting allowed).

    w_T = inv(S) (mu - rf*1) / (1' inv(S) (mu - rf*1))

    Args:
        mu: expected returns (N,) as array-like
        cov: covariance matrix (N×N)
        risk_free: risk-free rate per period (decimal)

    Returns:
        weights (N,) summing to 1

    Raises:
        ValueError: invalid inputs.
    """
    rf = _num(risk_free, "risk_free")
    m = _as_vector(mu, "mu")
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("cov must be a square matrix.")
    if S.shape[0] != m.shape[0]:
        raise ValueError("mu length must match covariance size.")
    if not np.all(np.isfinite(S)):
        raise ValueError("cov must be finite.")
    n = S.shape[0]
    ones = np.ones((n, 1))
    excess = m - rf * ones
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"covariance matrix is singular or ill-conditioned: {e}")
    num = invS @ excess
    denom = float(ones.T @ num)
    if abs(denom) < 1e-12:
        raise ValueError("Tangency portfolio undefined: denominator near zero (check mu and rf).")
    w = num / denom
    return w.flatten()

def portfolio_moments(mu: Any, cov: Any, weights: Any) -> Tuple[float, float]:
    """Compute portfolio mean and volatility.

    Args:
        mu: expected returns (N,)
        cov: covariance matrix (N×N)
        weights: weights (N,)

    Returns:
        (mean, vol)
    """
    m = _as_vector(mu, "mu")
    S = np.asarray(cov, dtype=float)
    w = _as_vector(weights, "weights")
    if S.shape[0] != S.shape[1] or S.shape[0] != m.shape[0] or w.shape[0] != m.shape[0]:
        raise ValueError("Dimensions must align.")
    mean = float((w.T @ m))
    var = float(w.T @ S @ w)
    if var < -1e-12:
        raise ValueError("Computed variance is negative; check covariance matrix.")
    vol = float(np.sqrt(max(var, 0.0)))
    return mean, vol

def efficient_frontier_unconstrained(mu: Any, cov: Any, target_returns: np.ndarray) -> pd.DataFrame:
    """Compute unconstrained efficient frontier using Lagrange multipliers (shorting allowed).

    For each target return R*, solve:
        min w' S w s.t. w'μ = R*, 1'w = 1

    Closed-form uses:
        A = 1' S^{-1} 1
        B = 1' S^{-1} μ
        C = μ' S^{-1} μ
        Δ = AC - B^2

        w(R) = S^{-1} [ (C - B R)/Δ * 1 + (A R - B)/Δ * μ ]

    Args:
        mu: expected returns (N,)
        cov: covariance matrix (N×N)
        target_returns: array of target returns (K,)

    Returns:
        DataFrame with columns: target_return, vol, weights...

    Raises:
        ValueError: invalid inputs.
    """
    m = _as_vector(mu, "mu")
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("cov must be square.")
    n = S.shape[0]
    if m.shape[0] != n:
        raise ValueError("mu length must match covariance size.")
    if n < 2:
        raise ValueError("Need at least 2 assets.")
    if not np.all(np.isfinite(S)):
        raise ValueError("cov must be finite.")
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"covariance matrix is singular or ill-conditioned: {e}")

    ones = np.ones((n, 1))
    A = float(ones.T @ invS @ ones)
    B = float(ones.T @ invS @ m)
    C = float(m.T @ invS @ m)
    Delta = A * C - B * B
    if abs(Delta) < 1e-12:
        raise ValueError("Delta is near zero; efficient frontier not well-defined for these inputs.")

    rows = []
    for R in np.asarray(target_returns, dtype=float).reshape(-1):
        if not np.isfinite(R):
            raise ValueError("target_returns must be finite.")
        g1 = (C - B * R) / Delta
        g2 = (A * R - B) / Delta
        w = invS @ (g1 * ones + g2 * m)
        mean, vol = portfolio_moments(m, S, w)
        row = {"target_return": float(R), "achieved_return": float(mean), "vol": float(vol)}
        for i in range(n):
            row[f"w{i+1}"] = float(w[i, 0])
        rows.append(row)
    return pd.DataFrame(rows)

# Example usage (illustrative; replace with real mu and cov)
np.random.seed(3)
n_assets = 4
# build a positive semi-definite covariance
A = np.random.normal(size=(n_assets, n_assets))
S = (A @ A.T) / 252.0  # daily cov scale
mu = np.array([0.08, 0.10, 0.06, 0.12]) / 252.0  # daily expected returns (illustrative)

w_gmv = global_minimum_variance_weights(S)
mu_gmv, vol_gmv = portfolio_moments(mu, S, w_gmv)
print("GMV weights:", np.round(w_gmv, 4), "sum:", w_gmv.sum())
print(f"GMV daily mean: {mu_gmv:.6f}, daily vol: {vol_gmv:.6f}")

rf_daily = 0.03 / 252.0
w_tan = tangency_portfolio_weights(mu, S, rf_daily)
mu_t, vol_t = portfolio_moments(mu, S, w_tan)
sharpe = (mu_t - rf_daily) / vol_t if vol_t > 0 else np.nan
print("Tangency weights:", np.round(w_tan, 4), "sum:", w_tan.sum())
print(f"Tangency daily mean: {mu_t:.6f}, daily vol: {vol_t:.6f}, daily Sharpe: {sharpe:.3f}")

# Efficient frontier (unconstrained)
targets = np.linspace(min(mu), max(mu), 10)
frontier = efficient_frontier_unconstrained(mu, S, targets)
print(frontier.head())
```

---

### Valuation Impact
Why this matters:
- Efficient frontier mechanics are the quantitative foundation for CAPM intuition: the “market” (tangency) portfolio concept underpins how beta and cost of equity are formed.
- For corporates with large pension/treasury portfolios, frontier analysis quantifies how asset allocation choices can change volatility, funding risk, and potentially credit metrics.

Impact on multiples:
- Higher systematic risk (higher beta) → higher discount rate → lower valuation multiples.
- Frontier stress scenarios (higher correlation/vol) can rationalize multiple compression during market stress periods.

Impact on DCF inputs:
- Cost of equity and WACC are sensitive to risk-free rate and risk premia; frontier/CAL narratives help defend these inputs under different rate/market regimes.
- If valuation uses scenario-weighted discount rates, frontier outputs can guide internally consistent “high risk / low risk” parameter sets.

Comparability issues across companies:
- Comparing firms in different risk regimes (high covariance with market vs defensive) without risk adjustment can mislead relative valuation conclusions.

Practical adjustments:
```python
def capm_cost_of_equity(risk_free: float, beta: float, equity_risk_premium: float) -> float:
    """CAPM: Re = Rf + beta * ERP."""
    rf = _num(risk_free, "risk_free")
    b = _num(beta, "beta")
    erp = _num(equity_risk_premium, "equity_risk_premium")
    return rf + b * erp
```

---

### Quality of Earnings Flags
⚠️ Pension accounting impacts: a risky pension asset mix can create volatile OCI and funding contributions, distorting “normalized” earnings and free cash flow.  
⚠️ Optimization output presented as “optimal” without showing sensitivity to μ assumptions (means are the weakest input).  
⚠️ Correlation assumptions remain static; crisis correlation spikes ignored in risk cases.  
✅ Frontier outputs accompanied by robustness checks, stress scenarios, and constraints reflecting real mandates (long-only, caps).

---

### Sector-Specific Considerations

| Sector | Frontier relevance | Typical handling |
|---|---|---|
| Insurers / pensions | ALM and solvency | include liability hedge assets; stress correlations; track funding ratio volatility |
| Banks | trading/treasury portfolios | overlay regulatory constraints; liquidity haircuts |
| Asset managers | product design | use constraints matching mandate; focus on tracking error and turnover |
| Corporates | pension/treasury risk | translate to funding and covenant impacts; use conservative μ inputs |

---

### Real-World Example
Scenario: A corporate pension plan is considering shifting from 60/40 equity/bonds to a lower-risk allocation. Use frontier logic to quantify how much expected return is sacrificed for volatility reduction.

```python
import numpy as np

# Simple 2-asset illustration using annual inputs
mu_eq, mu_bd = 0.08, 0.03
sig_eq, sig_bd = 0.16, 0.06
rho = 0.10

cov = np.array([
    [sig_eq**2, rho*sig_eq*sig_bd],
    [rho*sig_eq*sig_bd, sig_bd**2]
])
mu = np.array([mu_eq, mu_bd])

def portfolio_stats_2(w_eq):
    w_bd = 1.0 - w_eq
    w = np.array([w_eq, w_bd])
    mean = float(w @ mu)
    vol = float(np.sqrt(max(w.T @ cov @ w, 0.0)))
    return mean, vol

for w_eq in [0.6, 0.4, 0.2]:
    mean, vol = portfolio_stats_2(w_eq)
    print(f"Equity weight: {w_eq:.0%} -> E[R]: {mean:.2%}, Vol: {vol:.2%}")
```

Interpretation: Lower equity weight reduces volatility (funding risk) but lowers expected return. In valuation narratives, this can support lower risk (potentially lower beta/discount rate) but may imply higher expected contributions if returns undershoot liability growth—link to cash flow forecasts.

See also: Chapter 10 (portfolio fundamentals), Chapter 12 (variance-covariance matrix), Chapter 13 (beta and SML), Chapter 25 (VaR) for tail risk extensions.
