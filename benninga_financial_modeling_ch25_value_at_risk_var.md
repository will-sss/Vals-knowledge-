# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 25: Value at Risk (VaR) (quantiles, parametric vs historical vs Monte Carlo, backtesting)

### Core Concept
Value at Risk (VaR) estimates the loss threshold that will not be exceeded with a specified confidence level over a defined horizon. In valuation and deal work, VaR and related tail measures (CVaR/Expected Shortfall) translate market and portfolio risks into decision-ready limits (liquidity buffers, covenant headroom, discount-rate conservatism, stress scenarios).

---

### Formula/Methodology

#### 1) Define P&L and loss
Let ΔV be portfolio value change over horizon (P&L). Define loss L as:
```text
L = -ΔV
```
So higher L means worse outcome.

#### 2) VaR at confidence level q
VaR is the q-quantile of loss:
```text
VaR_q = quantile(L, q)
```

Where:
- q = confidence level (e.g., 0.95 or 0.99)
- L = loss distribution over the horizon

Interpretation:
- With q=0.95, VaR_95 is the loss level exceeded in 5% of outcomes (under the model/data).

#### 3) CVaR / Expected Shortfall (tail mean beyond VaR)
```text
CVaR_q = E[L | L ≥ VaR_q]
```

CVaR is more informative than VaR when tails are fat or skewed.

#### 4) VaR methods (how computed)

**A) Historical simulation**
```text
VaR_q = quantile({L_1, L_2, ..., L_N}, q)
```
Where losses come from historical return observations applied to current exposures.

**B) Parametric (variance–covariance) VaR**
Assume ΔV is approximately normal:
```text
VaR_q ≈ μ_L + z_q · σ_L
```
Where:
- μ_L = mean of loss
- σ_L = standard deviation of loss
- z_q = standard normal quantile at q (e.g., 1.645 for 95%)

If losses are defined as L = -ΔV and ΔV ~ Normal(μ, σ):
```text
VaR_q ≈ -(μ) + z_q · σ
```

**C) Monte Carlo VaR**
Simulate return scenarios (possibly multivariate), revalue the portfolio, compute losses, then take quantile:
```text
VaR_q = quantile(L_sim, q)
```

#### 5) Multi-day horizon scaling (use with caution)
If returns are i.i.d. normal and positions are linear:
```text
σ_h ≈ σ_1 · sqrt(h)
VaR_h ≈ VaR_1 · sqrt(h)
```
Where h is number of days/periods.

Guardrails:
- scaling fails with autocorrelation, volatility clustering, nonlinear payoffs (options), and fat tails

---

### Practical Application (How to apply VaR in valuation)

#### A) Choose the VaR you actually need
| Use case | Horizon | Confidence | Method preference |
|---|---:|---:|---|
| Liquidity buffer / margin | 1–10 days | 95–99% | historical or Monte Carlo |
| Covenant headroom stress | quarter/year | 95% + tail | Monte Carlo + scenarios |
| Portfolio risk reporting | day/month | 95–99% | parametric + backtest |
| Option-heavy exposures | day/month | 95–99% | Monte Carlo (revalue) |

#### B) Map VaR outputs to valuation decisions
- If VaR implies frequent large drawdowns, consider:
  - higher equity risk premium / discount rate overlay
  - tighter leverage in target capital structure
  - greater cash buffer (affects enterprise value if excess cash is constrained)
- For distressed/levered cases:
  - use VaR/CVaR to quantify probability of breaching liquidity/covenants, informing probability-weighted valuation or restructuring adjustments

#### C) Backtesting (minimum governance)
Use a rolling window:
- Compute daily VaR_95
- Count exceedances: days when realized loss > VaR_95
Expected exceedance rate:
```text
Expected exceedances ≈ (1 - q) · N_days
```
Too many exceedances suggests VaR model underestimates risk.

#### D) Guardrails and common pitfalls
- Fat tails: parametric VaR underestimates risk; consider t-distribution or stress overlays.
- Nonlinear portfolios: delta-normal approximations break; use full revaluation.
- Regime shifts: incorporate stress periods and avoid short calibration windows.
- Data frequency: match μ/σ to horizon; don’t mix monthly σ with daily VaR.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def var_cvar_from_pnl(pnl: np.ndarray, q: float = 0.95) -> Dict[str, float]:
    """Compute VaR/CVaR from P&L series.

    Args:
        pnl: array-like P&L (positive = gain, negative = loss)
        q: confidence level, e.g., 0.95

    Returns:
        dict with VaR and CVaR in loss units (positive numbers)

    Raises:
        ValueError: invalid inputs
    """
    x = np.array(pnl, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("pnl must have at least 2 observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError("pnl must be finite.")
    qq = _num(q, "q")
    if not (0 < qq < 1):
        raise ValueError("q must be between 0 and 1.")
    losses = -x  # loss = -P&L
    var = float(np.quantile(losses, qq))
    tail = losses[losses >= var]
    cvar = float(np.mean(tail)) if tail.size else var
    return {"VaR": var, "CVaR": cvar}

def historical_var(exposures: np.ndarray, returns: np.ndarray, q: float = 0.95) -> Dict[str, float]:
    """Historical simulation VaR for linear exposures.

    Args:
        exposures: vector of current positions in value terms (k,)
        returns: matrix of historical returns (N, k)
        q: confidence

    Returns:
        VaR/CVaR on portfolio loss distribution

    Notes:
        - Linear approximation: ΔV ≈ exposures · returns
        - For options/nonlinear, use full revaluation instead.
    """
    e = np.array(exposures, dtype=float).ravel()
    R = np.array(returns, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns must be (N, k).")
    if e.size != R.shape[1]:
        raise ValueError("exposures length must match returns columns.")
    if R.shape[0] < 10:
        raise ValueError("Need at least 10 return observations for stability.")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(R))):
        raise ValueError("exposures and returns must be finite.")
    pnl = R @ e  # ΔV
    return var_cvar_from_pnl(pnl, q=q)

def parametric_var_normal(mu_pnl: float, sigma_pnl: float, q: float = 0.95) -> Dict[str, float]:
    """Parametric VaR under Normal assumption for P&L.

    VaR_q (loss) ≈ -(mu) + z_q * sigma

    Args:
        mu_pnl: mean P&L per horizon
        sigma_pnl: std dev of P&L per horizon
        q: confidence

    Returns:
        VaR and CVaR (loss units). CVaR uses normal ES formula.
    """
    mu = _num(mu_pnl, "mu_pnl")
    sig = _num(sigma_pnl, "sigma_pnl")
    if sig < 0:
        raise ValueError("sigma_pnl must be >= 0.")
    qq = _num(q, "q")
    if not (0 < qq < 1):
        raise ValueError("q must be between 0 and 1.")

    # Standard normal quantile (approx) without scipy: use numpy's erfinv via normal approximation
    # z = sqrt(2) * erfinv(2q - 1)
    from math import sqrt
    z = sqrt(2.0) * float(np.erfinv(2.0 * qq - 1.0))

    var = float(-mu + z * sig)

    # Normal Expected Shortfall (loss): ES = -mu + sigma * φ(z)/(1-q)
    phi = float((1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z))
    cvar = float(-mu + sig * (phi / (1.0 - qq)))
    return {"VaR": var, "CVaR": cvar, "z": float(z)}

def backtest_var(pnl: np.ndarray, var_series: np.ndarray) -> Dict[str, float]:
    """Backtest VaR exceedances.

    Args:
        pnl: realized P&L series (N,)
        var_series: VaR thresholds in loss units (N,). Exceedance when loss > VaR.

    Returns:
        exceedance_rate and count

    Raises:
        ValueError: invalid inputs
    """
    p = np.array(pnl, dtype=float).ravel()
    v = np.array(var_series, dtype=float).ravel()
    if p.size != v.size or p.size < 10:
        raise ValueError("pnl and var_series must have same length >= 10.")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
        raise ValueError("inputs must be finite.")
    losses = -p
    exc = losses > v
    return {"n": float(p.size), "exceedances": float(np.sum(exc)), "exceedance_rate": float(np.mean(exc))}

# Example usage
rng = np.random.default_rng(1)
pnl = rng.normal(loc=0.0, scale=1.0, size=5000)  # toy P&L
risk = var_cvar_from_pnl(pnl, q=0.95)
print(f"VaR95: {risk['VaR']:.3f} | CVaR95: {risk['CVaR']:.3f}")

mu_hat = float(np.mean(pnl))
sig_hat = float(np.std(pnl, ddof=1))
risk_p = parametric_var_normal(mu_hat, sig_hat, q=0.95)
print(f"Parametric VaR95: {risk_p['VaR']:.3f} | Parametric CVaR95: {risk_p['CVaR']:.3f}")
```

---

### Valuation Impact
Why this matters:
- VaR translates market uncertainty into a quantifiable downside threshold, supporting liquidity buffers, leverage tolerance, and capital structure constraints that directly affect equity value.
- For capital-intensive or levered businesses, VaR/CVaR can quantify probability of value impairment via forced asset sales, covenant breaches, or higher refinancing spreads.

Impact on multiples:
- Higher tail risk (VaR/CVaR) can justify lower valuation multiples due to equity risk, optionality against shareholders, and constrained distributable cash.
- Comparable companies with similar EBITDA can trade at different EV/EBITDA due to different tail exposures and balance sheet fragility.

Impact on DCF inputs:
- Use VaR/CVaR to motivate stress discount rates or scenario-based WACC/spreads, rather than arbitrary “+100 bps” overlays.
- Supports probability-weighted valuation under multi-regime outcomes (base/downturn/severe).

Comparability issues across companies:
- VaR depends on horizon, confidence level, and method. Standardize (e.g., 1-month 95% historical) before comparing peers.

Practical adjustments:
```python
def apply_var_buffer_to_cash(cash: float, var_loss: float, buffer_multiple: float = 1.0) -> float:
    """Example: reduce 'excess cash' by a VaR-based liquidity buffer."""
    c = _num(cash, "cash")
    v = _num(var_loss, "var_loss")
    m = _num(buffer_multiple, "buffer_multiple")
    if m < 0:
        raise ValueError("buffer_multiple must be >= 0.")
    buffer = m * max(v, 0.0)
    return float(max(c - buffer, 0.0))
```

---

### Quality of Earnings Flags
⚠️ Management uses VaR to justify “safe leverage” while ignoring fat tails and nonlinearity (options/embedded leverage).  
⚠️ Calibration window excludes stress periods; VaR looks artificially low.  
⚠️ VaR reported without backtesting exceedance rate.  
✅ VaR method documented; backtested; supplemented with CVaR and stress scenarios.

---

### Sector-Specific Considerations

| Sector | Key VaR focus | Typical method | Notes |
|---|---|---|---|
| Banks | trading book market risk | parametric + stress + backtest | regulatory-style metrics |
| Insurers | ALM mismatch risk | Monte Carlo | tail risk dominates |
| Commodity | price-driven earnings | historical + scenario | non-normal tails common |
| PE/Holdco | NAV + liquidity | scenario + haircuts | include liquidity VaR |

---

### Real-World Example
Scenario: Compute 1-month historical VaR for a linear 3-asset exposure and apply a liquidity buffer.

```python
rng = np.random.default_rng(42)
N = 1000
k = 3

# Simulated historical returns (monthly)
returns = rng.normal(loc=[0.01, 0.004, 0.006], scale=[0.06, 0.02, 0.04], size=(N, k))

# Current exposures ($)
exposures = np.array([50_000_000, 30_000_000, 20_000_000], dtype=float)

risk = historical_var(exposures, returns, q=0.95)
print(f"Monthly VaR95 loss: ${risk['VaR']/1e6:.1f}M | CVaR95 loss: ${risk['CVaR']/1e6:.1f}M")

cash = 15_000_000
excess_cash = apply_var_buffer_to_cash(cash, risk['VaR'], buffer_multiple=1.0)
print(f"Excess cash after VaR buffer: ${excess_cash/1e6:.1f}M")
```

Interpretation: If VaR95 consumes most of available liquidity, dividend capacity and leverage tolerance are constrained, which can warrant conservative valuation assumptions (lower terminal multiple, higher discount rate overlay, or explicit distress scenarios).

See also: Chapter 24 (Monte Carlo for investments), Chapter 22 (Monte Carlo foundations), Chapter 9 (default-adjusted bond returns).
