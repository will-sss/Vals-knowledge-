# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 22: An Introduction to Monte Carlo Methods (simulation-based valuation, convergence, variance reduction)

### Core Concept
Monte Carlo methods estimate valuation outputs (NPV, option value, VaR) by simulating many plausible paths for uncertain inputs and aggregating the results. For valuation work, the focus is on building a defensible stochastic model (drivers + distributions + correlation), proving convergence, and translating the resulting distribution into decisions (expected value, downside percentiles, probability-weighted outcomes).

---

### Formula/Methodology

#### 1) Monte Carlo estimator (expected value)
Given an outcome function f(·) and random inputs X:
```text
θ = E[f(X)]
```

Monte Carlo estimate with N simulations:
```text
θ̂ = (1/N) · Σ_{i=1..N} f(X_i)
```

Where:
- X_i = simulated draw of inputs on simulation i
- f(X_i) = valuation output (e.g., PV of FCFs, NPV, terminal value, option payoff)

#### 2) Standard error and confidence interval for the mean
Sample standard deviation:
```text
s = sqrt( (1/(N-1)) · Σ (Y_i - Ȳ)^2 )
```

Standard error of the mean:
```text
SE(Ȳ) = s / sqrt(N)
```

Approx. 95% CI (normal approximation):
```text
CI_95% ≈ Ȳ ± 1.96 · SE(Ȳ)
```

Where:
- Y_i = f(X_i)
- Ȳ = mean of Y_i

#### 3) Discounted cash flow inside Monte Carlo
Simulated present value for one scenario:
```text
PV_i = Σ_{t=1..T} FCF_{i,t} / (1 + WACC)^t
```

Simulated enterprise value summary metrics:
```text
EV_mean = mean(PV_i)
EV_p10  = percentile(PV_i, 10)
EV_p90  = percentile(PV_i, 90)
```

#### 4) Terminal value under simulation (perpetuity growth)
```text
TV_{i,T} = FCF_{i,T+1} / (WACC - g_i)
PV_TV_i  = TV_{i,T} / (1 + WACC)^T
```

Guardrails:
- Require WACC > g_i (otherwise TV explodes / is invalid)
- Consider clipping or scenario filtering with explicit rationale

#### 5) Variance reduction (practical)
Antithetic variates (pair Z and -Z):
```text
Ȳ_antithetic = (Y(Z) + Y(-Z)) / 2
```

Effect:
- Reduces variance when f is monotone in Z (common in pricing/DCF drivers)

---

### Practical Application (How to apply Monte Carlo in valuation)

#### A) Build the driver model (not just the code)
1) Identify 3–8 key value drivers:
- Revenue growth, price, volume
- Gross margin / EBITDA margin
- Working capital intensity
- Capex intensity
- FX / commodity inputs

2) Choose distributions with bounds where needed:
- Growth: truncated normal, or normal with clipping (documented)
- Margins: normal with floor/ceiling or beta on [0,1] if modeled as a ratio
- Prices: lognormal / GBM-like for positive variables

3) Add correlation only where you can justify it:
- Revenue growth vs margin (positive for scale economies in some models)
- Commodity price vs costs (positive)
- FX vs revenue/cost mix

4) Define what “one simulation” means:
- A single year shock applied across forecast?
- A full path with year-by-year shocks?
Choose the simplest structure that matches the business.

#### B) Output interpretation: what to report
| Output | Use in valuation decisions | Typical use case |
|---|---|---|
| Mean EV | baseline “expected value” | internal planning |
| Median EV | robust central tendency | skewed distributions |
| P10 / P90 | downside/upside range | IC memos, stress |
| Probability(NPV < 0) | viability / distress signal | capex screening |
| CVaR (tail mean) | tail risk measure | risk committees |

#### C) Convergence and runtime in Excel/Python-in-Excel
- Use N that’s feasible (e.g., 5k–50k), then prove convergence by:
  - plotting/printing running mean every k steps
  - reporting SE and CI width
- Use variance reduction to improve precision for a given N:
  - antithetic normals
  - common random numbers for scenario comparisons
  - stratified uniforms for key drivers

#### D) Guardrails to avoid garbage valuations
- Terminal value sanity:
  - enforce WACC - g > 0 with minimum spread (e.g., 1.0% floor) or scenario rejection
- Margin and working-capital bounds:
  - clamp to plausible ranges
- Negative prices/volumes:
  - use lognormal or explicit floors
- Correlation matrix:
  - must be positive definite; otherwise fix with documented nearest-PD routine (or simplify)

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

def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be an int >= 0 or None.")
    return np.random.default_rng(seed)

def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    l = _num(lo, "lo"); h = _num(hi, "hi")
    if h < l:
        raise ValueError("hi must be >= lo.")
    return np.minimum(np.maximum(x, l), h)

def mc_summary(outcomes: np.ndarray) -> Dict[str, float]:
    x = np.array(outcomes, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("Need at least 2 outcomes.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Outcomes must be finite.")
    n = x.size
    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(n)
    p10 = float(np.percentile(x, 10))
    p90 = float(np.percentile(x, 90))
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    prob_neg = float(np.mean(x < 0))
    return {
        "n": float(n),
        "mean": mean,
        "median": median,
        "std": std,
        "se": float(se),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p10": p10,
        "p90": p90,
        "prob_neg": prob_neg,
    }

def dcf_pv(
    fcf: np.ndarray,
    wacc: float,
    terminal_g: Optional[float] = None,
    terminal_value_year: Optional[int] = None,
) -> float:
    """Discount an array of annual FCFs; optional perpetuity terminal value.

    Args:
        fcf: array-like of length T (FCF_1..FCF_T)
        wacc: discount rate as decimal (e.g., 0.10)
        terminal_g: perpetuity growth rate as decimal
        terminal_value_year: terminal year index (1..T). Default: T

    Returns:
        Present value as float

    Raises:
        ValueError: invalid rates or shapes
    """
    w = _num(wacc, "wacc")
    if w <= -0.999:
        raise ValueError("wacc is implausible (must be > -99.9%).")
    f = np.array(fcf, dtype=float).ravel()
    if f.size == 0:
        raise ValueError("fcf must not be empty.")
    if not np.all(np.isfinite(f)):
        raise ValueError("fcf must be finite.")
    T = f.size
    tv_year = T if terminal_value_year is None else int(terminal_value_year)
    if tv_year < 1 or tv_year > T:
        raise ValueError("terminal_value_year must be in 1..T.")

    disc = np.array([(1.0 + w) ** t for t in range(1, T + 1)], dtype=float)
    pv = float(np.sum(f / disc))

    if terminal_g is not None:
        g = _num(terminal_g, "terminal_g")
        if w <= g:
            raise ValueError("Need wacc > terminal_g for perpetuity terminal value.")
        # Terminal value at tv_year using FCF_{tv_year+1}. Approx as FCF_tv_year * (1+g)
        fcf_T = float(f[tv_year - 1])
        fcf_next = fcf_T * (1.0 + g)
        tv = fcf_next / (w - g)
        pv_tv = tv / ((1.0 + w) ** tv_year)
        pv += float(pv_tv)

    return pv

def simulate_dcf_value(
    n: int,
    base_revenue: float,
    base_margin: float,
    base_fcf_conversion: float,
    revenue_growth_mu: float,
    revenue_growth_sigma: float,
    margin_sigma: float,
    years: int,
    wacc: float,
    terminal_g: float,
    seed: int = 0,
    growth_floor: float = -0.50,
    growth_cap: float = 0.50,
    margin_floor: float = -0.20,
    margin_cap: float = 0.60,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Toy Monte Carlo DCF: simulate revenue and margin shocks -> FCF -> PV.

    Notes:
        - This is a template: replace with your driver tree (price/volume, WC, capex).
        - Clipping is explicit; document bounds as part of model governance.

    Returns:
        (values, summary)
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int.")
    if not isinstance(years, int) or years <= 0:
        raise ValueError("years must be a positive int.")
    R0 = _num(base_revenue, "base_revenue")
    m0 = _num(base_margin, "base_margin")
    conv = _num(base_fcf_conversion, "base_fcf_conversion")
    if R0 < 0:
        raise ValueError("base_revenue must be >= 0.")
    # margins/conversion can be negative, but keep sanity ranges
    mu = _num(revenue_growth_mu, "revenue_growth_mu")
    sig_g = _num(revenue_growth_sigma, "revenue_growth_sigma")
    sig_m = _num(margin_sigma, "margin_sigma")
    if sig_g < 0 or sig_m < 0:
        raise ValueError("sigmas must be >= 0.")

    rng = make_rng(seed)

    # Simulate constant annual growth and margin shocks (simple; can be path-based)
    g = mu + sig_g * rng.standard_normal(n)
    g = clamp(g, growth_floor, growth_cap)

    m = m0 + sig_m * rng.standard_normal(n)
    m = clamp(m, margin_floor, margin_cap)

    values = np.zeros(n, dtype=float)
    for i in range(n):
        rev = R0
        fcf = []
        for t in range(1, years + 1):
            rev *= (1.0 + float(g[i]))
            ebitda = rev * float(m[i])
            fcf_t = ebitda * conv
            fcf.append(fcf_t)
        values[i] = dcf_pv(np.array(fcf), wacc=wacc, terminal_g=terminal_g)

    summary = mc_summary(values)
    return values, summary

# Example usage
values, summ = simulate_dcf_value(
    n=10000,
    base_revenue=1_000_000_000,
    base_margin=0.20,
    base_fcf_conversion=0.60,
    revenue_growth_mu=0.05,
    revenue_growth_sigma=0.10,
    margin_sigma=0.05,
    years=5,
    wacc=0.10,
    terminal_g=0.03,
    seed=123,
)

print(f"Mean EV: ${summ['mean']/1e6:.1f}M")
print(f"Median EV: ${summ['median']/1e6:.1f}M")
print(f"P10-P90: ${summ['p10']/1e6:.1f}M to ${summ['p90']/1e6:.1f}M")
print(f"95% CI (mean): ${summ['ci_lo']/1e6:.1f}M to ${summ['ci_hi']/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- Monte Carlo converts single-point valuation into a distribution, revealing skewness, tail risk, and the probability of value impairment. This is especially relevant when terminal value dominates EV, when leverage is high, or when growth/commodity risk is material.
- It supports decision-grade outputs: downside percentiles, probability of NPV < 0, and sensitivity to key uncertain drivers.

Impact on multiples:
- Distributions can rationalize “rich” or “cheap” multiples when the market prices upside convexity (growth optionality) or penalizes downside tails (distress risk).
- Use Monte Carlo outputs to explain dispersion of peer multiples under different risk exposures.

Impact on DCF inputs:
- Helps stress-test terminal growth vs WACC spread and supports robust guardrails (e.g., minimum spread).
- Can inform normalized margins/growth by comparing simulated mid-cycle outcomes vs point forecasts.

Comparability issues across companies:
- Results depend on volatility and correlation calibration. Ensure consistent parameter-setting rules across a peer set and avoid “hand-tuning” σ to reach a target value.

Practical adjustments:
```python
def apply_conservatism_to_tail(values: np.ndarray, haircut: float = 0.10) -> float:
    """Example: apply a haircut to downside percentile for decision thresholds."""
    x = np.array(values, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("Need at least 2 values.")
    h = float(_num(haircut, "haircut"))
    if not (0 <= h < 1):
        raise ValueError("haircut must be in [0,1).")
    p10 = float(np.percentile(x, 10))
    return float((1 - h) * p10)
```

---

### Quality of Earnings Flags
⚠️ Driver distributions allow impossible values (e.g., sustained negative revenue or extreme margins) without explicit bounds.  
⚠️ Terminal value dominates EV but guardrails are absent (WACC ≤ g scenarios included).  
⚠️ Correlation assumptions are arbitrary and materially change tail outcomes.  
✅ Convergence evidence provided (CI width), distributions documented, and stress tests/guardrails applied transparently.

---

### Sector-Specific Considerations

| Sector | Key Monte Carlo focus | Typical driver set | Modeling notes |
|---|---|---|---|
| Resources | commodity price risk | price, volume, cost inflation | lognormal price + correlation |
| Infrastructure | demand & availability | volume, downtime, opex | bounded distributions |
| Financials | credit losses | PD, LGD, EAD | copula/correlation important |
| SaaS | retention/NRR | churn, NRR, CAC payback | heavy skew; scenario + MC |

---

### Real-World Example
Scenario: Use a Monte Carlo DCF to estimate EV distribution and report downside risk.

```python
values, summ = simulate_dcf_value(
    n=20000,
    base_revenue=800_000_000,
    base_margin=0.18,
    base_fcf_conversion=0.65,
    revenue_growth_mu=0.04,
    revenue_growth_sigma=0.12,
    margin_sigma=0.04,
    years=6,
    wacc=0.095,
    terminal_g=0.025,
    seed=2024,
)

p10_haircut = apply_conservatism_to_tail(values, haircut=0.10)

print(f"Mean EV: ${summ['mean']/1e6:.0f}M")
print(f"P10 EV:  ${summ['p10']/1e6:.0f}M")
print(f"P10 (haircut): ${p10_haircut/1e6:.0f}M")
print(f"Prob(EV < 0): {summ['prob_neg']:.2%}")
```

Interpretation: If P10 is close to debt or negative, the valuation case depends heavily on optimistic tails; consider restructuring-like discounting, tighter terminal assumptions, or scenario gating.

See also: Chapter 21 (random numbers and correlation), Chapter 20 (real options), Chapter 25 (VaR).
