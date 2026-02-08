# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 19: Option Greeks (sensitivities, risk management, valuation governance)

### Core Concept
Option Greeks measure how an option’s value changes with key drivers (underlying price, volatility, time, rates, dividends). In valuation, Greeks are less about trading and more about governance: they quantify which assumptions (especially volatility and time) dominate the value of warrants, employee options, and embedded derivatives. Practically, Greeks guide sensitivity tables, identify where model risk is concentrated, and help explain valuation movements between reporting dates.

---

### Formula/Methodology

#### 1) Definitions (core Greeks)
```text
Delta (Δ) = ∂V/∂S      (price sensitivity to underlying)
Gamma (Γ) = ∂²V/∂S²    (curvature; how delta changes with S)
Vega (ν)  = ∂V/∂σ      (sensitivity to volatility)
Theta (Θ) = ∂V/∂t      (time decay; sensitivity to passage of time)
Rho (ρ)   = ∂V/∂r      (sensitivity to interest rates)
```

Where:
- V = option value
- S = underlying price
- σ = volatility
- t = time (or time to maturity; conventions vary)
- r = risk-free rate

Practical note on sign conventions:
- Many references define Theta as the change in option value as calendar time passes (often negative). In implementation, be explicit whether you use ∂V/∂T (time-to-maturity) or ∂V/∂t (calendar time).

---

### Black–Scholes Greeks (European options)

Inputs:
```text
S0 = underlying price
K  = strike
T  = time to maturity (years)
r  = risk-free rate (cont. comp.)
σ  = volatility (annual)
q  = dividend yield (cont. comp.)
```

Helper terms:
```text
d1 = [ ln(S0/K) + (r - q + 0.5σ^2)T ] / (σ√T)
d2 = d1 - σ√T
```

Standard normal:
```text
N(.) = standard normal CDF
φ(.) = standard normal PDF
```

#### 1) Delta
```text
Δ_call = e^(-qT) · N(d1)
Δ_put  = e^(-qT) · (N(d1) - 1)
```

#### 2) Gamma (same for call and put)
```text
Γ = e^(-qT) · φ(d1) / (S0 · σ · √T)
```

#### 3) Vega (same for call and put)
```text
Vega = S0 · e^(-qT) · φ(d1) · √T
```

#### 4) Theta (calendar-time convention; commonly reported per day)
Call theta (per year):
```text
Θ_call = - [S0·e^(-qT)·φ(d1)·σ] / (2√T)  - rK·e^(-rT)·N(d2) + qS0·e^(-qT)·N(d1)
```

Put theta (per year):
```text
Θ_put  = - [S0·e^(-qT)·φ(d1)·σ] / (2√T)  + rK·e^(-rT)·N(-d2) - qS0·e^(-qT)·N(-d1)
```

Per day approximation:
```text
Theta_per_day ≈ Theta_per_year / 365
```

#### 5) Rho
```text
ρ_call =  K·T·e^(-rT)·N(d2)
ρ_put  = -K·T·e^(-rT)·N(-d2)
```

---

### Practical Application (How to apply Greeks in valuation work)

#### A) Build a sensitivity table that targets the dominant drivers
Most valuation errors for options come from σ and expected life (T). Use Vega and Theta to size sensitivities:
- High Vega → option value highly sensitive to volatility selection.
- High Theta magnitude → option value highly sensitive to expected life and reporting date.

Recommended sensitivities for disclosures/workpapers:
- σ: ±5pp and ±10pp
- T (expected life): ±25% (or ±1 year where reasonable)
- r: ±100 bps (usually small but include for long-dated warrants)

#### B) Explain period-to-period movements in option fair value
Decompose value change using a first-order approximation:
```text
ΔV ≈ Δ·ΔS + Vega·Δσ + Rho·Δr + Theta·Δt
```

This is not perfect (Gamma matters when ΔS is large), but it is useful for:
- Audit trails
- Management explanations
- Identifying which assumption drove changes

#### C) Use Gamma to assess non-linearity and why scenario tests matter
- High Gamma means Delta changes quickly with S; linear approximations fail.
- For near-the-money options, Gamma tends to be highest: scenario tables should include multiple S levels.

#### D) Use Delta as a dilution / economic exposure proxy (with caution)
Delta provides an “equivalent shares” view:
```text
Delta-adjusted shares ≈ number_of_options × Δ
```
Useful for:
- Communicating economic exposure
- Rough hedging intuition
Not a substitute for a proper equity bridge / fully diluted share count.

---

### Python Implementation
```python
from typing import Any, Dict
import math
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def norm_cdf(x: float) -> float:
    xx = _num(x, "x")
    return 0.5 * (1.0 + math.erf(xx / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    xx = _num(x, "x")
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * xx * xx)

def d1_d2(S0: float, K: float, r: float, T: float, sigma: float, q: float = 0.0) -> Dict[str, float]:
    s = _num(S0, "S0"); k = _num(K, "K"); rr = _num(r, "r"); tt = _num(T, "T"); sig = _num(sigma, "sigma"); qq = _num(q, "q")
    if s <= 0: raise ValueError("S0 must be > 0.")
    if k <= 0: raise ValueError("K must be > 0.")
    if tt <= 0: raise ValueError("T must be > 0.")
    if sig <= 0: raise ValueError("sigma must be > 0.")
    if qq < 0: raise ValueError("q must be >= 0.")
    vol_sqrt_t = sig * math.sqrt(tt)
    d1 = (math.log(s / k) + (rr - qq + 0.5 * sig * sig) * tt) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return {"d1": d1, "d2": d2}

def black_scholes_greeks(
    S0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    q: float = 0.0,
) -> Dict[str, float]:
    """Compute European Black–Scholes Greeks (annualized where applicable).

    Returns:
        dict with delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put

    Notes:
        - theta_* returned per year (calendar-time convention, typically negative)
        - vega returned per 1.0 change in sigma (i.e., per 100pp). For per 1pp, multiply by 0.01.
        - rho returned per 1.0 change in r (per 100pp). For per 1bp, multiply by 0.0001.
    """
    vals = d1_d2(S0, K, r, T, sigma, q=q)
    d1v, d2v = vals["d1"], vals["d2"]

    s = float(S0); k = float(K); rr = float(r); tt = float(T); sig = float(sigma); qq = float(q)
    df_r = math.exp(-rr * tt)
    df_q = math.exp(-qq * tt)
    Nd1 = norm_cdf(d1v)
    Nd2 = norm_cdf(d2v)
    Nmd1 = norm_cdf(-d1v)
    Nmd2 = norm_cdf(-d2v)
    phid1 = norm_pdf(d1v)

    delta_call = df_q * Nd1
    delta_put = df_q * (Nd1 - 1.0)

    gamma = (df_q * phid1) / (s * sig * math.sqrt(tt))

    vega = s * df_q * phid1 * math.sqrt(tt)

    # Theta per year (calendar time)
    common = - (s * df_q * phid1 * sig) / (2.0 * math.sqrt(tt))
    theta_call = common - rr * k * df_r * Nd2 + qq * s * df_q * Nd1
    theta_put  = common + rr * k * df_r * Nmd2 - qq * s * df_q * Nmd1

    rho_call = k * tt * df_r * Nd2
    rho_put  = -k * tt * df_r * Nmd2

    return {
        "delta_call": float(delta_call),
        "delta_put": float(delta_put),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta_call": float(theta_call),
        "theta_put": float(theta_put),
        "rho_call": float(rho_call),
        "rho_put": float(rho_put),
    }

def approx_value_change(greeks: Dict[str, float], dS: float, dSigma: float, dr: float, dt: float, is_call: bool = True) -> float:
    """First-order approximation of option value change.

    Args:
        greeks: output from black_scholes_greeks
        dS: change in underlying (same currency units as S0)
        dSigma: change in volatility (decimal; e.g., +0.05 for +5pp)
        dr: change in r (decimal; e.g., +0.01 for +100bps)
        dt: change in time (years; positive means time passes, so T decreases)

    Returns:
        float: approx change in option value
    """
    if not isinstance(greeks, dict):
        raise ValueError("greeks must be a dict.")
    delta = _num(greeks["delta_call" if is_call else "delta_put"], "delta")
    vega = _num(greeks["vega"], "vega")
    rho  = _num(greeks["rho_call" if is_call else "rho_put"], "rho")
    theta = _num(greeks["theta_call" if is_call else "theta_put"], "theta")

    dS_ = _num(dS, "dS")
    dSig_ = _num(dSigma, "dSigma")
    dr_ = _num(dr, "dr")
    dt_ = _num(dt, "dt")

    # Time passes -> option loses value by theta * dt (theta is per year)
    return delta * dS_ + vega * dSig_ + rho * dr_ + theta * dt_

# Example usage
params = {"S0": 120.0, "K": 110.0, "r": 0.04, "q": 0.01, "T": 2.0, "sigma": 0.30}
g = black_scholes_greeks(**params)
print({k: round(v, 6) for k, v in g.items()})

# Sensitivity: +5pp volatility, +$10 stock move, 3 months pass
dV = approx_value_change(g, dS=10.0, dSigma=0.05, dr=0.0, dt=0.25, is_call=True)
print(f"Approx ΔV: {dV:.3f}")
```

---

### Valuation Impact
Why this matters:
- Greeks quantify model risk: they show which assumptions drive value, supporting defensible sensitivity analysis and disclosures.
- They help reconcile period-to-period fair value changes for warrants and share-based payments (important in audit reviews and valuation memos).

Impact on multiples:
- High-dilution option programs can materially affect per-share metrics; Delta can be used as a quick “equivalent shares” cross-check, but must be reconciled to dilution methodology.
- Changes in underlying price and volatility can change option values and equity value, affecting P/E (via diluted EPS) and market-implied expectations.

Impact on DCF inputs:
- Greeks do not change enterprise DCF directly, but they affect the equity bridge (warrants/options) and explain why value per share can move even if operating forecasts are unchanged.
- For real options, Vega is a direct signal that uncertainty (volatility) adds value; if Vega is large, a single deterministic forecast is likely insufficient.

Comparability issues across companies:
- Different σ estimation methods produce different Vega exposure and option values, reducing comparability of “diluted” valuations.
- Short vs long expected life changes Theta materially; ensure consistent expected-life policy.

Practical adjustments:
```python
def vega_per_1pp(vega: float) -> float:
    """Convert vega per 100pp to vega per 1pp."""
    v = _num(vega, "vega")
    return v * 0.01

def rho_per_1bp(rho: float) -> float:
    """Convert rho per 100pp to rho per 1bp."""
    r = _num(rho, "rho")
    return r * 0.0001
```

---

### Quality of Earnings Flags
⚠️ Material swings in “other income/expense” from remeasurement of warrant liabilities not explained (Greeks can identify key drivers).  
⚠️ Option valuations use point estimates for σ and expected life with no sensitivity tables, despite high Vega/Theta.  
⚠️ Company reports diluted EPS without clear treasury stock method or treatment of in/out-of-the-money instruments.  
✅ Documented sensitivities aligned to Vega/Theta and clear reconciliation of diluted shares.

---

### Sector-Specific Considerations

| Sector | Typical instruments | Greek that dominates | Practical note |
|---|---|---|---|
| Tech | employee options, RSUs, warrants | Vega + Theta | expected life and σ drive fair value; disclose clearly |
| Biotech | warrants with long maturities | Vega | long-dated options are volatility-sensitive; scenario tests |
| Financials | structured products | Gamma + Rho | non-linear exposures; validate with stress scenarios |
| Distressed | equity as option on firm value | Gamma | non-linearity high near default boundary |

---

### Real-World Example
Scenario: Explain a quarter-on-quarter change in warrant liability fair value.

Assume:
- Underlying increases by $5
- Volatility increases by +3pp
- Risk-free increases by +25bps
- 0.25 years of time passes

```python
base = {"S0": 50.0, "K": 55.0, "r": 0.03, "q": 0.0, "T": 1.5, "sigma": 0.45}
g = black_scholes_greeks(**base)

approx = approx_value_change(
    g,
    dS=5.0,
    dSigma=0.03,
    dr=0.0025,
    dt=0.25,
    is_call=True
)
print(f"Approx change in warrant value: {approx:.3f}")
```

Interpretation: If the approximation is dominated by Vega, the volatility estimate (and its basis) should be the main focus of review and disclosure.

See also: Chapter 18 (Black–Scholes pricing), Chapter 17 (binomial for American/complex features), Chapter 20 (real options), Chapter 25 (risk and scenario analysis).
