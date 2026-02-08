# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 18: The Black-Scholes Model (European option pricing, inputs, sensitivity)

### Core Concept
Black–Scholes provides a closed-form price for European options under assumptions of lognormal returns, constant volatility, and continuous trading with no arbitrage. In valuation practice, it is widely used for employee share options (with adjustments), warrants, and as a baseline for real-options approximations when early exercise and discrete features are limited. The practical focus is not the derivation; it is correct input mapping (S, K, r, σ, T, q) and governance (sensitivity over σ and term, sanity checks against intrinsic value).

---

### Formula/Methodology

#### 1) Black–Scholes inputs (European options)
```text
S0 = current underlying price
K  = strike price
T  = time to maturity (years)
r  = risk-free rate (annual, continuously compounded)
σ  = volatility (annual standard deviation, decimal)
q  = dividend yield (annual, continuously compounded; 0 if none)
```

#### 2) d1 and d2
```text
d1 = [ ln(S0/K) + (r - q + 0.5σ^2)T ] / (σ√T)
d2 = d1 - σ√T
```

#### 3) European call and put prices
Call:
```text
C0 = S0·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
```

Put:
```text
P0 = K·e^(-rT)·N(-d2) - S0·e^(-qT)·N(-d1)
```

Where:
- N(.) is the standard normal CDF.

#### 4) Put–call parity (dividend yield form)
```text
C0 - P0 = S0·e^(-qT) - K·e^(-rT)
```

#### 5) Key “Greeks” (most used in valuation workflows)
Delta:
```text
Δ_call = e^(-qT)·N(d1)
Δ_put  = e^(-qT)·(N(d1) - 1)
```

Vega (sensitivity to σ):
```text
Vega = S0·e^(-qT)·φ(d1)·√T
```

Where:
- φ(.) is the standard normal PDF.

---

### Practical Application (How to apply Black–Scholes in valuation)

#### A) When Black–Scholes is appropriate
Use as a baseline for:
- European-style warrants/options
- Employee options with simplified terms (then adjust for vesting/forfeiture and early exercise behavior)
- Real-options approximations where exercise is at maturity (or where early exercise is not optimal)

Avoid using plain Black–Scholes when:
- American exercise is important (especially puts; dividend-paying calls)
- There are discrete dividends or complex payoff features (barriers, step-ups)
- Volatility is clearly non-constant or state-dependent

#### B) Input governance checklist (common failure points)
1) Risk-free rate r: match currency and maturity (use a curve if material)  
2) Volatility σ: choose a defendable method:
- Historical volatility (log returns), consistent window and frequency
- Peer/implied volatility for non-traded underlying
3) Dividend yield q: include if the underlying pays dividends or has cash yield (e.g., index)  
4) Time T: convert to years consistently (day count convention)  
5) Underlying S0 for private companies:
- If using equity value per share, ensure it is consistent with the option’s underlying definition (often common equity)
- Consider a volatility proxy and illiquidity adjustments (document explicitly)

#### C) Sensitivity and reasonableness checks
Minimum checks:
- Price ≥ intrinsic value (within numerical tolerance)
- Call price ≤ S0·e^(-qT)
- Put price ≤ K·e^(-rT)
- Put–call parity holds (within tolerance)

Deliverable standard: a small sensitivity table over σ (and sometimes r and T).

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
    """Standard normal CDF using erf (no SciPy dependency)."""
    xx = _num(x, "x")
    return 0.5 * (1.0 + math.erf(xx / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    xx = _num(x, "x")
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * xx * xx)

def black_scholes_d1_d2(S0: float, K: float, r: float, T: float, sigma: float, q: float = 0.0) -> Dict[str, float]:
    """Compute d1 and d2 for Black–Scholes.

    Args:
        S0: underlying price
        K: strike
        r: risk-free (cont. comp.)
        T: maturity (years)
        sigma: volatility (annual)
        q: dividend yield (cont. comp.)

    Returns:
        dict: {'d1':..., 'd2':...}

    Raises:
        ValueError: invalid inputs.
    """
    s = _num(S0, "S0")
    k = _num(K, "K")
    rr = _num(r, "r")
    tt = _num(T, "T")
    sig = _num(sigma, "sigma")
    qq = _num(q, "q")

    if s <= 0:
        raise ValueError("S0 must be > 0.")
    if k <= 0:
        raise ValueError("K must be > 0.")
    if tt <= 0:
        raise ValueError("T must be > 0.")
    if sig <= 0:
        raise ValueError("sigma must be > 0.")
    if qq < 0:
        raise ValueError("q must be >= 0.")

    vol_sqrt_t = sig * math.sqrt(tt)
    ln_sk = math.log(s / k)
    d1 = (ln_sk + (rr - qq + 0.5 * sig * sig) * tt) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return {"d1": d1, "d2": d2}

def black_scholes_price(
    S0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Price a European option with Black–Scholes.

    Args:
        S0: underlying price
        K: strike
        r: risk-free (cont. comp., annual)
        T: time to maturity (years)
        sigma: volatility (annual)
        option_type: 'call' or 'put'
        q: dividend yield (cont. comp., annual)

    Returns:
        float: option price

    Raises:
        ValueError: invalid inputs.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    vals = black_scholes_d1_d2(S0, K, r, T, sigma, q=q)
    d1, d2 = vals["d1"], vals["d2"]

    s = float(S0)
    k = float(K)
    rr = float(r)
    tt = float(T)
    qq = float(q)

    df_r = math.exp(-rr * tt)
    df_q = math.exp(-qq * tt)

    if option_type == "call":
        price = s * df_q * norm_cdf(d1) - k * df_r * norm_cdf(d2)
    else:
        price = k * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1)

    # Intrinsic sanity check (soft; allow tiny numerical noise)
    intrinsic = max(s - k, 0.0) if option_type == "call" else max(k - s, 0.0)
    if price + 1e-9 < intrinsic:
        raise ValueError("Price below intrinsic value (check inputs).")
    return float(price)

def black_scholes_greeks(S0: float, K: float, r: float, T: float, sigma: float, q: float = 0.0) -> Dict[str, float]:
    """Compute selected Greeks for European options."""
    vals = black_scholes_d1_d2(S0, K, r, T, sigma, q=q)
    d1 = vals["d1"]
    s = float(S0); tt = float(T); qq = float(q)
    df_q = math.exp(-qq * tt)

    delta_call = df_q * norm_cdf(d1)
    delta_put = df_q * (norm_cdf(d1) - 1.0)
    vega = s * df_q * norm_pdf(d1) * math.sqrt(tt)
    return {"delta_call": float(delta_call), "delta_put": float(delta_put), "vega": float(vega)}

def parity_deviation(S0: float, K: float, r: float, T: float, C0: float, P0: float, q: float = 0.0) -> float:
    """Deviation from put-call parity with dividend yield q."""
    s = _num(S0, "S0"); k = _num(K, "K"); rr = _num(r, "r"); tt = _num(T, "T")
    c = _num(C0, "C0"); p = _num(P0, "P0"); qq = _num(q, "q")
    lhs = c - p
    rhs = s * math.exp(-qq * tt) - k * math.exp(-rr * tt)
    return float(lhs - rhs)

# Example usage
S0 = 100.0
K = 105.0
r = 0.04
q = 0.01
T = 1.5
sigma = 0.25

call = black_scholes_price(S0, K, r, T, sigma, option_type="call", q=q)
put = black_scholes_price(S0, K, r, T, sigma, option_type="put", q=q)
greeks = black_scholes_greeks(S0, K, r, T, sigma, q=q)

print(f"Call: {call:.4f}")
print(f"Put : {put:.4f}")
print("Parity deviation:", parity_deviation(S0, K, r, T, call, put, q=q))
print("Delta call:", greeks["delta_call"])
print("Vega     :", greeks["vega"])
```

---

### Valuation Impact
Why this matters:
- Black–Scholes is commonly required for valuing share-based payments and warrants; getting inputs wrong (σ, T, q) can materially change equity value and compensation expense.
- It provides a baseline “option component” for complex securities and embedded features.

Impact on multiples:
- Option-related dilution affects per-share multiples (P/E, EV/share-derived metrics). If you ignore option value and dilution, multiples can be biased upward.
- Firms with high employee option overhang may appear “cheap” on basic-share metrics but not on fully diluted metrics.

Impact on DCF inputs:
- DCF provides enterprise value; option value and dilution are part of the equity bridge.
- For real options approximations, Black–Scholes can translate volatility and time-to-decision into incremental value (with governance caveats).

Comparability issues across companies:
- Different volatility estimation windows produce inconsistent option values.
- Private companies need proxy volatility; results depend on peer selection and leverage adjustments.

Practical adjustments:
```python
def levered_to_unlevered_beta(beta_l: float, tax_rate: float, debt: float, equity: float) -> float:
    """Hamada unlevering for consistency when selecting volatility proxies (simplified)."""
    b = _num(beta_l, "beta_l"); t = _num(tax_rate, "tax_rate")
    d = _num(debt, "debt"); e = _num(equity, "equity")
    if e <= 0:
        raise ValueError("equity must be > 0.")
    if t < 0 or t > 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    return b / (1.0 + (1.0 - t) * (d / e))
```

---

### Quality of Earnings Flags
⚠️ Share-based payment expense materially understated because σ or T is chosen too low, or forfeiture/vesting not reflected.  
⚠️ Per-share valuation presented without a clear dilution bridge (basic vs diluted).  
⚠️ Using Black–Scholes where early exercise or discrete dividends are material (model mismatch).  
✅ Clear disclosure: σ source, term/expected life, dividend yield, and sensitivity table.

---

### Sector-Specific Considerations

| Sector | Common option valuation use | Key issue | Typical handling |
|---|---|---|---|
| Tech | employee options/warrants | dilution & expected life | adjust expected life; sensitivity on σ |
| Biotech | warrants, milestone-linked instruments | high σ and binary risk | scenario overlays; consider binomial |
| Financials | structured products | model risk & calibration | validate vs market-implied σ |
| Real assets | real options | exercise flexibility | prefer binomial/Monte Carlo when early exercise matters |

---

### Real-World Example
Scenario: Value employee options to adjust fully diluted equity value per share.

1) Use Black–Scholes to estimate fair value per option (for a baseline).  
2) Multiply by number of options to gauge dilution/economic cost.  
3) Ensure valuation bridge reconciles to fully diluted share count.

```python
options_outstanding = 25_000_000
fair_value_per_option = call  # from example
total_option_value = options_outstanding * fair_value_per_option
print(f"Total option value (economic): ${total_option_value/1e6:.1f}M")
```

Interpretation: If total option value is material relative to equity value, your per-share valuation must explicitly incorporate dilution and the economic cost of options.

See also: Chapter 16 (option basics), Chapter 17 (binomial for American/complex features), Chapter 19 (Greeks), Chapter 20 (real options).
