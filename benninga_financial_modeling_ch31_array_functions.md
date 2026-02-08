# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 31: Array Functions (vectorised modeling for fast, auditable valuation)

### Core Concept
Array functions let you calculate entire rows/columns in one expression (vectorised logic), reducing helper columns and making models both faster and easier to audit. In valuation work, array logic is particularly useful for discount factors, cash flow discounting, debt schedules, scenario tables, peer statistics, and Monte Carlo-style simulations (where repeated calculations are required).

---

### Formula/Methodology

#### 1) Vectorised discounting (cash flows as arrays)
If CF is a vector of cash flows for periods 1..N and DF is a vector of discount factors:
```text
DF_t = 1 / (1 + r)^t
PV = Σ (CF_t × DF_t)
```

Vector form:
```text
PV = CF · DF   (dot product)
```

Where:
- CF = [CF_1, CF_2, ..., CF_N]
- DF = [DF_1, DF_2, ..., DF_N]
- r = discount rate per period

#### 2) Terminal value integration (explicit separation)
Perpetuity growth method at period N:
```text
TV_N = FCF_(N+1) / (WACC - g)
PV_TV = TV_N / (1 + WACC)^N
EV = PV_FCF + PV_TV
```

Exit multiple method at period N:
```text
TV_N = Metric_N × ExitMultiple
PV_TV = TV_N / (1 + WACC)^N
```

#### 3) Scenario tables (array cross-join logic)
If you have arrays of WACC values and g values, create a grid of terminal values:
```text
TV_grid[i,j] = FCF_(N+1) / (WACC_i - g_j)
```

#### 4) Peer stats and robust summaries
Median and percentile (preferred over mean when skewed):
```text
Median = P50
IQR = P75 - P25
```

Used to set valuation ranges and avoid outlier-driven conclusions.

---

### Practical Application (How to apply in valuation work)

#### A) DCF block as arrays (fewer cells, fewer mistakes)
Best practice:
- Keep forecast FCF in one contiguous range/series (CF array).
- Build a DF array once.
- PV is a single dot-product (SUMPRODUCT in Excel; dot in Python).

Advantages:
- avoids accidental omissions of a year
- prevents inconsistent discounting across line items
- makes sensitivity analysis trivial (swap r and recompute DF array)

#### B) Debt schedules with vector logic (structured + capped)
Key rules:
- Interest = opening balance × rate
- Repayment = MIN(available cash, opening balance)
- Closing balance = opening - repayment

Array modeling helps when:
- multiple tranches exist
- you run scenarios across multiple rates
- you need robust caps to prevent negative balances

#### C) Sensitivity tables: array-driven and audit-friendly
Typical “two-way” tables:
- WACC vs g (terminal value)
- WACC vs exit multiple
- Revenue CAGR vs margin (FCF outcomes)

Use consistent array shapes and label axes explicitly.

#### D) Python-in-Excel translation of array patterns
Excel dynamic arrays ≈ NumPy arrays:
- FILTER/UNIQUE/SORT → boolean masks / unique / sort
- SUMPRODUCT → dot product
- BYROW/BYCOL → apply along axis

---

### Python Implementation
```python
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import pandas as pd

def _finite_float(x: Any, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def discount_factors(rate: float, periods: int, start_t: int = 1) -> np.ndarray:
    """Create an array of discount factors DF_t = 1/(1+r)^t.

    Args:
        rate: discount rate per period (decimal)
        periods: number of periods N
        start_t: starting t index (1 for end-of-period CFs)

    Returns:
        np.ndarray: length-N array of discount factors

    Raises:
        ValueError: invalid inputs
    """
    r = _finite_float(rate, "rate")
    if r <= -0.999999:
        raise ValueError("rate must be > -100%.")
    if periods <= 0:
        raise ValueError("periods must be > 0.")
    if start_t < 0:
        raise ValueError("start_t must be >= 0.")
    t = np.arange(start_t, start_t + periods, dtype=float)
    return 1.0 / np.power(1.0 + r, t)

def pv_from_arrays(cashflows: Sequence[float], rate: float, start_t: int = 1) -> float:
    """PV via array dot-product.

    PV = Σ CF_t / (1+r)^t = CF · DF
    """
    if not isinstance(cashflows, (list, tuple, np.ndarray)):
        raise ValueError("cashflows must be a sequence.")
    cf = np.array([_finite_float(x, "cashflow") for x in cashflows], dtype=float)
    if cf.size == 0:
        raise ValueError("cashflows must be non-empty.")
    df = discount_factors(rate, periods=cf.size, start_t=start_t)
    return float(np.dot(cf, df))

def terminal_value_perpetuity(fcf_n_plus_1: float, wacc: float, g: float) -> float:
    """Terminal value (perpetuity growth): TV = FCF_(N+1) / (WACC - g)."""
    f = _finite_float(fcf_n_plus_1, "fcf_n_plus_1")
    w = _finite_float(wacc, "wacc")
    gg = _finite_float(g, "g")
    if w <= -0.999999:
        raise ValueError("wacc must be > -100%.")
    if gg >= w:
        raise ValueError("g must be < wacc to avoid infinite/negative TV.")
    return float(f / (w - gg))

def pv_terminal_value(tv_n: float, wacc: float, n: int) -> float:
    """PV of terminal value at year N: PV_TV = TV_N / (1+WACC)^N."""
    tv = _finite_float(tv_n, "tv_n")
    w = _finite_float(wacc, "wacc")
    if n <= 0:
        raise ValueError("n must be > 0.")
    if w <= -0.999999:
        raise ValueError("wacc must be > -100%.")
    return float(tv / ((1.0 + w) ** n))

def dcf_enterprise_value(fcf: Sequence[float], wacc: float, g: float) -> Dict[str, float]:
    """Compute EV from forecast FCFs + perpetuity terminal value.

    Assumes:
        - fcf = [FCF_1..FCF_N]
        - terminal value at N using FCF_(N+1) = FCF_N * (1+g)

    Returns:
        dict with PV_FCF, TV_N, PV_TV, EV
    """
    w = _finite_float(wacc, "wacc")
    gg = _finite_float(g, "g")
    if not isinstance(fcf, (list, tuple, np.ndarray)):
        raise ValueError("fcf must be a sequence.")
    cf = np.array([_finite_float(x, "fcf") for x in fcf], dtype=float)
    if cf.size == 0:
        raise ValueError("fcf must be non-empty.")
    n = int(cf.size)

    pv_fcf = pv_from_arrays(cf, w, start_t=1)
    fcf_n1 = float(cf[-1] * (1.0 + gg))
    tv_n = terminal_value_perpetuity(fcf_n1, w, gg)
    pv_tv = pv_terminal_value(tv_n, w, n)
    ev = pv_fcf + pv_tv

    return {"PV_FCF": float(pv_fcf), "TV_N": float(tv_n), "PV_TV": float(pv_tv), "EV": float(ev)}

def tv_sensitivity_grid(fcf_n_plus_1: float, wacc_values: Sequence[float], g_values: Sequence[float]) -> pd.DataFrame:
    """Create a WACC x g grid of perpetuity terminal values.

    Returns:
        DataFrame indexed by WACC with columns g
    """
    f = _finite_float(fcf_n_plus_1, "fcf_n_plus_1")
    w_arr = np.array([_finite_float(x, "wacc") for x in wacc_values], dtype=float)
    g_arr = np.array([_finite_float(x, "g") for x in g_values], dtype=float)
    if w_arr.size == 0 or g_arr.size == 0:
        raise ValueError("wacc_values and g_values must be non-empty.")
    # Broadcast: TV[i,j] = f / (w[i] - g[j])
    denom = w_arr.reshape(-1, 1) - g_arr.reshape(1, -1)
    if np.any(denom <= 0):
        raise ValueError("All WACC values must be > all g values in the grid.")
    tv = f / denom
    df = pd.DataFrame(tv, index=[f"{w:.3%}" for w in w_arr], columns=[f"{g:.2%}" for g in g_arr])
    return df

# Example usage: array DCF + sensitivity
fcf = [60, 70, 82, 95, 110]  # $m, years 1-5
wacc = 0.095
g = 0.03

out = dcf_enterprise_value(fcf, wacc, g)
print(out)
print(f"EV: ${out['EV']:.1f}m")

grid = tv_sensitivity_grid(fcf_n_plus_1=fcf[-1]*(1+g),
                           wacc_values=[0.085, 0.095, 0.105],
                           g_values=[0.02, 0.03, 0.04])
print(grid)
```

---

### Valuation Impact
Why this matters:
- Array modeling reduces implementation risk in DCFs (missing years, inconsistent discounting) and makes sensitivity analysis systematic.
- Vectorised logic enables faster scenario evaluation (e.g., WACC/g grids), improving decision quality and enabling tighter valuation ranges.

Impact on multiples:
- Peer statistics (median/IQR) are naturally “array” operations; using robust summaries reduces outlier bias in implied valuation.
- Scenario arrays can translate multiple ranges into valuation distributions (more realistic than single-point multiples).

Impact on DCF inputs:
- Terminal value dominates many DCFs; building an array-based sensitivity grid forces explicit checks that g < WACC and highlights fragility.
- Discount factor arrays make timing conventions transparent (end-of-period vs mid-year).

Comparability issues across companies:
- Using consistent array structures for discounting and normalisation ensures cross-company comparability and reduces analyst “style” differences.

Practical adjustments:
```python
def mid_year_adjustment_pv(pv_end_year: float, wacc: float) -> float:
    """Approximate mid-year convention: PV_mid ≈ PV_end × sqrt(1+WACC)."""
    import numpy as np
    pv = float(pv_end_year); w = float(wacc)
    if not np.isfinite(pv) or not np.isfinite(w):
        raise ValueError("Inputs must be finite.")
    if w <= -0.999999:
        raise ValueError("wacc must be > -100%.")
    return float(pv * np.sqrt(1.0 + w))
```

---

### Quality of Earnings Flags
⚠️ Terminal value sensitivity not tested; g too close to WACC (creates extreme valuations).  
⚠️ Discounting done with mixed timing conventions (some lines mid-year, some end-year) without documentation.  
⚠️ Array formulas in Excel spilling into populated cells (silent truncation or #SPILL! ignored).  
✅ Explicit DF arrays, TV grids with g < WACC checks, and labelled scenario axes.

---

### Sector-Specific Considerations

| Sector | Array-heavy use case | Typical treatment |
|---|---|---|
| High-growth tech | multi-scenario margins + reinvestment | driver arrays + distributions |
| Infrastructure | long-tenor cash flows | DF arrays + mid-year convention |
| Cyclicals | multiple cases across the cycle | scenario arrays + normalised year |
| Banks/Insurers | rate shock grids | matrix/array mapping across books |

---

### Real-World Example
Scenario: Build a full EV from array-discounted FCF and terminal value, then stress test WACC and g.

```python
fcf = [60, 70, 82, 95, 110]  # $m
base = dcf_enterprise_value(fcf, wacc=0.095, g=0.03)

# Stress test
waccs = [0.085, 0.095, 0.105]
gs = [0.02, 0.03, 0.04]
tv_grid = tv_sensitivity_grid(fcf[-1]*(1+0.03), waccs, gs)

print(f"Base EV: ${base['EV']:.1f}m")
print(tv_grid)
```

Interpretation: If small changes in WACC or g cause large EV swings, the valuation should be presented as a range and cross-checked using multiples and scenario narratives.

See also: Chapter 4 (DCF build), Chapter 28 (data tables/sensitivity), Chapter 21–24 (Monte Carlo methods and simulations).
