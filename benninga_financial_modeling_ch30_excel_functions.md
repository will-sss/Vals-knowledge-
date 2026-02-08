# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 30: Excel Functions (audit-safe function patterns for valuation models)

### Core Concept
Core Excel functions (time value, lookups, aggregation, and conditional logic) are the building blocks of valuation models, but they often fail due to unit inconsistencies, error propagation, and hidden assumptions. This chapter distils function patterns that are robust for DCFs, multiples, debt schedules, and sensitivity analysis, with Python equivalents for reproducible “Excel-to-code” workflows.

---

### Formula/Methodology

#### 1) Time value of money (PV/FV) function pattern
Present value of a cash flow stream:
```text
PV = Σ [ CF_t / (1 + r)^t ]  for t = 1..N
```

For a level annuity:
```text
PV_annuity = PMT * (1 - (1 + r)^(-N)) / r
```

Where:
- CF_t = cash flow in period t
- r = discount rate per period (decimal)
- N = number of periods
- PMT = periodic payment (same sign convention throughout)

Excel equivalents:
- PV: `NPV(rate, CF1:CFN) + CF0` (NPV excludes CF0)
- Annuity PV: `PV(rate, N, -PMT, 0, 0)` (watch sign conventions)

#### 2) Discount factors (defensive)
```text
DF_t = 1 / (1 + r)^t
```
Use explicit DF rows to audit discounting and prevent hidden logic.

#### 3) Growth and compounding
Compound annual growth rate (CAGR):
```text
CAGR = (End / Start)^(1/N) - 1
```

Forward projection:
```text
Value_t = Value_0 * (1 + g)^t
```

#### 4) Safe division and error control
```text
SafeRatio = IF(denominator = 0, NA(), numerator / denominator)
```
Goal: avoid silent 0s or hard Excel errors that corrupt valuation outputs.

#### 5) Lookups and mapping
Prefer exact matches for mapping assumptions:
```text
XLOOKUP(key, key_range, value_range, if_not_found, 0)
```
Avoid approximate match unless intentionally bucketed.

#### 6) Aggregation patterns (clean driver trees)
```text
SUMIFS(sum_range, criteria_range1, crit1, ...)
AVERAGEIFS(...)
```
Used for peer medians, segment rollups, and normalisation.

---

### Practical Application (How to apply in valuation work)

#### A) DCF and continuing value function patterns
- Always separate:
  1) operating model (drivers → financials → FCF)
  2) valuation layer (discounting, terminal value, EV → equity)
- Use explicit discount factors and isolate terminal value.

Common DCF worksheet structure:
| Block | Outputs | Key functions |
|---|---|---|
| Drivers | growth, margin, reinvestment | IF, MIN/MAX, CHOOSE |
| Financials | revenue, EBITDA, EBIT, tax | SUMIFS, XLOOKUP |
| FCF | NOPAT, ΔNWC, Capex | IFERROR/NA patterns |
| Valuation | PV, TV, EV | NPV, power, DF rows |
| Bridge | equity value/share | simple arithmetic + SAFE division |

#### B) Debt and leases schedules (function discipline)
- Use consistent sign conventions:
  - Cash outflows negative; inflows positive; EV and debt positive
- Use MIN/MAX to cap repayments and avoid negative balances:
  - repayment = MIN(requested, opening_balance)

#### C) Peer multiples and mapping
- Use `MEDIAN`/`PERCENTILE.INC` for peer stats (avoid average when skewed).
- Use exact lookup for peer group classification.

#### D) Error handling approach
- Prefer “propagate NA” over forcing zeros:
  - A missing WACC should result in NA outputs, not a valuation.
- Apply a top-level `Model_Status` flag (OK/ERROR) driven by validations.

---

### Python Implementation
```python
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def discount_factor(rate: float, t: int) -> float:
    """Discount factor DF_t = 1 / (1 + r)^t."""
    r = _num(rate, "rate")
    if t < 0:
        raise ValueError("t must be >= 0.")
    if r <= -0.999999:
        raise ValueError("rate must be > -100%.")
    return float(1.0 / ((1.0 + r) ** t))

def pv_cashflows(cashflows: Sequence[float], rate: float, start_t: int = 1) -> float:
    """Present value of a sequence of cashflows starting at period start_t.

    Args:
        cashflows: CFs for t=start_t..start_t+N-1
        rate: discount rate per period as decimal
        start_t: starting period index (1 for standard end-of-period cashflows)

    Returns:
        float: computes Σ CF_t / (1+r)^t

    Raises:
        ValueError: invalid inputs
    """
    r = _num(rate, "rate")
    if r <= -0.999999:
        raise ValueError("rate must be > -100%.")
    if start_t < 0:
        raise ValueError("start_t must be >= 0.")
    if not isinstance(cashflows, (list, tuple, np.ndarray)):
        raise ValueError("cashflows must be a sequence.")
    cfs = np.array([_num(x, f"cashflows[{i}]") for i, x in enumerate(cashflows)], dtype=float)
    if cfs.size == 0:
        raise ValueError("cashflows must be non-empty.")
    ts = np.arange(start_t, start_t + cfs.size, dtype=float)
    df = 1.0 / np.power(1.0 + r, ts)
    return float(np.sum(cfs * df))

def cagr(start: float, end: float, n_periods: int) -> float:
    """Compound annual growth rate."""
    s = _num(start, "start")
    e = _num(end, "end")
    if n_periods <= 0:
        raise ValueError("n_periods must be > 0.")
    if s <= 0 or e <= 0:
        raise ValueError("start and end must be > 0 for CAGR.")
    return float((e / s) ** (1.0 / n_periods) - 1.0)

def safe_divide(numerator: float, denominator: float, on_zero: Optional[float] = None) -> Optional[float]:
    """Safe division. Returns on_zero (default None) if denominator is 0."""
    num = _num(numerator, "numerator")
    den = _num(denominator, "denominator")
    if abs(den) < 1e-12:
        return on_zero
    return float(num / den)

def xlookup_exact(keys: Sequence[Any], values: Sequence[Any], query: Any, default: Any = None) -> Any:
    """Exact lookup (Python equivalent of XLOOKUP exact match).

    Args:
        keys: lookup keys
        values: parallel values
        query: key to find
        default: returned if not found

    Returns:
        matched value or default
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length.")
    for k, v in zip(keys, values):
        if k == query:
            return v
    return default

def median_peer_multiple(multiples: Sequence[float]) -> float:
    """Median of peer multiples with validation."""
    arr = np.array([_num(x, "multiple") for x in multiples], dtype=float)
    if arr.size == 0:
        raise ValueError("multiples must be non-empty.")
    return float(np.median(arr))

# Example usage
fcf = [50, 60, 70, 80, 90]   # $m
wacc = 0.10
pv = pv_cashflows(fcf, wacc, start_t=1)
print(f"PV of FCF: ${pv:.1f}m")

g = cagr(100, 150, 3)
print(f"CAGR: {g:.2%}")

ratio = safe_divide(200, 0, on_zero=np.nan)
print(f"Safe ratio: {ratio}")
```

---

### Valuation Impact
Why this matters:
- Most valuation errors are not “finance theory” errors; they are spreadsheet implementation errors (wrong sign, wrong period, hidden CF0 handling, broken lookups).
- Robust function patterns reduce audit risk and ensure outputs reflect assumptions transparently.

Impact on multiples:
- Peer multiple selection depends on clean mappings (sector, geography, business model) and robust aggregation (median vs mean).
- Lookup errors can misclassify peer sets and materially distort implied valuation.

Impact on DCF inputs:
- PV/NPV sign conventions and CF0 treatment affect EV and can flip investment decisions.
- CAGR and driver projections affect revenue/margin paths; poor error handling can silently default growth to 0.

Comparability issues across companies:
- Function-based normalisation (e.g., safe ratios, consistent denominators) avoids misleading comparisons when companies have volatile or near-zero denominators.

Practical adjustments:
```python
def normalise_metric(reported: float, adjustments: Sequence[float]) -> float:
    """Add back / subtract adjustments to produce a normalised metric."""
    x = _num(reported, "reported")
    adj = np.array([_num(a, "adjustment") for a in adjustments], dtype=float)
    return float(x + adj.sum())
```

---

### Quality of Earnings Flags
⚠️ Use of IFERROR to force zeros on missing inputs (hides problems and creates fake precision).  
⚠️ NPV applied to a range that incorrectly includes CF0 or omits the first forecast year.  
⚠️ Approximate lookups used for assumption mapping (silent misclassification).  
✅ Explicit discount factor rows; NA propagation for missing inputs; exact match lookups; documented sign conventions.

---

### Sector-Specific Considerations

| Sector | Common function risk | Typical treatment |
|---|---|---|
| Subscription/SaaS | cohort/ARR logic breaks with blanks | explicit NA + validation flags |
| Infrastructure | long horizons amplify PV errors | DF audit rows + sensitivity tables |
| Financials | ratio denominators near zero | safe_divide + percentile stats |
| Cyclicals | normalisation via medians | robust aggregation (median/IQR) |

---

### Real-World Example
Scenario: Build a clean DCF discounting block with explicit discount factors and NA-safe outputs.

```python
fcf = [50, 60, 70, 80, 90]  # $m
wacc = 0.095

dfs = [discount_factor(wacc, t) for t in range(1, len(fcf)+1)]
pv = pv_cashflows(fcf, wacc, start_t=1)

table = pd.DataFrame({"Year": range(1, len(fcf)+1), "FCF_$m": fcf, "DF": dfs})
table["PV_$m"] = table["FCF_$m"] * table["DF"]
print(table)

print(f"Total PV: ${pv:.1f}m")
```

Interpretation: The DF row makes discounting transparent and easy to audit; PV ties to the sum of PV line items.

See also: Chapter 4 (DCF valuation), Chapter 28 (data tables for sensitivity), Chapter 29 (matrices for regressions/portfolio risk).
