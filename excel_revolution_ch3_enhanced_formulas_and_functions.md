# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 3: Enhanced Formulas and Functions (Lookups, Dynamic Arrays, Financial Functions, Named Ranges)

### Core Concept
Chapter 3 focuses on building robust, scalable Excel models using advanced formulas (lookups, dynamic arrays, and financial functions) and worksheet design patterns (tables, named ranges, validation, and formatting). For valuation work, these mechanics make forecast models auditable, repeatable, and less error-prone when datasets expand, assumptions change, or scenarios are layered.

### Formula/Methodology
```text
Weighted Average = SUMPRODUCT(values, weights) / SUM(weights)
Where:
values = numeric series being averaged (e.g., prices, margins, growth rates)
weights = exposure weights (e.g., volumes, revenue shares)
```

```text
NPV (explicit t=0 investment) = -InitialInvestment + Σ[ CF_t / (1 + r)^t ]  for t = 1..T
Where:
InitialInvestment = cash outflow at t=0 (positive number in this definition)
CF_t = cash flow at period t
r = discount rate per period (as decimal)
T = number of periods
```

```text
PMT (annuity payment) = r * PV / (1 - (1 + r)^(-n))
Where:
r = periodic interest rate (as decimal)
PV = present value (loan principal)
n = number of periods
```

### Python Implementation
```python
from typing import Iterable, Sequence, Union
import math

Number = Union[int, float]

def _to_float_list(x: Iterable[Number], name: str) -> list:
    """Convert iterable to list[float] with validation."""
    try:
        xs = [float(v) for v in x]
    except Exception as e:
        raise ValueError(f"{name} must be an iterable of numbers: {e}")
    if len(xs) == 0:
        raise ValueError(f"{name} must not be empty")
    return xs

def weighted_average(values: Iterable[Number], weights: Iterable[Number]) -> float:
    """Compute weighted average (Excel SUMPRODUCT/SUM)."""
    v = _to_float_list(values, "values")
    w = _to_float_list(weights, "weights")
    if len(v) != len(w):
        raise ValueError("values and weights must have the same length")
    s_w = sum(w)
    if abs(s_w) < 1e-12:
        raise ValueError("sum(weights) must be non-zero")
    return sum(vi * wi for vi, wi in zip(v, w)) / s_w

def npv(rate: Number, cashflows: Sequence[Number], initial_investment: Number = 0.0) -> float:
    """NPV with explicit t=0 investment term."""
    r = float(rate)
    if r <= -1.0:
        raise ValueError("rate must be > -1.0")
    if cashflows is None or len(cashflows) == 0:
        raise ValueError("cashflows must be a non-empty sequence for t=1..T")
    pv = -float(initial_investment)
    for t, cf in enumerate(cashflows, start=1):
        pv += float(cf) / ((1.0 + r) ** t)
    return pv

def irr(cashflows_with_t0: Sequence[Number], guess: float = 0.1, tol: float = 1e-7, max_iter: int = 100) -> float:
    """IRR using safeguarded Newton method."""
    if cashflows_with_t0 is None or len(cashflows_with_t0) < 2:
        raise ValueError("cashflows_with_t0 must include at least t=0 and one later cash flow")
    cfs = [float(x) for x in cashflows_with_t0]
    if not (min(cfs) < 0 < max(cfs)):
        raise ValueError("IRR requires at least one negative and one positive cash flow (sign change)")

    r = float(guess)
    for _ in range(max_iter):
        npv_val = 0.0
        d_npv = 0.0
        for t, cf in enumerate(cfs):
            denom = (1.0 + r) ** t
            npv_val += cf / denom
            if t > 0:
                d_npv -= t * cf / ((1.0 + r) ** (t + 1))
        if abs(npv_val) < tol:
            return r
        if abs(d_npv) < 1e-12:
            r += 0.01
            continue
        r_next = r - npv_val / d_npv
        if r_next <= -0.999999:
            r_next = (r - 0.999999) / 2.0
        r = r_next
    raise ValueError("IRR did not converge; try a different guess or check cash flow pattern")

def pmt(rate: Number, nper: int, pv: Number, fv: Number = 0.0, when: str = "end") -> float:
    """Periodic payment (Excel PMT-like)."""
    r = float(rate)
    if nper <= 0:
        raise ValueError("nper must be > 0")
    when = when.lower().strip()
    if when not in ("end", "begin"):
        raise ValueError("when must be 'end' or 'begin'")
    pv = float(pv)
    fv = float(fv)

    if abs(r) < 1e-12:
        return -(pv + fv) / nper

    factor = (1.0 + r) ** nper
    adj = (1.0 + r) if when == "begin" else 1.0
    return -(r * (fv + pv * factor)) / ((factor - 1.0) * adj)

# Example usage
margins = [0.12, 0.18, 0.10]
weights = [0.50, 0.30, 0.20]
print(f"Weighted margin: {weighted_average(margins, weights):.2%}")

cfs = [15000, 15000, 15000, 15000, 15000]
print(f"NPV: ${npv(0.07, cfs, initial_investment=50000):,.0f}")
print(f"IRR: {irr([-50000] + cfs):.2%}")
print(f"Monthly payment: ${pmt(0.05/12, 5*12, 100000):,.0f}")
```

### Practical Excel Patterns for Valuation Models

#### 1) Lookups (XLOOKUP preferred)
Use lookups to map raw lines to normalized valuation lines (EBITDA, NOPAT, invested capital), connect assumptions to segments, and pull comps/multiples reliably.

```text
XLOOKUP(lookup_value, lookup_array, return_array, if_not_found, match_mode, search_mode)
Recommended defaults for valuation models:
if_not_found = "MISSING" (not 0)
match_mode   = 0 (exact match)
search_mode  = 1 (first-to-last) or -1 (latest-from-bottom when time-ordered)
```

**Valuation Impact**
- Improves auditability and reduces silent mapping errors in statement reorgs and comps mapping.
- Avoids structural breaks when data tables change shape (inserted/removed columns).

**Quality of Earnings Flags**
- ⚠️ Missing keys defaulting to 0 or blank (silently understates costs / overstates margins).
- ✅ Missing keys return a sentinel ("MISSING") and are surfaced in a review tab.

#### 2) Dynamic arrays (spill ranges)
Dynamic arrays scale forecasting blocks and scenario tables without copying formulas down.

```text
Forecast (concept) = BaseRange * (1 + GrowthAssumption)
```

**Valuation Impact**
- Faster scenario builds; fewer “drag down” errors across forecast horizons.
- Cleaner roll-forwards (working capital, debt, capex) when the period count changes.

**Quality of Earnings Flags**
- ⚠️ #SPILL! errors from cramped layouts can break linked calculations.
- ✅ Reserve output blocks; add explicit checks for spill errors.

#### 3) SUMPRODUCT for weighted blends
SUMPRODUCT is a compact tool for probability-weighted outcomes and blended assumptions (segment mix, WACC components).

```text
Weighted Average = SUMPRODUCT(values, weights) / SUM(weights)
```

**Valuation Impact**
- Enables probability-weighted valuation and mix-driven margin bridges without add-ins.
- Improves comparability by standardizing blended KPIs across companies with different segment exposure.

**Quality of Earnings Flags**
- ⚠️ Weights not reconciled (wrong units; sum=0) distort margins/ROIC.
- ✅ Add a reconciliation row: SUM(weights) must equal 1.0 (or a documented total).

#### 4) Financial functions (NPV/IRR/PMT)
Model capex/project economics (NPV/IRR) and debt service (PMT) with timing discipline.

**Modelling rule:** match compounding basis to cash flow frequency (monthly vs annual), and document timing (end vs begin).

**Valuation Impact**
- Investment decisions drive capex intensity and free cash flow; debt schedules drive leverage and equity value per share.
- Better stress testing under downside scenarios (coverage ratios, covenant headroom).

**Quality of Earnings Flags**
- ⚠️ Mixing nominal vs real rates (or misaligned periods) produces misleading NPVs.
- ✅ Explicit rate/period labels and consistent conventions across the model.

### Tables

#### Lookup Methods Comparison
| Method | Best Use | Strengths | Common Failure Mode | Guardrail |
|---|---|---|---|---|
| VLOOKUP | Legacy sheets | Simple | Breaks with column inserts; limited flexibility | Wrap with IFERROR; avoid hard-coded indices |
| INDEX+MATCH | Robust lookups | Stable to column changes | Higher complexity | Standardize template; name key ranges |
| XLOOKUP | Default choice | Exact match by default; explicit if_not_found | Poor defaults can mask gaps | Use sentinel "MISSING" + review checks |

#### Dynamic Range Approaches
| Approach | Pros | Cons | Valuation Use |
|---|---|---|---|
| Excel Tables | Auto-expands; structured references | Requires disciplined design | Comparable sets; transaction logs |
| Named ranges | Readable formulas; reusable | Can go stale if not dynamic | Assumption blocks; KPI definitions |
| Dynamic named range (OFFSET/COUNTA) | Auto-resize | OFFSET is volatile; slower | Expanding actuals feeding forecasts |
| Dynamic arrays | Minimal copying; scalable forecasts | Spill conflicts | Multi-scenario forecast matrices |

### Practical Adjustments
```python
import math

def safe_lookup(mapping: dict, key, default="MISSING"):
    """Defensive lookup for mapping raw accounts -> normalized lines."""
    return mapping.get(key, default)

def normalize_metric(reported_metric: float, adjustment: float) -> float:
    """Generic normalization hook for valuation (e.g., EBITDA add-backs)."""
    x = float(reported_metric)
    if not math.isfinite(x):
        raise ValueError("reported_metric must be finite")
    return x + float(adjustment)
```

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | Multi-segment KPIs and fast mix shifts | Use mapping tables (lookups) + spill-safe forecast blocks |
| Banking | Dense product/account hierarchies | Structured tables + explicit missing-key flags |
| Real Estate | Large asset/lease lists | Dynamic ranges + NPV/PMT for debt and lease schedules |
| Industrials | Price/mix and volume weighting | SUMPRODUCT blends + filters for outlier review |

### Real-World Example
Scenario: Build a defensible EBITDA bridge by blending segment margins (using revenue weights) and applying it to revenue forecasts.

```python
from typing import Sequence, Union

Number = Union[int, float]

def _to_float_list(x, name: str):
    try:
        xs = [float(v) for v in x]
    except Exception as e:
        raise ValueError(f"{name} must be an iterable of numbers: {e}")
    if len(xs) == 0:
        raise ValueError(f"{name} must not be empty")
    return xs

def weighted_average(values, weights) -> float:
    v = _to_float_list(values, "values")
    w = _to_float_list(weights, "weights")
    if len(v) != len(w):
        raise ValueError("values and weights must have the same length")
    s = sum(w)
    if abs(s) < 1e-12:
        raise ValueError("sum(weights) must be non-zero")
    return sum(vi * wi for vi, wi in zip(v, w)) / s

def forecast_ebitda(revenue: Sequence[Number], margin: float) -> list:
    rev = _to_float_list(revenue, "revenue")
    m = float(margin)
    if m < -1.0 or m > 1.0:
        raise ValueError("margin should be a decimal in [-1, 1]")
    return [r * m for r in rev]

company_data = {
    "segment_margins": [0.12, 0.18, 0.10],
    "segment_weights": [0.50, 0.30, 0.20],
    "revenue_forecast": [1_000_000_000, 1_070_000_000, 1_145_000_000],  # $
}

blended_margin = weighted_average(company_data["segment_margins"], company_data["segment_weights"])
ebitda = forecast_ebitda(company_data["revenue_forecast"], blended_margin)

print(f"Blended margin: {blended_margin:.2%}")
print([f"${x/1e6:,.0f}M" for x in ebitda])
```

Interpretation: A blended, segment-grounded margin assumption is easier to defend than a single top-down margin and reduces the risk of hidden mix-driven bias in EBITDA (and therefore EV/EBITDA and DCF cash flows).

See also: Chapter 4 (data analysis & visualization) and Chapter 5–7 (Python in Excel + pandas for scalable transformations).
