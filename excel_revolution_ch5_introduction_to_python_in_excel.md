# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 5: Introduction to Python in Excel (Workflow, Data Passing, Constraints, Finance Use-Cases)

### Core Concept
Chapter 5 introduces Python-in-Excel as a way to run Python calculations directly inside worksheets, enabling repeatable analytics (cleaning, transformations, valuation calculations, forecasting helpers) while keeping outputs in Excel for review and audit. For valuation teams, the key is using Python to automate error-prone steps (data prep, peer stats, sensitivities) without breaking spreadsheet governance: outputs must be deterministic, transparent, and easy to reconcile to source data.

See also: Chapter 6 (PY function), Chapter 7 (pandas in Excel), Chapter 8 (automation), McKinsey Chapters on forecasting and multiples.

---

### Formula/Methodology

#### 1) “Spreadsheet governance” rules for Python-in-Excel
```text
Rule 1: Separate inputs, transforms, and outputs.
Rule 2: Make every Python cell deterministic (no web calls, no random unless seeded).
Rule 3: Validate inputs (types, missing values, denominators).
Rule 4: Return small, auditable outputs (scalars, compact tables).
Rule 5: Reconcile to Excel formulas where possible (cross-checks).
```

#### 2) Common valuation calculations suited to Python-in-Excel
```text
Enterprise Value (EV) = MarketCap + NetDebt + PreferredEquity + MinorityInterest - NonOperatingAssets
Where:
MarketCap = SharePrice × SharesOutstanding
NetDebt = TotalDebt - Cash
NonOperatingAssets = assets not required for operations (excess cash, investments)
```

```text
EV/EBITDA = Enterprise Value / EBITDA
Where:
EBITDA should be normalized (QoE) and consistent across peers.
```

```text
DCF (simplified) = Σ[ FCF_t / (1 + WACC)^t ] + TV / (1 + WACC)^T
Where:
FCF_t = free cash flow
TV = terminal value (perpetuity or exit multiple)
WACC = discount rate
```

---

### Python Implementation
```python
from typing import Dict, Any, Optional, Sequence, Union
import numpy as np
import pandas as pd

Number = Union[int, float]

def validate_scalar(x: Any, name: str) -> float:
    """
    Validate a scalar input and convert to float.

    Args:
        x: input value
        name: variable name for error messages

    Returns:
        float: numeric value

    Raises:
        ValueError: if input is missing or not finite.
    """
    if x is None:
        raise ValueError(f"{name} is missing (None).")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def enterprise_value(
    share_price: Number,
    shares_outstanding: Number,
    total_debt: Number,
    cash: Number,
    preferred_equity: Number = 0.0,
    minority_interest: Number = 0.0,
    non_operating_assets: Number = 0.0
) -> float:
    """
    Compute enterprise value (EV) with defensive validation.

    Args:
        share_price: $ per share
        shares_outstanding: shares count
        total_debt: $ total debt (short + long)
        cash: $ cash and equivalents
        preferred_equity: $ preferred equity
        minority_interest: $ minority interest
        non_operating_assets: $ non-operating assets to subtract (e.g., excess cash, investments)

    Returns:
        float: enterprise value in $

    Raises:
        ValueError: for non-finite inputs or negative shares.
    """
    sp = validate_scalar(share_price, "share_price")
    sh = validate_scalar(shares_outstanding, "shares_outstanding")
    if sh < 0:
        raise ValueError("shares_outstanding must be >= 0.")
    debt = validate_scalar(total_debt, "total_debt")
    c = validate_scalar(cash, "cash")
    pe = validate_scalar(preferred_equity, "preferred_equity")
    mi = validate_scalar(minority_interest, "minority_interest")
    noa = validate_scalar(non_operating_assets, "non_operating_assets")

    market_cap = sp * sh
    net_debt = debt - c
    ev = market_cap + net_debt + pe + mi - noa
    return ev

def ev_multiple(ev: Number, ebitda: Number) -> float:
    """
    Compute EV/EBITDA multiple.

    Raises:
        ValueError: if EBITDA is zero or non-finite.
    """
    evv = validate_scalar(ev, "ev")
    e = validate_scalar(ebitda, "ebitda")
    if abs(e) < 1e-12:
        raise ValueError("ebitda must be non-zero for EV/EBITDA.")
    return evv / e

def dcf_value(fcf: Sequence[Number], wacc: Number, terminal_value: Number) -> float:
    """
    Simple DCF PV of forecast FCF plus terminal value.

    Args:
        fcf: sequence of free cash flows for t=1..T
        wacc: discount rate (decimal)
        terminal_value: value at time T (already in currency units)

    Returns:
        float: present value

    Raises:
        ValueError: if wacc <= -1 or fcf empty.
    """
    r = validate_scalar(wacc, "wacc")
    if r <= -1.0:
        raise ValueError("wacc must be > -1.0.")
    if fcf is None or len(fcf) == 0:
        raise ValueError("fcf must be non-empty for t=1..T.")
    tv = validate_scalar(terminal_value, "terminal_value")

    pv = 0.0
    for t, cf in enumerate(fcf, start=1):
        cft = validate_scalar(cf, f"fcf_t{t}")
        pv += cft / ((1.0 + r) ** t)

    T = len(fcf)
    pv += tv / ((1.0 + r) ** T)
    return pv

# Example usage (Python-in-Excel friendly: no file I/O, no web)
company = {
    "share_price": 25.0,
    "shares_outstanding": 120_000_000,
    "total_debt": 1_600_000_000,
    "cash": 300_000_000,
    "preferred_equity": 0.0,
    "minority_interest": 50_000_000,
    "non_operating_assets": 20_000_000,
    "ebitda": 420_000_000
}

ev = enterprise_value(
    share_price=company["share_price"],
    shares_outstanding=company["shares_outstanding"],
    total_debt=company["total_debt"],
    cash=company["cash"],
    preferred_equity=company["preferred_equity"],
    minority_interest=company["minority_interest"],
    non_operating_assets=company["non_operating_assets"],
)

mult = ev_multiple(ev, company["ebitda"])
print(f"EV: ${ev/1e9:.2f}B")
print(f"EV/EBITDA: {mult:.1f}x")

pv = dcf_value(fcf=[120_000_000, 135_000_000, 150_000_000, 165_000_000, 180_000_000], wacc=0.10, terminal_value=2_400_000_000)
print(f"DCF PV: ${pv/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- **Reproducible transforms:** Python-in-Excel can standardize how comps are cleaned, mapped, and summarized (quartiles, winsorized medians) across projects.
- **Model controls:** Defensive validation in code prevents silent spreadsheet failures (divide-by-zero, missing keys), improving review efficiency and reducing model risk.
- **Faster sensitivities:** Matrix outputs (DataFrames) are easier to generate in Python than via nested Excel formulas, and remain auditable.

Impact on multiples:
- Enables consistent normalization (e.g., remove one-offs) before computing EV/EBITDA and peer medians.
- Supports robust peer statistics (median, IQR) rather than single-point averages.

Impact on DCF inputs:
- Makes it easier to build repeatable forecast helper functions (growth/margin bridges, working capital normalization).
- Forces explicit assumptions and guards (e.g., WACC bounds, terminal constraints).

Practical adjustments:
```python
def normalize_ebitda(reported_ebitda: float, add_backs: float = 0.0, deductions: float = 0.0) -> float:
    \"\"\"Normalize EBITDA for QoE (simple add-back/deduction hook).\"\"\"
    e = float(reported_ebitda)
    if not np.isfinite(e):
        raise ValueError("reported_ebitda must be finite.")
    return e + float(add_backs) - float(deductions)
```

---

### Quality of Earnings Flags
⚠️ **Red flag: Python outputs used without reconciliation** to a basic Excel cross-check (e.g., recompute EV in Excel).  
⚠️ **Red flag: missing data silently coerced to zero** (e.g., NaNs filled without review), biasing margins/multiples.  
⚠️ **Red flag: randomness or non-determinism** in a model (e.g., Monte Carlo without fixed seed) used for a final valuation conclusion.  
✅ **Green flag: explicit validation + sentinels** (“MISSING”, errors raised) forcing review and improving audit trail.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | KPI-heavy datasets | Use Python for KPI reshaping and cohort tables, return compact outputs back to Excel for charts |
| Banking | Many ratios + definitions | Python standardizes definitions and peer stats; ensure consistent inputs (CET1, ROE) |
| Real Estate | Property-level rollups | Python groups property cash flows/NOI; Excel displays dashboards and sensitivity grids |
| Industrials | Complex segmentation | Python builds mix/price bridges and peer clusters; Excel remains the presentation layer |

---

### Real-World Example
Scenario: Pull a peer table into Excel, use Python-in-Excel to compute robust peer multiples (median, IQR), and compute an EV/EBITDA valuation range.

```python
import numpy as np
import pandas as pd

def robust_multiple_stats(multiples: Sequence[Number]) -> dict:
    arr = np.asarray(list(multiples), dtype=float)
    if arr.size == 0:
        raise ValueError("multiples must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("multiples contain non-finite values.")
    q1 = float(np.quantile(arr, 0.25))
    med = float(np.quantile(arr, 0.50))
    q3 = float(np.quantile(arr, 0.75))
    return {"q1": q1, "median": med, "q3": q3, "iqr": q3 - q1}

peer_multiples = [8.5, 9.2, 10.1, 11.7, 12.4, 14.0, 20.0]  # includes an outlier
stats = robust_multiple_stats(peer_multiples)

normalized_ebitda = 420_000_000
ev_low = stats["q1"] * normalized_ebitda
ev_mid = stats["median"] * normalized_ebitda
ev_high = stats["q3"] * normalized_ebitda

print(stats)
print(f"EV range: ${ev_low/1e9:.2f}B - ${ev_high/1e9:.2f}B (median ${ev_mid/1e9:.2f}B)")
```

Interpretation: Using quartiles rather than means reduces the influence of outliers and improves defensibility when selecting a multiple range.

See also: Chapter 6 (PY), Chapter 7 (pandas tasks), Chapter 10 (financial equations).
