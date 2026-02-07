# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 6: The PY Function (Inputs/Outputs, Excel Integration Patterns, Defensive Analytics)

### Core Concept
Chapter 6 focuses on using the `=PY()` function to execute Python inside Excel cells and return results into the grid. For valuation work, the value is in creating **reusable, validated calculation blocks** (peer stats, sensitivities, statement reorg helpers) while keeping the spreadsheet as the review surface. The key technical requirement is to design `=PY()` code to be **deterministic, small-output, and failure-obvious** (raise errors or return clear sentinels instead of silently coercing bad inputs).

See also: Chapter 5 (Python-in-Excel workflow), Chapter 7 (pandas tasks), Chapter 10 (financial equations), McKinsey forecasting/multiples chapters.

---

### Formula/Methodology

#### 1) Data boundary: Excel ↔ Python
In Python-in-Excel, the usual pattern is:
- Excel holds **inputs** (ranges, tables, assumption cells)
- Python reads them (often via `xl()`), transforms/calculates
- Python returns a scalar, list, or DataFrame back to Excel

Conceptually:
```text
PythonOutput = f(ExcelInputs)
Where:
ExcelInputs are values/ranges supplied to the Python cell
f is deterministic transformation / calculation
```

#### 2) Common valuation computations that fit `=PY()`
```text
Median Multiple = median(peer_multiples)
IQR = Q3 - Q1
Where:
Q1 = 25th percentile, Q3 = 75th percentile
```

```text
Winsorized Series:
x_wins = clamp(x, lower_quantile, upper_quantile)
Where:
clamp(a, L, U) = min(max(a, L), U)
```

```text
Sensitivity Grid:
V(i,j) = valuation_output(assumption_a_i, assumption_b_j)
```

---

### Python Implementation
```python
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union, Callable
import numpy as np
import pandas as pd

Number = Union[int, float]

def validate_array(x: Iterable[Any], name: str) -> np.ndarray:
    """
    Validate an iterable of values as a numeric numpy array.

    Args:
        x: iterable of values (numbers or numeric strings)
        name: name for error messages

    Returns:
        np.ndarray: float array

    Raises:
        ValueError: if empty or contains non-finite values.
    """
    try:
        arr = np.asarray(list(x), dtype=float)
    except Exception as e:
        raise ValueError(f"{name} must be numeric-convertible: {e}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf).")
    return arr

def robust_stats(x: Iterable[Any]) -> Dict[str, float]:
    """
    Robust summary stats for comparables and dashboards.

    Returns:
        dict: count, min, q1, median, q3, max, iqr
    """
    arr = validate_array(x, "x")
    q1 = float(np.quantile(arr, 0.25))
    med = float(np.quantile(arr, 0.50))
    q3 = float(np.quantile(arr, 0.75))
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "q1": q1,
        "median": med,
        "q3": q3,
        "max": float(np.max(arr)),
        "iqr": float(q3 - q1),
    }

def winsorize(x: Iterable[Any], lower_q: float = 0.05, upper_q: float = 0.95) -> np.ndarray:
    """
    Winsorize series by clamping to quantile bounds.

    Args:
        x: input series
        lower_q: lower quantile (0..1)
        upper_q: upper quantile (0..1)

    Returns:
        np.ndarray: winsorized series

    Raises:
        ValueError: invalid quantiles or inputs.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Quantiles must satisfy 0 <= lower_q < upper_q <= 1.")
    arr = validate_array(x, "x")
    lo = float(np.quantile(arr, lower_q))
    hi = float(np.quantile(arr, upper_q))
    return np.clip(arr, lo, hi)

def safe_div(numer: Any, denom: Any, name: str = "ratio") -> float:
    """
    Safe division with explicit zero-denominator handling.

    Raises:
        ValueError: if denom is zero or inputs non-finite.
    """
    n = float(numer)
    d = float(denom)
    if not np.isfinite(n) or not np.isfinite(d):
        raise ValueError(f"{name}: inputs must be finite.")
    if abs(d) < 1e-12:
        raise ValueError(f"{name}: denominator must be non-zero.")
    return n / d

def to_dataframe(d: Dict[str, float]) -> pd.DataFrame:
    """
    Convert a small dict to a 1-row DataFrame for Excel output.
    """
    return pd.DataFrame([d])

# Example usage: peer EV/EBITDA distribution (Python cell output to Excel)
peer_multiples = [8.5, 9.2, 10.1, 11.7, 12.4, 14.0, 20.0]  # outlier included
stats = robust_stats(peer_multiples)
print(stats)

wins = winsorize(peer_multiples, 0.05, 0.95)
print("Winsorized median:", float(np.median(wins)))

# Example: safe division for margin
ebitda = 210_000_000
revenue = 1_000_000_000
ebitda_margin = safe_div(ebitda, revenue, name="EBITDA margin")
print(f"EBITDA margin: {ebitda_margin:.1%}")

# Example: return a DataFrame for Excel display
out_df = to_dataframe(stats)
print(out_df)
```

---

### Valuation Impact
Why this matters:
- **Governance and auditability:** `=PY()` lets you replace fragile nested formulas with tested functions while still keeping calculations visible and reconcilable in the workbook.
- **Faster peer analysis:** robust stats and winsorization are quick to compute in Python, improving multiple selection defensibility.
- **Better sensitivities:** Python can generate full matrices (WACC × g, exit multiple × margin) with less model clutter.

Impact on multiples:
- Quartile-based ranges reduce outlier influence vs mean-based metrics.
- Winsorization improves stability of peer medians and valuation ranges.

Impact on DCF inputs:
- Enables rapid sensitivity matrices and scenario tables.
- Supports consistent data cleaning (missing handling, unit standardization) before forecasts.

Practical adjustments:
```python
def implied_ev_from_multiple(normalized_metric: float, multiple: float) -> float:
    \"\"\"Convert a multiple and metric into implied enterprise value.\"\"\"
    m = float(normalized_metric); mult = float(multiple)
    if not np.isfinite(m) or not np.isfinite(mult):
        raise ValueError("Inputs must be finite.")
    return m * mult
```

---

### Quality of Earnings Flags
⚠️ **Red flag: Python outputs that silently fill missing values** (NaNs → 0) without an explicit review gate.  
⚠️ **Red flag: inconsistent units** (e.g., some peers in $m, others in $) inside Python arrays.  
⚠️ **Red flag: returning huge tables** that obscure what changed and increase review friction.  
✅ **Green flag: small outputs + explicit errors** (raise ValueError) that force attention when data is invalid.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | Outliers in multiples (high growth dispersion) | Use robust stats + winsorization; show IQR and median rather than mean |
| Banking | Ratio definitions differ | Use Python to standardize inputs before calculating distributions; validate denominator rules |
| Real Estate | Cap rate and NOI distributions | Compute percentiles and trim/winsorize; report sensitivity to cap rates |
| Industrials | Cyclical volatility | Compute stats by cycle phase or year; avoid pooling regimes without flags |

---

### Real-World Example
Scenario: Use `=PY()` to compute a peer multiple range and implied EV range from normalized EBITDA, and return a compact table to Excel.

```python
import numpy as np
import pandas as pd

def peer_range_and_ev(normalized_ebitda: float, peer_multiples: list) -> pd.DataFrame:
    e = float(normalized_ebitda)
    if not np.isfinite(e) or e <= 0:
        raise ValueError("normalized_ebitda must be positive and finite.")
    arr = np.asarray(peer_multiples, dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        raise ValueError("peer_multiples must be non-empty and finite.")
    q1, med, q3 = np.quantile(arr, [0.25, 0.50, 0.75])
    ev_low, ev_mid, ev_high = e*q1, e*med, e*q3
    return pd.DataFrame([{
        "multiple_q1": float(q1),
        "multiple_median": float(med),
        "multiple_q3": float(q3),
        "ev_low": float(ev_low),
        "ev_mid": float(ev_mid),
        "ev_high": float(ev_high),
    }])

normalized_ebitda = 420_000_000
peer_multiples = [8.5, 9.2, 10.1, 11.7, 12.4, 14.0, 20.0]
df = peer_range_and_ev(normalized_ebitda, peer_multiples)
print(df)
print(f"EV range: ${df.loc[0,'ev_low']/1e9:.2f}B - ${df.loc[0,'ev_high']/1e9:.2f}B")
```

Interpretation: Returning a compact DataFrame with explicit quartiles and EV range is easy to audit and ties directly into a multiples-based valuation range.

See also: Chapter 7 (pandas-based tasks), Chapter 8 (automation), Chapter 10 (financial equations).
