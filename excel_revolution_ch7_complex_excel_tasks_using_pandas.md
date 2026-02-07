# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 7: Complex Excel Tasks Using Pandas (Tables, Joins, Cleaning, Grouping, Time Series)

### Core Concept
Chapter 7 applies pandas to tasks that are cumbersome or fragile in pure Excel: cleaning messy financial exports, joining multiple datasets (comps, FX, segment splits), grouping and aggregating, and producing analysis-ready tables for valuation models. For valuation, pandas is most useful for **statement reclassification**, **comparable set preparation**, **multiple calculations**, and **time-series transformations** that feed forecasts and sensitivity analysis.

See also: Chapter 6 (PY patterns), Chapter 8 (automation), McKinsey Chapter 11/12 (reorganising statements and performance analysis).

---

### Formula/Methodology

#### 1) Core transformations used in valuation pipelines
```text
Join (keyed merge) combines tables on identifiers:
Output = LEFT_JOIN(A, B, on=key)
Where:
A = primary table (e.g., peer list)
B = secondary table (e.g., multiples or financials)
key = shared identifier (ticker, company_id)
```

```text
Group aggregation:
GroupMetric = AGG(values) by group_key
Common AGG: sum, mean, median, count
```

```text
Trailing Twelve Months (TTM) metric:
TTM_t = Σ metric_{t-i} for i = 0..3 (quarterly series)
Where:
metric_{t} are quarterly values; use 4-quarter rolling sum
```

```text
Winsorized multiple:
x_wins = clamp(x, Q(lower), Q(upper))
Where:
Q(p) = p-quantile of the sample
```

---

### Python Implementation
```python
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

Number = Union[int, float]

def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str = "df") -> None:
    """
    Validate required columns exist.

    Raises:
        ValueError: if any required columns missing.
    """
    if df is None or df.empty:
        raise ValueError(f"{name} must be non-empty.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: Union[str, List[str]], how: str = "left") -> pd.DataFrame:
    """
    Defensive wrapper around pandas merge.

    Args:
        left: left DataFrame
        right: right DataFrame
        on: key column(s)
        how: merge type ('left', 'inner', 'right', 'outer')

    Returns:
        DataFrame merged

    Raises:
        ValueError: if inputs invalid or merge keys missing.
    """
    if how not in ("left", "inner", "right", "outer"):
        raise ValueError("how must be one of: left, inner, right, outer")
    if isinstance(on, str):
        on_cols = [on]
    else:
        on_cols = list(on)

    require_columns(left, on_cols, "left")
    require_columns(right, on_cols, "right")

    out = pd.merge(left, right, on=on_cols, how=how, validate=None)
    return out

def clean_numeric(df: pd.DataFrame, cols: Sequence[str], coerce_errors: bool = False) -> pd.DataFrame:
    """
    Convert selected columns to numeric.

    Args:
        df: input DataFrame
        cols: columns to convert
        coerce_errors: if True, invalid parses become NaN; if False, raise error

    Returns:
        DataFrame with converted columns

    Raises:
        ValueError: if conversion fails and coerce_errors is False
    """
    require_columns(df, cols, "df")
    out = df.copy()
    for c in cols:
        if coerce_errors:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            try:
                out[c] = pd.to_numeric(out[c], errors="raise")
            except Exception as e:
                raise ValueError(f"Failed to convert column {c} to numeric: {e}")
    return out

def compute_ev(
    df: pd.DataFrame,
    market_cap_col: str = "market_cap",
    total_debt_col: str = "total_debt",
    cash_col: str = "cash",
    minority_interest_col: Optional[str] = "minority_interest",
    preferred_equity_col: Optional[str] = "preferred_equity",
    non_operating_assets_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute enterprise value and add EV column.

    EV = MarketCap + TotalDebt - Cash + MinorityInterest + PreferredEquity - NonOperatingAssets

    Raises:
        ValueError: for missing columns or non-numeric values.
    """
    needed = [market_cap_col, total_debt_col, cash_col]
    opt = []
    if minority_interest_col:
        opt.append(minority_interest_col)
    if preferred_equity_col:
        opt.append(preferred_equity_col)
    if non_operating_assets_col:
        opt.append(non_operating_assets_col)

    require_columns(df, needed, "df")
    out = clean_numeric(df, needed + opt, coerce_errors=False).copy()

    mi = out[minority_interest_col] if minority_interest_col and minority_interest_col in out.columns else 0.0
    pe = out[preferred_equity_col] if preferred_equity_col and preferred_equity_col in out.columns else 0.0
    noa = out[non_operating_assets_col] if non_operating_assets_col and non_operating_assets_col in out.columns else 0.0

    out["ev"] = out[market_cap_col] + out[total_debt_col] - out[cash_col] + mi + pe - noa
    return out

def compute_multiple(df: pd.DataFrame, numer_col: str, denom_col: str, out_col: str) -> pd.DataFrame:
    """
    Compute a valuation multiple with denominator validation.

    Args:
        numer_col: numerator column (e.g., ev)
        denom_col: denominator column (e.g., ebitda)
        out_col: output column name (e.g., ev_ebitda)

    Returns:
        DataFrame with new out_col

    Raises:
        ValueError: if denom contains zeros or non-finite values.
    """
    require_columns(df, [numer_col, denom_col], "df")
    out = clean_numeric(df, [numer_col, denom_col], coerce_errors=False).copy()
    denom = out[denom_col].astype(float)
    if (denom.abs() < 1e-12).any():
        raise ValueError(f"{denom_col} contains zero values; cannot compute {out_col}.")
    out[out_col] = out[numer_col].astype(float) / denom
    if (~np.isfinite(out[out_col].to_numpy())).any():
        raise ValueError(f"{out_col} contains non-finite results.")
    return out

def ttm_from_quarters(df: pd.DataFrame, id_cols: Sequence[str], date_col: str, value_col: str, out_col: str = "ttm") -> pd.DataFrame:
    """
    Compute TTM rolling sum from quarterly series.

    Args:
        df: DataFrame with quarterly data
        id_cols: grouping keys (e.g., ['ticker'])
        date_col: sortable date column
        value_col: quarterly metric column
        out_col: output column name

    Returns:
        DataFrame with out_col added

    Raises:
        ValueError: if insufficient periods or missing columns.
    """
    require_columns(df, list(id_cols) + [date_col, value_col], "df")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    out = clean_numeric(out, [value_col], coerce_errors=False)
    out = out.sort_values(list(id_cols) + [date_col])
    out[out_col] = out.groupby(list(id_cols))[value_col].rolling(window=4, min_periods=4).sum().reset_index(level=list(range(len(id_cols))), drop=True)
    return out

def winsorize_series(x: Sequence[Number], lower_q: float = 0.05, upper_q: float = 0.95) -> np.ndarray:
    """
    Winsorize numeric series.

    Raises:
        ValueError: for invalid quantiles or non-finite inputs.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Quantiles must satisfy 0 <= lower_q < upper_q <= 1.")
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        raise ValueError("Input series must be non-empty and finite.")
    lo = float(np.quantile(arr, lower_q))
    hi = float(np.quantile(arr, upper_q))
    return np.clip(arr, lo, hi)

# Example: comps pipeline (clean -> EV -> EV/EBITDA -> robust stats)
comps = pd.DataFrame({
    "ticker": ["AAA", "BBB", "CCC", "DDD"],
    "market_cap": [2_500_000_000, 1_800_000_000, 3_200_000_000, 900_000_000],
    "total_debt": [900_000_000, 400_000_000, 1_200_000_000, 250_000_000],
    "cash": [200_000_000, 150_000_000, 300_000_000, 80_000_000],
    "minority_interest": [50_000_000, 0, 25_000_000, 0],
    "ebitda": [320_000_000, 210_000_000, 410_000_000, 70_000_000]
})

comps = compute_ev(comps)
comps = compute_multiple(comps, numer_col="ev", denom_col="ebitda", out_col="ev_ebitda")

wins = winsorize_series(comps["ev_ebitda"].tolist(), 0.05, 0.95)
print(comps[["ticker", "ev", "ev_ebitda"]])
print("Median (winsorized):", float(np.median(wins)))
```

---

### Valuation Impact
Why this matters:
- **Comparable prep at scale:** pandas makes it easy to clean, merge, and standardise peer data (units, definitions, missing values) before using multiples.
- **Statement reclassification:** reorging financials (operating vs non-operating, lease adjustments, one-off add-backs) is more reliable when done as explicit transformations.
- **Time-series readiness:** rolling TTM metrics and aligned period tables improve forecast baselining and prevent mixing quarterly and annual numbers.

Impact on multiples:
- Consistent EV calculation and denominator definitions reduce comparability noise.
- Robust stats (median/IQR, winsorization) reduce outlier influence and improve defensibility.

Impact on DCF inputs:
- Clean historical series enables driver-based forecasts (margins, working capital, reinvestment).
- Automated sensitivity grids and scenario tables are easier once inputs are structured.

Practical adjustments:
```python
def add_back_one_offs(df: pd.DataFrame, ebitda_col: str, one_off_col: str, out_col: str = "ebitda_normalized") -> pd.DataFrame:
    \"\"\"Normalize EBITDA by adding back one-offs (positive add-back increases EBITDA).\"\"\"
    require_columns(df, [ebitda_col, one_off_col], "df")
    out = clean_numeric(df, [ebitda_col, one_off_col], coerce_errors=False).copy()
    out[out_col] = out[ebitda_col] + out[one_off_col]
    return out
```

---

### Quality of Earnings Flags
⚠️ **Red flag: inconsistent units** across rows (some $m, others $) after merges—multiples become meaningless.  
⚠️ **Red flag: denominator mismatch** (EBITDA from different periods or definitions across peers).  
⚠️ **Red flag: aggressive missing-value handling** (filling NaNs with 0) that biases medians and ranges.  
✅ **Green flag: explicit validation** (required columns, non-zero denominators, finite outputs) and audit tables that show adjustments.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | Segment/KPI reshaping | Use melt/pivot to create clean KPI tables; compute ARR/NRR distributions; robust outlier handling |
| Banking | Peer alignment of ratios | Standardise definitions before aggregation; validate sign conventions and denominators |
| Real Estate | Property-level rollups | Groupby on property/region; compute NOI, occupancy, cap-rate distributions; TTM where relevant |
| Industrials | Cyclical time series | Use TTM/rolling averages; split samples by cycle phase; avoid pooling regimes without flags |

---

### Real-World Example
Scenario: Prepare a comps set in Excel (tickers + raw market cap, debt, cash, EBITDA), use pandas to compute EV and EV/EBITDA, winsorize outliers, and return a clean peer table and robust stats to Excel.

```python
# In Python-in-Excel, comps would come from xl("A1:F200") or a structured table.
# Steps:
# 1) clean_numeric()
# 2) compute_ev()
# 3) compute_multiple()
# 4) winsorize_series() and quantiles for range selection
```

Interpretation: A structured pandas pipeline creates a repeatable, reviewable comps process and reduces the risk of manual Excel errors.

See also: Chapter 8 (automation), Chapter 10 (financial equations), Chapter 12 (external data integration).
