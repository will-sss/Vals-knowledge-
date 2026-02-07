# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 4: Analyzing and Visualizing Data (PivotTables, Power Query, Charts, Dashboards, Auditability)

### Core Concept
Chapter 4 covers practical analysis and visualisation workflows in Excel—turning raw financial data into decision-ready insights using PivotTables, charts, conditional formatting, and Power Query-style transformations. In valuation, these tools are used to validate forecast drivers, explain performance, detect outliers (QoE), and build client-facing exhibits (bridge charts, margin trends, sensitivity tables) while maintaining auditability.

See also: Chapter 3 (model mechanics), Chapter 11 (financial reporting), McKinsey Chapter 12 (performance analysis), Damodaran relative valuation (peer comparisons).

---

### Formula/Methodology

#### 1) Growth, margins, and bridge metrics (core exhibit inputs)
```text
YoY Growth (%) = (Current - Prior) / Prior
Where:
Current = value in current period
Prior = value in prior period (must be non-zero)
```

```text
Margin (%) = Profit Metric / Revenue
Where:
Profit Metric = EBITDA, EBIT, Gross Profit, NOPAT (choose consistently)
Revenue = period revenue
```

```text
Contribution to change (bridge) = BaseWeight × (Change in Driver)
Common drivers:
Volume, Price, Mix, FX, Costs
```

#### 2) Basic statistical summaries (for dashboard tiles)
```text
Mean = SUM(x) / n
Median = middle value after sorting
Std Dev (population) = sqrt( SUM( (x - mean)^2 ) / n )
```

#### 3) Sensitivity table mechanics (what dashboards usually display)
```text
Sensitivity Grid: V(a_i, b_j) computed over i,j combinations
Where:
V = valuation output (e.g., terminal value, enterprise value, equity value per share)
a_i, b_j = scenario points (e.g., WACC and terminal growth)
```

---

### Python Implementation
```python
from typing import Dict, Sequence, Union, Callable
import numpy as np
import pandas as pd

Number = Union[int, float]

def _finite_float(x: Number, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def yoy_growth(current: Number, prior: Number) -> float:
    """
    Year-on-year growth rate.

    Args:
        current: Current period value.
        prior: Prior period value (must be non-zero).

    Returns:
        float: Growth as decimal (e.g., 0.12 = 12%).

    Raises:
        ValueError: When prior is zero or inputs are non-finite.
    """
    c = _finite_float(current, "current")
    p = _finite_float(prior, "prior")
    if abs(p) < 1e-12:
        raise ValueError("prior must be non-zero for YoY growth.")
    return (c - p) / p

def margin(metric: Number, revenue: Number) -> float:
    """
    Compute margin = metric / revenue.

    Args:
        metric: Profit metric (e.g., EBITDA).
        revenue: Revenue (must be non-zero).

    Returns:
        float: Margin as decimal.

    Raises:
        ValueError: If revenue is zero/non-finite.
    """
    m = _finite_float(metric, "metric")
    r = _finite_float(revenue, "revenue")
    if abs(r) < 1e-12:
        raise ValueError("revenue must be non-zero to compute margin.")
    return m / r

def describe_series(x: Sequence[Number]) -> Dict[str, float]:
    """
    Compact descriptive stats for dashboard tiles.

    Args:
        x: numeric sequence

    Returns:
        dict: mean, median, std, min, max

    Raises:
        ValueError: If empty or contains non-finite values.
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        raise ValueError("x must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("x contains non-finite values.")
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def pivot_like(df: pd.DataFrame, index: str, columns: str, values: str, agg: str = "sum") -> pd.DataFrame:
    """
    PivotTable-like operation using pandas pivot_table.

    Args:
        df: input DataFrame
        index: row field
        columns: column field
        values: value field
        agg: 'sum', 'mean', 'median', 'count'

    Returns:
        DataFrame pivot result

    Raises:
        ValueError: for missing columns, empty df, or invalid agg.
    """
    if df is None or df.empty:
        raise ValueError("df must be non-empty.")
    for c in (index, columns, values):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if agg not in ("sum", "mean", "median", "count"):
        raise ValueError("agg must be one of: sum, mean, median, count")

    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=agg, fill_value=0)

def sensitivity_grid(
    base_inputs: Dict[str, float],
    a_name: str,
    a_values: Sequence[Number],
    b_name: str,
    b_values: Sequence[Number],
    fn: Callable[..., float]
) -> pd.DataFrame:
    """
    Build a sensitivity grid for valuation outputs.

    Args:
        base_inputs: baseline inputs for fn (dict)
        a_name: first varied input name (e.g., 'wacc')
        a_values: values for a
        b_name: second varied input name (e.g., 'g')
        b_values: values for b
        fn: callable taking **inputs returning scalar output

    Returns:
        DataFrame: index=a_values, columns=b_values

    Raises:
        ValueError: if fn returns non-finite output or inputs invalid.
    """
    if not isinstance(base_inputs, dict) or len(base_inputs) == 0:
        raise ValueError("base_inputs must be a non-empty dict.")
    if a_name == b_name:
        raise ValueError("a_name and b_name must differ.")

    a_vals = [float(v) for v in a_values]
    b_vals = [float(v) for v in b_values]
    if len(a_vals) == 0 or len(b_vals) == 0:
        raise ValueError("a_values and b_values must be non-empty.")

    grid = np.zeros((len(a_vals), len(b_vals)), dtype=float)
    for i, av in enumerate(a_vals):
        for j, bv in enumerate(b_vals):
            inputs = dict(base_inputs)
            inputs[a_name] = av
            inputs[b_name] = bv
            out = float(fn(**inputs))
            if not np.isfinite(out):
                raise ValueError("fn returned non-finite output.")
            grid[i, j] = out
    return pd.DataFrame(grid, index=a_vals, columns=b_vals)

# Example: basic performance dashboard inputs
company = {
    "revenue_2024": 1_000_000_000,
    "revenue_2023": 920_000_000,
    "ebitda_2024": 210_000_000,
    "ebitda_2023": 175_000_000
}

rev_growth = yoy_growth(company["revenue_2024"], company["revenue_2023"])
marg_24 = margin(company["ebitda_2024"], company["revenue_2024"])
marg_23 = margin(company["ebitda_2023"], company["revenue_2023"])
print(f"Revenue growth: {rev_growth:.1%}")
print(f"EBITDA margin 2024: {marg_24:.1%} | 2023: {marg_23:.1%}")

# Example: pivot-like peer table
df = pd.DataFrame({
    "sector": ["Tech", "Tech", "Industrial", "Industrial", "Tech"],
    "metric": ["EV/EBITDA", "EV/Revenue", "EV/EBITDA", "EV/Revenue", "EV/EBITDA"],
    "value": [14.2, 5.1, 9.8, 1.9, 12.7]
})
pv = pivot_like(df, index="sector", columns="metric", values="value", agg="mean")
print(pv)

# Example: sensitivity grid for a perpetuity terminal value
def perpetuity_tv(fcf1: float, wacc: float, g: float) -> float:
    if wacc <= g:
        raise ValueError("wacc must be > g for perpetuity terminal value.")
    return fcf1 / (wacc - g)

base = {"fcf1": 120_000_000, "wacc": 0.10, "g": 0.03}
wacc_grid = [0.09, 0.10, 0.11]
g_grid = [0.02, 0.03, 0.04]
sens = sensitivity_grid(base, "wacc", wacc_grid, "g", g_grid, perpetuity_tv)
print(sens)
```

---

### Valuation Impact
Why this matters:
- **Audit and plausibility checks:** Dashboards reveal implausible trends (margin spikes, capex collapses, working capital swings) before they contaminate forecasts.
- **Comparable analysis:** Pivot-style tables and distributions help set ranges (median, quartiles) and identify outliers to exclude or normalise.
- **Communication:** Exhibits (bridges, sensitivities) are often the evidence base for valuation judgments and review sign-off.

Impact on multiples:
- Peer multiple distributions support selection of a defendable multiple range.
- Trend visuals help justify where within a peer range to land.

Impact on DCF inputs:
- Sensitivities quantify the value dependence on WACC, terminal growth, and near-term cash flows.
- Bridges provide mechanics for forecast integrity (volume/price/mix/cost).

Practical adjustments:
```python
import numpy as np

def safe_percent_change(current: float, prior: float, eps: float = 1e-12) -> float:
    \"\"\"Percent change with guardrail for near-zero prior.\"\"\"
    c = float(current); p = float(prior)
    if not np.isfinite(c) or not np.isfinite(p):
        raise ValueError("Inputs must be finite.")
    if abs(p) < eps:
        raise ValueError("prior is too close to zero; percent change is unstable.")
    return (c - p) / p
```

---

### Quality of Earnings Flags
⚠️ **Red flag: margin jumps** not supported by operational drivers (often one-offs, capitalisation, or classification changes).  
⚠️ **Red flag: working capital swings** inconsistent with revenue change (potential revenue recognition/provisioning effects).  
⚠️ **Red flag: cherry-picked periods** in visuals that hide volatility or downturns.  
✅ **Green flag: exhibit reconciliation** to source statements and consistent definitions over time.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | KPI-heavy reporting (ARR, NRR, churn) | Use cohort-style visuals; show distributions and retention curves; bridge ARR to revenue |
| Banking | Ratio comparability (ROE, CET1, NIM) | Pivot by business line; standardise definitions; show peer quartiles |
| Real Estate | Portfolio roll-forward (NOI, occupancy) | Waterfalls for NOI; heatmaps for occupancy/rent; sensitivity to cap rates |
| Industrials | Price/volume/mix and cyclicality | Bridge charts; separate cyclical vs structural; show utilisation and backlog trends |

---

### Real-World Example
Scenario: Build a valuation sanity-check dashboard: growth and margin tiles, a peer multiple pivot table, and a terminal value sensitivity to WACC and growth.

```python
# Use yoy_growth() and margin() for key tiles
# Use pivot_like() to build peer tables by sector and metric
# Use sensitivity_grid() to produce a WACC x g matrix for terminal value
```

Interpretation: If the output is extremely sensitive to terminal growth or a single period’s margin spike, tighten the forecast driver support and widen sensitivity disclosure in reporting.

See also: Chapter 5 (Python-in-Excel for repeatable analysis), Chapter 11 (financial reporting outputs).
