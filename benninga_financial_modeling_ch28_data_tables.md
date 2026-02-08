# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 28: Data Tables (scenario grids, sensitivity analysis, audit-ready model controls)

### Core Concept
Excel Data Tables are a fast, structured way to run one-variable or two-variable sensitivity analysis by recalculating a model across a grid of inputs and capturing outputs. In valuation work, Data Tables operationalise “what-if” analysis for key drivers (WACC, growth, margins, multiples), create audit-friendly evidence of model behaviour, and support decision-making without introducing manual copy/paste risk.

---

### Formula/Methodology

#### 1) One-variable sensitivity grid
Goal: Evaluate an output Y across a range of an input X.

Concept:
```text
For each x in X_grid:
    set model_input = x
    recompute model
    store output y(x)
```

Deliverable: a 2-column table (X, Y) and (optionally) a chart.

Common valuation examples:
- X = WACC, Y = Enterprise Value (EV)
- X = terminal growth g, Y = EV
- X = exit multiple, Y = EV
- X = EBITDA margin, Y = EV or equity value per share

#### 2) Two-variable sensitivity grid
Goal: Evaluate output Y across combinations of two inputs (X1, X2).

Concept:
```text
For each x1 in X1_grid:
  For each x2 in X2_grid:
      set input1 = x1
      set input2 = x2
      recompute model
      store y(x1, x2)
```

Deliverable: a matrix with X1 as rows and X2 as columns (or vice versa), typically heatmapped.

Canonical “football field” / “valuation table”:
- Rows = WACC
- Columns = terminal growth g
- Cells = EV (or equity value per share)

#### 3) Consistency checks for sensitivity tables
A sensitivity table should show “expected monotonicity” unless the model has real non-linearities:

| Output | Input change | Expected direction (typical DCF) |
|---|---|---|
| EV | WACC ↑ | EV ↓ |
| EV | g ↑ | EV ↑ (subject to g < WACC) |
| EV | Exit multiple ↑ | EV ↑ |
| Equity value/share | Net debt ↑ | ↓ |

If direction is wrong:
- sign errors (e.g., adding debt instead of subtracting)
- units mismatch (percent vs decimal)
- circularity/iteration instability
- wrong terminal year alignment

---

### Practical Application (How to apply in valuation work)

#### A) Standard “valuation bridge” sensitivity set
Minimum set that should exist in any DCF:
1) WACC vs terminal growth (2D table)
2) WACC vs exit multiple (2D table)
3) One-way tables for key operating drivers:
   - EBITDA margin
   - revenue growth
   - reinvestment rate or Capex/Revenue
   - working capital intensity

Suggested ranges (generic; tailor by sector and maturity):
| Driver | Typical grid | Notes |
|---|---|---|
| WACC | 7% to 14% (step 0.5%) | widen for small/private/high leverage |
| Terminal g | 1% to 4% (step 0.25%) | ensure g < WACC; align to inflation + real growth |
| Exit multiple | 6x to 14x (step 1x) | anchor to comps/transactions |
| EBITDA margin | -200 bps to +200 bps (step 50 bps) | use sustainable range, not one-off peak |

#### B) Model governance: keep Data Tables deterministic
- Ensure the output cell references a single “final” output (e.g., EV) with no manual overrides.
- Turn off volatile functions (RAND, NOW) or isolate them; otherwise tables are noisy.
- Fix iteration settings if circular references exist.
- Use a dedicated “Sensitivity” sheet to avoid accidental edits.

#### C) Convert Data Table logic to Python-in-Excel (for reproducibility)
If Data Tables are too slow or restricted, replicate the grid using Python loops over a function that computes valuation outputs. This also improves traceability and versioning.

---

### Python Implementation
```python
from typing import Any, Callable, Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def dcf_enterprise_value(
    fcf: List[float],
    wacc: float,
    terminal_growth: float,
    terminal_year_index: int = -1,
) -> float:
    """Simple enterprise value from forecast FCFs + perpetuity terminal value.

    Args:
        fcf: list of forecast free cash flows (must be >= 1 element)
        wacc: discount rate as decimal (e.g., 0.10 = 10%)
        terminal_growth: perpetuity growth rate as decimal (e.g., 0.03 = 3%)
        terminal_year_index: which element in fcf is terminal-year FCF (default last)

    Returns:
        float: enterprise value (same currency units as fcf)

    Raises:
        ValueError: invalid inputs (e.g., g >= wacc)
    """
    if not isinstance(fcf, (list, tuple)) or len(fcf) < 1:
        raise ValueError("fcf must be a non-empty list/tuple.")
    rr = _num(wacc, "wacc")
    g = _num(terminal_growth, "terminal_growth")
    if rr <= -0.99:
        raise ValueError("wacc is too low (must be > -99%).")
    if g >= rr:
        raise ValueError("terminal_growth must be < wacc for a stable perpetuity.")

    cashflows = np.array([_num(x, f"fcf[{i}]") for i, x in enumerate(fcf)], dtype=float)
    n = cashflows.size

    # discount factors (year 1..n)
    years = np.arange(1, n + 1, dtype=float)
    df = 1.0 / np.power(1.0 + rr, years)

    pv_forecast = float(np.sum(cashflows * df))

    # terminal value using terminal-year FCF
    tv_fcf = float(cashflows[terminal_year_index])
    tv = tv_fcf * (1.0 + g) / (rr - g)
    pv_tv = float(tv * df[terminal_year_index])

    return pv_forecast + pv_tv

def one_way_sensitivity(
    model_fn: Callable[[Dict[str, Any]], float],
    base_inputs: Dict[str, Any],
    var_name: str,
    grid: Iterable[float],
) -> pd.DataFrame:
    """Compute a one-way sensitivity table for a model function.

    Args:
        model_fn: function mapping inputs dict -> output float
        base_inputs: dict of base-case inputs
        var_name: input variable to override
        grid: iterable of values for var_name

    Returns:
        DataFrame with columns [var_name, 'output']

    Raises:
        ValueError: invalid inputs or model errors
    """
    if var_name not in base_inputs:
        raise ValueError(f"{var_name} must exist in base_inputs.")
    rows = []
    for x in grid:
        inp = dict(base_inputs)
        inp[var_name] = x
        y = float(model_fn(inp))
        if not np.isfinite(y):
            raise ValueError(f"Non-finite model output for {var_name}={x}.")
        rows.append({var_name: float(x), "output": y})
    return pd.DataFrame(rows)

def two_way_sensitivity(
    model_fn: Callable[[Dict[str, Any]], float],
    base_inputs: Dict[str, Any],
    var_row: str,
    grid_row: Iterable[float],
    var_col: str,
    grid_col: Iterable[float],
) -> pd.DataFrame:
    """Compute a two-way sensitivity matrix for a model function.

    Returns:
        DataFrame indexed by var_row values with columns as var_col values.
    """
    if var_row not in base_inputs or var_col not in base_inputs:
        raise ValueError("Both variables must exist in base_inputs.")
    grid_row = list(grid_row)
    grid_col = list(grid_col)
    mat = np.empty((len(grid_row), len(grid_col)), dtype=float)

    for i, xr in enumerate(grid_row):
        for j, xc in enumerate(grid_col):
            inp = dict(base_inputs)
            inp[var_row] = xr
            inp[var_col] = xc
            y = float(model_fn(inp))
            if not np.isfinite(y):
                raise ValueError(f"Non-finite output at {var_row}={xr}, {var_col}={xc}.")
            mat[i, j] = y

    df = pd.DataFrame(mat, index=[float(x) for x in grid_row], columns=[float(x) for x in grid_col])
    df.index.name = var_row
    return df

# Example usage: DCF EV sensitivity to WACC and terminal growth
base = {
    "fcf": [50, 60, 70, 80, 90],  # $m
    "wacc": 0.10,
    "g": 0.03,
}

def model(inputs: Dict[str, Any]) -> float:
    return dcf_enterprise_value(inputs["fcf"], inputs["wacc"], inputs["g"])

wacc_grid = np.arange(0.08, 0.13 + 1e-9, 0.005)
g_grid = np.arange(0.02, 0.04 + 1e-9, 0.0025)

two_way = two_way_sensitivity(model, base, "wacc", wacc_grid, "g", g_grid)
print(two_way.round(1))

one_way = one_way_sensitivity(model, base, "wacc", wacc_grid)
print(one_way.head())
```

---

### Valuation Impact
Why this matters:
- Sensitivity tables are often the *first* thing stakeholders look at to judge robustness. A base-case value without a driver grid is rarely decision-ready.
- They reveal whether value is driven primarily by operating performance (FCF) or assumptions that are hard to defend (terminal value inputs).

Impact on multiples:
- If DCF is highly sensitive to exit multiple, it indicates the valuation is effectively a multiples valuation in disguise; disclose and anchor to comps.
- High sensitivity to WACC suggests capital structure/risk assumptions dominate; ensure WACC is defensible.

Impact on DCF inputs:
- WACC/g grids test the “g < WACC” constraint and show how close the model is to instability.
- Sensitivity to margins and reinvestment tests whether ROIC assumptions are realistic.

Comparability issues across companies:
- Firms with different leverage, cyclicality, or accounting (leases, revenue recognition) may have structurally different sensitivities; compare like-for-like.

Practical adjustments:
```python
def equity_value_per_share(ev: float, net_debt: float, shares: float) -> float:
    """Convert enterprise value to equity value per share."""
    evv = _num(ev, "ev")
    nd = _num(net_debt, "net_debt")
    sh = _num(shares, "shares")
    if sh <= 0:
        raise ValueError("shares must be > 0.")
    return float((evv - nd) / sh)
```

---

### Quality of Earnings Flags
⚠️ Sensitivity table looks “too smooth” because it references the wrong output cell (e.g., a hard-coded value rather than the model result).  
⚠️ Outputs violate basic monotonicity (EV rises when WACC rises) indicating sign/unit errors.  
⚠️ Tables exclude the key valuation drivers (terminal assumptions) or use unreasonably narrow ranges to hide fragility.  
✅ Clear, well-ranged driver grids; monotonicity checks pass; sensitivity ranges aligned to peer evidence.

---

### Sector-Specific Considerations

| Sector | Key sensitivity driver | Typical table focus |
|---|---|---|
| SaaS / tech | margin + reinvestment / sales efficiency | margin vs growth; WACC vs g |
| Infrastructure | discount rate + terminal assumptions | WACC vs g; inflation vs g |
| Cyclicals | mid-cycle margin/volume | margin vs volume; WACC vs exit multiple |
| Financials | ROE / cost of equity | COE vs growth; P/B implied |

---

### Real-World Example
Scenario: Build a 2D valuation table (WACC × terminal growth) for a simple DCF and convert to equity value per share.

```python
base = {"fcf": [50, 60, 70, 80, 90], "wacc": 0.10, "g": 0.03}
net_debt = 120.0   # $m
shares = 80.0      # m shares

def model_ev(inp):
    return dcf_enterprise_value(inp["fcf"], inp["wacc"], inp["g"])

wacc_grid = np.arange(0.08, 0.13 + 1e-9, 0.005)
g_grid = np.arange(0.02, 0.04 + 1e-9, 0.0025)

ev_table = two_way_sensitivity(model_ev, base, "wacc", wacc_grid, "g", g_grid)

# Convert EV table to value per share
vps_table = ev_table.applymap(lambda ev: equity_value_per_share(ev, net_debt, shares))
print(vps_table.round(2))
```

Interpretation: The table quantifies how much valuation changes with plausible assumptions and highlights whether the base case is stable or depends on aggressive terminal inputs.

See also: Chapter 4 (DCF valuation), Chapter 3 (WACC), Chapter 14 (event studies as scenario impacts), Chapter 24 (Monte Carlo scenario grids).
