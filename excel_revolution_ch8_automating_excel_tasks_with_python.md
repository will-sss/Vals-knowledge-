# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 8: Automating Excel Tasks with Python (Reusable Pipelines, Scenario Runs, Validation, Reporting Outputs)

### Core Concept
Chapter 8 is about using Python to automate repeatable Excel workflows: pulling inputs from consistent ranges/tables, applying standardized transformations and calculations, and writing back clean outputs (tables, KPIs, sensitivity grids). For valuation work, automation reduces model risk by enforcing consistent assumptions, normalization rules (QoE), and comparable selection logic across cases—while producing outputs that remain reviewable in Excel.

See also: Chapter 7 (pandas transforms), Chapter 9 (VBA/macros), Chapter 11 (reporting), McKinsey Chapters on forecasting and multiples.

---

### Formula/Methodology

#### 1) Automation pattern: Inputs → Transform → Validate → Output
```text
Pipeline:
1) Read inputs (assumptions, actuals, comps)
2) Standardize units and definitions (e.g., $m vs $, fiscal year alignment)
3) Compute metrics (EV, multiples, ROIC drivers, forecast KPIs)
4) Validate (non-zero denominators, bounds, reconciliation checks)
5) Output (tables + dashboard tiles + sensitivity matrices)
```

#### 2) Scenario and sensitivity logic
```text
Scenario Output = f(BaseInputs, ScenarioOverrides)
Where:
ScenarioOverrides adjust assumptions (growth, margin, WACC, terminal g, exit multiple)
```

```text
Probability-Weighted Value = Σ (p_s × V_s)
Where:
p_s = probability of scenario s (sum to 1)
V_s = valuation output under scenario s
```

#### 3) Data validation gates (must pass before outputs used)
```text
Gate examples:
- Sum(probabilities) = 1.0
- WACC > terminal growth (perpetuity)
- EBITDA normalized > 0 (for EV/EBITDA usage)
- Balance sheet ties (Assets = Liabilities + Equity) if reorged
```

---

### Python Implementation
```python
from dataclasses import dataclass
from typing import Dict, Any, List, Sequence, Tuple, Union, Callable, Optional
import numpy as np
import pandas as pd

Number = Union[int, float]

def is_finite(x: Any) -> bool:
    try:
        v = float(x)
        return np.isfinite(v)
    except Exception:
        return False

def require_finite(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def require_positive(x: Any, name: str) -> float:
    v = require_finite(x, name)
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")
    return v

@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    overrides: Dict[str, float]

def validate_probabilities(scenarios: Sequence[Scenario], tol: float = 1e-6) -> None:
    if scenarios is None or len(scenarios) == 0:
        raise ValueError("scenarios must be non-empty.")
    ps = [require_finite(s.probability, f"probability[{s.name}]") for s in scenarios]
    if any(p < 0 for p in ps):
        raise ValueError("Scenario probabilities must be >= 0.")
    if abs(sum(ps) - 1.0) > tol:
        raise ValueError("Scenario probabilities must sum to 1.0 (within tolerance).")

def merge_inputs(base_inputs: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(base_inputs, dict) or len(base_inputs) == 0:
        raise ValueError("base_inputs must be a non-empty dict.")
    out = dict(base_inputs)
    for k, v in (overrides or {}).items():
        out[k] = v
    return out

def perpetuity_terminal_value(fcf1: float, wacc: float, g: float) -> float:
    f = require_finite(fcf1, "fcf1")
    r = require_finite(wacc, "wacc")
    gg = require_finite(g, "g")
    if r <= gg:
        raise ValueError("wacc must be > g for perpetuity terminal value.")
    return f / (r - gg)

def dcf_pv(fcf: Sequence[Number], wacc: float, tv: float) -> float:
    r = require_finite(wacc, "wacc")
    if r <= -1.0:
        raise ValueError("wacc must be > -1.0.")
    if fcf is None or len(fcf) == 0:
        raise ValueError("fcf must be non-empty.")
    pv = 0.0
    for t, cf in enumerate(fcf, start=1):
        c = require_finite(cf, f"fcf_t{t}")
        pv += c / ((1.0 + r) ** t)
    T = len(fcf)
    pv += require_finite(tv, "tv") / ((1.0 + r) ** T)
    return pv

def valuation_model(inputs: Dict[str, Any]) -> float:
    """
    Example 'model kernel' suitable for scenario automation.
    Uses: forecast FCF series + terminal value from perpetuity.

    Required inputs:
        fcf: list of forecast FCFs (t=1..T)
        wacc: discount rate
        g: terminal growth rate
        fcf1_terminal: FCF in year T+1 (or derived)
    """
    fcf = inputs.get("fcf")
    wacc = require_finite(inputs.get("wacc"), "wacc")
    g = require_finite(inputs.get("g"), "g")
    fcf1_term = require_finite(inputs.get("fcf1_terminal"), "fcf1_terminal")

    tv = perpetuity_terminal_value(fcf1_term, wacc, g)
    return dcf_pv(fcf, wacc, tv)

def run_scenarios(base_inputs: Dict[str, Any], scenarios: Sequence[Scenario]) -> pd.DataFrame:
    """
    Run valuation for multiple scenarios and return an auditable table.

    Returns:
        DataFrame: scenario, probability, value, weighted_value
    """
    validate_probabilities(scenarios)
    rows = []
    for s in scenarios:
        inp = merge_inputs(base_inputs, s.overrides)
        v = float(valuation_model(inp))
        rows.append({
            "scenario": s.name,
            "probability": float(s.probability),
            "value": v,
            "weighted_value": float(s.probability) * v
        })
    df = pd.DataFrame(rows)
    df.loc["TOTAL", "probability"] = df["probability"].sum()
    df.loc["TOTAL", "weighted_value"] = df["weighted_value"].sum()
    return df

# Example usage: automated scenario valuation
base = {
    "fcf": [120_000_000, 135_000_000, 150_000_000, 165_000_000, 180_000_000],
    "wacc": 0.10,
    "g": 0.03,
    "fcf1_terminal": 186_000_000  # year 6 FCF (or derived)
}

scenarios = [
    Scenario("Base", 0.60, {}),
    Scenario("Upside", 0.20, {"wacc": 0.095, "g": 0.035, "fcf1_terminal": 200_000_000}),
    Scenario("Downside", 0.20, {"wacc": 0.11, "g": 0.02, "fcf1_terminal": 165_000_000}),
]

df = run_scenarios(base, scenarios)
print(df)
print(f"Probability-weighted PV: ${df.loc['TOTAL','weighted_value']/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- **Consistency across cases:** automated pipelines ensure that the same definitions, filters, and adjustments are applied across multiple valuations, improving internal comparability and review efficiency.
- **Reduced spreadsheet risk:** automation decreases manual copy/paste and reduces hidden formula drift.
- **Scenario credibility:** probability-weighted outputs are easier to maintain and audit when scenario assumptions and results are generated from a consistent model kernel.

Impact on multiples:
- Automates peer cleaning + statistic computation (median/IQR) across multiple screens (industry, geography, size).
- Encourages consistent outlier handling and denominator rules.

Impact on DCF inputs:
- Makes sensitivities and scenarios first-class outputs rather than ad hoc tables.
- Enforces hard constraints (WACC > g) and reduces terminal value errors.

Practical adjustments:
```python
def probability_weighted_value(values: Sequence[float], probs: Sequence[float], tol: float = 1e-6) -> float:
    v = np.asarray(list(values), dtype=float)
    p = np.asarray(list(probs), dtype=float)
    if v.size == 0 or p.size == 0 or v.size != p.size:
        raise ValueError("values and probs must be same non-zero length.")
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(p)):
        raise ValueError("values and probs must be finite.")
    if np.any(p < 0):
        raise ValueError("probabilities must be >= 0.")
    if abs(float(p.sum()) - 1.0) > tol:
        raise ValueError("probabilities must sum to 1.0.")
    return float((v * p).sum())
```

---

### Quality of Earnings Flags
⚠️ **Red flag: automation hides judgment calls** (e.g., auto add-backs) without an audit table of adjustments.  
⚠️ **Red flag: scenario probabilities not justified** (or not summing to 1.0).  
⚠️ **Red flag: terminal value dominates** because automation defaults to aggressive g or low WACC.  
✅ **Green flag: audit outputs** (before/after adjustments, filters applied, constraint checks) generated automatically alongside results.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | Rapid KPI changes, high dispersion | Automate KPI cleaning and scenario assumptions; keep outputs compact and include outlier rules |
| Banking | Regulatory constraints and ratio definitions | Automate ratio calculation with strict denominator validation; separate business lines |
| Real Estate | Asset-level cash flows and cap rates | Automate roll-ups and sensitivities on cap rates/NOI; include data completeness checks |
| Industrials | Cyclical regimes | Automate cycle-phase tagging and scenario sets (mid-cycle vs peak/trough) |

---

### Real-World Example
Scenario: A team runs three valuation scenarios weekly (base/upside/downside). Automation pulls the latest actuals and comps table, refreshes peer stats, recomputes DCF outputs, and regenerates sensitivity matrices for review.

```python
# Use run_scenarios(base_inputs, scenarios) as the kernel.
# Extend pipeline:
# - clean and standardize comps (Chapter 7)
# - compute peer ranges
# - compute DCF and probability-weighted PV
# - return compact tables to Excel for dashboards
```

Interpretation: Automating these steps reduces drift between versions and makes the model easier to review because the scenario assumptions and results are generated from a single, validated workflow.

See also: Chapter 9 (VBA macros for UI automation), Chapter 11 (reporting outputs), Chapter 12 (external data).
