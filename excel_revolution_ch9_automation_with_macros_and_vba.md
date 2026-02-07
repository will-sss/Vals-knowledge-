# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 9: Automation with Macros and VBA (Model Controls, Repeatable Runs, Audit-Safe UX)

### Core Concept
Chapter 9 focuses on using Excel macros and VBA to automate workbook tasks (refreshes, formatting, navigation, report compilation) and to add controlled UI elements (buttons, drop-downs, scenario switches). In valuation work, VBA is most valuable for workflow automation and model controls (run base/upside/downside, refresh comps, rebuild outputs), while keeping calculations and key logic visible in Excel/Python for review.

See also: Chapter 8 (Python automation pipelines), Chapter 11 (financial reporting), Chapter 12 (external data), McKinsey chapters on forecasting and multiples.

---

### Formula/Methodology

#### 1) Automation design pattern for valuation workbooks
```text
VBA = Orchestration layer
Excel formulas / Python = Calculation layer
Outputs = Tables, charts, PDFs, sensitivity grids

Rule: VBA should not hide valuation logic; it should trigger runs, validate inputs, and package outputs.
```

#### 2) Scenario selection logic (common control pattern)
```text
SelectedScenario = {Base, Upside, Downside}
Assumptions = BaseAssumptions + ScenarioOverrides(SelectedScenario)
Outputs = Valuation(Assumptions)
```

#### 3) Audit-safe guardrails
```text
Guardrails:
- Input completeness check (no blanks in required assumptions)
- Denominator checks (no division by zero for margins/multiples)
- Timestamping and version tagging of runs
- Locking calculation sheets (protect) while leaving inputs editable
```

---

### Python Implementation
Even if orchestration is via VBA, keep the core model kernel in Python (or Excel formulas) and keep VBA minimal. Below is a Python kernel suitable for one-click scenario runs, with explicit validation and compact outputs.

```python
from typing import Dict, Any, Sequence
import numpy as np
import pandas as pd

def require_finite(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def perpetuity_tv(fcf1: float, wacc: float, g: float) -> float:
    f = require_finite(fcf1, "fcf1")
    r = require_finite(wacc, "wacc")
    gg = require_finite(g, "g")
    if r <= gg:
        raise ValueError("wacc must be > g for perpetuity terminal value.")
    return f / (r - gg)

def dcf_pv(fcf: Sequence[float], wacc: float, tv: float) -> float:
    r = require_finite(wacc, "wacc")
    if r <= -1.0:
        raise ValueError("wacc must be > -1.0.")
    if fcf is None or len(fcf) == 0:
        raise ValueError("fcf must be non-empty.")
    pv = 0.0
    for t, cf in enumerate(fcf, start=1):
        c = require_finite(cf, f"fcf_t{t}")
        pv += c / ((1 + r) ** t)
    pv += require_finite(tv, "tv") / ((1 + r) ** len(fcf))
    return pv

def run_valuation(inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns a compact 1-row table suitable for Excel output.

    Required keys:
        fcf: list of forecast FCFs (t=1..T)
        wacc: discount rate
        g: terminal growth
        fcf1_terminal: FCF in year T+1

    Output columns:
        pv, terminal_value, wacc, g, fcf1_terminal
    """
    fcf = inputs.get("fcf")
    wacc = require_finite(inputs.get("wacc"), "wacc")
    g = require_finite(inputs.get("g"), "g")
    fcf1_terminal = require_finite(inputs.get("fcf1_terminal"), "fcf1_terminal")
    tv = perpetuity_tv(fcf1_terminal, wacc, g)
    pv = dcf_pv(fcf, wacc, tv)
    return pd.DataFrame([{
        "pv": float(pv),
        "terminal_value": float(tv),
        "wacc": float(wacc),
        "g": float(g),
        "fcf1_terminal": float(fcf1_terminal)
    }])

# Example usage
inputs = {
    "fcf": [120_000_000, 135_000_000, 150_000_000, 165_000_000, 180_000_000],
    "wacc": 0.10,
    "g": 0.03,
    "fcf1_terminal": 186_000_000
}
out = run_valuation(inputs)
print(out)
print(f"PV: ${out.loc[0,'pv']/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- Repeatable runs: Macro buttons can standardize refresh → validate → calculate → export so outputs are consistent across users and versions.
- Reduced friction: Automates routine tasks (formatting, resetting scenarios, compiling exhibits), freeing analyst time for judgment and review.
- Governance: VBA can enforce process controls (input checks, run logs, protection) without burying valuation logic.

Impact on multiples:
- Macros can refresh comps tables, re-run filters, and rebuild peer distribution summaries on demand.
- Helps keep as-of timing consistent across all multiples and KPIs.

Impact on DCF inputs:
- Scenario switches and sensitivity rebuilds become one-click operations, reducing manual errors.
- Run logs create traceability of which assumptions produced which outputs.

Practical adjustments:
```python
import pandas as pd

def stamp_run_metadata(df: pd.DataFrame, model_version: str, run_id: str) -> pd.DataFrame:
    """Attach metadata fields to outputs for audit trail."""
    out = df.copy()
    out["model_version"] = str(model_version)
    out["run_id"] = str(run_id)
    return out
```

---

### Quality of Earnings Flags
⚠️ Red flag: VBA changes valuation logic invisibly (e.g., rewriting formulas or editing assumptions behind the scenes).  
⚠️ Red flag: macros that suppress errors and proceed with exports when validation fails.  
⚠️ Red flag: lack of run logging (no timestamp, no scenario/assumption record).  
✅ Green flag: orchestration only—VBA triggers calculations, validates inputs, and packages outputs; core logic stays in Excel/Python and is reviewable.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | Frequent scenario updates | Macro-driven scenario buttons and refresh of KPI tables; always log scenario + as-of date |
| Banking | Many linked sheets and controls | VBA navigation and protection controls; strict validation of denominators for ratios |
| Real Estate | Exhibit compilation | Macros to rebuild property rollups and export charts/tables per asset class |
| Industrials | Multi-entity or BU models | Macro routines to iterate business units and compile segment outputs |

---

### Real-World Example
Scenario: A valuation workbook has buttons: Refresh Data, Run Base, Run Upside, Run Downside, Export Pack. VBA orchestrates the workflow while Python/Excel perform the calculations.

```text
Button: Run Base
1) Validate required assumption cells are filled
2) Set scenario selector to "Base"
3) Trigger recalculation / Python cell refresh
4) Copy outputs into a locked "Results" sheet
5) Stamp timestamp + scenario + model version
```

Interpretation: The analyst reduces the risk of inconsistent manual runs and creates an audit trail for review and client queries.

See also: Chapter 8 (Python automation pipelines), Chapter 11 (financial reporting outputs), Chapter 13 (templates/add-ins).
