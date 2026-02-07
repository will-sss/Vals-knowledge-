# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 11: Financial Reporting (Model Outputs, Management Packs, Reconciliations, Controls)

### Core Concept
Chapter 11 focuses on using Excel (and Python/VBA where helpful) to produce financial reports: management packs, variance analysis, KPI dashboards, and valuation exhibits (forecast tables, multiple comps, sensitivity matrices). In valuation work, reporting is not “presentation only”—it is a control mechanism that forces reconciliations (to audited accounts, to model totals), makes adjustments explicit (QoE), and standardizes how conclusions are supported.

See also: Chapter 8 (automation), Chapter 9 (VBA orchestration), Chapter 12 (external data), McKinsey chapters on forecasting and multiples.

---

### Formula/Methodology

#### 1) Core reporting outputs used in valuation
```text
Outputs typically include:
- Historical performance summary (Revenue, EBITDA, EBIT, FCF)
- Bridge analysis (Revenue bridge, margin bridge, working capital bridge)
- Reconciliations (Reported → Normalized)
- Comparable multiples (median/IQR + selected range)
- DCF summary (PV, TV, implied multiples, sensitivities)
```

#### 2) Reconciliation frameworks (critical for QoE and comparability)

Reported to Normalized EBITDA:
```text
EBITDA_normalized = EBITDA_reported + Addbacks - Deductions
Where:
Addbacks = one-offs, non-recurring costs, normalization items (supported)
Deductions = non-recurring gains or items inflating EBITDA
```

Reported to Free Cash Flow:
```text
FCF = NOPAT + D&A - Capex - ΔNWC
Where:
NOPAT = EBIT × (1 - TaxRate)
ΔNWC = change in net working capital
```

Enterprise Value to Equity Value (reporting bridge):
```text
Equity Value = Enterprise Value - Net Debt - Preferred Equity - Minority Interest + Non-operating Assets
Where:
Net Debt = Total Debt - Cash
```

#### 3) Variance analysis (Actual vs Budget/Forecast)
```text
Variance = Actual - Plan
Variance % = (Actual - Plan) / Plan
Where:
Plan must be non-zero for % variance
```

---

### Python Implementation
```python
from typing import Dict, Any, Sequence, Optional, Union
import numpy as np
import pandas as pd

Number = Union[int, float]

def require_finite(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def safe_div(numer: Any, denom: Any, name: str) -> float:
    n = require_finite(numer, f"{name}_numer")
    d = require_finite(denom, f"{name}_denom")
    if abs(d) < 1e-12:
        raise ValueError(f"{name}: denominator must be non-zero.")
    return n / d

def normalize_ebitda(ebitda_reported: Number, addbacks: Number = 0.0, deductions: Number = 0.0) -> float:
    """
    Normalize EBITDA for QoE.

    Returns:
        float: normalized EBITDA
    """
    e = require_finite(ebitda_reported, "ebitda_reported")
    a = require_finite(addbacks, "addbacks")
    d = require_finite(deductions, "deductions")
    return e + a - d

def fcf_from_drivers(
    ebit: Number,
    tax_rate: Number,
    da: Number,
    capex: Number,
    delta_nwc: Number
) -> float:
    """
    Compute free cash flow from key drivers.

    FCF = EBIT*(1-T) + D&A - Capex - ΔNWC

    Raises:
        ValueError: invalid tax rate.
    """
    E = require_finite(ebit, "ebit")
    t = require_finite(tax_rate, "tax_rate")
    if not (0.0 <= t <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    DA = require_finite(da, "da")
    C = require_finite(capex, "capex")
    dNWC = require_finite(delta_nwc, "delta_nwc")
    nopat = E * (1.0 - t)
    return nopat + DA - C - dNWC

def ev_to_equity_value(
    enterprise_value: Number,
    total_debt: Number,
    cash: Number,
    preferred_equity: Number = 0.0,
    minority_interest: Number = 0.0,
    non_operating_assets: Number = 0.0
) -> float:
    """
    Bridge from enterprise value to equity value.

    Equity = EV - (Debt - Cash) - Pref - MI + Non-operating assets

    Raises:
        ValueError: non-finite inputs.
    """
    ev = require_finite(enterprise_value, "enterprise_value")
    debt = require_finite(total_debt, "total_debt")
    c = require_finite(cash, "cash")
    pe = require_finite(preferred_equity, "preferred_equity")
    mi = require_finite(minority_interest, "minority_interest")
    noa = require_finite(non_operating_assets, "non_operating_assets")

    net_debt = debt - c
    return ev - net_debt - pe - mi + noa

def variance_table(actual: pd.DataFrame, plan: pd.DataFrame, key_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    """
    Create a variance table (Actual vs Plan) with absolute and % variance.

    Args:
        actual: DataFrame with actuals
        plan: DataFrame with plan/forecast
        key_cols: merge keys (e.g., ['period'])
        value_cols: numeric columns to compare (e.g., ['revenue','ebitda'])

    Returns:
        DataFrame with columns: *_actual, *_plan, *_var, *_var_pct

    Raises:
        ValueError: missing columns or duplicates.
    """
    if actual is None or plan is None or actual.empty or plan.empty:
        raise ValueError("actual and plan must be non-empty DataFrames.")
    for c in list(key_cols) + list(value_cols):
        if c not in actual.columns:
            raise ValueError(f"actual missing column: {c}")
        if c not in plan.columns:
            raise ValueError(f"plan missing column: {c}")

    a = actual.copy()
    p = plan.copy()
    for c in value_cols:
        a[c] = pd.to_numeric(a[c], errors="raise")
        p[c] = pd.to_numeric(p[c], errors="raise")

    out = pd.merge(a, p, on=list(key_cols), how="inner", suffixes=("_actual", "_plan"))
    for c in value_cols:
        out[f"{c}_var"] = out[f"{c}_actual"] - out[f"{c}_plan"]
        # % variance: handle zero plan
        denom = out[f"{c}_plan"].astype(float)
        out[f"{c}_var_pct"] = np.where(np.abs(denom) < 1e-12, np.nan, out[f"{c}_var"] / denom)
    return out

# Example usage: QoE reconciliation + reporting outputs
reported_ebitda = 420_000_000
addbacks = 35_000_000
deductions = 10_000_000
ebitda_norm = normalize_ebitda(reported_ebitda, addbacks, deductions)
print(f"EBITDA normalized: ${ebitda_norm/1e6:.0f}M")

fcf = fcf_from_drivers(ebit=250_000_000, tax_rate=0.25, da=90_000_000, capex=110_000_000, delta_nwc=20_000_000)
print(f"FCF: ${fcf/1e6:.0f}M")

equity = ev_to_equity_value(enterprise_value=5_200_000_000, total_debt=1_600_000_000, cash=300_000_000, minority_interest=50_000_000)
print(f"Equity value: ${equity/1e9:.2f}B")

actual = pd.DataFrame({"period": ["2025A", "2026E"], "revenue": [1_000_000_000, 1_080_000_000], "ebitda": [420_000_000, 450_000_000]})
plan   = pd.DataFrame({"period": ["2025A", "2026E"], "revenue": [980_000_000, 1_050_000_000], "ebitda": [410_000_000, 440_000_000]})
vt = variance_table(actual, plan, key_cols=["period"], value_cols=["revenue", "ebitda"])
print(vt)
```

---

### Valuation Impact
Why this matters:
- **Defensibility:** Reporting forces explicit bridges (reported → normalized → valuation), reducing ambiguity and making assumptions reviewable.
- **Comparability:** Standard outputs (QoE reconciliation tables, peer stats, sensitivity matrices) reduce the risk of inconsistent treatment across companies.
- **Speed and control:** Once reporting templates exist, updating for new actuals or scenarios becomes a controlled refresh rather than a manual rebuild.

Impact on multiples:
- Normalized EBITDA and consistent EV bridges improve EV/EBITDA comparability.
- Reporting peer medians/IQR and the selected multiple range is essential for auditability.

Impact on DCF inputs:
- Reporting historical bridges (margin, working capital, capex) improves forecast rationale.
- Sensitivity outputs (WACC × g, exit multiple × margin) make value drivers explicit.

Practical adjustments:
```python
def build_qoe_bridge(reported: float, adjustments: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple QoE bridge table.

    adjustments columns required: ['item','amount','type'] where type in {'addback','deduction'}
    """
    if adjustments is None or adjustments.empty:
        raise ValueError("adjustments must be non-empty.")
    for c in ["item", "amount", "type"]:
        if c not in adjustments.columns:
            raise ValueError(f"adjustments missing column: {c}")
    adj = adjustments.copy()
    adj["amount"] = pd.to_numeric(adj["amount"], errors="raise")
    if not set(adj["type"].unique()).issubset({"addback", "deduction"}):
        raise ValueError("type must be 'addback' or 'deduction'.")
    addbacks = float(adj.loc[adj["type"] == "addback", "amount"].sum())
    deductions = float(adj.loc[adj["type"] == "deduction", "amount"].sum())
    normalized = float(reported) + addbacks - deductions
    summary = pd.DataFrame([{
        "ebitda_reported": float(reported),
        "addbacks": addbacks,
        "deductions": deductions,
        "ebitda_normalized": normalized
    }])
    return summary
```

---

### Quality of Earnings Flags
⚠️ Red flag: large add-backs without evidence or repeated “one-offs” every year (recurring one-offs).  
⚠️ Red flag: deductions omitted for non-recurring gains inflating EBITDA (e.g., asset sales).  
⚠️ Red flag: weak reconciliation between model outputs and audited accounts (no tie-outs).  
✅ Green flag: clear QoE bridge with itemized adjustments, consistent definitions, and cross-checks to source statements.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | KPI reporting + ARR/NRR | Include KPI definitions and tie ARR movements to revenue; show cohort or bridge tables |
| Banking | Regulatory metrics | Report ROE, CET1, NIM with consistent denominators; avoid mixing accounting bases |
| Real Estate | NAV and cap rates | Report NOI bridges, cap-rate sensitivities, and property roll-ups with reconciliation |
| Industrials | Segment reporting | Present segment margins and mix bridges; reconcile segment totals to consolidated numbers |

---

### Real-World Example
Scenario: Prepare a valuation pack page that shows (i) QoE EBITDA bridge, (ii) peer EV/EBITDA distribution, and (iii) EV→Equity bridge.

```python
# 1) QoE bridge: build_qoe_bridge()
# 2) Peer stats: robust median/IQR from Chapter 6/7 patterns
# 3) EV to equity: ev_to_equity_value()
```

Interpretation: A consistent reporting layer makes the valuation reproducible and reviewable, reducing turnaround time and improving credibility.

See also: Chapter 8 (automation), Chapter 9 (orchestration), Chapter 13 (templates/add-ins).
