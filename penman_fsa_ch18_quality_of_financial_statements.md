# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 18: Analysis of the Quality of Financial Statements (Earnings Quality, Accruals, Red Flags)

### Core Concept
Financial statement “quality” is about whether reported numbers are **reliable, sustainable, and valuation-relevant**. The practical goal is to separate **core, repeatable operating performance** from accounting outcomes driven by one-offs, estimation bias, classification games, or timing shifts that inflate earnings and/or understate invested capital.

---

### Formula/Methodology

#### 1) Clean surplus vs “dirty surplus” signals
```text
Ending BV_t = Beginning BV_{t-1} + Comprehensive Earnings_t - Net Distributions_t
```
Where:
- BV = common equity book value
- Comprehensive earnings includes items routed through OCI that still affect equity
- Net distributions = dividends + net share repurchases (if treated as equity transactions)

If this does not reconcile, inspect:
- OCI items, equity issuance/repurchase accounting, FX translation, pension remeasurements, hedging reserves.

#### 2) Total accruals (earnings vs cash diagnostics)
Common implementation:
```text
Total Accruals_t = Net Income_t - Cash Flow from Operations_t
Accrual Ratio_t = Total Accruals_t / Average Total Assets_t
```
Alternative operating focus:
```text
Operating Accruals_t ≈ ΔWorking Capital_t + Non-cash Expenses_t - Non-cash Revenue_t
```
Where:
- Working capital is operating (exclude cash, debt, and financing payables if reclassified)

Interpretation:
- Higher accrual ratios can indicate lower earnings quality (more estimate-driven earnings).
- Look for persistent accrual build rather than one period spikes.

#### 3) Cash conversion / earnings backed by cash
```text
Cash Conversion_t = CFO_t / Net Income_t
```
Interpretation:
- Persistently < 1.0 can flag aggressive revenue recognition, capitalisation, or working capital stress.
- Persistently > 1.0 can occur in downturns (working capital release) and is not automatically “good”.

#### 4) Working capital intensity and revenue timing risk
```text
DSO = (Accounts Receivable / Revenue) × 365
DIO = (Inventory / COGS) × 365
DPO = (Accounts Payable / COGS) × 365
Cash Conversion Cycle = DSO + DIO - DPO
```

#### 5) Core vs transitory earnings (“classification” quality)
Define an adjusted core earnings measure:
```text
Core Earnings = Reported Earnings - After-tax One-offs - Accounting Policy Effects (net)
```
Where one-offs include:
- restructuring charges, litigation settlements, disposal gains/losses, impairment charges (analyse carefully), fair value remeasurement gains, unusual tax items.

---

### Practical Application (How to apply)

#### A) Quality-of-earnings workflow for valuation
1) **Reclassify** operating vs financing consistently (avoid mixing interest-bearing items into operations).  
2) **Remove transitory items** and isolate core operating profit.  
3) **Check cash support** (CFO vs earnings; accrual ratios; working capital movements).  
4) **Test capitalisation choices** (R&D, software dev, customer acquisition costs, leases).  
5) **Inspect “cookie jar” reserves** (provisions and reversals).  
6) **Assess persistence**: which components are repeatable and which revert.

#### B) Key “games” that affect valuation outputs
| Tactic | What changes | Typical valuation distortion |
|---|---|---|
| Aggressive revenue timing | Revenue, receivables, deferred revenue | Inflates growth and margins; misleads DCF and multiples |
| Capitalising costs | EBITDA/EBIT up, assets up (later amortisation) | Overstates near-term profitability; understates maintenance reinvestment |
| Under-provisioning | Expenses down, liabilities down | Inflates earnings and ROIC; future drag when catch-up occurs |
| “Below-the-line” classification | Moves expenses out of operating profit | Inflates EBITDA multiples and peer comparison |
| One-off gains | Earnings up (non-recurring) | Inflates P/E and residual income; misprices sustainability |

#### C) Adjustments that matter most in valuation models
- **Normalize operating profit (NOPAT/OPAT)**: remove unusual items and policy effects.
- **Normalize operating capital (NOA/IC)**: correct for off-balance-sheet obligations (leases), capitalised costs, and misclassified liabilities.
- **Rebuild sustainable margins and reinvestment**: align “core earnings” with required reinvestment (capex, working capital, maintenance R&D).

---

### Python Implementation
```python
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def total_accruals(net_income: float, cfo: float) -> float:
    """Total accruals = Net income - CFO."""
    ni = _f(net_income, "net_income")
    cfo = _f(cfo, "cfo")
    return float(ni - cfo)

def accrual_ratio(net_income: float, cfo: float, avg_total_assets: float) -> float:
    """Accrual ratio = (NI - CFO) / Avg total assets."""
    ta = _f(avg_total_assets, "avg_total_assets")
    if ta <= 0:
        raise ValueError("avg_total_assets must be > 0.")
    return float(total_accruals(net_income, cfo) / ta)

def cash_conversion(cfo: float, net_income: float) -> float:
    """Cash conversion = CFO / Net income."""
    cfo = _f(cfo, "cfo")
    ni = _f(net_income, "net_income")
    if abs(ni) < 1e-12:
        raise ValueError("net_income is ~0; cash conversion is not meaningful.")
    return float(cfo / ni)

def days_outstanding(balance: float, flow: float, days: int = 365) -> float:
    """Generic days metric: (Balance / Flow) * days."""
    bal = _f(balance, "balance")
    flw = _f(flow, "flow")
    if flw <= 0:
        raise ValueError("flow must be > 0 for a days metric.")
    return float((bal / flw) * days)

def cash_conversion_cycle(ar: float, revenue: float, inventory: float, cogs: float, ap: float) -> Dict[str, float]:
    """Compute DSO, DIO, DPO and cash conversion cycle."""
    dso = days_outstanding(ar, revenue)
    dio = days_outstanding(inventory, cogs)
    dpo = days_outstanding(ap, cogs)
    ccc = dso + dio - dpo
    return {"dso": dso, "dio": dio, "dpo": dpo, "ccc": ccc}

def core_earnings(reported_earnings: float, one_offs_pre_tax: float = 0.0, tax_rate: float = 0.25) -> float:
    """Core earnings = reported - after-tax one-offs (positive one_offs reduce core)."""
    rep = _f(reported_earnings, "reported_earnings")
    oo = _f(one_offs_pre_tax, "one_offs_pre_tax")
    tr = _f(tax_rate, "tax_rate")
    if tr < 0 or tr > 0.6:
        raise ValueError("tax_rate must be a decimal between 0 and 0.6.")
    after_tax_oo = oo * (1.0 - tr)
    return float(rep - after_tax_oo)

def earnings_quality_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact earnings-quality panel from a time series DataFrame.

    Required columns: net_income, cfo, total_assets
    Optional columns: revenue, ar, inventory, cogs, ap

    Returns: DataFrame with accruals, accrual ratio, cash conversion, and CCC if inputs exist.
    """
    required = {"net_income", "cfo", "total_assets"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index).copy()
    out["accruals"] = df["net_income"] - df["cfo"]
    out["avg_assets"] = (df["total_assets"] + df["total_assets"].shift(1)) / 2.0
    out["accrual_ratio"] = out["accruals"] / out["avg_assets"]
    out["cash_conversion"] = df["cfo"] / df["net_income"].replace({0: np.nan})

    # CCC if available
    ccc_cols = {"revenue", "ar", "inventory", "cogs", "ap"}
    if ccc_cols.issubset(df.columns):
        out["dso"] = (df["ar"] / df["revenue"]) * 365.0
        out["dio"] = (df["inventory"] / df["cogs"]) * 365.0
        out["dpo"] = (df["ap"] / df["cogs"]) * 365.0
        out["ccc"] = out["dso"] + out["dio"] - out["dpo"]

    return out

# Example usage
example = {
    "net_income": 120.0,
    "cfo": 70.0,
    "avg_total_assets": 1_000.0,
    "reported_earnings": 120.0,
    "one_offs_pre_tax": 40.0,
    "tax_rate": 0.25
}

print(f"Total accruals: ${total_accruals(example['net_income'], example['cfo']):.1f}m")
print(f"Accrual ratio: {accrual_ratio(example['net_income'], example['cfo'], example['avg_total_assets']):.2%}")
print(f"Cash conversion: {cash_conversion(example['cfo'], example['net_income']):.2f}x")
print(f"Core earnings: ${core_earnings(example['reported_earnings'], example['one_offs_pre_tax'], example['tax_rate']):.1f}m")
```

---

### Valuation Impact
Why this matters:
- Quality adjustments prevent overvaluing firms with inflated earnings (transitory items, aggressive accruals) and undervaluing firms with conservative accounting.
- Improves forecasting accuracy by anchoring projections on **sustainable core earnings** and realistic reinvestment needs.

Impact on multiples:
- EV/EBITDA: classification and capitalisation can inflate EBITDA; adjust comparables for consistent definitions.
- P/E: one-offs and tax anomalies distort earnings; use core earnings to compute “core P/E”.
- P/B: book value reliability matters (write-downs, OCI items, aggressive asset capitalisation).

Impact on DCF inputs:
- Cash conversion affects **working capital forecasts** and free cash flow.
- Accrual build often signals future cash drag or earnings reversal, affecting growth and margins.

Comparability issues:
- Different revenue policies, provisioning, and capitalisation practices can dominate ratio differences more than economics.
- For cross-company comps, standardise:
  - operating vs financing classification,
  - lease treatment,
  - capitalised development/R&D policy,
  - provisions and restructuring.

Practical adjustments:
```python
def adjust_ebitda_for_capitalized_costs(reported_ebitda: float, capitalized_costs: float, amortization_of_capitalized_costs: float) -> float:
    """Normalize EBITDA by expensing capitalized costs and adding back related amortization."""
    e = float(reported_ebitda)
    cap = float(capitalized_costs)
    amo = float(amortization_of_capitalized_costs)
    if not np.isfinite(e + cap + amo):
        raise ValueError("Inputs must be finite.")
    # Expensing capitalized costs reduces EBITDA; removing amortization increases EBITDA
    return e - cap + amo
```

---

### Quality of Earnings Flags
⚠️ Accrual ratio persistently high (NI rising while CFO lags).  
⚠️ DSO rising faster than sales (revenue pulled forward; collection risk).  
⚠️ Large “non-recurring” items every year (recurring one-offs).  
⚠️ Big swings in provisions/reserves with later reversals (cookie-jar accounting).  
⚠️ EBITDA improving while capex, capitalised development, or working capital balloon (hidden reinvestment).  
⚠️ Frequent classification changes (operating vs exceptional vs financing) making trends hard to interpret.  
✅ Stable cash conversion over multiple years with consistent accounting policies.  
✅ Transparent reconciliation from reported earnings to core earnings with clear one-off separation.  

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS / software | revenue timing (contract liabilities), capitalised dev, SBC | reconcile ARR/contract liabilities; normalise dev capitalisation and SBC policy |
| Construction / project-based | percentage-of-completion estimates | stress-test margins and claims; watch receivables and unbilled revenue |
| Retail / distribution | inventory valuation and markdowns | track inventory days and write-downs; normalise for shrinkage/markdown policy |
| Financials | provisions and fair value through P&L | focus on credit loss metrics and reserve builds; separate market-driven FV effects |

---

### Real-World Example
Scenario: A firm reports improving earnings, but working capital is absorbing cash and accruals are rising.

```python
import pandas as pd

ts = pd.DataFrame(
    {
        "net_income": [80, 100, 120],
        "cfo": [90, 75, 70],
        "total_assets": [900, 980, 1100],
        "revenue": [700, 760, 820],
        "ar": [90, 115, 140],
        "inventory": [120, 135, 160],
        "cogs": [420, 456, 492],
        "ap": [85, 90, 95],
    },
    index=[2022, 2023, 2024],
)

panel = earnings_quality_panel(ts)
print(panel.round(3))
```

Interpretation:
- If accrual ratio increases and cash conversion falls while DSO/DIO rise, earnings quality is deteriorating.
- In valuation, respond by:
  - forecasting slower cash conversion or higher working capital needs,
  - reducing confidence in margin expansion,
  - and applying more conservative terminal assumptions unless evidence supports reversal.

See also: Chapter 10 (analysis of BS/IS), Chapter 11 (cash flow analysis), Chapter 12 (profitability), Chapter 13 (growth and sustainable earnings), Chapter 17 (economic vs accounting value creation).
