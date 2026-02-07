# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 8: Inventories (IAS 2, costing, NRV write-downs, margin quality)

### Core Concept
IAS 2 governs how inventories are measured (cost vs net realisable value) and what costs can be included. For valuation, inventory accounting directly affects gross margin, EBITDA, working capital, and cash conversion; inventory write-downs and reversals are classic quality-of-earnings flags and can signal pricing pressure, obsolescence, or poor demand forecasting.

### Formula/Methodology

#### 1) Inventory measurement
```text
Inventory carrying amount = min(Cost, Net Realisable Value (NRV))
Where:
NRV = Estimated selling price - Estimated costs of completion - Estimated costs to sell
```

#### 2) Cost formulas (policy choice; must be consistent)
```text
Cost of Inventory = Purchase costs + Conversion costs + Other costs to bring to present location/condition
Exclude: abnormal waste, storage (unless necessary in production), administrative overheads not contributing to inventory, selling costs.
```

Cost assignment methods (typical):
```text
FIFO: earliest costs assigned to COGS first; ending inventory reflects latest costs
Weighted Average Cost: COGS and ending inventory based on average cost per unit
Specific Identification: used for non-interchangeable items
```

#### 3) Gross margin and inventory turns (valuation KPIs)
```text
Gross Margin = (Revenue - COGS) / Revenue
Inventory Turnover = COGS / Average Inventory
Days Inventory Outstanding (DIO) = 365 × Average Inventory / COGS
```

---

### Practical Application for Valuation Analysts

#### 1) Detect margin distortions from write-downs and reversals
IAS 2 write-downs typically hit COGS (or expense line) and reduce margins. Reversals (when NRV improves) increase profit and can inflate margins temporarily.

Analyst actions:
- Identify write-downs/reversals in notes and quantify.
- Normalise gross margin if write-downs are unusually large or non-recurring (policy-driven judgement).
- Treat repeated write-downs as structural (pricing power or inventory management issue).

#### 2) Normalise working capital for forecasting
Inventory is often the largest working-capital component in industrials/retail. For forecasting:
- Model inventory as days of sales or days of COGS (DIO).
- Use average balances (seasonality).
- Adjust for non-recurring build/release events (e.g., supply disruption, one-off stock clearance).

#### 3) Comparability across peers
Differences in costing policies affect comparability:
- What is capitalised into inventory (e.g., certain overheads) changes timing of expense recognition, inflating current EBITDA.
- IFRS prohibits LIFO (good for comparability vs US GAAP peers that historically used LIFO).

---

### Python Implementation
```python
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def nrv(selling_price: float, completion_costs: float, selling_costs: float) -> float:
    """
    NRV = selling price - completion costs - selling costs.
    """
    sp = _num(selling_price, "selling_price")
    cc = _num(completion_costs, "completion_costs")
    sc = _num(selling_costs, "selling_costs")
    return sp - cc - sc

def inventory_carrying_amount(cost: float, nrv_value: float) -> float:
    """
    Carrying amount = min(cost, NRV).
    """
    c = _num(cost, "cost")
    n = _num(nrv_value, "nrv_value")
    return min(c, n)

def inventory_write_down(cost: float, nrv_value: float) -> float:
    """
    Write-down amount = max(0, cost - NRV).
    """
    c = _num(cost, "cost")
    n = _num(nrv_value, "nrv_value")
    return max(0.0, c - n)

def gross_margin(revenue: float, cogs: float) -> float:
    """
    Gross margin = (revenue - cogs) / revenue.
    """
    r = _num(revenue, "revenue")
    c = _num(cogs, "cogs")
    if abs(r) < 1e-12:
        raise ValueError("revenue must not be zero.")
    return (r - c) / r

def inventory_turnover(cogs: float, inv_begin: float, inv_end: float) -> Dict[str, float]:
    """
    Inventory turnover and DIO.

    Returns:
        dict with avg_inventory, turnover, dio

    Raises:
        ValueError: cogs must not be zero for turnover metrics.
    """
    cg = _num(cogs, "cogs")
    b = _num(inv_begin, "inv_begin")
    e = _num(inv_end, "inv_end")
    avg_inv = (b + e) / 2.0
    if abs(avg_inv) < 1e-12:
        turnover = float("inf")
        dio = 0.0
    else:
        turnover = cg / avg_inv
        dio = 365.0 * avg_inv / cg if abs(cg) > 1e-12 else float("inf")
    return {"avg_inventory": avg_inv, "turnover": turnover, "dio": dio}

def normalize_cogs_for_write_downs(cogs_reported: float, write_downs_included: float, reversals_included: float = 0.0) -> float:
    """
    Normalise COGS by removing inventory write-downs and reversals.

    Convention:
      - write_downs_included: positive expense included in COGS
      - reversals_included: positive credit reducing COGS (improves gross profit)

    Normalised COGS = reported COGS - write_downs + reversals

    Raises:
        ValueError: invalid inputs.
    """
    c = _num(cogs_reported, "cogs_reported")
    wd = _num(write_downs_included, "write_downs_included")
    rv = _num(reversals_included, "reversals_included")
    if wd < 0 or rv < 0:
        raise ValueError("write_downs_included and reversals_included must be >= 0.")
    return c - wd + rv

# Example usage
inv_cost = 52.0
nrv_val = nrv(selling_price=50.0, completion_costs=1.0, selling_costs=2.0)  # 47
carrying = inventory_carrying_amount(inv_cost, nrv_val)
wd = inventory_write_down(inv_cost, nrv_val)

print(f"NRV: ${nrv_val:.1f}m")
print(f"Carrying amount: ${carrying:.1f}m")
print(f"Write-down: ${wd:.1f}m")

rev = 400.0
cogs = 280.0
gm = gross_margin(rev, cogs)
print(f"Gross margin: {gm:.1%}")

turn = inventory_turnover(cogs=280.0, inv_begin=90.0, inv_end=110.0)
print(turn)

cogs_norm = normalize_cogs_for_write_downs(cogs_reported=280.0, write_downs_included=12.0, reversals_included=3.0)
gm_norm = gross_margin(rev, cogs_norm)
print(f"Normalised gross margin: {gm_norm:.1%}")
```

---

### Valuation Impact
Why this matters:
- Inventory accounting affects **gross margin** and therefore EV/EBITDA and EV/EBIT comparability across peers.
- Inventory is a major component of **net working capital**, driving reinvestment needs and FCF.
- Persistent write-downs indicate weaker pricing power, poor demand forecasting, or obsolete product risk (future cash flows and terminal value risk).

Impact on multiples:
- Companies capitalising more costs into inventory may show higher current EBITDA and later margin compression when inventory sells through.
- Write-downs can depress current earnings; repeated write-downs should not be treated as one-off.

Impact on DCF inputs:
- ΔInventory drives ΔNWC. If inventory increases faster than sales, FCF falls even if profits rise.
- Terminal margin assumptions should reflect structural inventory risk (e.g., high obsolescence categories).

Practical adjustments:
```python
def normalize_working_capital_for_inventory(inventory_reported: float, excess_inventory: float) -> float:
    """
    Adjust inventory to a 'normal' level by removing excess build.

    Args:
        inventory_reported: period-end inventory
        excess_inventory: portion considered non-recurring / abnormal build

    Returns:
        float: adjusted inventory
    """
    inv = _num(inventory_reported, "inventory_reported")
    ex = _num(excess_inventory, "excess_inventory")
    if ex < 0:
        raise ValueError("excess_inventory must be >= 0.")
    if ex > inv:
        raise ValueError("excess_inventory cannot exceed inventory_reported.")
    return inv - ex
```

---

### Quality of Earnings Flags
⚠️ Large inventory write-downs framed as “non-recurring” but occurring frequently (structural demand/price issues).  
⚠️ Significant reversals of prior write-downs boosting margins (may indicate timing management or volatile pricing).  
⚠️ Inventory growth outpacing revenue/COGS without clear strategic rationale (risk of future write-downs).  
⚠️ Gross margin improvement driven by capitalised overheads into inventory (timing shift, not economics).  
✅ Stable DIO and limited write-downs with clear ageing and obsolescence policies.

---

### Sector-Specific Considerations

| Sector | Key inventory issue | Typical valuation treatment |
|---|---|---|
| Retail / Apparel | High obsolescence and markdown risk | Normalise margins for markdown cycles; model DIO and clearance effects explicitly |
| Manufacturing | Overhead capitalisation into inventory | Standardise cost policy; watch for absorption accounting boosting EBIT in build periods |
| Commodities / Chemicals | Price volatility | Evaluate NRV sensitivity; consider mid-cycle margins and inventory valuation effects |
| Pharma / Medtech | Shelf-life and regulatory write-offs | Higher scrutiny on obsolescence provisions; conservative terminal margins |

---

### Real-World Example
Scenario: A retailer reports gross margin improvement, but it includes a large reversal of prior inventory write-downs. You want run-rate margin for a multiple.

```python
revenue = 1_000.0
cogs_reported = 720.0
write_downs = 0.0
reversals = 25.0  # credit inside COGS

cogs_run_rate = normalize_cogs_for_write_downs(cogs_reported, write_downs_included=write_downs, reversals_included=reversals)
gm_reported = gross_margin(revenue, cogs_reported)
gm_run_rate = gross_margin(revenue, cogs_run_rate)

print(f"Reported GM: {gm_reported:.1%}")
print(f"Run-rate GM (ex reversal): {gm_run_rate:.1%}")
```

Interpretation: If margin improvement is mainly reversal-driven, a peer multiple based on reported EBIT/EBITDA may be overstated; use run-rate margin.

See also: Chapter 4 (balance sheet) for NWC definitions; Chapter 6 (cash flows) for working-capital movements; Chapter 7 (IAS 8) for reclassifications and restatements affecting COGS and inventory disclosures.
