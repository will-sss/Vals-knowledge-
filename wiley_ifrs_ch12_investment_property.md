# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 12: Investment Property (IAS 40, fair value vs cost, EBITDA noise, NAV vs DCF bridges)

### Core Concept
IAS 40 governs investment property—property held to earn rentals and/or for capital appreciation—allowing measurement either at fair value (with changes in profit or loss) or at cost (with depreciation and impairment). For valuation, IAS 40 is critical because fair value movements can create large non-cash P&L volatility, distort EBITDA/EBIT comparability, and shift the appropriate valuation anchor toward NAV-style approaches rather than operating multiples.

### Formula/Methodology

#### 1) Classification (practical)
```text
Investment property: held to earn rentals/capital appreciation (not owner-occupied, not held for sale in ordinary course)
Owner-occupied property: IAS 16 PPE
Inventory/property held for sale in ordinary course: IAS 2 / IFRS 15 context
```

#### 2) Measurement models
Fair value model:
```text
Carrying Amount = Fair Value
Fair Value Change (P&L) = Fair Value_end - Fair Value_begin - Capital Expenditure + Disposals (simplified bridge)
No depreciation under fair value model
```

Cost model:
```text
Carrying Amount = Cost - Accumulated Depreciation - Accumulated Impairment
Depreciation impacts EBIT; impairment tested under IAS 36 as relevant
```

#### 3) Yield and valuation building blocks (commonly used in practice)
```text
Net Initial Yield (NIY) = Net Operating Income / Property Value
Capitalisation (income) approach: Property Value ≈ Stabilised NOI / Capitalisation Rate
Where:
NOI = rental income - property operating costs (exclude financing)
Capitalisation Rate = market yield / cap rate
```

#### 4) DCF framework (property-level)
```text
Property Value = sum(FCF_t / (1 + r)^t) + Terminal Value
Where:
FCF_t = net rental cash flows - capex/letting costs - taxes (depending on modelling choice)
r = discount rate consistent with property risk (often WACC-like for property cash flows)
Terminal Value (cap-rate): TV = Stabilised NOI_(t+1) / Cap Rate
```

---

### Practical Application for Valuation Analysts

#### 1) Decide the right valuation anchor: operating multiple vs NAV vs property DCF
Common decision rules:
- If the business is predominantly investment property (REIT-like), use **NAV and property-level cap rates/DCF** as primary.
- If property is ancillary to operations (e.g., retailer with owned real estate), treat property as **non-operating / surplus assets** and value separately (valuation by parts).

#### 2) Normalise earnings for fair value movements
Under fair value model, profit includes FV gains/losses that are not operating cash earnings.
For multiples:
- Use a “cash earnings” metric (e.g., rental EBITDA/NOI/FFO-like proxy) rather than IFRS profit including FV changes.
- Separate: (i) recurring rental margin, (ii) one-off revaluation movements.

#### 3) Comparability across companies using different models
- Fair value model: higher P&L volatility, no depreciation.
- Cost model: smoother P&L but potential hidden value changes not in earnings.
Analyst action:
- Bring both onto a comparable basis: focus on cash NOI/EBITDA and disclosed fair values, or build NAV.

#### 4) Watch for classification boundary issues
Boundary between IAS 40 and IAS 16 can change ratios:
- Reclassifications affect depreciation vs FV changes.
- For mixed-use property, split where possible; otherwise classify based on predominant use.

#### 5) Sensitivity: cap rate / yield is a key driver
Small changes in cap rates can materially change values.
- Always run sensitivity on cap rate and rent growth.
- Cross-check implied cap rate from market value vs NOI.

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

def net_initial_yield(noi: float, property_value: float) -> float:
    """
    NIY = NOI / Property Value

    Raises:
        ValueError: if property_value is zero/negative.
    """
    n = _num(noi, "noi")
    v = _num(property_value, "property_value")
    if v <= 0:
        raise ValueError("property_value must be > 0.")
    return n / v

def capitalised_value(noi: float, cap_rate: float) -> float:
    """
    Value = NOI / cap_rate

    Raises:
        ValueError: if cap_rate <= 0.
    """
    n = _num(noi, "noi")
    c = _num(cap_rate, "cap_rate")
    if c <= 0:
        raise ValueError("cap_rate must be > 0.")
    return n / c

def fair_value_change(fv_begin: float, fv_end: float, capex: float = 0.0, disposals_at_fv: float = 0.0) -> float:
    """
    Approximate FV movement:
      FV gain/loss ≈ FV_end - FV_begin - capex + disposals_at_fv

    Args:
        fv_begin: opening fair value
        fv_end: closing fair value
        capex: capital expenditure added during period (reduces 'pure' valuation gain)
        disposals_at_fv: FV of disposed assets (adds back to isolate valuation effect on remaining)

    Returns:
        float: estimated FV gain/loss

    Raises:
        ValueError: invalid inputs.
    """
    b = _num(fv_begin, "fv_begin")
    e = _num(fv_end, "fv_end")
    cx = _num(capex, "capex")
    d = _num(disposals_at_fv, "disposals_at_fv")
    return e - b - cx + d

def property_dcf(cash_flows: np.ndarray, discount_rate: float, terminal_value: float = 0.0) -> float:
    """
    Discount a stream of property-level cash flows.

    Args:
        cash_flows: array-like, cash flow at t=1..n
        discount_rate: per-period discount rate (decimal)
        terminal_value: value at end of final period (undiscounted)

    Returns:
        float: present value

    Raises:
        ValueError: invalid inputs.
    """
    if cash_flows is None:
        raise ValueError("cash_flows is missing.")
    r = _num(discount_rate, "discount_rate")
    if r <= -0.999:
        raise ValueError("discount_rate must be > -0.999.")
    tv = _num(terminal_value, "terminal_value")
    cf = np.asarray(cash_flows, dtype=float)
    if cf.ndim != 1 or cf.size == 0:
        raise ValueError("cash_flows must be a 1D non-empty array.")
    if not np.all(np.isfinite(cf)):
        raise ValueError("cash_flows must be finite.")

    periods = np.arange(1, cf.size + 1)
    pv = np.sum(cf / ((1.0 + r) ** periods))
    pv_tv = tv / ((1.0 + r) ** cf.size)
    return pv + pv_tv

def nav_bridge(investment_property_fv: float, other_assets: float, liabilities: float) -> float:
    """
    Simple NAV:
      NAV = (Investment property FV + other assets) - liabilities

    Raises:
        ValueError: invalid inputs.
    """
    ip = _num(investment_property_fv, "investment_property_fv")
    oa = _num(other_assets, "other_assets")
    l = _num(liabilities, "liabilities")
    return ip + oa - l

# Example usage
noi = 48_000_000
pv = 800_000_000
niy = net_initial_yield(noi, pv)
print(f"NIY: {niy:.2%}")

value_cap = capitalised_value(noi, cap_rate=0.06)
print(f"Capitalised value: ${value_cap/1e6:.0f}M")

fv_gain = fair_value_change(fv_begin=780_000_000, fv_end=820_000_000, capex=10_000_000, disposals_at_fv=0.0)
print(f"Estimated FV gain/loss: ${fv_gain/1e6:.1f}M")

cfs = np.array([35e6, 36e6, 37e6, 38e6, 39e6])
tv = capitalised_value(noi=40e6, cap_rate=0.065)
pv_dcf = property_dcf(cfs, discount_rate=0.075, terminal_value=tv)
print(f"Property DCF value: ${pv_dcf/1e6:.0f}M")

nav = nav_bridge(investment_property_fv=820e6, other_assets=90e6, liabilities=520e6)
print(f"NAV: ${nav/1e6:.0f}M")
```

---

### Valuation Impact
Why this matters:
- Fair value gains/losses under IAS 40 can dominate reported profit but are not operating cash earnings; using P/E or EV/EBIT without normalisation can mislead.
- Choice of model affects leverage and return metrics; compare on **cash NOI**, **cap rates**, and **NAV** rather than accounting profit.
- Cap rate sensitivity is often the largest single driver of equity value for property-heavy businesses.

Impact on multiples:
- EV/EBITDA: if EBITDA includes only rental operations, it can be comparable; if profit includes FV movements, exclude FV movements for “operating” multiple analysis.
- P/E: problematic under FV model due to volatility; prefer P/NAV or discount to NAV for REIT-like profiles (if applicable to your framework).

Impact on DCF inputs:
- Rent growth, occupancy, operating costs, capex (letting/refurbishment), and terminal cap rate dominate value.
- Ensure discount rate is consistent with property risk; do not mix corporate WACC with property cash flows without justification.

Practical adjustments:
```python
def normalize_operating_profit_for_fair_value(reported_profit: float, fair_value_gains: float, fair_value_losses: float = 0.0) -> float:
    """
    Remove FV movements to approximate operating profit.

    Convention:
      fair_value_gains: positive amount included in profit
      fair_value_losses: positive amount included in profit (as expense)

    Operating profit proxy = reported_profit - fair_value_gains + fair_value_losses
    """
    p = _num(reported_profit, "reported_profit")
    g = _num(fair_value_gains, "fair_value_gains")
    l = _num(fair_value_losses, "fair_value_losses")
    if g < 0 or l < 0:
        raise ValueError("fair_value_gains and fair_value_losses must be >= 0.")
    return p - g + l
```

---

### Quality of Earnings Flags
⚠️ Large FV gains used to support dividends or “underlying earnings” narrative without cash support.  
⚠️ Material cap rate compression assumptions not aligned with market evidence.  
⚠️ Frequent reclassifications between IAS 40 and IAS 16 (ratio management risk).  
⚠️ Limited disclosure of valuation inputs (yields, discount rates, occupancy assumptions).  
✅ Transparent valuation methodology, sensitivity analysis, and reconciliation of FV movements.

---

### Sector-Specific Considerations

| Sector | Key IAS 40 issue | Typical valuation treatment |
|---|---|---|
| REIT / Property investment | FV model common, P&L volatility | NAV and cap-rate approach primary; use cash NOI/FFO proxy for coverage |
| Hospitality / Mixed-use | Classification between owner-occupied and investment | Split cash flows by use; value operating hotel separately from property |
| Retailers with owned property | Property may be non-operating | Value property separately; adjust EV bridge for surplus assets |
| Infrastructure with property elements | Yield-based valuation | Ensure correct asset classification; apply appropriate discount rate |

---

### Real-World Example
Scenario: Two property companies report different profits because one uses FV model. You want comparable cash earnings for EV/EBITDA screens.

```python
reported_profit = 120.0
fv_gains = 70.0
fv_losses = 0.0

operating_profit_proxy = normalize_operating_profit_for_fair_value(reported_profit, fv_gains, fv_losses)
print(f"Operating profit proxy: ${operating_profit_proxy:.1f}m")
```

Interpretation: High profit driven by FV gains should not be capitalised at a standard earnings multiple; use NAV/cap-rate approach and stress-test yields.

See also: Chapter 25 (fair value) for measurement principles; Chapter 13 (impairment/non-current assets held for sale) for valuation downside; Chapter 6 (cash flows) for capex and rental cash conversion.
