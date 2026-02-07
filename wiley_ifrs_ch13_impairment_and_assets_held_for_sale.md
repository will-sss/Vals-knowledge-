# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 13: Impairment of Assets and Non-Current Assets Held for Sale (IAS 36 & IFRS 5, recoverable amount, CGUs, terminal assumptions)

### Core Concept
IAS 36 requires entities to ensure assets are not carried above recoverable amount (the higher of fair value less costs of disposal and value in use). IFRS 5 governs classification and measurement of assets held for sale and discontinued operations. For valuation, impairment accounting is a major quality-of-earnings and balance-sheet integrity topic: impairments reveal over-optimistic past capitalisation and can signal structurally weaker cash flows, while IFRS 5 classification can create one-off measurement effects and reclassifications that complicate comparability.

### Formula/Methodology

#### 1) Recoverable amount (IAS 36)
```text
Recoverable Amount = max(Value in Use (VIU), Fair Value Less Costs of Disposal (FVLCD))

Impairment Loss = max(0, Carrying Amount - Recoverable Amount)
```

Value in Use (DCF style):
```text
VIU = sum(FCF_t / (1 + r)^t) + Terminal Value
Where:
FCF_t = cash flows from continuing use (pre-tax or post-tax consistent with r)
r = discount rate consistent with cash flows (often pre-tax rate in IAS 36 context)
Terminal Value = value beyond explicit period (e.g., perpetuity, exit multiple), consistent with cash flows
```

FVLCD (market-based):
```text
FVLCD = Fair Value - Costs of Disposal
Where:
Fair Value: observable market price or valuation technique consistent with market participant assumptions
Costs of Disposal: incremental costs directly attributable to disposal (excluding finance costs and income tax)
```

#### 2) Cash-Generating Unit (CGU) allocation
```text
If an asset does not generate independent cash inflows, test at CGU level:
CGU = smallest identifiable group of assets generating largely independent cash inflows
Allocate goodwill and corporate assets to CGUs (or groups of CGUs) for testing.
```

#### 3) Impairment reversal rules (IAS 36)
```text
Reversal allowed for assets (excluding goodwill) if recoverable amount increases due to change in estimates.
Goodwill impairment is NOT reversed.
Reversal limited: carrying amount after reversal cannot exceed carrying amount that would have been determined (net of depreciation/amortisation) if no impairment had been recognized.
```

#### 4) IFRS 5 held-for-sale measurement
```text
Held-for-sale carrying amount = min(Current carrying amount, Fair Value Less Costs to Sell (FVLCTS))
Depreciation/amortisation stops when classified as held for sale.
Loss on initial classification = max(0, Carrying Amount - FVLCTS)
Gains on subsequent remeasurement recognized only to the extent of cumulative losses previously recognized.
```

---

### Practical Application for Valuation Analysts

#### 1) Impairments are not “non-cash therefore ignore” in valuation context
While impairments are non-cash in the period, they are evidence that:
- Prior capex / acquisitions / capitalised costs did not generate expected returns
- Forecasts, terminal value, or discount rates used historically may have been aggressive
Implication: incorporate impairment signals into forward-looking assumptions (margins, growth, reinvestment, and risk).

#### 2) Normalise earnings while preserving economic message
For multiples:
- Remove impairment charges from run-rate EBIT/EBITDA when clearly non-recurring.
- But do NOT automatically treat repeated impairments as one-offs; repeated impairments can indicate a structurally poor business model or overpayment pattern.

Analyst approach:
- Create an “underlying EBIT” excluding impairments.
- Separately add an “impairment frequency” QoE note and adjust valuation risk (multiple discount or higher WACC / lower terminal growth if warranted by evidence).

#### 3) Cross-check IAS 36 VIU assumptions against your valuation model
IAS 36 VIU is DCF-like; it is a useful consistency check:
- Compare management’s terminal growth, margins, and discount rate to your valuation case.
- Large headroom can be sensitive to small changes; focus on sensitivity and key drivers.

#### 4) Goodwill impairment: interpret as acquisition discipline signal
Goodwill impairment is not reversed. It often signals:
- Overpayment or synergy overestimation
- Integration challenges
- Downturn in acquired segment economics
For valuation: revisit acquisition-driven growth assumptions and synergy credibility.

#### 5) IFRS 5: beware classification effects on metrics
Held-for-sale classification:
- Reclassifies assets/liabilities to current, changing working capital ratios.
- Stops depreciation (boosts EBIT) while reducing asset base (changes ROIC).
- Discontinued operations are presented separately; ensure you value continuing operations consistently.

---

### Python Implementation
```python
from typing import Dict, Any, Optional, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def present_value_cash_flows(cash_flows: np.ndarray, discount_rate: float) -> float:
    """
    Present value of cash flows at t=1..n.

    Args:
        cash_flows: array of cash flows (t=1..n)
        discount_rate: per-period discount rate (decimal)

    Returns:
        float: PV

    Raises:
        ValueError: invalid inputs.
    """
    if cash_flows is None:
        raise ValueError("cash_flows is missing.")
    r = _num(discount_rate, "discount_rate")
    if r <= -0.999:
        raise ValueError("discount_rate must be > -0.999.")
    cf = np.asarray(cash_flows, dtype=float)
    if cf.ndim != 1 or cf.size == 0:
        raise ValueError("cash_flows must be a 1D non-empty array.")
    if not np.all(np.isfinite(cf)):
        raise ValueError("cash_flows must be finite.")
    t = np.arange(1, cf.size + 1)
    return float(np.sum(cf / ((1.0 + r) ** t)))

def value_in_use(cash_flows: np.ndarray, discount_rate: float, terminal_value: float = 0.0) -> float:
    """
    IAS 36 VIU-style value: PV of explicit cash flows plus PV of terminal value.

    Args:
        cash_flows: cash flows t=1..n
        discount_rate: per-period discount rate
        terminal_value: terminal value at end of period n (undiscounted)

    Returns:
        float: VIU

    Raises:
        ValueError: invalid inputs.
    """
    pv = present_value_cash_flows(cash_flows, discount_rate)
    tv = _num(terminal_value, "terminal_value")
    n = len(np.asarray(cash_flows))
    pv_tv = tv / ((1.0 + discount_rate) ** n)
    return pv + pv_tv

def fvlcd(fair_value: float, costs_of_disposal: float) -> float:
    """
    Fair value less costs of disposal.

    Raises:
        ValueError: invalid inputs.
    """
    fv = _num(fair_value, "fair_value")
    c = _num(costs_of_disposal, "costs_of_disposal")
    if c < 0:
        raise ValueError("costs_of_disposal must be >= 0.")
    return fv - c

def recoverable_amount(viu_value: float, fvlcd_value: float) -> float:
    """
    Recoverable amount is max(VIU, FVLCD).
    """
    v = _num(viu_value, "viu_value")
    f = _num(fvlcd_value, "fvlcd_value")
    return max(v, f)

def impairment_loss(carrying_amount: float, recoverable: float) -> float:
    """
    Impairment = max(0, carrying - recoverable).
    """
    ca = _num(carrying_amount, "carrying_amount")
    ra = _num(recoverable, "recoverable")
    return max(0.0, ca - ra)

def ifrs5_held_for_sale_measurement(carrying_amount: float, fvlcts_value: float) -> Dict[str, float]:
    """
    IFRS 5 measurement on classification:
      new_carrying = min(carrying_amount, fvlcts)
      loss = max(0, carrying_amount - fvlcts)

    Args:
        carrying_amount: current carrying amount
        fvlcts_value: fair value less costs to sell

    Returns:
        dict with new_carrying, initial_loss

    Raises:
        ValueError: invalid inputs.
    """
    ca = _num(carrying_amount, "carrying_amount")
    fv = _num(fvlcts_value, "fvlcts_value")
    new_carrying = min(ca, fv)
    loss = max(0.0, ca - fv)
    return {"new_carrying": new_carrying, "initial_loss": loss}

def normalize_ebit_for_impairments(reported_ebit: float, impairment_charges: float) -> float:
    """
    Underlying EBIT proxy by adding back impairment charges (expense).

    Raises:
        ValueError: impairment must be >=0.
    """
    e = _num(reported_ebit, "reported_ebit")
    imp = _num(impairment_charges, "impairment_charges")
    if imp < 0:
        raise ValueError("impairment_charges must be >= 0.")
    return e + imp

# Example usage (CGU test)
cash_flows = np.array([55e6, 58e6, 60e6, 62e6, 64e6])
discount_rate = 0.095
terminal_value = 900e6

viu = value_in_use(cash_flows, discount_rate, terminal_value)
market_fvlcd = fvlcd(fair_value=980e6, costs_of_disposal=20e6)
ra = recoverable_amount(viu, market_fvlcd)

carrying = 1_080e6
imp = impairment_loss(carrying, ra)
print(f"VIU: ${viu/1e6:.0f}M")
print(f"FVLCD: ${market_fvlcd/1e6:.0f}M")
print(f"Recoverable amount: ${ra/1e6:.0f}M")
print(f"Impairment: ${imp/1e6:.0f}M")

# IFRS 5 example
hfs = ifrs5_held_for_sale_measurement(carrying_amount=260e6, fvlcts_value=235e6)
print(hfs)

# Earnings normalisation
underlying_ebit = normalize_ebit_for_impairments(reported_ebit=120.0, impairment_charges=45.0)
print(f"Underlying EBIT: ${underlying_ebit:.1f}m")
```

---

### Valuation Impact
Why this matters:
- IAS 36 impairment is a **signal about future cash flow expectations** and capital allocation discipline, even if non-cash today.
- Impairment charges distort EBIT/EBITDA and ROIC trends; valuation models must separate “run-rate” from “capital misallocation history.”
- IFRS 5 classification changes depreciation, asset base, and segment presentation—impacting multiples and DCF continuity.

Impact on multiples:
- Remove one-off impairment charges from EBIT/EBITDA for trading multiples, but adjust the multiple (or risk view) if impairments recur.
- Goodwill impairment can affect equity value narratives; be careful using P/B or ROE screens without adjusting for impairment-driven equity changes.

Impact on DCF inputs:
- Use impairment assumptions as a sanity check: if management’s VIU implies aggressive terminal growth or margins, your DCF should challenge those inputs.
- For held-for-sale assets, exclude discontinued cash flows and treat expected disposal proceeds in EV bridge (or valuation by parts).

Practical adjustments:
```python
def adjust_enterprise_value_for_disposal(ev_continuing: float, expected_disposal_proceeds: float, disposal_costs: float = 0.0) -> float:
    """
    Add expected net proceeds from disposal (if not already reflected) to move from continuing EV to total EV.

    Args:
        ev_continuing: EV from continuing operations valuation
        expected_disposal_proceeds: gross proceeds
        disposal_costs: incremental costs to sell

    Returns:
        float: adjusted EV
    """
    ev = _num(ev_continuing, "ev_continuing")
    p = _num(expected_disposal_proceeds, "expected_disposal_proceeds")
    c = _num(disposal_costs, "disposal_costs")
    if c < 0:
        raise ValueError("disposal_costs must be >= 0.")
    return ev + (p - c)
```

---

### Quality of Earnings Flags
⚠️ Repeated impairments in the same CGU or repeated “restructuring + impairment” cycles.  
⚠️ Large “headroom” disclosures that disappear with modest sensitivity changes (fragile value).  
⚠️ Discount rates/terminal assumptions that appear inconsistent with industry conditions or company risk.  
⚠️ Goodwill impairments shortly after acquisitions (overpayment or integration failure).  
⚠️ IFRS 5 classification used to boost EBIT (depreciation stops) without clear strategic rationale.  
✅ Detailed disclosure of CGU composition, key assumptions, sensitivity, and allocation of goodwill/corporate assets.

---

### Sector-Specific Considerations

| Sector | Key impairment driver | Typical valuation treatment |
|---|---|---|
| Consumer / Retail | Store closures, brand erosion, margin pressure | Treat impairments as signal of weaker unit economics; update terminal margins and reinvestment assumptions |
| Industrials | Overcapacity, commodity cycles, project underperformance | Use mid-cycle normalisation; stress-test capex and margin recovery |
| Technology | Capitalised development and acquired intangibles | Scrutinise capitalisation and product obsolescence; align DCF growth with reinvestment reality |
| Financial services | Different impairment standards (IFRS 9 expected credit loss) | Separate from IAS 36; apply sector-specific credit modelling |

---

### Real-World Example
Scenario: Company reports $80m impairment in EBIT line. You need run-rate EBIT for EV/EBIT, and you also want to see how sensitive recoverable amount is.

```python
reported_ebit = 210.0
impairment = 80.0
underlying = normalize_ebit_for_impairments(reported_ebit, impairment)
print(f"Underlying EBIT: ${underlying:.1f}m")

# Sensitivity: reduce terminal value by 10%
cash_flows = np.array([55e6, 58e6, 60e6, 62e6, 64e6])
viu_base = value_in_use(cash_flows, 0.095, terminal_value=900e6)
viu_down = value_in_use(cash_flows, 0.095, terminal_value=900e6 * 0.9)
print(f"VIU base: ${viu_base/1e6:.0f}M")
print(f"VIU -10% TV: ${viu_down/1e6:.0f}M")
```

Interpretation: If a small terminal value change drives large impairment swings, the asset value is fragile; reflect that in your valuation risk adjustments.

See also: Chapter 15 (business combinations) for goodwill allocation; Chapter 25 (fair value) for market-based measurement; Chapter 12 (investment property) for FV model interactions; Chapter 6 (cash flows) for discontinued operations cash flow treatment.
