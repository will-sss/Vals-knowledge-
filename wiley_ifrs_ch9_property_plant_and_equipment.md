# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 9: Property, Plant and Equipment (IAS 16, capex vs opex, depreciation, revaluation, impairment linkage)

### Core Concept
IAS 16 governs recognition and measurement of property, plant and equipment (PPE), including what gets capitalised, depreciation methods, and optional revaluation. For valuation, PPE accounting drives EBITDA/EBIT comparability (capex vs opex timing), invested capital (ROIC), and free cash flow (maintenance vs growth capex). Differences in useful lives, residual values, and capitalisation policies can materially shift reported margins and reinvestment rates.

### Formula/Methodology

#### 1) Recognition and initial measurement (cost model baseline)
```text
PPE at initial recognition = Purchase price + directly attributable costs + initial estimate of dismantling/restoration (if applicable)
Exclude: abnormal costs, inefficiencies, general admin not directly attributable, start-up costs (unless directly attributable), certain training costs.
```

#### 2) Subsequent measurement
Cost model:
```text
Carrying Amount = Cost - Accumulated Depreciation - Accumulated Impairment
```

Revaluation model (if applied to a class of assets):
```text
Carrying Amount = Fair Value at revaluation date - Subsequent Accumulated Depreciation - Subsequent Accumulated Impairment
Revaluation Surplus (equity) generally recorded in OCI, unless reversing a prior loss in P&L.
```

#### 3) Depreciation (systematic allocation)
Straight-line (common):
```text
Annual Depreciation = (Cost - Residual Value) / Useful Life
```

Units of production (when usage drives economic consumption):
```text
Depreciation per unit = (Cost - Residual Value) / Total expected units
Period Depreciation = Units produced in period × Depreciation per unit
```

#### 4) Capex and maintenance (valuation modelling)
```text
Gross Capex = Cash paid for PPE and other capital additions (from cash flow statement)
Maintenance Capex (concept) = Capex required to sustain current operations (often approximated via D&A or disclosed guidance)
Growth Capex (concept) = Capex beyond maintenance to grow capacity or enter new markets
```

#### 5) ROIC linkage (invested capital impact)
```text
Invested Capital includes Net PPE (often: PPE net of accumulated depreciation and impairment) as an operating asset.
ROIC = NOPAT / Invested Capital
```

---

### Practical Application for Valuation Analysts

#### 1) Normalise EBITDA/EBIT for capitalisation policy differences
Common distortion: capitalising costs that peers expense (e.g., major overhauls, certain installation or project costs).
- Capitalising increases current EBIT/EBITDA (lower opex) but increases depreciation later.
- For peer multiples, consider adjusting to a consistent policy: either (i) expense-like treatment (reduce EBITDA) or (ii) capitalise-like treatment (increase invested capital and depreciation consistently).

#### 2) Depreciation is an estimate: treat “useful life extensions” as a potential earnings management lever
Useful lives and residual values are estimates. Extending lives:
- Reduces depreciation and increases EBIT, without immediate cash benefit.
- May require a maintenance capex cross-check: if capex does not fall, “earnings improvement” may be cosmetic.

#### 3) Componentisation matters (especially in asset-heavy industries)
IAS 16 requires significant components to be depreciated separately when they have different useful lives.
- If component accounting is weak, depreciation and future capex profiles may be misstated.
- Analysts should check disclosure quality and capex cycles.

#### 4) Revaluation model: impacts equity and ratios
Revaluation increases assets and equity (revaluation surplus), reduces leverage ratios, and can change ROA/ROE mechanically.
- For valuation, revaluation is usually not an operating cash flow driver; treat carefully in ROIC comparisons and EV bridges.
- If valuation relies on NAV (e.g., some infrastructure), revaluation may be more relevant, but still requires scrutiny of valuation methodology.

#### 5) Disposal gains/losses: separate from operating performance
Gains on disposal of PPE are not operating earnings; remove from run-rate EBIT/EBITDA unless disposals are part of normal operations (rare; even then, treat cautiously).

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

def straight_line_depreciation(cost: float, residual_value: float, useful_life_years: float) -> float:
    """
    Annual straight-line depreciation.

    Args:
        cost: gross cost of the asset
        residual_value: expected residual value at end of life
        useful_life_years: useful life in years (must be > 0)

    Returns:
        float: annual depreciation expense

    Raises:
        ValueError: invalid inputs (life <= 0, residual > cost).
    """
    c = _num(cost, "cost")
    r = _num(residual_value, "residual_value")
    life = _num(useful_life_years, "useful_life_years")
    if life <= 0:
        raise ValueError("useful_life_years must be > 0.")
    if r < 0:
        raise ValueError("residual_value must be >= 0.")
    if r > c:
        raise ValueError("residual_value cannot exceed cost.")
    return (c - r) / life

def units_of_production_depreciation(cost: float, residual_value: float, total_expected_units: float, units_this_period: float) -> float:
    """
    Units of production depreciation.

    Raises:
        ValueError: invalid inputs.
    """
    c = _num(cost, "cost")
    r = _num(residual_value, "residual_value")
    total_units = _num(total_expected_units, "total_expected_units")
    units = _num(units_this_period, "units_this_period")
    if total_units <= 0:
        raise ValueError("total_expected_units must be > 0.")
    if units < 0:
        raise ValueError("units_this_period must be >= 0.")
    if r > c:
        raise ValueError("residual_value cannot exceed cost.")
    dep_per_unit = (c - r) / total_units
    return units * dep_per_unit

def net_ppe(gross_ppe: float, accumulated_depreciation: float, accumulated_impairment: float = 0.0) -> float:
    """
    Net PPE = gross PPE - accumulated depreciation - accumulated impairment.
    """
    g = _num(gross_ppe, "gross_ppe")
    ad = _num(accumulated_depreciation, "accumulated_depreciation")
    ai = _num(accumulated_impairment, "accumulated_impairment")
    return g - ad - ai

def capex_intensity(capex: float, revenue: float) -> float:
    """
    Capex intensity = capex / revenue.

    Raises:
        ValueError: revenue zero.
    """
    cx = _num(capex, "capex")
    r = _num(revenue, "revenue")
    if abs(r) < 1e-12:
        raise ValueError("revenue must not be zero.")
    return cx / r

def maintenance_capex_proxy(depreciation: float, adjustment_factor: float = 1.0) -> float:
    """
    Simple proxy: maintenance capex ≈ depreciation × factor.
    Use factor > 1 where inflation/ageing requires higher replacement spending.

    Raises:
        ValueError: invalid factor.
    """
    d = _num(depreciation, "depreciation")
    f = _num(adjustment_factor, "adjustment_factor")
    if f <= 0:
        raise ValueError("adjustment_factor must be > 0.")
    return d * f

# Example usage
asset = {"cost": 120_000_000, "residual": 10_000_000, "life": 10}
dep_sl = straight_line_depreciation(asset["cost"], asset["residual"], asset["life"])
print(f"Annual depreciation (SL): ${dep_sl/1e6:.1f}M")

dep_uop = units_of_production_depreciation(
    cost=120_000_000, residual_value=10_000_000,
    total_expected_units=5_000_000, units_this_period=420_000
)
print(f"Period depreciation (UoP): ${dep_uop/1e6:.1f}M")

nppe = net_ppe(gross_ppe=2_300_000_000, accumulated_depreciation=1_050_000_000, accumulated_impairment=60_000_000)
print(f"Net PPE: ${nppe/1e9:.2f}B")

cx_int = capex_intensity(capex=310_000_000, revenue=2_400_000_000)
print(f"Capex intensity: {cx_int:.1%}")

maint = maintenance_capex_proxy(depreciation=260_000_000, adjustment_factor=1.1)
print(f"Maintenance capex proxy: ${maint/1e6:.0f}M")
```

---

### Valuation Impact
Why this matters:
- PPE accounting influences **EBITDA/EBIT** (expense timing) and **invested capital** (ROIC denominator).
- Forecasts require realistic reinvestment: depreciation is not cash, but it often proxies for replacement needs.
- Revaluation can inflate equity and reduce apparent leverage; EV-to-equity bridges and ROIC comparability must be adjusted to avoid false “improvement.”

Impact on multiples:
- Aggressive capitalisation and long useful lives inflate EBITDA/EBIT, potentially lowering EV/EBITDA and EV/EBIT artificially.
- Disposal gains can temporarily boost EBIT and bias multiples downward unless removed.

Impact on DCF inputs:
- Capex and maintenance assumptions drive FCF; check capex vs depreciation and asset age.
- ROIC-based growth: Net PPE affects invested capital and reinvestment rate calculations.

Practical adjustments:
```python
def normalize_ebit_for_capitalized_costs(reported_ebit: float, capitalized_costs: float, annual_depreciation_on_capitalized: float) -> float:
    """
    Approximate adjustment to restate EBIT as if costs were expensed (policy comparison).
    If costs were capitalized, reported EBIT is higher by (capitalized_costs - depreciation_on_that_cost).
    To expense: subtract (capitalized_costs - depreciation_on_that_cost).

    Args:
        reported_ebit: reported EBIT
        capitalized_costs: costs capitalized in the period that peers might expense
        annual_depreciation_on_capitalized: depreciation/amortization recorded on those capitalized costs in the period

    Returns:
        float: normalized EBIT

    Raises:
        ValueError: invalid inputs.
    """
    re = _num(reported_ebit, "reported_ebit")
    cap = _num(capitalized_costs, "capitalized_costs")
    dep = _num(annual_depreciation_on_capitalized, "annual_depreciation_on_capitalized")
    return re - (cap - dep)
```

---

### Quality of Earnings Flags
⚠️ Useful life extensions or residual value increases that materially boost EBIT without corresponding capex reductions.  
⚠️ High levels of capitalised project costs with limited disclosure (risk of “cost parking”).  
⚠️ Large impairment charges following years of aggressive capitalisation (suggests overstated prior earnings).  
⚠️ Frequent “gains on disposal” supporting operating profit.  
✅ Clear capex disclosure (maintenance vs growth where available) and stable capex-to-depreciation relationship consistent with asset base.

---

### Sector-Specific Considerations

| Sector | Key IAS 16 sensitivity | Typical valuation treatment |
|---|---|---|
| Industrials / Manufacturing | Component accounting and overhaul capitalization | Validate depreciation profile vs capex cycles; adjust for capitalization differences |
| Utilities / Infrastructure | Asset lives and regulatory replacement cycles | Use long-term capex plans; cross-check ROIC vs allowed returns; treat revaluation carefully |
| Airlines / Transport | Heavy asset base; major checks/overhauls | Separate maintenance events; model capex cycles explicitly |
| Real Estate | PPE vs investment property classification | Ensure correct standard applied (IAS 16 vs IAS 40); align valuation approach accordingly |

---

### Real-World Example
Scenario: Two peers have the same EBITDA, but one capitalises $40m of maintenance costs that the other expenses. You want comparable EBIT for EV/EBIT.

```python
reported_ebit = 180.0
capitalized_maint = 40.0
dep_on_capitalized = 5.0

ebit_norm = normalize_ebit_for_capitalized_costs(reported_ebit, capitalized_maint, dep_on_capitalized)
print(f"Reported EBIT: ${reported_ebit:.1f}m")
print(f"Normalized EBIT (expense-like): ${ebit_norm:.1f}m")
```

Interpretation: Normalised EBIT is lower if capitalised maintenance is treated as period expense; this increases EV/EBIT and can change peer ranking.

See also: Chapter 13 (impairment) for recoverable amount testing; Chapter 6 (cash flows) for capex cash line items; Chapter 7 (IAS 8) for depreciation method and useful life estimate changes.
