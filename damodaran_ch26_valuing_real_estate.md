# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 26: Valuing Real Estate (NOI, Cap Rates, DCF, NAV, Leases, Development Risk)

### Core Concept
Real estate valuation is a cash-flow problem anchored on **property-level operating cash flows (NOI)**, **capitalisation rates**, and **asset-specific risks** (lease terms, vacancies, capex, and market cycles). Practically, you value property using either (1) **DCF of unlevered property cash flows** or (2) **capitalisation (cap-rate) methods**, and then reconcile to **NAV** and financing structure. The key is consistency: cap rates, growth, and reinvestment (capex/leasing costs) must align.

See also: Chapter 12 (terminal value), Chapter 17–20 (multiples as cap-rate analogs), Chapter 15 (WACC/APV), IFRS 16/IAS 40 mechanics (if applicable).

---

### Formula/Methodology

#### 1) Property income: NOI (unlevered operating cash flow proxy)
```text
NOI = Gross Rental Income
    − Vacancy & Credit Loss
    − Operating Expenses (property-level)
    − Property Taxes (if treated as operating)
    + Other Property Income (parking, service fees, etc.)

Notes:
- NOI excludes: depreciation, interest, corporate overhead, income taxes.
- Keep NOI consistent with cap-rate comps (some markets quote “NOI before capex” vs “NOI after reserves”).
```
Common variants (must define):
- NOI (pre-reserves) vs NOI (after reserves for replacements).
- EBITDAre (REIT convention) vs NOI (property convention).

#### 2) Capitalisation and cap rates
```text
Cap Rate = NOI_1 / Property Value

Property Value (cap method) = NOI_1 / Cap Rate
```
Where:
- NOI_1 is forward (next 12 months) stabilised NOI (not trailing if lease-up is underway).
- Cap rate reflects required unlevered return net of growth and risk expectations (market-based).

Cap-rate decomposition (intuition):
```text
Cap Rate ≈ Required Unlevered Return − Expected NOI Growth
```
Use as a check:
- If you assume high growth, cap rate should be lower (all else equal).
- If cap rate is high, either growth is low or risk is high (or both).

#### 3) DCF for property (unlevered)
Unlevered free cash flow to the property (before debt):
```text
UFCF_t = NOI_t
       − Capital Expenditures (maintenance + value-add)
       − Leasing/tenant improvement costs
       − Leasing commissions
       ± Change in working capital (often small)
```
Then:
```text
Property Value = Σ [ UFCF_t / (1 + r_u)^t ] + Terminal Value

Terminal Value (exit cap) = NOI_{T+1} / Exit Cap Rate
or
Terminal Value (perpetuity) = UFCF_{T+1} / (r_u − g)
```
Critical consistency:
- If terminal uses exit cap, NOI and cap definitions must match market practice.
- If terminal uses perpetuity, r_u must be greater than g and reinvestment must be sustainable.

#### 4) Stabilised vs in-place vs as-is value
Define the state of cash flows:
```text
In-place value: current leases/cash flows (with expiries and step-ups)
Stabilised value: assumes normalised occupancy and market rent levels
As-is value: current state including near-term capex/lease-up required
```
For vacancy/lease-up, DCF is usually superior to a single cap-rate method.

#### 5) NAV logic (portfolio / REIT style)
```text
NAV (equity) = Market Value of Properties
             + Other assets
             − Net Debt
             − Other claims (preferred, minorities)
```
Then:
```text
NAV per share = NAV equity / Diluted shares
```
NAV is a bridge between property values and equity valuation.

#### 6) Development and value-add (option-like risk)
Development value is highly path-dependent:
- construction cost overruns,
- timing delays,
- leasing risk,
- exit cap-rate risk.
Use scenario analysis with probabilities:
```text
Expected Value = Σ [ p_s × Value_s ]
```
Avoid single-point values where risks are binary.

---

### Python Implementation
```python
from typing import List, Optional, Dict, Tuple
import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def noi(
    gross_rent: float,
    vacancy_rate: float,
    operating_expenses: float,
    other_income: float = 0.0
) -> float:
    """Compute NOI using a simple structure."""
    for name, v in {"gross_rent": gross_rent, "vacancy_rate": vacancy_rate, "operating_expenses": operating_expenses, "other_income": other_income}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if gross_rent < 0 or operating_expenses < 0 or other_income < 0:
        raise ValueError("gross_rent, operating_expenses, other_income must be >= 0")
    if not (0.0 <= vacancy_rate < 1.0):
        raise ValueError("vacancy_rate must be in [0,1)")
    effective_rent = gross_rent * (1.0 - vacancy_rate)
    return effective_rent + other_income - operating_expenses


def cap_rate_value(noi_forward: float, cap_rate: float) -> Optional[float]:
    """Value = NOI_1 / cap_rate. Returns None if cap_rate <= 0."""
    if not all(_num(v) for v in [noi_forward, cap_rate]):
        raise ValueError("inputs must be numeric")
    if noi_forward < 0:
        raise ValueError("noi_forward must be >= 0")
    if cap_rate <= 0:
        return None
    return noi_forward / cap_rate


def pv_cash_flows(cash_flows: List[float], r: float) -> float:
    """PV of cash flows discounted at r."""
    if not _num(r) or r <= -1:
        raise ValueError("r must be numeric and > -100%")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values")
        pv += cf / ((1.0 + r) ** t)
    return pv


def terminal_value_exit_cap(noi_next: float, exit_cap_rate: float) -> Optional[float]:
    """TV = NOI_{T+1}/Exit Cap. Returns None if invalid."""
    if not all(_num(v) for v in [noi_next, exit_cap_rate]):
        raise ValueError("inputs must be numeric")
    if noi_next < 0:
        raise ValueError("noi_next must be >= 0")
    if exit_cap_rate <= 0:
        return None
    return noi_next / exit_cap_rate


def property_value_dcf_exit_cap(
    ufcf: List[float],
    r_u: float,
    noi_t: float,
    noi_growth: float,
    exit_cap_rate: float
) -> float:
    """Unlevered property DCF with exit cap terminal."""
    if not isinstance(ufcf, list) or len(ufcf) == 0:
        raise ValueError("ufcf must be a non-empty list")
    if not all(_num(v) for v in [r_u, noi_t, noi_growth, exit_cap_rate]):
        raise ValueError("inputs must be numeric")
    if r_u <= -1:
        raise ValueError("r_u must be > -100%")
    if not ( -0.5 < noi_growth < 0.5 ):
        raise ValueError("noi_growth seems out of bounds; check units")
    if exit_cap_rate <= 0:
        raise ValueError("exit_cap_rate must be > 0")
    if noi_t < 0:
        raise ValueError("noi_t must be >= 0")

    pv_stage = pv_cash_flows(ufcf, r_u)
    T = len(ufcf)

    noi_next = noi_t * (1.0 + noi_growth)
    tv = terminal_value_exit_cap(noi_next, exit_cap_rate)
    if tv is None:
        raise ValueError("terminal value could not be computed")
    pv_tv = tv / ((1.0 + r_u) ** T)

    return pv_stage + pv_tv


def nav_equity(property_values: float, other_assets: float, net_debt: float, other_claims: float = 0.0) -> float:
    """NAV equity = property values + other assets - net debt - other claims."""
    for name, v in {"property_values": property_values, "other_assets": other_assets, "net_debt": net_debt, "other_claims": other_claims}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return property_values + other_assets - net_debt - other_claims


def expected_value_scenarios(values: List[Tuple[float, float]]) -> float:
    """Expected value from (probability, value) pairs."""
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("values must be a non-empty list")
    total_p = 0.0
    ev = 0.0
    for p, v in values:
        if not _num(p) or not _num(v):
            raise ValueError("pairs must be numeric")
        if not (0.0 <= p <= 1.0):
            raise ValueError("probabilities must be in [0,1]")
        total_p += p
        ev += p * v
    if total_p > 1.0001:
        raise ValueError("sum of probabilities exceeds 1; check inputs")
    return ev


# Example usage (all $m)
gross_rent = 120.0
vacancy = 0.08
opex = 35.0
other_inc = 5.0

noi1 = noi(gross_rent, vacancy, opex, other_inc)
cap = 0.055
cap_val = cap_rate_value(noi1, cap)
print(f"NOI_1: ${noi1:,.1f}m")
print(f"Cap-rate value: ${cap_val:,.0f}m" if cap_val else "Cap-rate value: n/a")

# DCF with value-add capex and leasing costs (simplified)
ufcf = [noi1 - 10.0, noi1 - 8.0, noi1 - 6.0, noi1 - 4.0, noi1 - 3.0]  # includes capex/lease-up costs
r_u = 0.075
noi_t = noi1  # last year NOI base for terminal
g_noi = 0.03
exit_cap = 0.058

dcf_val = property_value_dcf_exit_cap(ufcf, r_u, noi_t, g_noi, exit_cap)
print(f"DCF property value: ${dcf_val:,.0f}m")

# NAV bridge for a simple portfolio
properties = dcf_val
other_assets = 20.0
net_debt = 180.0
other_claims = 15.0

nav = nav_equity(properties, other_assets, net_debt, other_claims)
shares = 100.0  # million
print(f"NAV equity: ${nav:,.0f}m, NAV/share: ${nav/shares:,.2f}")

# Development scenario valuation (diagnostic)
dev_ev = expected_value_scenarios([
    (0.20, 0.0),     # fail
    (0.50, 350.0),   # base
    (0.30, 500.0),   # upside
])
print(f"Expected development value: ${dev_ev:,.0f}m")
```

---

### Valuation Impact
Why this matters:
- NOI definitions drive cap-rate comparability. A 50 bps cap-rate error can create large value swings; mis-defining NOI can be equivalent to mis-estimating cap rate.
- DCF is essential for lease-up, vacancies, redevelopment, and properties with non-stabilised cash flows; cap-rate snapshots often misprice transitional assets.
- NAV provides a clean bridge from asset values to equity value, but only if net debt and claims (preferred, minorities, lease-like obligations) are fully captured.

Impact on multiples:
- Cap rate is the real-estate analog of an earnings yield (1 / multiple). Lower cap rate implies higher multiple and higher implied growth/quality.
- Comparing cap rates across markets requires controlling for lease length, tenant credit, and capex/reserves assumptions.

Impact on DCF inputs:
- Discount rate should be unlevered and property-risk consistent (location, tenant, lease structure).
- Terminal value typically dominates; exit cap should be consistent with long-run assumptions (no “high growth + high exit cap” contradictions).

Practical adjustments:
```python
def implied_cap_rate(value: float, noi_forward: float) -> Optional[float]:
    """Implied cap rate = NOI_1 / value."""
    if not all(_num(v) for v in [value, noi_forward]):
        raise ValueError("inputs must be numeric")
    if value <= 0:
        return None
    if noi_forward < 0:
        raise ValueError("noi_forward must be >= 0")
    return noi_forward / value
```

---

### Quality of Earnings Flags (Real Estate Cash Flows)
⚠️ NOI inflated by under-reserving capex (roof/HVAC, replacements) or ignoring leasing costs.  
⚠️ Vacancy assumptions too low vs market; rent roll not reconciled to market rents.  
⚠️ Tenant concentration risk (single tenant, weak credit) not reflected in discount rate or exit cap.  
⚠️ Rent growth assumes above-inflation increases without demand evidence; re-leasing downtime ignored.  
⚠️ One-time concessions (free rent, TI allowances) excluded from effective rent.  
✅ Green flag: NOI reconciled to rent roll, market rent comps, and includes recurring capex/leasing costs consistently.

---

### Sector-Specific Considerations
| Property Type | Key Issue | Typical Treatment |
|---|---|---|
| Office | Lease-up and capex risk | DCF with downtime, TI/LC; conservative exit cap. |
| Retail | Tenant credit + turnover | Higher discount rate; model percentage rent if relevant. |
| Industrial/logistics | Strong demand but cap-rate sensitivity | Use market exit cap ranges; check growth assumptions. |
| Multifamily | High turnover, operating cost inflation | Model renewals and expenses; rent control scenarios if relevant. |
| Hotels | Highly cyclical cash flows | Use through-cycle NOI and scenario analysis; cap-rate snapshot risky. |

---

### Practical Comparison Tables

#### 1) Cap-rate vs DCF: when to use what
| Situation | Preferred method | Why |
|---|---|---|
| Stabilised, long leases, prime tenant | Cap-rate + DCF cross-check | Market pricing reliable |
| Vacancy/lease-up underway | DCF | Cash flows transitional |
| Redevelopment/value-add | DCF + scenarios | Cap-rate ignores capex timing and risk |
| Portfolio/REIT | NAV | Bridges asset values to equity claims |
| Highly cyclical (hotel) | Through-cycle DCF | Single-year NOI misleading |

#### 2) NOI definition consistency
| Item | Include in NOI? | Why it matters |
|---|---|---|
| Property operating expenses | Yes | core to ongoing cash generation |
| Corporate overhead | Usually no | belongs above property level |
| Maintenance capex reserves | Market-dependent | affects “NOI after reserves” comparability |
| Leasing commissions/TI | Often excluded from NOI but included in UFCF | must appear somewhere |
| Vacancy & credit loss | Yes | avoids overstating effective rent |

---

### Real-World Example
Scenario: A value-add office asset has 8% vacancy and near-term leasing costs. You compute NOI_1, estimate value via cap rate, then use a 5-year DCF with capex/leasing costs and an exit cap. You reconcile to NAV and compute NAV per share.

```python
# Outputs printed by the example code above:
# - NOI_1
# - Cap-rate value
# - DCF property value
# - NAV equity and NAV/share
# - Expected development value (scenario)
```
Interpretation: If cap-rate value materially exceeds DCF value, the cap-rate method likely assumes stabilisation without accounting for capex/leasing costs or uses an overly aggressive cap rate. Use the DCF to make the stabilisation assumptions explicit and defensible.
