# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 30: Valuing Equity in Distressed Firms (Distress Costs, Default Risk, Restructuring Value, Claimholder Waterfall)

### Core Concept
Distressed firms require valuing the business while explicitly recognising **default risk**, **distress costs**, and the **priority of claims**. Practically, equity can be worth near-zero even when enterprise value is positive if debt claims absorb value. A defensible distressed valuation separates (1) operating value under survival/restructuring scenarios, (2) the effect of distress on cash flows and discount rates, and (3) the claimholder waterfall (debt, preferred, pensions, leases, etc.).

See also: Chapter 12 (terminal value), Chapter 15 (APV), Chapter 22 (money-losing firms), Chapter 25 (takeovers), Chapter 33 (scenario analysis), Chapter 16 (per-share value).

---

### Formula/Methodology

#### 1) Distress is about probability + consequences
Distress affects value through:
```text
A) Higher probability of default (PD)
B) Lower cash flows (operating disruption)
C) Higher financing costs / tighter constraints
D) Direct and indirect distress costs
```
The valuation must reflect both likelihood and impact.

#### 2) Expected value framework (probability-weighted)
A clean approach is probability-weighted enterprise value:
```text
Expected EV = Σ [ p_s × EV_s ]

Where:
- scenarios s include survival/restructuring/liquidation outcomes
- probabilities p_s sum to 1
```
Then equity value is derived from EV using the claimholder waterfall per scenario.

#### 3) Distress costs (explicit subtraction)
Distress costs can be modelled as cash flow reductions and/or one-off restructuring costs:
```text
FCFF_t(distressed) = FCFF_t(normal) − Distress Cost_t
```
Common distress cost categories:
- Lost customers, supplier tightening (indirect)
- Legal/advisory and restructuring fees (direct)
- Forced asset sales (value leakage)
- Underinvestment due to liquidity constraints (long-run value loss)

#### 4) Valuing equity in distress: waterfall logic
Equity is a residual claim:
```text
Equity Value = max(0, Enterprise Value − Debt-like Claims − Other Senior Claims)

More generally:
Recoveries follow priority:
1) Secured debt
2) Senior unsecured
3) Subordinated
4) Preferred
5) Common equity
```
In practice you must include hidden debt-like claims:
- leases, pensions deficits, environmental provisions, litigation, deferred taxes (context-dependent).

#### 5) Restructuring value vs liquidation value
Distress options:
```text
Liquidation: value based on asset sale proceeds (often discounted, time + fire-sale)
Restructuring: enterprise survives with new capital structure and reduced distress costs
```
If restructuring is plausible, value can increase materially relative to liquidation, but equity may still be wiped out depending on bargaining and priority.

#### 6) Probability of default and discounting choices
Two common approaches:
```text
Approach A (probability-weighted CF): discount at “going concern” rate; multiply CF by survival probabilities.
Approach B (higher discount rate): embed default risk in discount rate.
```
Practical preference:
- Use probability-weighted cash flows when default risk is high and discrete (transparent and auditable).
- Avoid “cranking WACC” without modelling recovery; it can misstate equity value and distort terminal value.

#### 7) Debt valuation and implied recovery
If you need to value debt claims (to solve for equity):
```text
Value of Debt = Σ [ Expected Coupon/Principal Payments / (1 + r)^t ] + Expected Recovery PV
Recovery = Recovery Rate × Face Value (or based on enterprise value in default)
```
Even a simple recovery assumption can prevent equity being overstated.

---

### Python Implementation
```python
from typing import List, Dict, Tuple, Optional
import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def expected_value_scenarios(scenarios: List[Tuple[float, float]]) -> float:
    """Expected value from (probability, value) pairs."""
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        raise ValueError("scenarios must be a non-empty list")
    total_p = 0.0
    ev = 0.0
    for p, v in scenarios:
        if not _num(p) or not _num(v):
            raise ValueError("probability and value must be numeric")
        if not (0.0 <= p <= 1.0):
            raise ValueError("probability must be in [0,1]")
        total_p += p
        ev += p * v
    if total_p > 1.0001:
        raise ValueError("sum of probabilities exceeds 1; check inputs")
    return ev


def pv_cash_flows(cash_flows: List[float], r: float, survival: Optional[List[float]] = None) -> float:
    """PV of cash flows with optional survival probabilities per period."""
    if not _num(r) or r <= -1:
        raise ValueError("r must be numeric and > -100%")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list")
    if survival is not None:
        if len(survival) != len(cash_flows):
            raise ValueError("survival must match cash_flows length")
        for p in survival:
            if not _num(p) or not (0.0 <= p <= 1.0):
                raise ValueError("survival probabilities must be in [0,1]")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values")
        p = 1.0 if survival is None else survival[t - 1]
        pv += (p * cf) / ((1.0 + r) ** t)
    return pv


def terminal_value_perpetuity(cf_next: float, r: float, g: float) -> float:
    """TV = CF_{n+1}/(r-g)."""
    if not all(_num(v) for v in [cf_next, r, g]):
        raise ValueError("inputs must be numeric")
    if r <= g:
        raise ValueError("Need r > g for finite TV")
    return cf_next / (r - g)


def equity_value_from_ev(ev: float, senior_claims: float) -> float:
    """Equity as residual: max(0, EV - senior_claims)."""
    if not all(_num(v) for v in [ev, senior_claims]):
        raise ValueError("inputs must be numeric")
    return max(0.0, ev - senior_claims)


def waterfall_recovery(ev: float, claims: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Allocate enterprise value down the priority waterfall.
    claims: list of (claim_name, claim_amount) ordered from most senior to most junior.
    Returns recoveries per claim; residual goes to last claimant if included (e.g., equity).
    """
    if not _num(ev):
        raise ValueError("ev must be numeric")
    if ev < 0:
        raise ValueError("ev must be >= 0")
    if not isinstance(claims, list) or len(claims) == 0:
        raise ValueError("claims must be a non-empty list")

    remaining = ev
    rec = {}
    for name, amt in claims:
        if not isinstance(name, str) or name.strip() == "":
            raise ValueError("claim names must be non-empty strings")
        if not _num(amt) or amt < 0:
            raise ValueError("claim amounts must be numeric and >= 0")
        pay = min(remaining, amt)
        rec[name] = pay
        remaining -= pay
    rec["_unallocated"] = remaining
    return rec


def distressed_equity_probability_weighted(
    scenario_evs: List[Tuple[float, float]],
    senior_claims: float
) -> float:
    """Expected equity = sum(p * max(0, EV_s - senior_claims))."""
    if not _num(senior_claims) or senior_claims < 0:
        raise ValueError("senior_claims must be numeric and >= 0")
    total_p = 0.0
    exp_eq = 0.0
    for p, ev in scenario_evs:
        if not _num(p) or not _num(ev):
            raise ValueError("scenario inputs must be numeric")
        if not (0.0 <= p <= 1.0):
            raise ValueError("probabilities must be in [0,1]")
        total_p += p
        exp_eq += p * equity_value_from_ev(ev, senior_claims)
    if total_p > 1.0001:
        raise ValueError("sum of probabilities exceeds 1; check inputs")
    return exp_eq


# Example usage (all $m)
# Scenario EVs: survival/restructure/liquidation
scenario_evs = [
    (0.55, 2_600.0),  # survive/restructure
    (0.25, 1_900.0),  # weak restructure / partial asset sales
    (0.20, 1_200.0),  # liquidation / distressed sale
]

# Senior claims includes debt-like claims (debt + leases + pensions + provisions as appropriate)
senior_claims = 2_050.0

exp_ev = expected_value_scenarios(scenario_evs)
exp_eq = distressed_equity_probability_weighted(scenario_evs, senior_claims)

print(f"Expected EV: ${exp_ev:,.0f}m")
print(f"Expected equity (residual, scenario-weighted): ${exp_eq:,.0f}m")

# Waterfall illustration for one scenario (survival case)
ev_survive = 2_600.0
claims = [
    ("Secured Debt", 900.0),
    ("Senior Unsecured", 900.0),
    ("Leases (debt-like)", 120.0),
    ("Pensions/Other", 130.0),
    ("Subordinated", 200.0),
    ("Preferred", 50.0),
    ("Common Equity", 10_000.0),  # placeholder to capture residual as recovery
]
rec = waterfall_recovery(ev_survive, claims)
print("Waterfall recoveries (survival scenario):")
for k, v in rec.items():
    if k != "_unallocated":
        print(f"  {k}: ${v:,.0f}m")
print(f"  Residual unallocated: ${rec['_unallocated']:,.0f}m")
```

---

### Valuation Impact
Why this matters:
- Distressed valuations frequently overstate equity by valuing the firm as a going concern without subtracting senior claims and without modelling default outcomes.
- Recovery and priority drive equity outcomes; two firms with the same EV can have radically different equity values depending on leverage and off-balance-sheet claims.
- Probability-weighting separates operating risk from default risk and avoids inconsistent terminal values that arise from simply raising discount rates.

Impact on multiples:
- Multiples based on current earnings are often meaningless in distress (earnings depressed, one-offs, covenant-driven behaviour).
- If using EV/EBITDA in distress, normalise EBITDA (remove restructuring charges) and adjust EV for debt-like obligations; interpret as a restructuring valuation, not a steady-state trading comp.
- Distressed sale comps embed forced-sale discounts; do not mix with going-concern comps without adjustment.

Impact on DCF inputs:
- Distress costs reduce FCFF and can compress margins and reinvestment capacity; model explicitly rather than assuming normalisation without cost.
- Terminal assumptions must reflect post-restructure economics (capital structure, reinvestment) and must still satisfy r > g.

Practical adjustments:
```python
def senior_claims_bridge(debt: float, leases_debt_like: float = 0.0, pensions: float = 0.0, provisions: float = 0.0, other: float = 0.0) -> float:
    """Build a debt-like senior claims estimate."""
    for name, v in {"debt": debt, "leases_debt_like": leases_debt_like, "pensions": pensions, "provisions": provisions, "other": other}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
        if v < 0:
            raise ValueError(f"{name} must be >= 0")
    return debt + leases_debt_like + pensions + provisions + other
```

---

### Quality of Earnings Flags (Distressed Firms)
⚠️ EBITDA “normalisation” removes recurring costs (maintenance capex, working capital) to make covenants look passable.  
⚠️ One-off restructuring charges repeat year after year (not truly non-recurring).  
⚠️ Aggressive working capital actions near reporting date (stretch payables, factoring receivables) to show liquidity.  
⚠️ Capitalised costs increase materially (development costs, commissions) to support earnings.  
⚠️ Off-balance-sheet obligations (leases, pensions, provisions) omitted from senior claims.  
✅ Green flag: transparent bridge from reported to normalised operating cash flow, and explicit claimholder waterfall in valuation.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Retail | Lease obligations dominate | Treat leases as debt-like; use store-level cash flows and closure costs. |
| Airlines/transport | Cyclical + asset values | Scenario analysis with liquidation values; consider fleet marketability. |
| Construction/contracting | Working capital volatility | Model contract assets/liabilities and cash conversion carefully. |
| Energy | Asset abandonment + commodity risk | Include decommissioning obligations; use mid-cycle scenarios. |
| Financials | Different capital structure | Use equity-based valuation and regulatory capital constraints; distress mechanics differ. |

---

### Practical Comparison Tables

#### 1) Distressed valuation approaches
| Approach | Best when | Key output | Main risk |
|---|---|---|---|
| Scenario-weighted EV + waterfall | Default is meaningful | Expected equity value | Probabilities subjective |
| Going-concern DCF with distress costs | Business likely survives | EV and post-restructure value | Understates default risk |
| Liquidation / break-up | Asset sale likely | Liquidation value | Over/under-estimating fire-sale discounts |
| Market-based (distressed comps) | Many comparable distress deals | Range | Mixed deal terms and forced-sale bias |

#### 2) Hidden senior claims checklist
| Item | Why it matters | Typical handling |
|---|---|---|
| Leases | Debt-like fixed claims | Capitalise or treat as debt-like PV |
| Pensions | Contractual/priority features | Add deficit as claim; check priority |
| Provisions (legal/env) | Cash drain and priority | Include expected PV |
| Factoring / supply chain finance | Hidden leverage | Reclassify to debt-like |
| Guarantees | Contingent claims | Scenario-based expected value |

---

### Real-World Example
Scenario: A distressed firm has three possible outcomes: restructure successfully, partial restructure, or liquidation. You compute expected EV and then expected equity as the scenario-weighted residual after senior claims. You also illustrate a priority waterfall for the survival scenario to show who gets paid and whether equity receives anything.

```python
# Outputs printed by the example code above:
# - Expected EV
# - Expected equity (scenario-weighted residual)
# - Waterfall recoveries for the survival scenario
```
Interpretation: Equity value is highly sensitive to senior claims and liquidation outcomes. If equity is out of the money in most plausible scenarios, equity should be valued like an option (deep out-of-the-money), and any single-point going-concern DCF that ignores default risk is likely overstating value.
