# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 13: Narrative and Numbers — Story to Value (Value Drivers, Scenario Trees, Assumption Discipline)

### Core Concept
A valuation is a translation of a business **story** (why the company will win/lose) into **numbers** (revenue growth, margins, reinvestment, risk). The practical discipline is to make every key assumption **traceable** to a driver, to quantify uncertainty with **scenarios** (not just point estimates), and to run consistency checks so the story does not violate accounting, economics, or industry constraints.

See also: Chapter 9 (measuring earnings), Chapter 12 (terminal value closure), Chapter 33 (probabilistic approaches: scenarios/trees/simulations), Chapter 17 (relative valuation cross-checks).

---

### Formula/Methodology

#### 1) Value driver translation map (story → inputs)
```text
Revenue_t = Revenue_{t-1} × (1 + Growth_t)

Operating Income_t = Revenue_t × Operating Margin_t

NOPAT_t = Operating Income_t × (1 − Tax Rate_t)

FCFF_t = NOPAT_t + D&A_t − Capex_t − ΔNWC_t
```
Core idea:
- “We will grow faster” must show up as higher Growth_t.
- “We have pricing power” must show up as higher Margin_t (or slower margin fade).
- “We scale efficiently” must show up as lower reinvestment per $ of growth (higher sales-to-capital, lower ΔNWC intensity, lower capex intensity), not just higher margins.

#### 2) Reinvestment and growth consistency (economic constraints)
A practical closure relationship (firm-level, stable economics):
```text
Reinvestment Rate_t = Reinvestment_t / NOPAT_t
Growth_t ≈ Reinvestment Rate_t × ROIC_t

Equivalently:
Reinvestment Rate_t ≈ Growth_t / ROIC_t
```
This prevents “free growth” assumptions (high growth without reinvestment).

#### 3) Scenario-weighted valuation (instead of one story)
```text
Expected Value = Σ p_s × Value_s
Where:
p_s = probability of scenario s (Σ p_s = 1)
Value_s = valuation output under scenario s (EV or Equity)
```
Use scenarios when outcomes are discrete (approval/no approval, platform adoption yes/no, commodity up/down).

#### 4) Sensitivity vs scenario
```text
Sensitivity: vary one input (e.g., WACC ± 1%) holding others fixed
Scenario: change a coherent set of linked drivers (growth, margin, reinvestment, risk) consistent with a story
```
Prefer scenarios for “story risk,” sensitivities for parameter uncertainty.

---

### Python Implementation
```python
from typing import Dict, List, Optional, Tuple


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def project_financials(
    revenue0: float,
    growth: List[float],
    margin: List[float],
    tax_rate: List[float],
    da_pct_rev: List[float],
    capex_pct_rev: List[float],
    nwc_pct_rev: List[float],
) -> List[Dict[str, float]]:
    """
    Project a simple operating model from story-driven inputs.

    Args:
        revenue0: starting revenue at t=0 (currency).
        growth: list of growth rates by year (decimal).
        margin: list of operating margins by year (decimal).
        tax_rate: list of tax rates by year (decimal).
        da_pct_rev: list of D&A as % of revenue (decimal).
        capex_pct_rev: list of capex as % of revenue (decimal).
        nwc_pct_rev: list of NWC as % of revenue (decimal).

    Returns:
        list of dicts per year with revenue, op_income, nopat, da, capex, nwc, delta_nwc, fcff.

    Raises:
        ValueError: for inconsistent lengths or invalid inputs.
    """
    series = [growth, margin, tax_rate, da_pct_rev, capex_pct_rev, nwc_pct_rev]
    n = len(growth)
    if any(len(s) != n for s in series):
        raise ValueError("All input lists must have the same length.")
    if not _num(revenue0) or revenue0 <= 0:
        raise ValueError("revenue0 must be numeric and > 0.")
    for name, arr in [
        ("growth", growth), ("margin", margin), ("tax_rate", tax_rate),
        ("da_pct_rev", da_pct_rev), ("capex_pct_rev", capex_pct_rev), ("nwc_pct_rev", nwc_pct_rev)
    ]:
        for v in arr:
            if not _num(v):
                raise ValueError(f"{name} contains non-numeric values.")
    if any(v < -0.9 for v in growth):
        raise ValueError("growth contains implausibly low values (< -90%).")
    if any((v < -0.5 or v > 0.8) for v in margin):
        raise ValueError("margin values look out of bounds; check units (use decimals).")
    if any((v < 0 or v >= 1) for v in tax_rate):
        raise ValueError("tax_rate must be in [0,1).")
    if any(v < 0 for v in da_pct_rev + capex_pct_rev):
        raise ValueError("da_pct_rev and capex_pct_rev must be >= 0.")
    if any((v < -0.5 or v > 0.8) for v in nwc_pct_rev):
        raise ValueError("nwc_pct_rev values look out of bounds; check units.")

    out = []
    revenue_prev = revenue0
    nwc_prev = revenue0 * nwc_pct_rev[0]  # initial anchor; year-1 not observed in this simple template

    for t in range(n):
        revenue = revenue_prev * (1.0 + growth[t])
        op_income = revenue * margin[t]
        nopat = op_income * (1.0 - tax_rate[t])
        da = revenue * da_pct_rev[t]
        capex = revenue * capex_pct_rev[t]
        nwc = revenue * nwc_pct_rev[t]
        delta_nwc = nwc - nwc_prev
        fcff = nopat + da - capex - delta_nwc

        out.append({
            "year": t + 1,
            "revenue": revenue,
            "op_income": op_income,
            "nopat": nopat,
            "da": da,
            "capex": capex,
            "nwc": nwc,
            "delta_nwc": delta_nwc,
            "fcff": fcff,
        })

        revenue_prev = revenue
        nwc_prev = nwc

    return out


def pv_cash_flows(cash_flows: List[float], discount_rate: float) -> float:
    """PV of a list of cash flows at rate r."""
    if not _num(discount_rate) or discount_rate <= -1:
        raise ValueError("discount_rate must be numeric and > -100%.")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list.")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values.")
        pv += cf / ((1.0 + discount_rate) ** t)
    return pv


def scenario_weighted_value(scenario_values: List[Tuple[str, float, float]]) -> float:
    """
    Expected value from scenarios.

    Args:
        scenario_values: list of (name, probability, value).

    Returns:
        float: expected value.

    Raises:
        ValueError: if probabilities don't sum to 1 or invalid values.
    """
    if not scenario_values:
        raise ValueError("scenario_values must be non-empty.")
    p_sum = 0.0
    ev = 0.0
    for name, p, v in scenario_values:
        if not _num(p) or p < 0 or p > 1:
            raise ValueError("probabilities must be in [0,1].")
        if not _num(v):
            raise ValueError("scenario value must be numeric.")
        p_sum += p
        ev += p * v
    if abs(p_sum - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1 (got {p_sum}).")
    return ev


# Example usage: story-driven model + scenario valuation (all $m)
base = project_financials(
    revenue0=1_000.0,
    growth=[0.12, 0.10, 0.08, 0.06, 0.04],
    margin=[0.18, 0.19, 0.20, 0.20, 0.20],
    tax_rate=[0.24, 0.24, 0.24, 0.24, 0.24],
    da_pct_rev=[0.03]*5,
    capex_pct_rev=[0.04]*5,
    nwc_pct_rev=[0.10]*5,
)

fcff_base = [row["fcff"] for row in base]
pv_base = pv_cash_flows(fcff_base, discount_rate=0.09)
print(f"PV(FCFF) base: ${pv_base:,.0f}m")

# Two coherent scenarios (not single-parameter sensitivities)
bull = project_financials(
    revenue0=1_000.0,
    growth=[0.16, 0.14, 0.10, 0.08, 0.05],
    margin=[0.18, 0.20, 0.22, 0.22, 0.22],
    tax_rate=[0.24]*5,
    da_pct_rev=[0.03]*5,
    capex_pct_rev=[0.045]*5,  # higher reinvestment to support growth
    nwc_pct_rev=[0.11]*5,
)
bear = project_financials(
    revenue0=1_000.0,
    growth=[0.06, 0.04, 0.03, 0.03, 0.03],
    margin=[0.16, 0.16, 0.17, 0.17, 0.17],
    tax_rate=[0.24]*5,
    da_pct_rev=[0.03]*5,
    capex_pct_rev=[0.035]*5,
    nwc_pct_rev=[0.10]*5,
)

pv_bull = pv_cash_flows([r["fcff"] for r in bull], 0.09)
pv_bear = pv_cash_flows([r["fcff"] for r in bear], 0.09)

expected = scenario_weighted_value([
    ("Bull", 0.30, pv_bull),
    ("Base", 0.50, pv_base),
    ("Bear", 0.20, pv_bear),
])

print(f"PV base: ${pv_base:,.0f}m | PV bull: ${pv_bull:,.0f}m | PV bear: ${pv_bear:,.0f}m")
print(f"Scenario-weighted PV: ${expected:,.0f}m")
```

---

### Valuation Impact
Why this matters:
- “Spreadsheet drift” (changing assumptions without a story) creates valuations that cannot be defended. Driver-linked inputs make the valuation auditable and easier to challenge.
- Scenario-weighting is often superior to a single point estimate when outcomes are discrete, and it reduces reliance on arbitrary “conservatism” embedded in one set of assumptions.
- A coherent story-to-value translation improves comparability: you can explain why this firm deserves higher/lower growth, margins, reinvestment, and risk than peers.

Impact on multiples:
- Multiples reflect stories too (market narratives). A driver-based story lets you test whether a high multiple implies credible drivers (growth + margins + reinvestment + risk) or is pure sentiment.
- If the story implies superior ROIC and growth, then higher multiples may be justified; if not, treat high multiples as fragile.

Impact on DCF inputs:
- Growth must be supported by reinvestment and capacity (capex, working capital, customer acquisition costs).
- Margin expansion should be tied to scale, pricing, mix, or cost structure, and should fade if competitive forces suggest it.
- Risk (discount rate) should respond to story uncertainty (regulatory, cyclicality, customer concentration).

Practical adjustments:
```python
def story_consistency_check(growth_rate: float, roic: float) -> None:
    """Fast check that growth is not 'free' given ROIC."""
    if not _num(growth_rate) or not _num(roic):
        raise ValueError("inputs must be numeric")
    if roic <= 0:
        raise ValueError("ROIC must be > 0")
    reinvest_rate = growth_rate / roic
    if reinvest_rate > 1:
        raise ValueError("Growth requires reinvestment > 100% of NOPAT; revise story assumptions.")
```

---

### Quality of Earnings Flags (Story/Numbers Mismatch)
⚠️ High growth story with flat capex/working capital and no explanation (implied free growth).  
⚠️ Margin expansion story without unit economics support (pricing, mix, CAC/LTV, scale economies).  
⚠️ “Adjusted” earnings remove recurring costs (e.g., SBC, ongoing restructuring), inflating the narrative.  
⚠️ Scenario probabilities chosen to “force” a desired value rather than reflecting decision-relevant likelihoods.  
✅ Green flag: every forecast driver has an explicit narrative justification and a peer/industry anchor, and the valuation includes scenario and sensitivity grids.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Start-ups | Binary/step-change outcomes | Scenario trees (product success/fail), staged margins and reinvestment. |
| Cyclicals | Mean reversion dominates | Use mid-cycle margins and working capital dynamics; scenario on cycle depth/length. |
| Regulated utilities | Policy shocks | Scenario on allowed returns, capex plans, regulatory resets. |
| Platforms/marketplaces | Network effects and take rates | Translate narrative into take-rate, retention, CAC/LTV, and margin scaling path. |

---

### Practical Comparison Tables

#### 1) Story-to-driver translation checklist
| Story claim | Driver(s) that must move | Evidence to look for |
|---|---|---|
| “Pricing power” | Higher margin, stable volume | Customer churn, differentiation, competitor pricing |
| “Scale economies” | Margin rises as revenue rises | Gross margin trend, fixed cost leverage |
| “New markets” | Higher growth + higher reinvestment | Salesforce expansion, capex/NWC needs |
| “Moat” | Sustained ROIC > WACC | High switching costs, IP, brand, cost advantage |

#### 2) Scenario vs sensitivity guidance
| Use case | Better tool | Why |
|---|---|---|
| Discrete outcomes | Scenario-weighted | Coherent bundles of drivers |
| Parameter uncertainty | Sensitivity grid | One input at a time (WACC, g) |
| Multiple uncertainties | Simulation | Captures distribution, not just points |

---

### Real-World Example
Scenario: Management claims “international expansion + premium pricing” will drive a step-up in growth and margins. You translate that into growth, margin, and reinvestment paths, build bull/base/bear scenarios, and compute a probability-weighted value.

```python
# Using the example functions above:
# - base: moderate growth and margin improvement
# - bull: faster growth with higher reinvestment
# - bear: slower adoption and weaker margins

# If the resulting expected value is still below market EV, the market is pricing
# either higher probabilities on the bull case or a structurally higher terminal value.
# Your next step is to reconcile: implied drivers vs achievable drivers.
```
