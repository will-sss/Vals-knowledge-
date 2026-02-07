# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 19: Employee Benefits (IAS 19, pensions, service cost vs finance cost, valuation adjustments, QoE)

### Core Concept
IAS 19 requires entities to recognise employee benefit obligations and expenses, including defined contribution plans (expense as incurred) and defined benefit (DB) plans (recognise present value of obligation, fair value of plan assets, and actuarial gains/losses in OCI). For valuation, DB pensions are often large, long-dated, and economically debt-like. They affect EV bridges (pension deficits), earnings quality (service cost vs finance cost), and cash flow forecasting (contributions vs accounting expense).

### Formula/Methodology

#### 1) Defined contribution vs defined benefit (high-level)
```text
Defined contribution (DC):
- Expense = contributions due for the period
- No long-term obligation beyond contributions

Defined benefit (DB):
- Recognise net defined benefit liability/asset:
Net DB position = Present value of DB obligation (DBO) - Fair value of plan assets
- Expense includes service cost + net interest + remeasurements (OCI)
```

#### 2) Present value of defined benefit obligation (concept)
```text
DBO = Σ [ Expected benefit payments_t × probability × discount factor_t ]
Discount rate: high-quality corporate bond yields (or government yields where deep market absent), consistent with currency and term of obligation
```

#### 3) Net interest on the net DB liability/asset
```text
Net interest = Discount rate × Net DB liability (or asset) at start of period
Where:
Net DB liability = DBO - Plan assets (opening balance)
Presentation: finance cost/income (below operating profit) in many analyses
```

#### 4) Pension expense components (practical breakdown)
```text
Profit or loss (P&L):
- Service cost (current + past service cost, settlements/curtailments)
- Net interest on net DB liability/asset

Other comprehensive income (OCI):
- Remeasurements:
  - actuarial gains/losses on DBO
  - return on plan assets excluding interest income
  - changes in asset ceiling (if relevant)
```

#### 5) Pension deficit as debt-like claim (valuation bridge)
```text
Adjusted EV = EV + Pension deficit (debt-like) - Pension surplus (if accessible, case-by-case)
Adjusted net debt = Reported net debt + Pension deficit (if treated as debt-like)

Key: avoid double counting (if you add deficit to EV, do not also subtract pension contributions from FCF unless consistent with your cash flow definition)
```

---

### Practical Application for Valuation Analysts

#### 1) Treat DB pension deficit as debt-like unless clearly immaterial
Practical rule:
- If DB pension deficit is material relative to market cap/EV, treat it as debt-like for EV and leverage metrics.
- Use net DB liability (DBO minus plan assets) as starting point, but verify:
  - Funding status and contribution schedule
  - Asset allocation and risk (equity-heavy assets increase volatility)
  - Discount rate sensitivity and longevity assumptions

#### 2) Separate operating performance from pension financing effects
For operating comparability:
- Treat service cost as operating (it is compensation).
- Treat net interest as finance cost (like debt interest).
This improves comparability between firms with different pension structures.

Practical adjustment:
- If reported EBIT includes pension interest (rare), reclassify.
- If EBITDA excludes pension service cost (via “adjusted” metrics), ensure consistent peer policy.

#### 3) Cash flow forecasting: contributions vs IAS 19 expense
IAS 19 expense ≠ cash contributions.
For DCF:
- Base cash flows on expected contributions (cash funding schedule), not accounting expense, if pension is material and has a known funding plan.
- Alternatively, treat pension deficit as debt-like and assume contributions are part of financing cash flows (requires consistency).

#### 4) Actuarial assumptions are a major QoE risk
Key sensitivities:
- Discount rate (bps changes can swing DBO)
- Salary growth (for final salary plans)
- Longevity assumptions
- Inflation indexation
Valuation action:
- Perform sensitivity analysis and consider downside scenario if assumptions look aggressive.

#### 5) M&A and carve-outs
Pension obligations can be retained by seller or transferred.
Valuation action:
- Confirm which entity retains the obligation.
- Adjust EV bridge and forecast contributions accordingly.

---

### Python Implementation
```python
from typing import Any, Dict, List, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def pv_benefit_payments(payments: List[Tuple[float, float]], discount_rate: float) -> float:
    """
    Present value of expected benefit payments (simplified DBO proxy).

    Args:
        payments: list of (t_years, expected_payment) with payments as positive amounts
        discount_rate: annual discount rate (decimal), must be > -1

    Returns:
        float: present value

    Raises:
        ValueError: invalid inputs.
    """
    r = _num(discount_rate, "discount_rate")
    if r <= -1.0:
        raise ValueError("discount_rate must be > -1.0.")
    pv = 0.0
    for t, p in payments:
        tt = _num(t, "t_years")
        pay = _num(p, "expected_payment")
        if tt < 0:
            raise ValueError("t_years must be >= 0.")
        if pay < 0:
            raise ValueError("expected_payment must be >= 0.")
        pv += pay / ((1.0 + r) ** tt)
    return pv

def net_db_position(dbo: float, plan_assets: float) -> float:
    """
    Net defined benefit liability (positive) or asset (negative).

    Net DB = DBO - Plan assets

    Raises:
        ValueError: negative inputs not allowed for dbo/plan assets.
    """
    d = _num(dbo, "dbo")
    a = _num(plan_assets, "plan_assets")
    if d < 0 or a < 0:
        raise ValueError("dbo and plan_assets must be >= 0.")
    return d - a

def net_interest(opening_net_db: float, discount_rate: float) -> float:
    """
    Net interest on net DB liability/asset:
      net interest = discount_rate × opening_net_db

    Returns:
        float: interest expense if opening_net_db > 0, interest income if < 0
    """
    nd = _num(opening_net_db, "opening_net_db")
    r = _num(discount_rate, "discount_rate")
    return nd * r

def pension_ev_adjustment(ev: float, net_db_liability: float, treat_as_debt_like: bool = True) -> float:
    """
    Adjust EV for pension deficit if treated as debt-like.

    Args:
        ev: enterprise value
        net_db_liability: DBO - plan assets (positive deficit, negative surplus)

    Returns:
        float: adjusted EV

    Raises:
        ValueError: invalid inputs.
    """
    E = _num(ev, "ev")
    nd = _num(net_db_liability, "net_db_liability")
    if not treat_as_debt_like:
        return E
    # Add deficit; handle surplus cautiously (access restrictions may apply)
    return E + max(nd, 0.0)

def reclassify_operating_profit(ebit_reported: float, pension_net_interest_included: float = 0.0) -> float:
    """
    If reported EBIT includes pension net interest, remove it to make EBIT operating.

    Args:
        pension_net_interest_included: amount included in EBIT (positive if expense included; negative if income)

    Returns:
        float: EBIT operating proxy
    """
    e = _num(ebit_reported, "ebit_reported")
    pni = _num(pension_net_interest_included, "pension_net_interest_included")
    return e - pni

# Example usage
# Simplified PV of benefits (not a full actuarial model)
benefits = [(5, 12_000_000), (6, 12_500_000), (7, 13_000_000), (8, 13_500_000)]
dbo_proxy = pv_benefit_payments(benefits, discount_rate=0.045)
assets = 180_000_000
net_db = net_db_position(dbo_proxy, assets)
print(f"DBO proxy PV: ${dbo_proxy/1e6:.1f}M")
print(f"Net DB position (deficit + / surplus -): ${net_db/1e6:.1f}M")

ni = net_interest(opening_net_db=net_db, discount_rate=0.045)
print(f"Net interest (annual, approx): ${ni/1e6:.1f}M")

adj_ev = pension_ev_adjustment(ev=2_900_000_000, net_db_liability=net_db, treat_as_debt_like=True)
print(f"Adjusted EV (pension deficit added): ${adj_ev/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- DB pension deficits are economically similar to long-dated debt and can materially change enterprise value and leverage.
- Pension accounting can distort operating profitability if finance components are embedded in operating lines or if companies promote “adjusted” metrics.
- Cash contributions drive value; IAS 19 expense can diverge significantly from cash funding.

Impact on multiples:
- EV/EBITDA: add pension deficits to EV where treated as debt-like; ensure EBITDA is not simultaneously “adjusted” in a way that double counts pension costs.
- P/E: pension remeasurements bypass P&L (OCI) but can foreshadow future contributions and risk; do not ignore large OCI swings.

Impact on DCF inputs:
- If modelling FCFF, treat pension service cost as operating cost; model cash contributions separately if material.
- If adding pension deficit to EV as debt-like, keep related contributions in financing or exclude from operating FCF to avoid double counting (choose one consistent framework).

Practical adjustments:
```python
def adjust_net_debt_for_pension(net_debt_reported: float, net_db_deficit: float) -> float:
    """
    Add pension deficit to net debt for EV bridge comparability.

    Raises:
        ValueError: deficit negative.
    """
    nd = _num(net_debt_reported, "net_debt_reported")
    deficit = _num(net_db_deficit, "net_db_deficit")
    if deficit < 0:
        raise ValueError("net_db_deficit must be >= 0 for this adjustment.")
    return nd + deficit
```

---

### Quality of Earnings Flags
⚠️ Large pension deficits disclosed but excluded from “net debt” without explanation.  
⚠️ Aggressive discount rate assumptions (high rates reduce DBO) relative to market yields.  
⚠️ Plan assets heavily allocated to equities/illiquid assets while deficit is large (volatility risk).  
⚠️ Repeated “one-off” curtailment/settlement gains boosting profit.  
⚠️ Significant OCI pension remeasurements ignored in narrative and forecasting.  
✅ Clear disclosure of assumptions, sensitivity tables (discount rate, inflation, longevity), and a credible contribution schedule.

---

### Sector-Specific Considerations

| Sector | Typical IAS 19 exposure | Typical valuation handling |
|---|---|---|
| Industrials / Legacy corporates | Large DB plans, closed to new entrants | Treat deficit as debt-like; stress test discount rate and asset return assumptions |
| Airlines / Transport | Material DB and post-employment benefits | Scenario analysis for funding contributions and downturn resilience |
| Financial services | Often smaller DB, but still material for some | Ensure classification consistency and regulatory capital considerations |
| Tech / Growth | Limited DB, mostly DC and SBC | IAS 19 less material; focus more on IFRS 2 dilution |

---

### Real-World Example
Scenario: Company has EV of $5.0bn and a net DB pension deficit of $600m (DBO $2.1bn, plan assets $1.5bn). You value on EV/EBITDA and want comparability versus peers without DB plans.

```python
ev = 5_000.0
dbo = 2_100.0
assets = 1_500.0
deficit = net_db_position(dbo, assets)  # in $m
adj_ev = ev + max(deficit, 0.0)

print(f"Reported EV: ${ev:.0f}m")
print(f"Pension deficit: ${deficit:.0f}m")
print(f"Adjusted EV: ${adj_ev:.0f}m")
```

Interpretation: Adjusted EV provides a cleaner view of enterprise claims; peers without DB obligations are more comparable.

See also: Chapter 16 (equity and capital stack) for EV-to-equity bridges; Chapter 18 (provisions) for other long-dated debt-like obligations; Chapter 25 (fair value) for plan asset measurement and valuation of obligations.
