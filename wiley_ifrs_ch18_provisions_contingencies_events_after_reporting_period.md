# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 18: Current Liabilities, Provisions, Contingencies and Events After the Reporting Period (IAS 37, IAS 10, debt-like adjustments, QoE risks)

### Core Concept
IAS 37 governs when to recognise provisions (present obligation from a past event with probable outflow and reliable estimate), when to disclose contingencies, and how to measure obligations. IAS 10 governs adjusting vs non-adjusting events after the reporting period. For valuation, these topics directly affect “net debt” (debt-like provisions), sustainable earnings (recurring vs one-off charges/releases), and reliability of cash flow forecasts (hidden obligations, litigation, onerous contracts).

### Formula/Methodology

#### 1) Recognition: provision vs contingent liability (IAS 37 decision rule)
```text
Recognise a provision if ALL are met:
1) Present obligation (legal or constructive) from a past event
2) Probable outflow of resources (>50% likelihood in practice)
3) Reliable estimate can be made

If obligation is possible (not probable) OR cannot be measured reliably:
- Do NOT recognise; disclose as contingent liability (unless remote)

If no present obligation:
- No provision; no contingent liability (may disclose commitments separately)
```

#### 2) Measurement (best estimate; expected value vs most likely outcome)
```text
Provision = Best estimate of expenditure required to settle the present obligation

Measurement approach:
- Single obligation: most likely outcome may be best estimate
- Large population: expected value (probability-weighted)

Discounting:
If time value of money is material:
Provision = Σ [ Expected cash flow_t / (1 + r)^t ]
Where r is a pre-tax discount rate reflecting time value and liability-specific risks (not double-counted)
Unwinding of discount is recognised as a finance cost.
```

#### 3) Onerous contracts (IAS 37)
```text
Onerous contract provision = Lower of:
A) Cost to fulfil remaining obligation
B) Cost to exit/terminate (penalties/compensation)
minus
Economic benefits expected from the contract
```

#### 4) Restructuring provisions (key constraints)
```text
A restructuring provision requires:
- Detailed formal plan, and
- Valid expectation created in affected parties that the entity will carry out restructuring

Costs included:
- Direct expenditures necessarily entailed by restructuring (not associated with ongoing activities)

Costs excluded:
- Future operating losses
- Training/marketing/investment for future operations
```

#### 5) Contingent assets
```text
Do not recognise contingent assets.
Disclose only when inflow is probable.
Recognise an asset only when realization is virtually certain.
```

#### 6) Events after reporting period (IAS 10)
```text
Adjusting events: provide evidence of conditions existing at reporting date -> adjust financial statements
Non-adjusting events: indicative of conditions arising after reporting date -> disclose if material
Dividends declared after reporting date: generally non-adjusting (disclose, not recognise as liability at reporting date)
```

---

### Practical Application for Valuation Analysts

#### 1) Build a “debt-like provisions” schedule for EV bridges
Many IAS 37 provisions behave like debt (contractual/constructive cash outflows).
Common debt-like candidates (case-by-case):
- Onerous contract provisions
- Decommissioning / asset retirement obligations (often IAS 37/IFRIC 1 linkage)
- Litigation/settlement provisions (where highly probable and cash-timed)
- Environmental remediation provisions

Not always debt-like:
- Warranty provisions tied to revenue (may be operating working capital-like)
- Customer refunds/returns (often linked to revenue recognition; may be operating)

Analyst rule:
- If provision is a financing-like claim with predictable cash timing, treat as debt-like in EV.
- If provision is a normal operating accrual linked to sales/cost of sales, keep within working capital.

#### 2) Quality of earnings: provisions are a major “earnings management” channel
Red flags:
- Large provisions booked in good years (“cookie jar”) then released to boost future profits
- Reclassifications between provisions and other payables to manage leverage metrics
- Frequent “non-recurring” restructuring charges that recur every year

Practical steps:
- Build a provisions roll-forward: opening + new provisions - utilisation ± FX/discount unwind - releases
- Compare utilisation to actual cash outflows in the cash flow statement
- Scrutinise discount unwind: it increases finance costs; treat consistently with debt in valuation

#### 3) Forecasting: align cash flow timing and discount rates
For DCF:
- If you treat a provision as debt-like, ensure the associated cash outflows are excluded from operating FCF (to avoid double counting).
- If you leave it in operating working capital, include expected cash settlement in operating cash flows.

#### 4) Events after reporting period: ensure “as-of date” integrity
Before finalising valuation:
- Check for major post-balance-sheet events (litigation settlement, acquisition, fire, covenant breach, refinancing).
- Adjusting events can change baseline financials; non-adjusting events may require scenario impacts.

#### 5) Onerous contracts and impairment interplay
Onerous contract provisions often arise when expected future cash flows deteriorate.
Valuation action:
- Stress test margins/cash flows and compare to impairment indicators (see IAS 36 chapter).
- Ensure you do not count both an onerous provision and a separate “restructuring adjustment” for the same economics.

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

def present_value_cashflows(cashflows: List[Tuple[float, float]], discount_rate: float) -> float:
    """
    Discount expected cash flows for provision measurement.

    Args:
        cashflows: list of (t_years, expected_cash_outflow) with outflows as positive numbers
        discount_rate: pre-tax discount rate (decimal), must be > -1

    Returns:
        float: present value

    Raises:
        ValueError: invalid inputs.
    """
    r = _num(discount_rate, "discount_rate")
    if r <= -1.0:
        raise ValueError("discount_rate must be > -1.0.")
    pv = 0.0
    for (t, cf) in cashflows:
        tt = _num(t, "t_years")
        c = _num(cf, "expected_cash_outflow")
        if tt < 0:
            raise ValueError("t_years must be >= 0.")
        if c < 0:
            raise ValueError("expected_cash_outflow must be >= 0 (use positive outflows).")
        pv += c / ((1.0 + r) ** tt)
    return pv

def expected_value(outcomes: List[Tuple[float, float]]) -> float:
    """
    Expected value for a large population of obligations.

    Args:
        outcomes: list of (probability, cash_outflow) with probabilities summing to 1.

    Returns:
        float: expected cash outflow

    Raises:
        ValueError: invalid probabilities.
    """
    total_p = 0.0
    ev = 0.0
    for p, cf in outcomes:
        pp = _num(p, "probability")
        c = _num(cf, "cash_outflow")
        if pp < 0 or pp > 1:
            raise ValueError("probability must be between 0 and 1.")
        if c < 0:
            raise ValueError("cash_outflow must be >= 0.")
        total_p += pp
        ev += pp * c
    if not np.isclose(total_p, 1.0, atol=1e-6):
        raise ValueError(f"probabilities must sum to 1. Got {total_p}.")
    return ev

def provision_rollforward(opening: float, new: float, utilised: float, released: float,
                          discount_unwind: float = 0.0, other: float = 0.0) -> float:
    """
    Provision roll-forward.

    Closing = Opening + New + Discount unwind + Other - Utilised - Released

    Args:
        utilised: actual usage/cash settlement during period (>=0)
        released: reversals of unused provisions (>=0)

    Raises:
        ValueError: negative inputs where not allowed.
    """
    o = _num(opening, "opening")
    n = _num(new, "new")
    u = _num(utilised, "utilised")
    rel = _num(released, "released")
    du = _num(discount_unwind, "discount_unwind")
    oth = _num(other, "other")
    if u < 0 or rel < 0:
        raise ValueError("utilised and released must be >= 0.")
    return o + n + du + oth - u - rel

def debt_like_adjustment_for_provisions(selected_provisions: float) -> float:
    """
    Treat selected provisions as debt-like for EV bridge.

    Raises:
        ValueError: negative input.
    """
    v = _num(selected_provisions, "selected_provisions")
    if v < 0:
        raise ValueError("selected_provisions must be >= 0.")
    return v

def classify_event_after_reporting_period(condition_existed_at_reporting_date: bool, material: bool) -> str:
    """
    IAS 10 event classification helper.

    Returns:
        str: 'adjusting', 'non_adjusting_disclose', or 'ignore'

    Raises:
        ValueError: if material is None.
    """
    if material is None:
        raise ValueError("material must be True/False.")
    if condition_existed_at_reporting_date:
        return "adjusting"
    return "non_adjusting_disclose" if material else "ignore"

# Example usage
# Litigation: outcomes and PV timing
lit_ev = expected_value([(0.2, 0.0), (0.5, 25_000_000), (0.3, 60_000_000)])
pv = present_value_cashflows([(1.5, lit_ev)], discount_rate=0.05)
print(f"Expected litigation outflow: ${lit_ev/1e6:.1f}M")
print(f"Provision PV (discounted): ${pv/1e6:.1f}M")

closing = provision_rollforward(opening=120, new=55, utilised=40, released=10, discount_unwind=6)
print(f"Closing provision balance: ${closing:.1f}m")

event_type = classify_event_after_reporting_period(condition_existed_at_reporting_date=False, material=True)
print(f"IAS 10 classification: {event_type}")
```

---

### Valuation Impact
Why this matters:
- Provisions can be effectively hidden leverage; ignoring debt-like provisions understates enterprise value and overstates equity value.
- Provision movements affect reported earnings; releases can inflate margins and mask weakening fundamentals.
- Discounting and unwinding create finance costs and timing effects that must be reflected consistently in DCF and multiples.

Impact on multiples:
- EV/EBITDA: if you add debt-like provisions to EV, ensure EBITDA is not adjusted for related operating costs unless consistent.
- P/E: provision releases can temporarily boost earnings; normalise for sustainable profitability.

Impact on DCF inputs:
- If provisions represent future cash outflows, include them in forecast cash flows OR treat them as debt-like adjustments (not both).
- For long-dated obligations, ensure discount rates and inflation assumptions are aligned (real vs nominal).

Practical adjustments:
```python
def adjust_net_debt_for_debt_like_provisions(net_debt_reported: float, debt_like_provisions: float) -> float:
    """
    Add debt-like provisions to net debt for EV bridge.

    Raises:
        ValueError: negative provisions.
    """
    nd = _num(net_debt_reported, "net_debt_reported")
    p = _num(debt_like_provisions, "debt_like_provisions")
    if p < 0:
        raise ValueError("debt_like_provisions must be >= 0.")
    return nd + p
```

---

### Quality of Earnings Flags
⚠️ Large net releases of provisions boosting profit without corresponding improvement in cash flow.  
⚠️ Repeated “one-off” restructuring provisions every year (suggests recurring cost base issues).  
⚠️ Material contingent liabilities with limited disclosure on probability, range, or timing.  
⚠️ Big increases in provisions while operating cash flow remains strong (potential underinvestment in accrual discipline).  
⚠️ Significant discount unwind (finance cost) not reflected in debt-like adjustments or valuation bridge.  
✅ Transparent roll-forward tables, clear nature of obligations, discount rates used, and sensitivity/range disclosures.

---

### Sector-Specific Considerations

| Sector | Key provision/contingency issue | Typical treatment |
|---|---|---|
| Industrials / Chemicals | Environmental remediation, decommissioning | Treat as debt-like; model long-dated cash flows and discounting consistently |
| Construction / Services | Onerous contracts, warranties | Separate operating provisions (warranty) vs debt-like onerous exits; scrutinise margin forecasts |
| Financial services | Litigation/regulatory provisions | Stress scenarios; consider tail risks and disclosure quality |
| Retail / Consumer | Returns/refunds, restructuring | Evaluate “recurring” restructuring; ensure returns align with revenue recognition |

---

### Real-World Example
Scenario: Company reports net debt of $900m and IAS 37 provisions of $260m. You identify $140m as debt-like (onerous contracts + environmental) and $120m as operating (warranty/returns).

```python
net_debt_reported = 900.0
debt_like = 140.0
adj_net_debt = adjust_net_debt_for_debt_like_provisions(net_debt_reported, debt_like)
print(f"Adjusted net debt: ${adj_net_debt:.0f}m")

# Avoid double counting: if you treat debt_like as net debt, do not subtract the same cash outflows from operating FCF again.
```

Interpretation: Adjusted net debt improves EV comparability, especially versus peers with different provisioning practices.

See also: Chapter 26 (income taxes) for uncertain tax positions and deferred tax; Chapter 13 (impairment) for loss recognition; Chapter 20 (revenue) for refunds/returns provisions; Chapter 24 (financial instruments) for other liabilities and measurement.
