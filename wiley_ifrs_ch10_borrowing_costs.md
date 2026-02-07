# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 10: Borrowing Costs (IAS 23, capitalization, interest normalization, project economics)

### Core Concept
IAS 23 requires capitalisation of borrowing costs that are directly attributable to the acquisition, construction, or production of a qualifying asset. For valuation, IAS 23 can shift interest expense out of profit or loss into the balance sheet (PPE, intangibles, inventory for certain long-cycle production), inflating EBITDA/EBIT during build periods and affecting invested capital, depreciation, and future cash flows.

### Formula/Methodology

#### 1) Qualifying asset definition (practical)
```text
Qualifying asset = an asset that necessarily takes a substantial period of time to get ready for its intended use or sale
Examples: large construction projects, certain internally developed intangibles, long-cycle inventory (e.g., wine, aircraft manufacturing)
Non-examples: assets ready for use when acquired, inventories produced in short time frames
```

#### 2) Capitalised borrowing costs (basic concept)
```text
Capitalised Borrowing Costs (period) = Eligible Borrowing Costs incurred - Investment income on temporary investment of borrowings (if applicable)
Eligible Borrowing Costs:
- Specific borrowings: actual interest on specific project debt, net of related income
- General borrowings: weighted-average capitalisation rate × qualifying expenditures
```

#### 3) Capitalisation rate (general borrowings)
```text
Capitalisation Rate = Weighted Average Cost of General Borrowings
Where:
Weighted Average Cost = sum(Interest_i) / sum(Principal_i) for borrowings outstanding during the period (time-weighted if needed)
```

#### 4) Avoidable interest (project expenditure approach)
```text
Avoidable Interest = Capitalisation Rate × Average Accumulated Expenditures (AAE)
Where:
AAE = time-weighted average of qualifying expenditures during the period
```

#### 5) Carrying amount effects
```text
Asset cost increases by capitalised interest, then impacts future P&L via:
- Higher depreciation/amortisation (if depreciable asset)
- Potential impairment risk if project underperforms
```

---

### Practical Application for Valuation Analysts

#### 1) Normalise operating earnings during build periods
Capitalising borrowing costs can:
- Increase EBIT (because interest is not expensed)
- Improve EBITDA/interest coverage temporarily
- Reduce reported interest expense and increase “project asset” carrying value

For peer multiples:
- Consider whether to treat capitalised borrowing costs as operating or financing for comparability.
- A common approach in enterprise valuation: keep EBITDA/EBIT “operating,” but ensure project economics are captured via capex and future depreciation.

#### 2) Link to capex and project cash flows
If a company is in heavy build-out:
- The economic cost of capital during construction is real; it affects project IRR and value.
- In DCF, ensure you capture construction period capex and the timing of cash flows; do not double-count capitalised interest if your capex already reflects total cash spend including interest.

Practical rule:
- If you forecast cash capex (cash paid), capitalised interest should not be added again to project investment.
- If you forecast accounting capex (additions to PPE per note), reconcile to cash capex.

#### 3) Detect policy choices and risk signals
IAS 23 requires suspension of capitalisation when active development is interrupted and cessation when substantially complete. Red flags include:
- Capitalisation continuing despite delays or halted projects
- Large capitalised interest with weak disclosure of rates and qualifying expenditures
- Capitalised interest masking weak profitability in core operations

---

### Python Implementation
```python
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def weighted_average_borrowing_rate(borrowings: List[Dict[str, Any]]) -> float:
    """
    Compute weighted average borrowing rate for general borrowings.

    Args:
        borrowings: list of dicts with keys:
            - principal: outstanding principal (positive)
            - annual_interest_rate: nominal annual rate as decimal (e.g., 0.06)
            - time_weight: fraction of year outstanding in the period (0..1), optional (default 1)

    Returns:
        float: weighted average rate for the period.

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(borrowings, list) or len(borrowings) == 0:
        raise ValueError("borrowings must be a non-empty list.")
    total_principal = 0.0
    total_interest = 0.0

    for i, b in enumerate(borrowings):
        if not isinstance(b, dict):
            raise ValueError(f"Borrowing {i} must be a dict.")
        p = _num(b.get("principal"), f"principal[{i}]")
        r = _num(b.get("annual_interest_rate"), f"annual_interest_rate[{i}]")
        tw = _num(b.get("time_weight", 1.0), f"time_weight[{i}]")
        if p <= 0:
            raise ValueError("principal must be > 0.")
        if r < 0:
            raise ValueError("annual_interest_rate must be >= 0.")
        if not (0.0 <= tw <= 1.0):
            raise ValueError("time_weight must be between 0 and 1.")

        total_principal += p * tw
        total_interest += p * r * tw

    if total_principal <= 0:
        raise ValueError("Total time-weighted principal must be > 0.")
    return total_interest / total_principal

def average_accumulated_expenditures(expenditures: List[Dict[str, Any]]) -> float:
    """
    Compute average accumulated expenditures (AAE) using time-weighted expenditures.

    Args:
        expenditures: list of dicts with keys:
            - amount: expenditure incurred (positive)
            - fraction_of_year_outstanding: fraction of period the expenditure is outstanding (0..1)

    Returns:
        float: AAE

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(expenditures, list) or len(expenditures) == 0:
        raise ValueError("expenditures must be a non-empty list.")
    aae = 0.0
    for i, e in enumerate(expenditures):
        if not isinstance(e, dict):
            raise ValueError(f"Expenditure {i} must be a dict.")
        amt = _num(e.get("amount"), f"amount[{i}]")
        f = _num(e.get("fraction_of_year_outstanding"), f"fraction_of_year_outstanding[{i}]")
        if amt < 0:
            raise ValueError("expenditure amount must be >= 0.")
        if not (0.0 <= f <= 1.0):
            raise ValueError("fraction_of_year_outstanding must be between 0 and 1.")
        aae += amt * f
    return aae

def capitalised_borrowing_costs_general(aae: float, capitalisation_rate: float, investment_income: float = 0.0) -> float:
    """
    Capitalised borrowing costs for general borrowings.

    Capitalised = AAE * capitalisation_rate - investment_income

    Args:
        aae: average accumulated expenditures
        capitalisation_rate: weighted average borrowing rate (decimal)
        investment_income: income from temporary investment of borrowings (positive reduces capitalised amount)

    Returns:
        float: capitalised borrowing costs (floor at 0)

    Raises:
        ValueError: invalid inputs.
    """
    a = _num(aae, "aae")
    r = _num(capitalisation_rate, "capitalisation_rate")
    ii = _num(investment_income, "investment_income")
    if a < 0:
        raise ValueError("aae must be >= 0.")
    if r < 0:
        raise ValueError("capitalisation_rate must be >= 0.")
    if ii < 0:
        raise ValueError("investment_income must be >= 0.")
    cap = a * r - ii
    return max(0.0, cap)

def normalize_interest_expense(reported_interest_expense: float, capitalised_interest: float) -> float:
    """
    Adjust interest expense to an 'economic' interest cost by adding back capitalised interest.

    Args:
        reported_interest_expense: interest expense in P&L
        capitalised_interest: interest capitalised into assets

    Returns:
        float: normalized interest expense

    Raises:
        ValueError: invalid inputs.
    """
    ie = _num(reported_interest_expense, "reported_interest_expense")
    ci = _num(capitalised_interest, "capitalised_interest")
    if ci < 0:
        raise ValueError("capitalised_interest must be >= 0.")
    return ie + ci

# Example usage
borrowings = [
    {"principal": 600_000_000, "annual_interest_rate": 0.055, "time_weight": 1.0},
    {"principal": 300_000_000, "annual_interest_rate": 0.065, "time_weight": 0.5},
]
cap_rate = weighted_average_borrowing_rate(borrowings)

expenditures = [
    {"amount": 200_000_000, "fraction_of_year_outstanding": 0.75},
    {"amount": 150_000_000, "fraction_of_year_outstanding": 0.40},
    {"amount": 100_000_000, "fraction_of_year_outstanding": 0.20},
]
aae = average_accumulated_expenditures(expenditures)

cap_int = capitalised_borrowing_costs_general(aae, cap_rate, investment_income=2_000_000)

print(f"Capitalisation rate: {cap_rate:.2%}")
print(f"AAE: ${aae/1e6:.1f}M")
print(f"Capitalised borrowing costs: ${cap_int/1e6:.1f}M")

reported_ie = 38_000_000
normalized_ie = normalize_interest_expense(reported_ie, cap_int)
print(f"Reported interest expense: ${reported_ie/1e6:.1f}M")
print(f"Normalized interest expense: ${normalized_ie/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- Capitalised borrowing costs can **inflate EBIT/EBITDA** and improve coverage ratios during construction, affecting multiples and credit screens.
- It increases **invested capital** (asset base), lowering ROIC mechanically unless returns increase commensurately.
- It shifts expenses to future periods via depreciation/amortisation, affecting forecasts and terminal value assumptions.

Impact on multiples:
- EV/EBITDA and EV/EBIT may look cheaper during build phases; adjust for capitalised interest when comparing to peers that expense more financing costs (or have fewer qualifying assets).

Impact on DCF inputs:
- If your DCF uses **cash capex**, do not add capitalised interest again; it is already part of financing/cash cost depending on how capex is defined.
- For project valuation, capitalised interest affects the project’s “all-in” investment base and therefore IRR/NPV if you model on accounting capex.

Practical adjustments:
```python
def adjust_ebit_for_capitalised_interest(reported_ebit: float, capitalised_interest: float) -> float:
    """
    Approximate adjustment for comparability: treat capitalised interest as period cost.
    Adjusted EBIT = Reported EBIT - Capitalised Interest

    Note:
      This is a heuristic for peer comparability, not a substitute for full project cash flow modelling.

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(reported_ebit, "reported_ebit")
    ci = _num(capitalised_interest, "capitalised_interest")
    if ci < 0:
        raise ValueError("capitalised_interest must be >= 0.")
    return e - ci
```

---

### Quality of Earnings Flags
⚠️ Capitalised interest rising while projects are delayed or not “actively” developed (risk of non-compliance and earnings management).  
⚠️ High capitalised interest with limited disclosure of capitalisation rate and qualifying expenditure base.  
⚠️ Sudden drop in interest expense without debt reduction, driven by higher capitalisation (presentation-driven improvement).  
✅ Clear disclosure of capitalisation rate, qualifying asset categories, and total capitalised borrowing costs with consistent policy.

---

### Sector-Specific Considerations

| Sector | Typical IAS 23 exposure | Typical valuation treatment |
|---|---|---|
| Infrastructure / Utilities | Long construction periods | Model construction cash flows explicitly; scrutinise capitalised interest and project completion risk |
| Real Estate Development | Qualifying assets common | Use project-level cash flows or NAV; avoid relying on EBIT inflated by capitalisation |
| Manufacturing (long-cycle) | Aircraft/ship/build-to-order | Ensure inventory qualification assessed; normalise margins for capitalisation practices |
| Technology | Usually low | Watch for internally generated assets; capitalised interest may be small but still distorts comparability |

---

### Real-World Example
Scenario: A company is constructing a plant and capitalises $20m of borrowing costs. You are valuing on EV/EBIT and want a run-rate EBIT.

```python
reported_ebit = 260.0
capitalised_interest = 20.0

ebit_run_rate = adjust_ebit_for_capitalised_interest(reported_ebit, capitalised_interest)
print(f"Reported EBIT: ${reported_ebit:.1f}m")
print(f"Run-rate EBIT (treat cap. interest as expense): ${ebit_run_rate:.1f}m")
```

Interpretation: During build phases, adjusted EBIT provides a more conservative peer-comparable measure; the definitive approach is modelling the project cash flows and commissioning timing.

See also: Chapter 9 (PPE) for capitalised cost mechanics and depreciation; Chapter 13 (impairment) for project underperformance risk; Chapter 6 (cash flows) for interest and capex cash classification.
