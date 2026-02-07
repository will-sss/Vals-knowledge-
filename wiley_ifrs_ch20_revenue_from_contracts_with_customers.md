# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 20: Revenue from Contracts with Customers (IFRS 15, revenue quality, timing, margins, working capital)

### Core Concept
IFRS 15 sets a single model for recognising revenue from contracts with customers based on transfer of control and the entity’s performance obligations. For valuation, IFRS 15 affects the timing of revenue and profit recognition, contract assets/liabilities (working capital), and the comparability of margins across companies with different contract structures (software/SaaS, construction, OEMs, services). Misapplication can create material quality-of-earnings risks.

### Formula/Methodology

#### 1) IFRS 15 five-step model (application checklist)
```text
Step 1: Identify the contract(s) with a customer
Step 2: Identify the performance obligations (POs)
Step 3: Determine the transaction price (TP)
Step 4: Allocate TP to POs (based on stand-alone selling prices)
Step 5: Recognise revenue when (or as) each PO is satisfied (over time or at a point in time)
```

#### 2) Transaction price and variable consideration
```text
Transaction Price (TP) = Fixed consideration
                       + Estimated variable consideration (subject to constraint)
                       + Other adjustments (significant financing component, non-cash consideration, consideration payable to customer)

Variable consideration estimate:
- Expected value (probability-weighted) OR Most likely amount
Constraint: include only amounts that are highly probable not to result in a significant revenue reversal
```

#### 3) Allocation to performance obligations (stand-alone selling price method)
```text
Allocated revenue to PO_i = TP × (SSP_i / Σ SSP_all)

Where:
TP = total transaction price
SSP_i = stand-alone selling price of performance obligation i
```

#### 4) Over-time recognition (progress toward completion)
```text
Revenue recognized to date = TP_allocated × % complete

% complete (input method example) = Costs incurred to date / Total expected costs
% complete (output method example) = Units delivered / Total units promised

Key: choose method that best depicts transfer of control and avoid including inefficiencies that do not depict performance.
```

#### 5) Contract assets and contract liabilities (working capital)
```text
Contract asset: entity has transferred goods/services but does not yet have an unconditional right to consideration (e.g., unbilled revenue)
Contract liability: entity has received consideration (or has unconditional right) before transferring goods/services (e.g., deferred revenue)

Movement in contract assets/liabilities affects operating working capital and cash conversion.
```

---

### Practical Application for Valuation Analysts

#### 1) Diagnose revenue recognition pattern and map to cash
For each major revenue stream:
- Identify whether revenue is recognised over time or point-in-time
- Understand billing terms vs revenue recognition (creates contract assets/liabilities)
- Link contract liability run-rate to future revenue visibility (especially SaaS)

Valuation actions:
- For DCF, build a bridge from recognised revenue to billings/cash collections:
  - Change in contract liabilities is often a leading indicator for growth in subscription models.
  - Rising contract assets can inflate revenue without equivalent cash.

#### 2) Normalise revenue and margins for comparability
Common comparability issues:
- Principal vs agent (gross vs net revenue)
- Bundled arrangements (hardware + service + software)
- Capitalisation of contract costs (incremental costs of obtaining a contract; fulfilment costs)
Analyst adjustments:
- Restate revenue on a consistent gross/net basis for comps where material.
- Separate recurring vs non-recurring revenue; treat one-time implementation differently from subscription where relevant.
- Evaluate impact of capitalised contract costs on margins (amortisation vs immediate expense).

#### 3) Variable consideration is a key QoE risk
Areas:
- Discounts, rebates, returns, refunds
- Performance bonuses/penalties
- Usage-based fees
Valuation actions:
- Compare estimates vs subsequent reversals; look for repeated downward true-ups.
- Check whether constraint is applied conservatively; aggressive estimates boost near-term revenue.

#### 4) Significant financing components (SFC)
If timing between transfer and payment is significant:
- Recognise interest income/expense separately from revenue.
Valuation actions:
- Remove financing component from operating revenue/EBITDA to avoid overstating operating performance.
- Treat financing effect as interest-like (affects WACC bridge).

#### 5) Contract modifications and multi-year deals
Contract modifications can create step-changes in revenue:
- Separate contract vs modification; reallocate TP; cumulative catch-up adjustments may occur.
Valuation actions:
- Identify “catch-up” revenue as non-recurring; adjust run-rate.

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

def allocate_transaction_price(transaction_price: float, stand_alone_selling_prices: Dict[str, float]) -> Dict[str, float]:
    """
    Allocate transaction price to performance obligations based on relative stand-alone selling prices.

    Args:
        transaction_price: total consideration (TP)
        stand_alone_selling_prices: mapping of PO name -> SSP

    Returns:
        dict: PO -> allocated TP

    Raises:
        ValueError: if SSP sum <= 0 or any SSP negative.
    """
    tp = _num(transaction_price, "transaction_price")
    if tp < 0:
        raise ValueError("transaction_price must be >= 0.")
    if not stand_alone_selling_prices:
        raise ValueError("stand_alone_selling_prices cannot be empty.")
    ssp_sum = 0.0
    for k, v in stand_alone_selling_prices.items():
        vv = _num(v, f"SSP[{k}]")
        if vv < 0:
            raise ValueError(f"SSP for {k} must be >= 0.")
        ssp_sum += vv
    if ssp_sum <= 0:
        raise ValueError("Sum of SSPs must be > 0.")
    return {k: tp * (_num(v, f"SSP[{k}]") / ssp_sum) for k, v in stand_alone_selling_prices.items()}

def revenue_over_time_input_method(allocated_tp: float, costs_incurred_to_date: float, total_expected_costs: float,
                                   exclude_inefficiency_costs: float = 0.0) -> float:
    """
    Over-time revenue using an input method (% complete = costs incurred / total expected).

    Args:
        allocated_tp: transaction price allocated to the PO
        costs_incurred_to_date: cumulative costs incurred
        total_expected_costs: total expected costs to satisfy PO
        exclude_inefficiency_costs: costs that should not depict performance (optional)

    Returns:
        float: cumulative revenue recognised to date

    Raises:
        ValueError: invalid costs or totals.
    """
    tp = _num(allocated_tp, "allocated_tp")
    ci = _num(costs_incurred_to_date, "costs_incurred_to_date")
    te = _num(total_expected_costs, "total_expected_costs")
    ex = _num(exclude_inefficiency_costs, "exclude_inefficiency_costs")
    if tp < 0:
        raise ValueError("allocated_tp must be >= 0.")
    if ci < 0 or te <= 0:
        raise ValueError("costs_incurred_to_date must be >= 0 and total_expected_costs must be > 0.")
    if ex < 0 or ex > ci:
        raise ValueError("exclude_inefficiency_costs must be between 0 and costs_incurred_to_date.")
    progress = (ci - ex) / te
    progress = max(0.0, min(progress, 1.0))
    return tp * progress

def variable_consideration_expected_value(outcomes: List[Tuple[float, float]]) -> float:
    """
    Expected value estimate for variable consideration.

    Args:
        outcomes: list of (probability, amount)

    Returns:
        float: expected amount

    Raises:
        ValueError: probabilities must sum to 1.
    """
    total_p = 0.0
    ev = 0.0
    for p, amt in outcomes:
        pp = _num(p, "probability")
        a = _num(amt, "amount")
        if pp < 0 or pp > 1:
            raise ValueError("probability must be between 0 and 1.")
        total_p += pp
        ev += pp * a
    if not np.isclose(total_p, 1.0, atol=1e-6):
        raise ValueError(f"probabilities must sum to 1. Got {total_p}.")
    return ev

def contract_balance_rollforward(opening: float, additions: float, billings: float, cash_collections: float = 0.0) -> float:
    """
    Simplified roll-forward for contract assets or liabilities.

    For contract asset (unbilled receivable):
      Closing = Opening + Revenue recognized but not billed (additions) - Billings

    For contract liability (deferred revenue):
      Closing = Opening + Billings - Revenue recognized (use sign convention)

    This function is generic; define additions/billings consistently.

    Raises:
        ValueError: invalid input.
    """
    o = _num(opening, "opening")
    a = _num(additions, "additions")
    b = _num(billings, "billings")
    c = _num(cash_collections, "cash_collections")
    # cash_collections included only if user wants a check; not used in balance calculation
    return o + a - b

# Example usage
ssp = {"hardware": 600.0, "installation": 200.0, "subscription": 700.0}
alloc = allocate_transaction_price(transaction_price=1_200.0, stand_alone_selling_prices=ssp)
print("Allocation:", {k: round(v, 1) for k, v in alloc.items()})

# Over-time service: revenue to date using input method
rev_to_date = revenue_over_time_input_method(
    allocated_tp=alloc["installation"],
    costs_incurred_to_date=60.0,
    total_expected_costs=120.0,
    exclude_inefficiency_costs=0.0
)
print(f"Installation revenue recognised to date: ${rev_to_date:.1f}")

# Variable consideration
vc = variable_consideration_expected_value([(0.6, 50.0), (0.3, 20.0), (0.1, 0.0)])
print(f"Expected variable consideration: ${vc:.1f}")
```

---

### Valuation Impact
Why this matters:
- Revenue timing affects growth narratives, margins, and forecast baselines; aggressive recognition can inflate near-term performance and terminal value assumptions.
- Contract liabilities (deferred revenue) can indicate future revenue visibility in subscription businesses; contract assets can indicate revenue recognised ahead of billing/cash.
- Principal vs agent judgements can swing revenue scale and margin structure, affecting comparables.

Impact on multiples:
- EV/Revenue: sensitive to gross vs net presentation; ensure consistent principal/agent basis across comps.
- EV/EBITDA: contract cost capitalisation and revenue timing can shift reported margins; normalise if peers differ.

Impact on DCF inputs:
- Working capital: model contract liabilities/assets explicitly if material (especially SaaS, construction).
- Growth: distinguish bookings/billings from recognised revenue; use billings as leading indicator where relevant.

Practical adjustments:
```python
def normalize_revenue_gross_to_net(reported_revenue: float, agent_net_factor: float) -> float:
    """
    Convert gross revenue to a net presentation for comparability (simplified).

    Args:
        agent_net_factor: between 0 and 1 (net = gross × factor)

    Raises:
        ValueError: invalid factor.
    """
    rev = _num(reported_revenue, "reported_revenue")
    f = _num(agent_net_factor, "agent_net_factor")
    if not (0.0 <= f <= 1.0):
        raise ValueError("agent_net_factor must be between 0 and 1.")
    return rev * f
```

---

### Quality of Earnings Flags
⚠️ Large growth in contract assets (unbilled revenue) without matching cash collections.  
⚠️ Frequent “cumulative catch-up” adjustments or reversals of variable consideration.  
⚠️ Aggressive principal vs agent classification leading to inflated revenue scale.  
⚠️ Significant financing components embedded in revenue (should be separated into interest).  
⚠️ High capitalisation of contract acquisition/fulfilment costs inflating margins.  
✅ Stable revenue recognition policies with clear disclosures, consistent billings-to-revenue relationship, and transparent contract balance roll-forwards.

---

### Sector-Specific Considerations

| Sector | Key IFRS 15 issue | Typical treatment |
|---|---|---|
| SaaS / Software | Multiple POs (licence, support, subscription), deferred revenue | Contract liabilities are key; treat billings as leading indicator; assess churn/renewals |
| Construction / Engineering | Over-time recognition, input method, claims/variations | Scrutinise cost-to-complete assumptions and variable consideration constraint |
| Retail / Consumer | Returns, loyalty programs | Ensure refunds/returns and loyalty obligations reflected; watch for revenue reversals |
| Marketplaces / Platforms | Principal vs agent | Ensure gross vs net comparability in EV/Revenue screens |

---

### Real-World Example
Scenario: Company sells a bundled contract (hardware + installation + 3-year subscription). You need to allocate consideration and understand working capital implications.

```python
contract = {
    "transaction_price": 1_200.0,
    "ssp": {"hardware": 600.0, "installation": 200.0, "subscription": 700.0}
}

alloc = allocate_transaction_price(contract["transaction_price"], contract["ssp"])
hardware_rev = alloc["hardware"]  # point-in-time (delivery)
install_rev_to_date = revenue_over_time_input_method(
    allocated_tp=alloc["installation"],
    costs_incurred_to_date=80.0,
    total_expected_costs=160.0
)

print(f"Hardware revenue (delivery): ${hardware_rev:.1f}")
print(f"Installation revenue to date: ${install_rev_to_date:.1f}")

# Deferred revenue proxy: if billed upfront for subscription, contract liability increases
opening_deferred = 300.0
billings_sub = alloc["subscription"]  # billed
rev_recognized_sub_year1 = alloc["subscription"] / 3.0
closing_deferred = opening_deferred + billings_sub - rev_recognized_sub_year1
print(f"Closing deferred revenue (proxy): ${closing_deferred:.1f}")
```

Interpretation: Deferred revenue can signal future recognised revenue; rising unbilled revenue can signal timing risk if cash does not follow.

See also: Chapter 6 (cash flows) for cash vs revenue reconciliation; Chapter 8 (inventories) for cost recognition; Chapter 18 (provisions) for onerous contracts; Chapter 28 (segments) for revenue disaggregation and comparability.
