# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 6: Financial Analysis of Leasing (lease vs buy, lease PV, IFRS 16 adjustments, valuation impacts)

### Core Concept
Leasing is a financing decision that changes the timing and classification of cash flows and reported earnings. For valuation, the key is to treat leases consistently across companies and over time: either capitalise operating leases (treat as debt-like) and adjust EBITDA/EBIT, or keep reported numbers but adjust enterprise value (EV) and cash flows to avoid mixing accounting treatments.

### Formula/Methodology

#### 1) Present value of lease payments (lease liability proxy)
```text
PV_lease = Σ_{t=1..N} Payment_t / (1 + r)^t

Where:
Payment_t = contractual lease payment in period t (use post-incentive, fixed portion; treat variable lease payments separately)
r = discount rate (incremental borrowing rate or lease rate implicit in the lease)
N = number of periods
```

#### 2) Lease vs buy (NPV comparison framework)
```text
NPV_buy = -PurchasePrice + Σ_{t=1..N} (After-tax operating savings_t) / (1 + r)^t + After-tax residual value / (1 + r)^N

NPV_lease = Σ_{t=1..N} (After-tax lease benefit_t) / (1 + r)^t

Decision rule:
Choose the alternative with the higher NPV (or lower PV cost).
```

Practical notes:
- “After-tax lease benefit” often reflects tax deductibility of lease payments.
- For comparability, use a consistent discount rate basis (usually pre-tax or after-tax coherently).

#### 3) IFRS 16-style capitalisation mechanics (simplified)
If you treat operating leases as capitalised:
```text
Lease liability (opening) ≈ PV_lease
Interest expense_t = r × Liability_(t-1)
Liability_t = Liability_(t-1) + Interest expense_t - Payment_t

ROU depreciation_t (simplified) = ROU asset / Useful life (often straight-line)
```

#### 4) EBITDA adjustment (when capitalising operating leases)
```text
Reported EBITDA includes operating lease expense.
If capitalised:
Adjusted EBITDA = Reported EBITDA + Operating lease expense (fixed portion)

Because:
Lease expense is replaced by depreciation (below EBITDA) and interest (below EBIT).
```

#### 5) Enterprise value bridge (consistency)
```text
If you capitalise leases as debt-like:
EV (operating) = Market cap + Gross debt + Lease liability - Cash (plus other claims)

Then use:
EV / EBITDA where EBITDA is lease-adjusted to match EV definition.
```

---

### Practical Application (How to handle leases in valuation work)

#### Step 1: Decide the valuation consistency approach
Two consistent approaches:

**A) Capitalise leases (recommended for cross-company comparability)**
- Add lease liability to debt-like items (affects EV and leverage).
- Add back operating lease expense to EBITDA and EBIT (then replace with depreciation/interest in below-the-line analysis if needed).

**B) Do not capitalise leases**
- Keep reported EBITDA/EBIT.
- Use reported EV definition consistently (do not add lease liability).
- This is less comparable across IFRS 16 adopters / disclosure quality differences.

#### Step 2: Build a lease liability estimate (if disclosures are partial)
- Prefer disclosed lease liabilities under IFRS 16.
- If only operating lease commitments disclosed (legacy): estimate PV using a chosen discount rate and approximate timing (front-loaded vs straight-line).

#### Step 3: Identify which lease payments to include
Include:
- fixed payments (including in-substance fixed)
- fixed escalators where contractual

Exclude / treat separately:
- variable payments linked to sales/usage (operating cost)
- short-term/low-value leases if immaterial

#### Step 4: Adjust metrics used in multiples and credit analysis
- EV/EBITDA: adjust both EV (add leases) and EBITDA (add back lease expense).
- Net debt/EBITDA: include leases in debt-like numerator, ensure EBITDA is lease-adjusted.
- Interest cover: if you capitalise leases, include imputed lease interest in interest expense for coverage ratios.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def pv_lease_payments(payments: List[float], discount_rate: float) -> float:
    """Present value of lease payments.

    Args:
        payments: lease payments for periods 1..N (currency, typically annual)
        discount_rate: discount rate per period (decimal)

    Returns:
        float: present value (currency)

    Raises:
        ValueError: invalid inputs.
    """
    r = _num(discount_rate, "discount_rate")
    if r <= -0.99:
        raise ValueError("discount_rate must be > -0.99.")
    if not payments:
        raise ValueError("payments must not be empty.")
    pv = 0.0
    for t, p in enumerate(payments, start=1):
        pay = _num(p, f"payments[{t}]")
        pv += pay / ((1.0 + r) ** t)
    return pv

def lease_amortization_schedule(
    payments: List[float],
    discount_rate: float,
    opening_liability: Optional[float] = None
) -> List[Dict[str, float]]:
    """Build a simple lease liability amortization schedule.

    Args:
        payments: lease payments for periods 1..N
        discount_rate: rate per period (decimal)
        opening_liability: if None, set to PV of payments

    Returns:
        List[Dict[str,float]]: schedule with period, opening, interest, payment, closing

    Raises:
        ValueError: invalid inputs.
    """
    r = _num(discount_rate, "discount_rate")
    if not payments:
        raise ValueError("payments must not be empty.")
    if r <= -0.99:
        raise ValueError("discount_rate must be > -0.99.")

    liab = pv_lease_payments(payments, r) if opening_liability is None else _num(opening_liability, "opening_liability")
    if liab < 0:
        raise ValueError("opening_liability must be >= 0.")

    sched = []
    opening = liab
    for i, p in enumerate(payments, start=1):
        pay = _num(p, f"payments[{i}]")
        interest = opening * r
        closing = opening + interest - pay
        # guard against tiny negative due to rounding
        if closing < -1e-6:
            raise ValueError("Schedule produced negative liability; check payments/discount_rate.")
        closing = max(0.0, closing)
        sched.append({
            "period": float(i),
            "opening_liability": float(opening),
            "interest": float(interest),
            "payment": float(pay),
            "closing_liability": float(closing),
        })
        opening = closing
    return sched

def lease_adjusted_ebitda(reported_ebitda: float, operating_lease_expense: float) -> float:
    """Add back operating lease expense to reported EBITDA.

    Args:
        reported_ebitda: EBITDA as reported (currency)
        operating_lease_expense: fixed/committed lease expense (currency, >=0)

    Returns:
        float: lease-adjusted EBITDA

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(reported_ebitda, "reported_ebitda")
    le = _num(operating_lease_expense, "operating_lease_expense")
    if le < 0:
        raise ValueError("operating_lease_expense must be >= 0.")
    return e + le

def lease_adjusted_ev(market_cap: float, gross_debt: float, cash: float, lease_liability: float,
                     other_claims: float = 0.0) -> float:
    """Compute an EV definition that treats leases as debt-like.

    EV = market cap + gross debt + lease liability + other claims - cash
    """
    mc = _num(market_cap, "market_cap")
    d = _num(gross_debt, "gross_debt")
    c = _num(cash, "cash")
    ll = _num(lease_liability, "lease_liability")
    oc = _num(other_claims, "other_claims")
    if mc < 0 or d < 0 or ll < 0:
        raise ValueError("market_cap, gross_debt, lease_liability must be >= 0.")
    if c < 0:
        raise ValueError("cash must be >= 0.")
    return mc + d + ll + oc - c

def ev_to_ebitda_multiple(ev: float, ebitda: float) -> float:
    """Compute EV/EBITDA multiple with denominator checks."""
    ev = _num(ev, "ev")
    e = _num(ebitda, "ebitda")
    if e == 0:
        raise ValueError("ebitda must not be 0.")
    return ev / e

# Example usage (illustrative)
payments = [55_000_000, 55_000_000, 55_000_000, 55_000_000, 55_000_000]  # 5-year lease
r = 0.06

lease_liab = pv_lease_payments(payments, r)
sched = lease_amortization_schedule(payments, r)

reported_ebitda = 220_000_000
lease_expense = 55_000_000
adj_ebitda = lease_adjusted_ebitda(reported_ebitda, lease_expense)

ev_adj = lease_adjusted_ev(market_cap=3_000_000_000, gross_debt=900_000_000, cash=200_000_000, lease_liability=lease_liab)
multiple = ev_to_ebitda_multiple(ev_adj, adj_ebitda)

print(f"Lease liability (PV): ${lease_liab/1e6:.0f}M")
print(f"Lease-adjusted EBITDA: ${adj_ebitda/1e6:.0f}M")
print(f"Lease-adjusted EV/EBITDA: {multiple:.1f}x")
print(f"First period schedule row: {sched[0]}")
```

---

### Valuation Impact
Why this matters:
- Leases can materially change EV, leverage, and EBITDA, driving comparability issues in EV/EBITDA and credit metrics.
- In DCF, inconsistent lease treatment can double-count or omit financing-like cash flows.

Impact on multiples:
- If one company capitalises leases (IFRS 16) and another has different disclosure or treatment, unadjusted EV/EBITDA is not comparable.
- Lease-heavy sectors (retail, logistics, airlines) can look “low leverage” on book metrics but are highly leveraged once leases are capitalised.

Impact on DCF inputs:
- FCFF should be unlevered; if leases are treated as debt, interest-like portions belong below FCFF (financing).
- If you add lease liabilities to EV, ensure cash flows/EBITDA align (lease expense add-back).

Comparability issues across companies:
- Differences in lease term assumptions, discount rates, and variable lease components can change reported lease liabilities.
- Consider using a consistent discount rate proxy when re-estimating liabilities from commitments.

Practical adjustments:
```python
def lease_consistency_check(ev: float, ebitda: float, lease_adjusted: bool) -> str:
    """Flag potential inconsistency between EV definition and EBITDA basis."""
    if lease_adjusted:
        return "Ensure EV includes lease liability AND EBITDA adds back lease expense."
    return "Ensure EV excludes lease liability AND EBITDA is reported (includes lease expense)."
```

---

### Quality of Earnings Flags
⚠️ Large differences between disclosed lease commitments and recognised lease liabilities without clear explanation (term, discount rate, variable leases).  
⚠️ “Adjusted EBITDA” adds back lease expense but EV is not adjusted for leases (inconsistent multiple).  
⚠️ Lease classification choices that appear aggressive (short terms, high variable components) to suppress liabilities.  
✅ Transparent disclosure of lease payments split (fixed vs variable) and discount rate assumptions; consistent metric definitions.

---

### Sector-Specific Considerations

| Sector | Key lease issue | Typical handling |
|---|---|---|
| Retail | store leases dominate liabilities | capitalise leases; compare lease-adjusted EV/EBITDA and fixed-charge cover |
| Airlines | aircraft leases; complex terms | include lease liabilities; separate maintenance reserves; stress residuals |
| Logistics | warehouse/fleet leases | capitalise; verify term assumptions and renewal options |
| Software | usually low lease intensity | adjustments often immaterial; focus on SBC and revenue recognition |

---

### Real-World Example
Scenario: Two retailers have identical reported EV/EBITDA, but one has $500M of lease liabilities and higher lease expense. Normalise to compare.

```python
company_a = {"market_cap": 2_500_000_000, "debt": 700_000_000, "cash": 150_000_000, "lease_liab": 500_000_000,
             "ebitda_rep": 180_000_000, "lease_expense": 70_000_000}

ev_a = lease_adjusted_ev(company_a["market_cap"], company_a["debt"], company_a["cash"], company_a["lease_liab"])
ebitda_a = lease_adjusted_ebitda(company_a["ebitda_rep"], company_a["lease_expense"])
print(f"Lease-adjusted EV/EBITDA: {ev_to_ebitda_multiple(ev_a, ebitda_a):.1f}x")
```

Interpretation: Lease-adjusting often increases both EV and EBITDA; the net effect on multiples varies by lease intensity and accounting choices. Use consistent definitions to avoid misleading relative conclusions.

See also: Chapter 3 (WACC) for discount rate selection; Chapter 4 (DCF) for FCFF consistency; IFRS 16 (leases) guidance for financial statement adjustments and disclosures.
