# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 22: Leases (IFRS 16, lease capitalization, EBITDA uplift, debt-like liabilities, comparability)

### Core Concept
IFRS 16 brings most leases onto the balance sheet by recognising a right-of-use (ROU) asset and a lease liability for lessees (with limited exemptions). This materially changes EBITDA (operating lease expense replaced by depreciation + interest), leverage, and cash flow classification. For valuation, the key is comparability across periods (pre/post IFRS 16) and across peers with different lease intensity.

### Formula/Methodology

#### 1) Initial measurement (lessee)
```text
Lease liability (initial) = PV of future lease payments

PV = Σ [ Payment_t / (1 + r)^t ]

Where:
Payment_t = lease payments at time t (fixed payments, in-substance fixed, index/ rate-linked using index at commencement, residual value guarantees, purchase/termination options if reasonably certain)
r = discount rate (lessee incremental borrowing rate, unless implicit rate readily determinable)

ROU asset (initial) = Lease liability
                    + lease payments made at or before commencement (less incentives)
                    + initial direct costs
                    + estimated dismantling/restoration costs (if applicable)
```

#### 2) Subsequent measurement (lessee)
```text
Interest expense_t = Opening lease liability × r
Lease liability closing = Opening liability + Interest expense - Lease payments

ROU depreciation (straight-line example) = ROU asset / lease term
(Or over useful life if purchase option reasonably certain)

Total P&L effect (typical):
- Depreciation (operating)
- Interest (finance)
```

#### 3) EBITDA / EBIT effects
```text
Pre-IFRS 16 (operating lease):
- Operating lease expense in operating costs -> reduces EBITDA

Post-IFRS 16:
- Depreciation replaces lease expense -> EBITDA increases (no longer includes operating lease expense)
- Interest below EBITDA -> increases finance cost
```

#### 4) Cash flow effects (typical)
```text
Operating cash flow increases (lease payments removed from operating, depending on policy)
Financing cash flow decreases (principal portion of lease payments classified as financing)
Interest portion: operating or financing depending on accounting policy (IAS 7 choices)
```

#### 5) Practical “lease debt” approximation (when full schedule not available)
```text
Lease debt proxy (rough) = Annual operating lease expense × Multiple

Common heuristics:
- 6x to 8x for retail/airlines/longer leases
- 4x to 6x for shorter leases

Prefer: PV-based using disclosed maturity analysis where available.
```

---

### Practical Application for Valuation Analysts

#### 1) Enterprise value bridge: treat lease liabilities as debt-like
Default valuation convention:
- Add lease liabilities to net debt (or to EV bridge) when using EV multiples
- Ensure cash / working capital definitions are consistent

Avoid double counting:
- If EBITDA is IFRS 16 (post), EV includes lease debt-like claim; do not also capitalise lease expense again.
- If you “de-IFRS16” EBITDA to a pre-IFRS16 basis, reverse the EV adjustment accordingly.

#### 2) Comparable multiples: standardise either on “post-IFRS16” or “pre-IFRS16”
Two consistent approaches:
- Approach A (recommended): Use reported IFRS 16 EBITDA and include lease liabilities in net debt/EV.
- Approach B: Restate to pre-IFRS16 operating lease basis:
  - Subtract ROU depreciation and interest; add back operating lease expense (or reconstruct)
  - Remove lease liabilities from net debt
Use one approach consistently across comp set and historical periods.

#### 3) DCF: include lease economics once
For FCFF:
- Treat lease payments as financing (like debt) and include lease liability as debt-like in EV-to-equity bridge, OR
- Treat lease payments as operating (pre-IFRS16 style) and exclude lease debt from net debt
Consistency matters more than the specific method.

#### 4) Lease term and discount rate assumptions are valuation-sensitive
Key judgement areas:
- Lease term (renewal options “reasonably certain”)
- Discount rate (incremental borrowing rate)
- Variable lease payments (excluded unless in-substance fixed; index-based uses index at commencement)
Analyst actions:
- Review lease maturity table and sensitivity disclosures
- If rates used are unusually low/high relative to borrowing costs, challenge comparability

---

### Python Implementation
```python
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def lease_liability_pv(payments: List[Tuple[float, float]], discount_rate: float) -> float:
    """
    Present value of lease payments (lease liability at commencement).

    Args:
        payments: list of (t_years, payment) where payment is positive
        discount_rate: annual discount rate (decimal), must be > -1

    Returns:
        float: PV of payments

    Raises:
        ValueError: invalid inputs.
    """
    r = _num(discount_rate, "discount_rate")
    if r <= -1.0:
        raise ValueError("discount_rate must be > -1.0.")
    pv = 0.0
    for t, p in payments:
        tt = _num(t, "t_years")
        pay = _num(p, "payment")
        if tt < 0:
            raise ValueError("t_years must be >= 0.")
        if pay < 0:
            raise ValueError("payment must be >= 0.")
        pv += pay / ((1.0 + r) ** tt)
    return pv

def amortize_lease_liability(opening_liability: float, discount_rate: float, lease_payment: float) -> Dict[str, float]:
    """
    One-period lease liability roll-forward.

    closing = opening + interest - payment

    Args:
        opening_liability: opening balance
        discount_rate: annual rate (decimal)
        lease_payment: cash payment for the period (positive)

    Returns:
        dict: {'interest': ..., 'closing_liability': ...}

    Raises:
        ValueError: invalid inputs.
    """
    L0 = _num(opening_liability, "opening_liability")
    r = _num(discount_rate, "discount_rate")
    p = _num(lease_payment, "lease_payment")
    if L0 < 0:
        raise ValueError("opening_liability must be >= 0.")
    if p < 0:
        raise ValueError("lease_payment must be >= 0.")
    interest = L0 * r
    closing = L0 + interest - p
    # allow small negative due to rounding; clamp at 0
    closing = 0.0 if closing < 0 and abs(closing) < 1e-9 else closing
    if closing < 0:
        raise ValueError("lease_payment too large relative to liability; check inputs.")
    return {"interest": interest, "closing_liability": closing}

def straight_line_depreciation(rou_asset: float, lease_term_years: float) -> float:
    """
    Straight-line depreciation for ROU asset.

    Raises:
        ValueError: non-positive term or negative asset.
    """
    a = _num(rou_asset, "rou_asset")
    t = _num(lease_term_years, "lease_term_years")
    if a < 0:
        raise ValueError("rou_asset must be >= 0.")
    if t <= 0:
        raise ValueError("lease_term_years must be > 0.")
    return a / t

def ifrs16_to_pre_ifrs16_ebitda(reported_ebitda: float, operating_lease_expense_pre_ifrs16: float) -> float:
    """
    Approximate pre-IFRS16 EBITDA by subtracting the operating lease expense that would have been included.

    Note: This requires an estimate of operating lease expense (often disclosed pre-adoption or via note).

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(reported_ebitda, "reported_ebitda")
    ol = _num(operating_lease_expense_pre_ifrs16, "operating_lease_expense_pre_ifrs16")
    if ol < 0:
        raise ValueError("operating_lease_expense_pre_ifrs16 must be >= 0.")
    return e - ol

def lease_debt_proxy(annual_lease_expense: float, multiple: float) -> float:
    """
    Heuristic lease debt proxy = annual lease expense × multiple.

    Args:
        annual_lease_expense: annual operating lease cost
        multiple: heuristic multiple (e.g., 6-8)

    Raises:
        ValueError: negative inputs.
    """
    exp = _num(annual_lease_expense, "annual_lease_expense")
    m = _num(multiple, "multiple")
    if exp < 0 or m < 0:
        raise ValueError("annual_lease_expense and multiple must be >= 0.")
    return exp * m

def adjust_net_debt_for_leases(net_debt_reported: float, lease_liability: float) -> float:
    """
    Add lease liability to net debt for EV bridge.

    Raises:
        ValueError: negative lease liability.
    """
    nd = _num(net_debt_reported, "net_debt_reported")
    ll = _num(lease_liability, "lease_liability")
    if ll < 0:
        raise ValueError("lease_liability must be >= 0.")
    return nd + ll

# Example usage
payments = [(1, 12_000_000), (2, 12_000_000), (3, 12_000_000), (4, 12_000_000), (5, 12_000_000)]
r = 0.055
pv = lease_liability_pv(payments, r)
print(f"Lease liability PV: ${pv/1e6:.1f}M")

step = amortize_lease_liability(opening_liability=pv, discount_rate=r, lease_payment=12_000_000)
print(f"Year 1 interest: ${step['interest']/1e6:.1f}M")
print(f"Closing liability (Y1): ${step['closing_liability']/1e6:.1f}M")

dep = straight_line_depreciation(rou_asset=pv, lease_term_years=5)
print(f"ROU depreciation (annual): ${dep/1e6:.1f}M")

proxy = lease_debt_proxy(annual_lease_expense=12_000_000, multiple=7.0)
print(f"Lease debt proxy (7x): ${proxy/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- IFRS 16 increases EBITDA mechanically; without adjusting EV for lease liabilities, EV/EBITDA multiples become overstated (appearing “cheaper”).
- Lease liabilities are debt-like claims; ignoring them overstates equity value and understates leverage.
- Lease intensity affects comparability across business models (retail, airlines, logistics vs asset-heavy owners).

Impact on multiples:
- EV/EBITDA: use post-IFRS16 EBITDA with EV including lease liabilities, or restate both to pre-IFRS16 basis.
- P/E: net income impact can be front-loaded (interest higher early); be cautious comparing periods immediately after adoption.

Impact on DCF inputs:
- If modelling FCFF, decide whether leases are operating (treat as rent) or financing (treat as debt) and apply consistently through cash flows and EV-to-equity bridge.
- Ensure reinvestment and ROIC calculations reflect ROU assets consistently if using operating capital definitions.

Practical adjustments:
```python
def valuation_adjustment_for_ifrs16(ev: float, lease_liability: float, ebitda: float, operating_lease_expense: float,
                                   approach: str = "post_ifrs16") -> Dict[str, float]:
    """
    Produce consistent EV/EBITDA inputs under two approaches.

    approach:
      - 'post_ifrs16': EV includes lease liability; EBITDA is reported (IFRS16)
      - 'pre_ifrs16': EV excludes lease liability; EBITDA is adjusted down by operating lease expense

    Returns:
        dict with keys: 'ev', 'ebitda'
    """
    E = _num(ev, "ev")
    LL = _num(lease_liability, "lease_liability")
    EBITDA = _num(ebitda, "ebitda")
    rent = _num(operating_lease_expense, "operating_lease_expense")
    if LL < 0 or rent < 0:
        raise ValueError("lease_liability and operating_lease_expense must be >= 0.")
    if approach not in {"post_ifrs16", "pre_ifrs16"}:
        raise ValueError("approach must be 'post_ifrs16' or 'pre_ifrs16'.")
    if approach == "post_ifrs16":
        return {"ev": E + LL, "ebitda": EBITDA}
    return {"ev": E, "ebitda": EBITDA - rent}
```

---

### Quality of Earnings Flags
⚠️ Large EBITDA uplift post-IFRS16 highlighted in KPIs without explaining mechanical accounting effect.  
⚠️ Lease term assumptions unusually long (inflates ROU asset and liability) relative to peer practice.  
⚠️ Significant variable lease payments excluded from liability but economically unavoidable (e.g., sales-linked rents in stable stores).  
⚠️ Significant modification activity causing catch-up adjustments and volatility.  
✅ Clear maturity analysis, discount rate disclosure, and reconciliation of lease liabilities and ROU assets.

---

### Sector-Specific Considerations

| Sector | Key lease issue | Typical treatment |
|---|---|---|
| Retail | High store lease intensity; variable rents | Treat lease liabilities as debt-like; consider variable rents in cash flow forecasts |
| Airlines | Aircraft leases; long terms | Lease liabilities are central to EV; ensure comparability with owned fleets |
| Logistics | Warehouse leases; renewal options | Scrutinise “reasonably certain” extensions; align forecast lease costs |
| Tech / Professional services | Office leases (often smaller) | Usually less material; still adjust for comps where needed |

---

### Real-World Example
Scenario: Company reports EV $8.0bn, net debt $1.2bn, lease liabilities $1.5bn, EBITDA $900m (IFRS 16). Peer set uses EV/EBITDA; you want a consistent multiple.

```python
ev = 8_000_000_000
lease_liab = 1_500_000_000
ebitda = 900_000_000

inputs = valuation_adjustment_for_ifrs16(
    ev=ev,
    lease_liability=lease_liab,
    ebitda=ebitda,
    operating_lease_expense=0.0,
    approach="post_ifrs16"
)
multiple = inputs["ev"] / inputs["ebitda"]
print(f"EV incl leases: ${inputs['ev']/1e9:.1f}B")
print(f"EV/EBITDA (post-IFRS16 consistent): {multiple:.1f}x")
```

Interpretation: Including lease liabilities avoids understating enterprise claims and prevents “false cheapness” from EBITDA uplift.

See also: Chapter 4 (SOFP) for presentation of lease liabilities/ROU assets; Chapter 6 (cash flows) for lease payment classification; Chapter 10 (borrowing costs) for interest classification considerations; Chapter 24 (financial instruments) for discount rate comparability.
