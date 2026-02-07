# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 25: Fair Value (IFRS 13, valuation hierarchy, inputs, DCF alignment, observable vs unobservable)

### Core Concept
IFRS 13 defines fair value as an exit price in an orderly transaction between market participants at the measurement date and establishes a framework for measuring fair value. For valuation, IFRS 13 is the bridge between accounting fair values (e.g., FV instruments, investment property under FV model, business combination fair values) and market-consistent valuation techniques (market approach, income approach, cost approach). The critical practical value is understanding the fair value hierarchy (Level 1/2/3), how inputs are sourced, and how to assess reliability and bias in reported fair values.

### Formula/Methodology

#### 1) Fair value definition and measurement perspective
```text
Fair value = exit price (sell an asset / transfer a liability) in an orderly transaction
            between market participants at the measurement date.

Key constraints:
- Market participant assumptions (not entity-specific synergies)
- Principal (or most advantageous) market
- Measurement date conditions (not future events)
```

#### 2) Fair value hierarchy (input observability)
```text
Level 1: quoted prices in active markets for identical assets/liabilities (most reliable)
Level 2: observable inputs other than Level 1 (e.g., quoted prices for similar items, yield curves, credit spreads)
Level 3: unobservable inputs (significant management judgement), e.g., internally-developed DCF assumptions
```

#### 3) Common valuation approaches under IFRS 13
```text
Market approach:
- Uses prices and other relevant information from market transactions involving identical or comparable assets.

Income approach:
- Converts future amounts (cash flows/earnings) to a single present value amount.
- Techniques include discounted cash flows (DCF), option pricing, excess earnings methods.

Cost approach:
- Reflects the amount required to replace the service capacity of an asset (replacement cost).
```

#### 4) DCF under IFRS 13 (income approach)
```text
PV = Σ [ CF_t / (1 + r)^t ]

Where:
CF_t = cash flow in period t consistent with market participant assumptions
r = discount rate reflecting time value + risk (consistent with CF definition and measurement basis)
t = period index

Consistency rules:
- Currency consistency: cash flows and discount rate in same currency
- Nominal vs real: cash flows and discount rate both nominal or both real
- Pre-tax vs post-tax: align cash flows and discount rate basis consistently
```

#### 5) Valuation adjustments and calibration
```text
If transaction price differs from model value at initial recognition:
- Consider whether differences reflect bid-ask, liquidity, credit, model risk, or market conditions
- IFRS 13 emphasizes calibration where possible (align model to observable transaction pricing at measurement date)

Valuation adjustments (examples):
- Credit valuation adjustment (CVA) for counterparty default risk (asset)
- Debit valuation adjustment (DVA) for own credit risk (liability), where applicable
- Liquidity/marketability adjustments (often embedded via discount rate or explicit discount)
```

---

### Practical Application for Valuation Analysts

#### 1) Interpret Level 3 disclosures as “risk of estimation error”
When fair values rely on Level 3 inputs, use disclosures to:
- Identify key unobservable assumptions (discount rates, growth rates, margins, exit multiples, occupancy, cap rates)
- Assess sensitivity ranges and whether assumptions are symmetric vs optimistic

Valuation actions:
- Build a “disclosure-driven sensitivity” (use the same drivers and ranges) to bound valuation uncertainty.
- Compare Level 3 discount rates to implied WACC/cost of equity from market comps for plausibility.

#### 2) Align IFRS 13 fair value with enterprise valuation work
IFRS 13 fair value is a market participant exit price at measurement date.
Enterprise valuation for deals can include:
- synergies (buyer-specific)
- control premiums
- strategic value
Actions:
- Do not treat accounting fair value as a proxy for deal value without considering synergies and control.
- When using reported fair values (e.g., investment property), validate that they reflect “as-is” market assumptions.

#### 3) Common pitfalls in fair value analysis
- Using entity-specific forecast improvements in a market-participant fair value without support
- Double counting risk: adding both a high discount rate and explicit risk discounts
- Terminal value inconsistencies (growth > economy / exit multiple inconsistent with margins and reinvestment)
- Ignoring liquidity/marketability conditions for private instruments

#### 4) Reconcile fair value movements to earnings quality
Fair value changes can be:
- realised vs unrealised
- in P&L vs OCI
Actions:
- Separate realised cash outcomes from mark-to-market.
- When valuing operating companies, treat investment fair value swings as non-core unless central to business model.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def dcf_pv(cashflows: List[Tuple[float, float]], discount_rate: float) -> float:
    """
    Present value of a set of dated cash flows.

    Args:
        cashflows: list of (t_years, cf) where cf can be positive or negative
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
    for t, cf in cashflows:
        tt = _num(t, "t_years")
        c = _num(cf, "cf")
        if tt < 0:
            raise ValueError("t_years must be >= 0.")
        pv += c / ((1.0 + r) ** tt)
    return pv

def terminal_value_perpetuity(fcf_next: float, discount_rate: float, growth_rate: float) -> float:
    """
    Perpetuity-growth terminal value.

    TV = FCF_{t+1} / (r - g)

    Raises:
        ValueError: if r <= g or invalid inputs.
    """
    f = _num(fcf_next, "fcf_next")
    r = _num(discount_rate, "discount_rate")
    g = _num(growth_rate, "growth_rate")
    if r <= g:
        raise ValueError("discount_rate must be greater than growth_rate.")
    return f / (r - g)

def fair_value_hierarchy_level(level: int) -> str:
    """
    Map numeric hierarchy level to label.

    Args:
        level: 1, 2, or 3

    Returns:
        str: 'level_1', 'level_2', or 'level_3'

    Raises:
        ValueError: invalid level.
    """
    if level == 1:
        return "level_1"
    if level == 2:
        return "level_2"
    if level == 3:
        return "level_3"
    raise ValueError("level must be 1, 2, or 3.")

def apply_liquidity_discount(fair_value: float, discount_pct: float) -> float:
    """
    Apply a liquidity/marketability discount as a simple percentage reduction.

    Args:
        fair_value: base fair value
        discount_pct: discount as decimal (e.g., 0.15 for 15%)

    Returns:
        float: discounted value

    Raises:
        ValueError: discount out of bounds.
    """
    v = _num(fair_value, "fair_value")
    d = _num(discount_pct, "discount_pct")
    if not (0.0 <= d < 1.0):
        raise ValueError("discount_pct must be in [0, 1).")
    return v * (1.0 - d)

# Example usage: Level 3 DCF with terminal value
fcfs = [(1, 50_000_000), (2, 60_000_000), (3, 70_000_000), (4, 80_000_000)]
r = 0.11
pv_stage = dcf_pv(fcfs, r)

tv = terminal_value_perpetuity(fcf_next=85_000_000, discount_rate=r, growth_rate=0.03)
pv_tv = dcf_pv([(4, tv)], r)  # discount TV from end of year 4 to today

fv = pv_stage + pv_tv
fv_liq = apply_liquidity_discount(fv, 0.10)

print(f"PV explicit period: ${pv_stage/1e6:.1f}M")
print(f"PV terminal value: ${pv_tv/1e6:.1f}M")
print(f"Fair value (Level 3 DCF): ${fv/1e6:.1f}M")
print(f"Fair value after liquidity discount: ${fv_liq/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- IFRS 13 is the accounting framework most aligned with valuation practice; it governs the “inputs” you will often reuse (discount rates, cap rates, exit multiples).
- Fair value hierarchy drives reliability: Level 1 is market-based; Level 3 is model-based and more susceptible to bias.
- For asset-heavy businesses (property, infrastructure) fair values can drive NAV-based valuations and debt capacity; for financials, FV categories drive earnings and capital sensitivity.

Impact on multiples:
- If a business includes investment fair value gains in earnings, P/E can be distorted; isolate operating earnings.
- NAV and P/B multiples depend on fair value credibility (Level 3 assumptions can inflate book).

Impact on DCF inputs:
- Calibrate discount rates and terminal assumptions against observable inputs (Level 2 yield curves, credit spreads).
- Ensure the measurement basis (market participant) is consistent with cash flows (no entity-only synergies unless doing deal value, not IFRS 13).

Practical adjustments:
```python
def implied_exit_multiple(tv: float, metric: float) -> float:
    """
    Implied exit multiple = terminal value / terminal metric (e.g., EBITDA).

    Raises:
        ValueError: metric <= 0.
    """
    T = _num(tv, "tv")
    m = _num(metric, "metric")
    if m <= 0:
        raise ValueError("metric must be > 0.")
    return T / m
```

---

### Quality of Earnings Flags
⚠️ Large Level 3 gains recognised in P&L with weak sensitivity disclosure.  
⚠️ Discount rates materially below market comparables without justification (overvaluation risk).  
⚠️ Terminal growth assumptions inconsistent with reinvestment needs or industry maturity.  
⚠️ “Valuation adjustments” applied inconsistently period-to-period (management discretion).  
✅ Clear hierarchy breakdown, sensitivity ranges, and reconciliation of Level 3 roll-forward (opening FV, purchases/sales, gains/losses, transfers).

---

### Sector-Specific Considerations

| Sector | Key fair value driver | Typical treatment |
|---|---|---|
| Real estate | cap rates, occupancy, rent growth | Cross-check to market transactions; sensitivity to cap rate ±25–50 bps |
| Private credit / lending | discount margins, PD/LGD, illiquidity | Reconcile to ECL and credit spreads; separate market moves vs credit impairment |
| Infrastructure | long-dated cash flows, inflation linkage | Verify inflation consistency (nominal vs real); terminal assumptions |
| Financial services | Level 2 curves, Level 3 models | Focus on observable inputs; stress test spread shocks |

---

### Real-World Example
Scenario: Investment property carried at fair value (Level 3). Valuer uses 5.25% cap rate. You want a quick cross-check and sensitivity.

```python
noi = 40_000_000  # stabilized net operating income
cap_rate = 0.0525
value = noi / cap_rate
value_up = noi / 0.0500
value_down = noi / 0.0550
print(f"Value @ 5.25% cap: ${value/1e6:.1f}M")
print(f"Value @ 5.00% cap: ${value_up/1e6:.1f}M")
print(f"Value @ 5.50% cap: ${value_down/1e6:.1f}M")
```

Interpretation: A 25–50 bps move in cap rate can swing valuation materially; incorporate this into downside cases and assess covenant headroom.

See also: Chapter 24 (financial instruments) for FVOCI/FVTPL routing; Chapter 13 (impairment) for recoverable amount tests (VIU vs FVLCD); Chapter 22 (leases) where discount rate consistency matters; Chapter 26 (taxes) for deferred tax on fair value movements.
