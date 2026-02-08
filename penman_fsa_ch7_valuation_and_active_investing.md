# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 7: Valuation and Active Investing (Mispricing, Anchoring, Fundamental Signals)

### Core Concept
Active investing aims to identify mispricing: situations where market price deviates from fundamental value implied by financial statements. In practice, this means translating accounting information into explicit valuation anchors (book value, earnings, and operating profitability) and then testing whether the market price embeds assumptions that are inconsistent with a disciplined forecast. The key operational skill is separating value-relevant information (persistent drivers) from noise (transitory items and accounting artefacts).

---

### Formula/Methodology

#### 1) Mispricing spread (valuation vs price)
```text
Mispricing Spread = (Intrinsic Value - Market Price) / Market Price
```

Where:
- Intrinsic Value = value from a valuation model (e.g., residual income, DCF, multiples with fundamentals)
- Market Price = current traded price

#### 2) Price decomposition using accounting anchors
A practical “anchor + premium” framing:
```text
Price = Accounting Anchor + PV(Future Value Creation)
```

Common anchors:
- Book value (equity) for residual income valuation
- Capital employed (IC) for enterprise value framing
- Current earnings level for earnings-based multiples (only if sustainable)

Residual income identity (equity):
```text
Equity Value_0 = BV_0 + Σ [ AE_t / (1 + Re)^t ] + PV(continuing AE)
```

Where:
```text
AE_t = Earnings_t - Re × BV_(t-1)
```

Interpretation: “Value creation” is abnormal earnings. Price can be high even with low current earnings if future abnormal earnings are expected to be large and persistent.

#### 3) Market-implied expectations (reverse engineering)
For a stable perpetuity-style check (use only as a rough diagnostic):
```text
Implied Long-Run ROE ≈ Re + (Price - BV) × Re / BV
```

This is not a theorem; it is a quick consistency check to infer how optimistic the market must be relative to book value.

A more robust approach is to solve for an implied growth rate in a continuing-value expression:
```text
TV_N = FCF_(N+1) / (WACC - g)
```

Then infer g that reconciles enterprise value (EV) to market capitalisation (plus net debt) using your explicit forecast horizon.

#### 4) Signal strength: persistence vs transitory components
A practical persistence view:
```text
Sustainable Earnings = Core Operating Earnings + Persistent Other Items
Transitory Earnings  = One-off / volatile items (low persistence)
```

Active investing relies on identifying:
- “market overreacts to transitory bad news” (value opportunities), or
- “market underreacts to deteriorating fundamentals” (short/avoid signals).

---

### Practical Application (How to apply)

#### A) Build an “active investing” valuation workflow (repeatable)
1) **Pick anchor model**: residual income (BV anchor) or DCF (cash anchor) or a fundamentals-based multiple.
2) **Normalise accounting**: remove transitory items, align operating vs financing classification.
3) **Forecast only the drivers that matter**:
   - operating profitability (margins/ROIC/ROE),
   - reinvestment (ΔIC or capex + working capital),
   - growth (revenue/volume/price) consistent with reinvestment.
4) **Compare to price**:
   - compute mispricing spread,
   - reverse engineer market-implied assumptions (growth, margins, ROIC fade).
5) **Decide**: invest when your base-case fundamentals + conservative ranges still support upside.

#### B) Use financial statements to find mispricing signals (example checklist)
Value-relevant signals (typically higher persistence):
- improving operating margins with credible driver (mix/scale/pricing)
- improving asset turnover (working capital discipline, capacity utilisation)
- sustainable ROE/ROIC above cost of capital
- conservative accounting (timely loss recognition, cautious capitalisation)

Potential mispricing triggers:
- market penalises earnings drop driven by **transitory** charge while cash generation remains intact
- market rewards earnings growth driven by **accrual inflation** (weak cash conversion)

#### C) Avoid “multiple-only” investing
Multiples are shorthand for expectations. For active investing:
- Translate multiple into implied fundamentals (growth, margins, ROE/ROIC persistence).
- If the implied assumptions are unrealistic, the stock is not “cheap” even at a low multiple.

#### D) Document a defendable thesis in valuation terms
Write the thesis as:
- What the market is implicitly assuming,
- What the financial statements suggest is more likely,
- The valuation bridge from assumptions → intrinsic value.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Sequence
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def mispricing_spread(intrinsic_value: float, market_price: float) -> float:
    """(V - P) / P."""
    v = _f(intrinsic_value, "intrinsic_value")
    p = _f(market_price, "market_price")
    if abs(p) < 1e-12:
        raise ValueError("market_price must be non-zero.")
    return float((v - p) / p)

def implied_perpetuity_growth_from_ev(ev: float,
                                      fcf_next: float,
                                      wacc: float) -> float:
    """Solve g from EV = FCF1 / (WACC - g) => g = WACC - FCF1/EV.

    This is a diagnostic, not a full valuation. Requires EV > 0 and WACC > 0.
    """
    EV = _f(ev, "ev")
    F1 = _f(fcf_next, "fcf_next")
    r = _f(wacc, "wacc")
    if EV <= 0:
        raise ValueError("ev must be > 0 for implied growth.")
    if r <= 0:
        raise ValueError("wacc must be > 0.")
    g = r - (F1 / EV)
    return float(g)

def reverse_engineer_margin(required_ev: float,
                            revenue_next: float,
                            margin_assumption: float,
                            tax_rate: float,
                            reinvestment_rate: float,
                            wacc: float,
                            g: float) -> float:
    """Diagnostic: compute EV implied by a simple steady-state driver set.

    Steady-state approximation:
    - EBIT = Revenue × margin
    - NOPAT = EBIT × (1 - tax)
    - FCF = NOPAT × (1 - reinvestment_rate)
    - EV = FCF / (WACC - g)

    Returns:
        float: implied EV
    """
    rev = _f(revenue_next, "revenue_next")
    m = _f(margin_assumption, "margin_assumption")
    t = _f(tax_rate, "tax_rate")
    rr = _f(reinvestment_rate, "reinvestment_rate")
    r = _f(wacc, "wacc")
    gg = _f(g, "g")

    if rev < 0:
        raise ValueError("revenue_next must be >= 0.")
    if t < 0 or t > 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    if rr < 0 or rr > 1:
        raise ValueError("reinvestment_rate must be between 0 and 1.")
    if r <= 0:
        raise ValueError("wacc must be > 0.")
    if gg >= r:
        raise ValueError("g must be < wacc for a finite perpetuity.")

    ebit = rev * m
    nopat = ebit * (1.0 - t)
    fcf = nopat * (1.0 - rr)
    ev_implied = fcf / (r - gg)

    # Compare to required EV if you want to solve for a driver
    _ = _f(required_ev, "required_ev")  # validated but not used directly here
    return float(ev_implied)

# Example usage: mispricing + market-implied growth
inputs = {
    "intrinsic_value": 32.0,  # $/share
    "market_price": 24.0,     # $/share
    "ev": 12_000e6,           # $ enterprise value
    "fcf_next": 650e6,        # $ FCF next year
    "wacc": 0.09
}

spread = mispricing_spread(inputs["intrinsic_value"], inputs["market_price"])
g_implied = implied_perpetuity_growth_from_ev(inputs["ev"], inputs["fcf_next"], inputs["wacc"])

print(f"Mispricing spread: {spread:.1%}")
print(f"Implied perpetuity growth (diagnostic): {g_implied:.1%}")
```

---

### Valuation Impact
Why this matters:
- Active investing is fundamentally about **assumption gaps**. Financial statements provide disciplined anchors (book value, earnings, ROE/ROIC) to quantify what must be true for the current price to be reasonable.
- Reverse-engineering price forces explicit assumptions (growth, margins, reinvestment, fade), which reduces narrative bias.

Impact on multiples:
- “Cheap” P/E may simply reflect low expected persistence (or poor earnings quality). The actionable question is what persistence/growth is implied and whether it is too pessimistic/optimistic.
- Book-to-price can be interpreted as the market’s view on future abnormal earnings; low P/B can signal either value or expected value destruction.

Impact on DCF inputs:
- The active-investing edge often comes from forecasting operational drivers better than the market: margin trajectory, competitive erosion, reinvestment needs, and working capital behaviour.
- Accounting analysis improves DCF quality by preventing overstatement of sustainable margins and understatement of reinvestment.

Comparability issues across companies:
- Different accounting policies can create apparent differences in margins/earnings that are not economic; normalise before interpreting “cheap vs expensive”.

Practical adjustments:
```python
def price_to_value_bridge(market_price: float, intrinsic_value: float, shares: float) -> float:
    """Dollar upside = (V - P) × shares."""
    import numpy as np
    p = float(market_price); v = float(intrinsic_value); s = float(shares)
    if not np.isfinite(p) or not np.isfinite(v) or not np.isfinite(s):
        raise ValueError("Inputs must be finite.")
    if s < 0:
        raise ValueError("shares must be >= 0.")
    return (v - p) * s
```

---

### Quality of Earnings Flags (Active Investing Lens)
⚠️ “Cheap” multiple driven by aggressive revenue recognition or capitalisation policies (not economics).  
⚠️ Rising earnings with deteriorating cash conversion (working capital build, capitalised costs).  
⚠️ Profitability improvement driven by underinvestment (maintenance capex too low, R&D cuts) that will reverse.  
⚠️ Large fair-value gains or non-operating gains boosting EPS.  
✅ Consistent cash conversion over the cycle; conservative accruals; transparent segment disclosures.

---

### Sector-Specific Considerations

| Sector | Key mispricing driver | Practical approach |
|---|---|---|
| Cyclicals | peak/trough earnings distort P/E | normalise earnings and margins to mid-cycle; stress-test demand |
| Software | SBC and capitalised dev costs distort earnings | treat SBC consistently; adjust for capitalised R&D if comparing |
| Banks | earnings reflect credit cycle + provisions | focus on through-cycle ROE, credit costs, capital adequacy |
| Real estate | fair value changes can dominate profit | separate operating income from revaluation; focus on NAV/cash yields |

---

### Real-World Example
Scenario: The market price implies high perpetual growth; test whether that is plausible.

```python
ev = 12_000e6
fcf_next = 650e6
wacc = 0.09

g = implied_perpetuity_growth_from_ev(ev, fcf_next, wacc)
print(f"Implied g: {g:.2%}")

# If implied g is near or above long-run nominal GDP, treat as an optimism flag unless the firm has a clear moat and reinvestment capacity.
```

Interpretation: If implied growth is unrealistically high, the stock may be priced for perfection; the active thesis would require either higher near-term cash flows than expected or a lower risk profile than the market assumes.

See also: Chapter 4 (cash vs accrual and DCF), Chapter 6 (pricing earnings), Chapter 18 (quality of financial statements).
