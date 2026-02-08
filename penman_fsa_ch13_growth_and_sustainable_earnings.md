# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 13: The Analysis of Growth and Sustainable Earnings (Core vs Transitory, Sustainable Growth, Persistence)

### Core Concept
Growth only adds value when it is **profitable and repeatable**. This chapter’s practical objective is to separate **core (sustainable) earnings** from **transitory items**, identify the **drivers of growth** (margins, turnover, reinvestment), and build forecasts that align with clean, defensible valuation inputs.

---

### Formula/Methodology

#### 1) Sustainable earnings (core earnings concept)
```text
Sustainable Earnings ≈ Reported Earnings - Transitory Earnings Components
```

Transitory components commonly include:
- Gains/losses on asset sales
- Unusual restructuring/impairment charges (may be recurring in practice—diagnose!)
- Fair value remeasurement swings not tied to core operations
- One-off tax effects (rate changes, settlements)

#### 2) Growth rate (generic)
```text
Growth Rate (g) = (Value_t - Value_(t-1)) / Value_(t-1)
```

Use for revenue, operating profit, book value, NOA, etc. Ensure the numerator and denominator are consistent (operating vs total).

#### 3) Sustainable growth in equity (retention-based)
```text
Sustainable Equity Growth (g_E) ≈ ROE × Retention Ratio
Retention Ratio = 1 - Payout Ratio
Payout Ratio = Dividends / Net Income
```

Where:
- ROE = Net Income / Average Equity
- Use normalised ROE (remove transitory items) for sustainability.

#### 4) Growth in operating earnings via reinvestment efficiency (operating perspective)
```text
Reinvestment Rate ≈ Net Investment in Operations / OPAT
Growth in OPAT (approx) ≈ ROCE × Reinvestment Rate
```

Where:
- OPAT = EBIT × (1 - operating tax rate)
- ROCE (or RNOA) = OPAT / Average NOA
- Net Investment in Operations ≈ ΔNOA (adjust for acquisitions/disposals and reclassifications)

Practical implication:
- For a given growth target, required reinvestment rises as ROCE falls.

#### 5) Persistence (earnings quality / forecasting weight)
```text
Forecast Weight ∝ Earnings Persistence
```

High persistence: earnings likely to recur (core operating).  
Low persistence: earnings likely to mean-revert (special items, temporary spreads, cyclical peaks).

---

### Practical Application (How to apply)

#### A) Build a “core earnings bridge” (minimal viable)
1) Start with reported operating profit (EBIT) and net income.
2) Identify transitory items from notes:
   - “exceptional”, “non-recurring”, “other income/expense”, “fair value”, “disposals”
3) Reclassify each item:
   - Operating (core) vs operating (non-core) vs financing vs tax-only
4) Convert to after-tax where needed for comparability.
5) Output:
   - Core OPAT
   - Core net income
   - Core margins (core OPAT / revenue)

#### B) Diagnose what drives growth
Separate growth into:
- Price/mix vs volume (if disclosed)
- Margin change vs turnover change (see ROCE decomposition)
- Reinvestment (ΔNOA) vs efficiency (ROCE)

A practical table to maintain in models:

| Driver | Metric | How to compute | What “good” looks like |
|---|---|---|---|
| Profitability | ROCE (RNOA) | OPAT / avg NOA | stable or improving vs peers |
| Margin | OPAT margin | OPAT / revenue | improves without one-offs |
| Turnover | NOA turnover | revenue / NOA | stable; not boosted by underinvestment |
| Reinvestment | Reinvestment rate | ΔNOA / OPAT | consistent with growth plans |
| Cash conversion | CFO / core earnings | operating cash / core profit | stable, near 1x over cycle |

#### C) Forecast sustainability explicitly
- Fade transitory margins back to a mid-cycle or peer-consistent level.
- Fade abnormal growth to an industry or economy-consistent range.
- Where growth is acquisition-driven, forecast **organic** and **inorganic** separately; avoid embedding acquisition growth into “core” trends without reinvestment/capital implications.

#### D) Cross-check against balance sheet growth
If earnings are “growing” but operating assets and working capital are not, question sustainability.
Conversely, if NOA is growing fast without earnings growth, profitability may compress.

---

### Python Implementation
```python
from typing import Dict, Any, Optional
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def growth_rate(current: float, prior: float) -> float:
    """Compute growth rate with validation.

    Args:
        current: current period value
        prior: prior period value (must be non-zero)

    Returns:
        float: growth rate as decimal

    Raises:
        ValueError: if prior is zero or inputs invalid
    """
    c = _f(current, "current")
    p = _f(prior, "prior")
    if abs(p) < 1e-12:
        raise ValueError("prior must be non-zero to compute growth.")
    return float((c - p) / p)

def core_earnings(reported: float, transitory_after_tax: float) -> float:
    """Remove transitory after-tax components from reported earnings.

    Args:
        reported: reported earnings ($)
        transitory_after_tax: items to remove (after-tax) ($)

    Returns:
        float: core earnings ($)

    Raises:
        ValueError: invalid inputs
    """
    r = _f(reported, "reported")
    t = _f(transitory_after_tax, "transitory_after_tax")
    return float(r - t)

def retention_ratio(dividends: float, net_income: float) -> float:
    """Compute retention ratio = 1 - dividends/net income.

    Notes:
        If net income is <= 0, retention is not meaningful for sustainable growth.
        In that case, raise and handle explicitly in your model.

    Args:
        dividends: cash dividends ($)
        net_income: net income ($)

    Returns:
        float: retention ratio (0-1 typical)

    Raises:
        ValueError: if net_income <= 0 or inputs invalid
    """
    d = _f(dividends, "dividends")
    ni = _f(net_income, "net_income")
    if ni <= 0:
        raise ValueError("net_income must be > 0 for retention-based sustainable growth.")
    rr = 1.0 - (d / ni)
    return float(rr)

def sustainable_growth_roe(ro_e: float, retention: float) -> float:
    """g ≈ ROE × retention."""
    roe = _f(ro_e, "roe")
    r = _f(retention, "retention")
    if r < 0 or r > 1.5:
        raise ValueError("retention looks implausible; check dividends and net income.")
    return float(roe * r)

def reinvestment_rate(delta_noa: float, opat: float) -> float:
    """Reinvestment rate ≈ ΔNOA / OPAT."""
    dn = _f(delta_noa, "delta_noa")
    o = _f(opat, "opat")
    if abs(o) < 1e-12:
        raise ValueError("OPAT must be non-zero.")
    return float(dn / o)

def growth_from_roce(roce: float, reinvest_rate: float) -> float:
    """Approximate operating earnings growth ≈ ROCE × reinvestment rate."""
    r = _f(roce, "roce")
    rr = _f(reinvest_rate, "reinvest_rate")
    return float(r * rr)

# Example usage ($m)
company = {
    "reported_net_income": 420.0,
    "transitory_after_tax": 60.0,   # e.g., after-tax disposal gain + unusual tax credit
    "dividends": 120.0,
    "avg_equity": 2_000.0,
    "opat": 520.0,
    "noa_prior": 3_100.0,
    "noa_current": 3_450.0,
    "roce": 0.16
}

core_ni = core_earnings(company["reported_net_income"], company["transitory_after_tax"])
roe_core = core_ni / company["avg_equity"]

rr = retention_ratio(company["dividends"], core_ni)
g_equity = sustainable_growth_roe(roe_core, rr)

delta_noa = company["noa_current"] - company["noa_prior"]
reinv = reinvestment_rate(delta_noa, company["opat"])
g_opat = growth_from_roce(company["roce"], reinv)

print(f"Core NI: ${core_ni:.0f}m | Core ROE: {roe_core:.1%}")
print(f"Retention: {rr:.1%} | Sustainable equity growth (approx): {g_equity:.1%}")
print(f"Reinvestment rate: {reinv:.1%} | Operating earnings growth (approx): {g_opat:.1%}")
```

---

### Valuation Impact
Why this matters:
- Sustainable earnings drive terminal value: overestimating persistence inflates value disproportionately.
- Growth forecasts must reconcile with reinvestment: high growth with low reinvestment is rarely sustainable unless the business is genuinely capital-light.

Impact on multiples:
- P/E and EV/EBITDA are sensitive to “core vs reported” earnings. One-off gains can create optically cheap multiples.
- Firms with high persistence typically command higher multiples; low persistence gets discounted.

Impact on DCF inputs:
- Growth affects terminal value and fade period assumptions.
- Reinvestment assumptions determine FCFF; using ROCE to translate growth into reinvestment is a practical consistency check.

Comparability issues:
- Different accounting policies and classification (capitalised costs, leases, pensions) shift both “earnings” and the “capital base”.
- Acquisition-heavy groups may show growth that is not comparable with organic peers without separating inorganic effects.

Practical adjustments:
```python
def normalise_growth(reported_growth: float, acquisition_growth: float) -> float:
    """Estimate organic growth by stripping acquisition contribution."""
    import numpy as np
    g = float(reported_growth)
    a = float(acquisition_growth)
    if not np.isfinite(g) or not np.isfinite(a):
        raise ValueError("Inputs must be finite.")
    return g - a
```

---

### Quality of Earnings Flags (growth-focused)
⚠️ Earnings growth driven by transitory gains (asset sales, fair value uplifts, tax windfalls).  
⚠️ Revenue growth without corresponding working capital/capacity investment (may indicate aggressive recognition or channel stuffing).  
⚠️ Margin expansion coincides with capitalisation of costs or sharp cut in maintenance capex.  
⚠️ “Core” adjustments repeat every year (recurring “non-recurring” items).  
⚠️ Growth driven by acquisitions but treated as organic in forecasts (double counting synergies without reinvestment).  
✅ Core earnings growth aligns with cash conversion and stable working-capital intensity.  
✅ ROCE remains above cost of capital while the firm grows (value-creating growth).  

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Consumer / retail | promotional pull-forward and inventory build can mimic growth | normalise sales and working-capital intensity; watch returns and markdowns |
| Software / subscription | contract assets/deferred revenue timing distorts growth | reconcile billings, deferred revenue, and revenue; separate SBC and capitalised dev |
| Industrials | cyclical peaks inflate “sustainable” margins | use mid-cycle margins and capacity utilisation; fade to normal |
| Financials | earnings include credit-cycle and fair value effects | isolate core spread income vs mark-to-market; stress credit losses |

---

### Real-World Example
Scenario: Reported net income rises 25%, but most of the change comes from a one-off disposal gain.

```python
reported_ni_t = 500.0
reported_ni_t1 = 400.0
one_off_after_tax = 90.0

reported_g = growth_rate(reported_ni_t, reported_ni_t1)
core_t = core_earnings(reported_ni_t, one_off_after_tax)
core_g = growth_rate(core_t, reported_ni_t1)

print(f"Reported NI growth: {reported_g:.1%}")
print(f"Core NI (adjusted): ${core_t:.0f}m | Core growth: {core_g:.1%}")
```

Interpretation: A large gap between reported and core growth indicates low persistence; valuation should anchor on core profitability and a defensible reinvestment path, not headline earnings growth.

See also: Chapter 12 (profitability drivers), Chapter 11 (cash flow diagnostics), Chapter 18 (quality of financial statements), Chapter 15 (simple forecasting and valuation), Chapter 16 (full-information forecasting).
