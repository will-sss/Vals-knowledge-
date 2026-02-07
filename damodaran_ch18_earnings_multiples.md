# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 18: Earnings Multiples (P/E, Forward P/E, PEG, Normalisation, Negative Earnings Workarounds)

### Core Concept
Earnings multiples (especially P/E) are the most widely used relative valuation tools because they are intuitive and data is available. Practically, P/E only works when earnings are **meaningful, sustainable, and comparable**; otherwise you must normalise earnings, switch to forward earnings, or use alternative multiples (EV/Sales, EV/EBITDA) and tie the multiple back to fundamentals (growth, risk, payout, ROE).

See also: Chapter 9 (earnings measurement/normalisation), Chapter 17 (peer selection and standardisation), Chapter 22 (money-losing firms), Chapter 14 (equity intrinsic models justify P/E).

---

### Formula/Methodology

#### 1) P/E and forward P/E (definitions)
```text
P/E = Price per Share / EPS
P/E = Equity Value / Net Income

Forward P/E = Price / Expected EPS_{next year}
```
Minimum requirements:
- EPS is positive and not dominated by one-offs.
- Accounting is comparable (tax, SBC, capitalisation, impairment charges).

#### 2) Normalised P/E (replace noisy denominator)
```text
Normalised EPS = Reported EPS + One-off charges − One-off gains + Classification adjustments

Normalised P/E = Price / Normalised EPS
```
Practical normalisation:
- Use multi-year average earnings for cyclicals (mid-cycle EPS).
- Remove restructuring charges if truly non-recurring; beware repeated “one-offs.”

#### 3) Relationship to fundamentals (stable growth intuition)
In stable growth equity models, P/E relates to payout, growth, and risk.
```text
Justified P/E increases with:
- Higher expected growth (g)
- Higher payout (for same growth, implies lower reinvestment needs)
- Lower cost of equity (k_e)

Justified P/E decreases with:
- Higher risk (k_e)
- Lower quality/sustainability of earnings
```
A practical stable-growth mapping (directional; do not treat as exact without a model):
```text
Value ≈ Div_1 / (k_e − g)
If Div_1 = Payout × Earnings_1 then:
P/E ≈ Payout / (k_e − g)
```
Where:
- Payout is total payout ratio consistent with how “earnings” is defined (dividends only vs dividends+buybacks).

#### 4) PEG (growth-adjusted P/E) and cautions
```text
PEG = (P/E) / Expected Growth Rate

Where:
Expected Growth Rate is usually in % (e.g., 10) or decimal (0.10) — be consistent.
```
Practical usage:
- Use only when growth is positive and measured consistently across peers.
- PEG fails when earnings quality differs, leverage differs materially, or growth is short-lived/hard to forecast.

#### 5) When P/E breaks (negative or tiny earnings)
```text
If earnings <= 0:
P/E is not meaningful (negative or extremely large).
Use:
- EV/Sales with margin path
- EV/EBITDA if EBITDA positive and comparable
- P/B with ROE (for financials)
```
Also consider forward earnings if credible profitability is near-term.

---

### Python Implementation
```python
from typing import Optional, List, Dict
import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def normalize_earnings(reported: float, charges: float = 0.0, gains: float = 0.0, adjustments: float = 0.0) -> float:
    """Normalised = Reported + Charges - Gains + Adjustments."""
    for name, v in {"reported": reported, "charges": charges, "gains": gains, "adjustments": adjustments}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if charges < 0 or gains < 0:
        raise ValueError("charges and gains must be >= 0")
    return reported + charges - gains + adjustments


def pe_ratio(price: float, eps: float) -> Optional[float]:
    """P/E = price / eps. Returns None if eps <= 0."""
    for name, v in {"price": price, "eps": eps}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if price < 0:
        raise ValueError("price must be >= 0")
    if eps <= 0:
        return None
    return price / eps


def pe_from_equity_value(equity_value: float, net_income: float) -> Optional[float]:
    """P/E = equity value / net income. Returns None if net income <= 0."""
    for name, v in {"equity_value": equity_value, "net_income": net_income}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if equity_value < 0:
        raise ValueError("equity_value must be >= 0")
    if net_income <= 0:
        return None
    return equity_value / net_income


def peg_ratio(pe: float, growth: float, growth_is_percent: bool = False) -> Optional[float]:
    """PEG = P/E divided by growth. Returns None if growth <= 0."""
    if not _num(pe) or not _num(growth):
        raise ValueError("inputs must be numeric")
    if pe <= 0:
        return None
    g = growth / 100.0 if growth_is_percent else growth
    if g <= 0:
        return None
    return pe / g


def justified_pe_simple(payout: float, ke: float, g: float) -> Optional[float]:
    """Simple justified P/E proxy: payout / (ke - g). Returns None if ke<=g."""
    for name, v in {"payout": payout, "ke": ke, "g": g}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if not (0.0 <= payout <= 1.0):
        raise ValueError("payout must be in [0,1]")
    if ke <= g:
        return None
    return payout / (ke - g)


def mid_cycle_eps(eps_series: List[float]) -> Optional[float]:
    """Compute a simple mid-cycle EPS as average of positive EPS values; returns None if none positive."""
    if not isinstance(eps_series, list) or len(eps_series) == 0:
        raise ValueError("eps_series must be a non-empty list")
    pos = [e for e in eps_series if _num(e) and e > 0]
    if len(pos) == 0:
        return None
    return sum(pos) / len(pos)


# Example usage
price = 42.0
reported_eps = 2.10
one_off_charges_per_share = 0.35
one_off_gains_per_share = 0.10
adj_eps = normalize_earnings(reported_eps, charges=one_off_charges_per_share, gains=one_off_gains_per_share)

pe_reported = pe_ratio(price, reported_eps)
pe_adj = pe_ratio(price, adj_eps)

print(f"Reported EPS: ${reported_eps:.2f} -> P/E: {pe_reported:.1f}x" if pe_reported else "Reported P/E: n/a")
print(f"Normalised EPS: ${adj_eps:.2f} -> P/E: {pe_adj:.1f}x" if pe_adj else "Normalised P/E: n/a")

# PEG example
growth = 0.10  # 10% as decimal
peg = peg_ratio(pe_adj if pe_adj else 0.0, growth, growth_is_percent=False)
print(f"PEG: {peg:.1f}" if peg else "PEG: n/a")

# Justified P/E proxy (quick sanity check)
payout = 0.35
ke = 0.10
g = 0.03
jpe = justified_pe_simple(payout, ke, g)
print(f"Justified P/E (proxy): {jpe:.1f}x" if jpe else "Justified P/E: n/a")
```

---

### Valuation Impact
Why this matters:
- P/E is often used as a shortcut for intrinsic value, but it embeds assumptions about growth, payout, ROE, and risk. Normalising earnings and aligning peer definitions are essential to avoid false conclusions.
- For cyclicals, mid-cycle earnings prevent valuing on peak or trough EPS, which can invert your buy/sell signal.
- For high option overhang or heavy buybacks, per-share metrics can be distorted; ensure share count and payout definitions are consistent.

Impact on multiples:
- Normalised EPS shifts P/E materially; using reported P/E can mis-rank companies if one-offs differ across firms.
- Forward P/E can be more informative when near-term earnings are depressed or inflated, but only if forecasts are credible and comparable.

Impact on DCF inputs:
- P/E can be reverse-engineered into implied growth and k_e; large gaps vs DCF often indicate inconsistent earnings definitions or hidden assumptions.
- Use P/E as a cross-check on your equity valuation (Chapters 14–16), not as a substitute for understanding cash flows and reinvestment.

Practical adjustments:
```python
def implied_growth_from_pe(price: float, eps_next: float, payout: float, ke: float) -> Optional[float]:
    """
    From proxy P/E ≈ payout/(ke - g) and price/eps_next = P/E, solve g ≈ ke - payout/(P/E).
    Returns None if inputs invalid.
    """
    if not all(_num(v) for v in [price, eps_next, payout, ke]):
        raise ValueError("inputs must be numeric")
    if eps_next <= 0 or price <= 0:
        return None
    if not (0.0 <= payout <= 1.0):
        raise ValueError("payout must be in [0,1]")
    pe = price / eps_next
    if pe <= 0:
        return None
    g = ke - payout / pe
    return g
```

---

### Quality of Earnings Flags (Earnings Multiples)
⚠️ Repeated “non-recurring” charges—normalisation becomes permanent.  
⚠️ EPS boosted by share buybacks funded with debt while operating earnings are flat (financial engineering).  
⚠️ Large impairment reversals or asset sale gains in earnings (non-operating).  
⚠️ Tax rate anomalies (temporary credits/one-offs) inflate net income and P/E.  
⚠️ P/E comparisons across very different leverage (P/E penalises high leverage differently).  
✅ Green flag: stable operating performance, transparent earnings reconciliation, and a consistent peer definition of “earnings.”

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Cyclicals | Earnings swing with cycle | Use mid-cycle EPS/earnings; avoid peak/trough P/E. |
| Financials | Earnings tied to leverage/regulation | P/E useful with ROE and risk controls; check provisioning cycle. |
| Tech | SBC and intangible amortisation | Normalise for SBC policy and intangible charges; consider EV/Sales when early-stage. |
| Utilities | Stable payout | P/E can work; tie to payout and allowed returns. |

---

### Practical Comparison Tables

#### 1) When to use P/E vs alternatives
| Condition | Use P/E? | Better alternative |
|---|---|---|
| Positive, stable earnings | Yes | EV/EBITDA for cap-structure neutrality |
| Earnings volatile (cyclical) | Only with mid-cycle EPS | EV/EBIT on normalised EBIT |
| Earnings near zero/negative | No | EV/Sales + margin path; EV/GP |
| Heavy one-offs | Only after normalising | Use adjusted earnings or switch metric |

#### 2) Normalisation checklist for EPS
| Adjustment | Add/Subtract | Why |
|---|---|---|
| Restructuring | Add back (if truly one-off) | Sustainable earnings |
| Asset sale gains | Subtract | Not recurring |
| Tax one-offs | Remove | Avoid distorted net income |
| SBC | Policy decision | Comparability and true cost |
| Amortisation of acquired intangibles | Case-by-case | Can be large, non-cash, but recurring from M&A strategy |

---

### Real-World Example
Scenario: A company reports EPS depressed by a one-time legal settlement and boosted by a one-time asset sale gain. You normalise EPS and compare P/E to peers, then compute an implied growth sanity check.

```python
price = 42.0
eps_next = 2.30
payout = 0.35
ke = 0.10

g_implied = implied_growth_from_pe(price, eps_next, payout, ke)
print(f"Implied growth (proxy): {g_implied:.1%}" if g_implied is not None else "Implied growth: n/a")
```

Interpretation: If implied growth is unrealistically high for a mature firm, the market P/E is likely supported by other factors (lower perceived risk, expected ROE improvement) or the earnings denominator is temporarily depressed. Normalisation and forward earnings context are essential before concluding mispricing.
