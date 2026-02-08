# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 3: How Financial Statements Are Used in Valuation (Multiples, Comparables, NPV, Cost of Capital, Going-Concern Valuation)

### Core Concept
Financial statements are used in valuation in three main ways: (1) to anchor relative valuation via multiples, (2) to support asset-based valuation when balance-sheet values matter, and (3) to power fundamental valuation by forecasting and converting expected payoffs into intrinsic value. The practical challenge is separating operating performance from financing effects and ensuring that the metric being priced is aligned with the claim being valued (equity vs enterprise).

---

### Formula/Methodology

#### 1) Multiple analysis (pricing relative to an attribute)
```text
P/E = Price per share / Earnings per share

P/B = Price per share / Book value per share

EV/EBIT = Enterprise value / EBIT

EV/EBITDA = Enterprise value / EBITDA
```

Where:
- Price per share = market price of common equity per share
- Earnings per share (EPS) = earnings attributable to common shareholders / weighted avg shares
- Book value per share (BVPS) = common shareholders’ equity / shares
- Enterprise value (EV) = market value of equity + net debt + minority interest - cash (definition must be consistent)
- EBIT/EBITDA = operating profit measures (ensure comparability across firms)

Alignment rule (non-negotiable):
- Equity multiples (P/E, P/B) must use equity numerator (price or equity value) with equity denominator (earnings to common, book value).
- Enterprise multiples (EV/EBIT, EV/EBITDA) must use enterprise numerator with operating denominators (before financing).

#### 2) Discounted cash flow / present value logic (terminal investments)
```text
PV = Σ [ CF_t / (1 + r)^t ]  for t = 1..N
```

Project net present value:
```text
NPV = Σ [ CF_t / (1 + r)^t ] - Initial Investment
```

Where:
- CF_t = cash flow in period t (consistent definition: to equity or to firm)
- r = required return matching the risk of CF_t
- N = forecast horizon (finite for terminal investments; practical horizon for going concerns)

Value created:
```text
Value Added = PV(payoffs) - Cost
```

#### 3) Required return (cost of capital) framing
```text
Required Return = Risk-free Rate + Risk Premium
```

CAPM form (common):
```text
Required Return = Risk-free Rate + Beta × Market Risk Premium
```

Where:
- Beta = sensitivity of asset returns to market returns
- Market risk premium = expected market excess return over risk-free

---

### Practical Application (How to apply)

#### A) Build a valuation “map” from financial statements
1) Identify the object being valued:
- Equity value (common shareholders) vs enterprise value (operations financed by debt + equity)

2) Separate operating vs financing:
- Operating performance metrics: revenue, operating margin, EBIT, NOPAT
- Financing items: interest expense, net debt, share count, dividends/repurchases

3) Choose the valuation tool and enforce alignment:
- Multiples: pick numerator/denominator consistent with claim
- Asset-based: adjust book values to economic values (only when justified)
- Fundamental valuation: forecast payoff attribute and discount with matched required return

#### B) Using multiples without misleading yourself
Peer selection checklist (practical):
- Similar business model and value drivers (pricing power, churn, unit economics)
- Similar risk profile (cyclicality, leverage, operating leverage)
- Similar accounting (revenue recognition, capitalisation vs expensing, leases/pensions)
- Similar growth phase (early/high-growth vs mature/stable)

Use robust peer statistics:
- Prefer median and IQR over simple average when distributions are skewed.

```text
Median Multiple = P50(peer multiples)
IQR = P75(peer multiples) - P25(peer multiples)
```

#### C) When asset-based valuation is appropriate
Use asset-based / breakup valuation when:
- Assets are separable and marketable (real estate-heavy, investment holdings)
- Going-concern earnings are not reliable (distress, restructuring, liquidation)
- A sum-of-parts logic is economically meaningful (diversified groups)

Breakup framing:
```text
Equity Value ≈ Σ (Fair Value of Assets) - Σ (Fair Value of Liabilities and Senior Claims)
```

#### D) Going-concern valuation: the two hard problems and the practical fix
Going concerns create two implementation problems:
- Forecast horizon: value exists beyond explicit forecasts
- What to forecast: cash flows are not directly reported; accounting provides observable anchors (earnings, book value, operating assets)

Practical model criteria:
- Finite forecast horizon (explicit years) + a disciplined continuation assumption
- Validation: forecast items that will show up in statements
- Parsimony: avoid overfitting with too many speculative drivers

---

### Python Implementation
```python
from typing import Any, Dict, Iterable, List, Optional, Sequence
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def safe_divide(numerator: float, denominator: float, *, on_zero: Optional[float] = None) -> Optional[float]:
    """Safe division to avoid silent spreadsheet errors.

    Args:
        numerator: value in numerator
        denominator: value in denominator
        on_zero: returned if denominator is 0 (default None)

    Returns:
        numerator/denominator or on_zero

    Raises:
        ValueError: non-finite inputs
    """
    num = _f(numerator, "numerator")
    den = _f(denominator, "denominator")
    if abs(den) < 1e-12:
        return on_zero
    return float(num / den)

def enterprise_value(market_cap: float, net_debt: float, *, minority_interest: float = 0.0, cash: float = 0.0) -> float:
    """Compute a consistent EV definition: EV = MarketCap + NetDebt + MI - Cash.

    Args:
        market_cap: market value of equity ($)
        net_debt: total debt minus cash-like items ($)
        minority_interest: minority interest ($)
        cash: excess cash to subtract ($)

    Returns:
        float: enterprise value ($)
    """
    mc = _f(market_cap, "market_cap")
    nd = _f(net_debt, "net_debt")
    mi = _f(minority_interest, "minority_interest")
    c = _f(cash, "cash")
    return float(mc + nd + mi - c)

def multiple_ev_to_metric(ev: float, metric: float) -> float:
    """Enterprise multiple (EV/Metric) with validation."""
    e = _f(ev, "ev")
    m = _f(metric, "metric")
    out = safe_divide(e, m, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Metric must be non-zero to compute multiple.")
    return float(out)

def pv_cashflows(cashflows: Sequence[float], rate: float, *, start_t: int = 1) -> float:
    """Present value of end-of-period cashflows.

    PV = Σ CF_t / (1+r)^t

    Args:
        cashflows: CFs for t=start_t..start_t+N-1
        rate: discount rate per period (decimal)
        start_t: starting period index (1 typical)

    Returns:
        float: present value

    Raises:
        ValueError: invalid inputs (rate <= -100%, empty cashflows, non-finite values)
    """
    r = _f(rate, "rate")
    if r <= -0.999999:
        raise ValueError("rate must be > -100%.")
    if start_t < 0:
        raise ValueError("start_t must be >= 0.")
    if not isinstance(cashflows, (list, tuple, np.ndarray)):
        raise ValueError("cashflows must be a sequence.")
    cf = np.array([_f(x, "cashflow") for x in cashflows], dtype=float)
    if cf.size == 0:
        raise ValueError("cashflows must be non-empty.")
    t = np.arange(start_t, start_t + cf.size, dtype=float)
    df = 1.0 / np.power(1.0 + r, t)
    return float(np.sum(cf * df))

# Example usage (multiples + PV)
company = {
    "market_cap": 2_000_000_000,     # $2.0bn
    "net_debt": 500_000_000,         # $0.5bn
    "minority_interest": 50_000_000, # $0.05bn
    "cash": 100_000_000,             # $0.10bn
    "ebitda": 300_000_000            # $0.30bn
}

ev = enterprise_value(company["market_cap"], company["net_debt"],
                      minority_interest=company["minority_interest"],
                      cash=company["cash"])
ev_ebitda = multiple_ev_to_metric(ev, company["ebitda"])
print(f"EV: ${ev/1e9:.2f}bn")
print(f"EV/EBITDA: {ev_ebitda:.1f}x")

fcf = [120e6, 135e6, 150e6, 165e6, 180e6]  # $m
pv = pv_cashflows(fcf, rate=0.095, start_t=1)
print(f"PV of FCF: ${pv/1e9:.2f}bn")
```

---

### Valuation Impact
Why this matters:
- Financial statements anchor valuation inputs that are observable and auditable (earnings, book values, operating assets), helping keep forecasts disciplined and reducing “story-based” overreach.
- Correct separation of operations and financing prevents numerator/denominator mismatches that can materially distort multiples and enterprise-to-equity bridges.

Impact on multiples:
- Misaligned multiples (e.g., EV divided by after-interest earnings) produce meaningless comparisons.
- Accounting differences (leases, capitalised development, revenue timing) can drive multiples more than economics—requiring normalisation when comparing peers.

Impact on DCF inputs:
- Discount rate must match the cash flow being discounted (to equity vs to firm).
- Forecast horizon discipline and continuation assumptions can dominate intrinsic value; statement-based forecasting improves validation.

Comparability issues across companies:
- Differences in capitalisation vs expensing shift earnings and book values (and therefore P/E and P/B).
- Different financing structures affect equity multiples more than enterprise multiples.

Practical adjustments:
```python
def normalise_earnings(reported_earnings: float, unusual_items: Sequence[float]) -> float:
    """Remove unusual/nonrecurring items to estimate sustainable earnings."""
    rep = _f(reported_earnings, "reported_earnings")
    adj = np.array([_f(x, "unusual_item") for x in unusual_items], dtype=float)
    return float(rep - adj.sum())
```

---

### Quality of Earnings Flags
⚠️ Low P/E driven by temporarily inflated earnings (one-offs, reversals, gains on disposals).  
⚠️ High P/E driven by depressed earnings (restructuring, write-downs, front-loaded expenses) rather than sustainable growth.  
⚠️ Peer multiple differences explained mainly by accounting policy differences (leases, capitalisation, revenue timing).  
✅ Multiples tie to sustainable earnings/operating profit and are consistent with claim (equity vs enterprise).

---

### Sector-Specific Considerations

| Sector | Key comparability issue | Typical treatment |
|---|---|---|
| Technology / software | expensing vs capitalising development; revenue timing | normalise EBITDA/EBIT; reconcile deferred revenue and capitalised costs |
| Retail / airlines | lease intensity distorts EV/EBITDA and leverage | treat leases consistently; adjust EV and EBITDA if peer set mixes standards |
| Banks/insurers | book value and earnings are closer to “operating” | prefer P/B, ROE-based comparisons; avoid EV/EBITDA |
| Asset-heavy real estate | fair value vs cost model differences | align asset measurement basis; consider NAV-style approaches |

---

### Real-World Example
Scenario: Use enterprise multiple and a statement-anchored PV check as a cross-validation.

```python
company = {"market_cap": 2_000_000_000, "net_debt": 500_000_000, "ebitda": 300_000_000}
ev = enterprise_value(company["market_cap"], company["net_debt"])
multiple = multiple_ev_to_metric(ev, company["ebitda"])

fcf = [120e6, 135e6, 150e6, 165e6, 180e6]
pv = pv_cashflows(fcf, rate=0.095)

print(f"EV: ${ev/1e9:.2f}bn | EV/EBITDA: {multiple:.1f}x | PV(FCF): ${pv/1e9:.2f}bn")
```

Interpretation: If EV/EBITDA implies a valuation far outside the PV-based range, investigate whether the peer set, earnings quality, or financing/lease treatment is driving the difference.

See also: Chapter 4 (accrual vs cash accounting and DCF), Chapter 18 (quality of financial statements).
