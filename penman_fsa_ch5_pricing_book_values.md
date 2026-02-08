# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 5: Accrual Accounting and Valuation — Pricing Book Values (P/B, Residual Income Logic, Clean Surplus, Balance-Sheet Anchors)

### Core Concept
Book value (common shareholders’ equity) is both an anchor for valuation and a scorecard of value creation through retained earnings and other equity changes. In practice, price-to-book (P/B) is most informative when book value is measured close to economic value and when profitability (ROE) and growth in book value are stable and interpretable. Accrual accounting matters because it determines what is on the balance sheet and when gains/losses hit equity.

---

### Formula/Methodology

#### 1) Book value and clean surplus relation (CSR)
Book value of equity evolves with earnings and dividends (plus “dirty surplus” items if present).
```text
BV_t = BV_(t-1) + Earnings_t - Dividends_t  (+/- Other comprehensive income & direct-to-equity items)
```

Clean surplus (idealised, for many valuation identities):
```text
BV_t = BV_(t-1) + Earnings_t - Dividends_t
```

Where:
- BV_t = book value of common equity at time t
- Earnings_t = earnings attributable to common shareholders for period t
- Dividends_t = dividends to common shareholders for period t

Practical note:
- IFRS/US GAAP include items that bypass the income statement (OCI, hedging reserves, FX translation, remeasurements). For valuation, you either (a) model them explicitly, or (b) treat them as part of comprehensive income and adjust ROE.

#### 2) Price-to-book and its driver: ROE
```text
P/B = Market value of equity / Book value of equity
```

Return on equity:
```text
ROE = Earnings to common / Average common equity
```

Growth in book value (approx, under clean surplus):
```text
g_BV ≈ ROE × (1 - Payout Ratio)
```

Where:
- Payout Ratio = Dividends / Earnings (if earnings > 0)

#### 3) Residual income (RI) / abnormal earnings framing (equity intrinsic value)
Residual income is earnings in excess of the required return on beginning book value:
```text
RI_t = Earnings_t - Re × BV_(t-1)
```

Equity value identity (residual income model):
```text
Equity Value_0 = BV_0 + Σ [ RI_t / (1 + Re)^t ]  for t = 1..N  + PV(continuing RI)
```

Where:
- Re = cost of equity (required return for equity holders)

Continuation value (perpetuity of residual income, simplified):
```text
CV_N = RI_(N+1) / (Re - g_RI)
```

Constraints:
- g_RI must be < Re
- RI should converge toward 0 in competitive markets unless durable advantage exists

#### 4) Balance-sheet “anchors” vs “noise”: when BV is more/less useful
BV is typically more informative when:
- assets/liabilities are close to fair value (financial firms, investment property at fair value)
- earnings are volatile but equity base is measurable (cyclical troughs)
- liquidation or breakup valuation is plausible

BV is less informative when:
- large internally generated intangibles are expensed (brand, R&D-heavy, platform effects)
- aggressive accounting inflates assets (capitalised costs, weak impairment discipline)
- off-balance-sheet obligations are material (leases, pensions) unless adjusted

---

### Practical Application (How to apply)

#### A) Use P/B with ROE and growth as a three-part diagnostic
1) Compute P/B, ROE, and book value growth
2) Check consistency:
- High P/B should be supported by ROE materially above Re and/or strong growth in residual income
- Low P/B can reflect ROE below Re, asset quality concerns, or accounting conservatism

3) Decompose P/B judgementally:
- Franchise value (PV of future RI) vs book value (current invested equity)

#### B) Build an “economic book value” when BV is distorted
Common adjustments (case-by-case, based on disclosures):
- Capitalise and amortise R&D (or certain development costs) to make BV reflect invested capital
- Reclassify operating leases / pensions onto balance sheet (for comparability)
- Normalise goodwill/intangibles treatment if peer set differs materially (use tangible book when appropriate)

Adjustment discipline:
- Only adjust when it improves comparability and you can defend the mechanics from disclosures.
- Track tax effects where relevant (deferred tax impacts of remeasurement/capitalisation).

#### C) Residual income is often more robust than DCF when:
- short-term cash flows are noisy (working capital swings, capex timing)
- earnings and book value are more stable and forecastable than cash
- you need a statement-anchored model that reconciles to equity book value

#### D) Implementation checklist (statement-based forecasting)
For each forecast year:
- Forecast earnings to common (or comprehensive income if material OCI)
- Forecast dividends/repurchases (payout)
- Roll forward BV using clean surplus (plus explicit OCI if needed)
- Compute RI each year and discount at Re
- Add BV_0

---

### Python Implementation
```python
from typing import Any, Dict, Sequence, Optional
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def safe_divide(n: float, d: float, *, on_zero: Optional[float] = None) -> Optional[float]:
    n = _f(n, "numerator")
    d = _f(d, "denominator")
    if abs(d) < 1e-12:
        return on_zero
    return float(n / d)

def pb_ratio(market_cap: float, book_value_equity: float) -> float:
    """P/B = Market value of equity / Book value of equity."""
    p = _f(market_cap, "market_cap")
    bv = _f(book_value_equity, "book_value_equity")
    out = safe_divide(p, bv, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("book_value_equity must be non-zero.")
    return float(out)

def roe(earnings: float, avg_equity: float) -> float:
    """ROE = Earnings / Average equity."""
    e = _f(earnings, "earnings")
    ae = _f(avg_equity, "avg_equity")
    out = safe_divide(e, ae, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("avg_equity must be non-zero.")
    return float(out)

def roll_forward_book_value(bv0: float, earnings: Sequence[float], dividends: Sequence[float], *,
                            oci: Optional[Sequence[float]] = None) -> np.ndarray:
    """Roll forward book value using clean surplus (+ optional OCI/direct-to-equity items).

    BV_t = BV_(t-1) + Earnings_t - Dividends_t (+ OCI_t)

    Args:
        bv0: starting book value ($)
        earnings: sequence of earnings per period ($)
        dividends: sequence of dividends per period ($)
        oci: optional sequence of other equity changes bypassing earnings ($)

    Returns:
        np.ndarray: book value path including BV0 at index 0

    Raises:
        ValueError: length mismatch, invalid inputs
    """
    b0 = _f(bv0, "bv0")
    if not isinstance(earnings, (list, tuple, np.ndarray)) or not isinstance(dividends, (list, tuple, np.ndarray)):
        raise ValueError("earnings and dividends must be sequences.")
    e = np.array([_f(x, "earnings") for x in earnings], dtype=float)
    d = np.array([_f(x, "dividends") for x in dividends], dtype=float)
    if e.size != d.size:
        raise ValueError("earnings and dividends must have the same length.")
    if oci is None:
        o = np.zeros_like(e)
    else:
        o = np.array([_f(x, "oci") for x in oci], dtype=float)
        if o.size != e.size:
            raise ValueError("oci must match length of earnings.")
    bv = np.empty(e.size + 1, dtype=float)
    bv[0] = b0
    for t in range(1, bv.size):
        bv[t] = bv[t-1] + e[t-1] - d[t-1] + o[t-1]
    return bv

def residual_income(earnings: float, cost_of_equity: float, opening_bv: float) -> float:
    """RI = Earnings - Re * BV_(t-1)."""
    e = _f(earnings, "earnings")
    re = _f(cost_of_equity, "cost_of_equity")
    bv = _f(opening_bv, "opening_bv")
    if re <= -0.999999:
        raise ValueError("cost_of_equity must be > -100%.")
    return float(e - re * bv)

def pv_residual_income_model(bv0: float, earnings: Sequence[float], dividends: Sequence[float],
                             cost_of_equity: float, *, g_ri: float = 0.0,
                             include_terminal: bool = True) -> Dict[str, float]:
    """Equity value = BV0 + PV(RI) + PV(Terminal RI), using simple perpetuity of RI.

    Assumptions:
    - Clean surplus holds (or OCI is captured separately before calling)
    - Terminal RI grows at g_ri and g_ri < cost_of_equity

    Returns:
        dict with components
    """
    re = _f(cost_of_equity, "cost_of_equity")
    g = _f(g_ri, "g_ri")
    if g >= re:
        raise ValueError("g_ri must be < cost_of_equity.")
    bv_path = roll_forward_book_value(bv0, earnings, dividends)
    e = np.array([_f(x, "earnings") for x in earnings], dtype=float)

    ri = np.array([residual_income(e[t], re, bv_path[t]) for t in range(e.size)], dtype=float)
    t = np.arange(1, ri.size + 1, dtype=float)
    pv_ri = float(np.sum(ri / np.power(1.0 + re, t)))

    pv_term = 0.0
    term_val = 0.0
    if include_terminal:
        ri_next = float(ri[-1] * (1.0 + g))
        term_val = float(ri_next / (re - g))
        pv_term = float(term_val / np.power(1.0 + re, ri.size))

    total = float(_f(bv0, "bv0") + pv_ri + pv_term)
    return {"BV0": float(bv0), "PV_RI": pv_ri, "Terminal_RI_Value": term_val, "PV_Terminal_RI": pv_term, "Equity_Value": total}

# Example usage: P/B diagnostic + residual income valuation
example = {
    "market_cap": 3_600_000_000,   # $3.6bn
    "bv0": 2_000_000_000,          # $2.0bn
    "earnings": [260e6, 280e6, 300e6, 320e6, 340e6],
    "dividends": [80e6, 90e6, 100e6, 110e6, 120e6],
    "re": 0.10,
    "g_ri": 0.02
}

pb = pb_ratio(example["market_cap"], example["bv0"])
bv_path = roll_forward_book_value(example["bv0"], example["earnings"], example["dividends"])
avg_eq_y1 = 0.5 * (bv_path[0] + bv_path[1])
roe_y1 = roe(example["earnings"][0], avg_eq_y1)

val = pv_residual_income_model(example["bv0"], example["earnings"], example["dividends"], example["re"], g_ri=example["g_ri"])
print(f"P/B: {pb:.2f}x | ROE (Y1): {roe_y1:.1%} | Equity Value: ${val['Equity_Value']/1e9:.2f}bn")
```

---

### Valuation Impact
Why this matters:
- Book value is the “anchor” component of equity value; residual income explains the premium/discount to book. This is directly useful when cash-flow forecasts are noisy or when valuing firms where balance-sheet values are central (financials, asset-heavy firms).
- P/B without ROE (and a required return benchmark) is incomplete; the economic question is whether the firm earns returns above its cost of equity on the equity capital employed.

Impact on multiples:
- Higher sustainable ROE relative to Re supports higher P/B; falling ROE compresses P/B even if earnings are growing.
- P/B is sensitive to accounting conservatism: expensing intangibles lowers BV and can inflate P/B without changing economics.

Impact on DCF inputs:
- Residual income valuation provides a cross-check on DCF: if DCF implies a large premium to BV but RI forecasts show little abnormal earnings, reconcile assumptions.
- Equity discount rate (Re) is the key rate; ensure consistency with risk assumptions.

Comparability issues across companies:
- Different goodwill/intangible accounting (impairment timing, acquisitions vs organic growth) affects BV and ROE.
- Off-balance-sheet obligations (leases/pensions) distort BV-based leverage and ROE unless adjusted.

Practical adjustments:
```python
def adjust_book_value_for_capitalised_rnd(bv_reported: float, unamortised_rnd_asset: float, deferred_tax_effect: float = 0.0) -> float:
    """Increase book value by adding back an imputed R&D asset (net of tax effects if applicable)."""
    import numpy as np
    bv = float(bv_reported); a = float(unamortised_rnd_asset); dt = float(deferred_tax_effect)
    if not np.isfinite(bv) or not np.isfinite(a) or not np.isfinite(dt):
        raise ValueError("Inputs must be finite.")
    return float(bv + a - dt)
```

---

### Quality of Earnings Flags
⚠️ P/B looks “cheap” because book value is overstated (capitalised costs, delayed impairments, aggressive fair values).  
⚠️ High ROE driven primarily by shrinking equity base (buybacks funded by leverage) rather than improved operating performance.  
⚠️ Large OCI swings (FX, hedges, revaluations) not incorporated in ROE/forecast roll-forward.  
✅ ROE consistently above Re with transparent drivers (margins, asset turnover, prudent leverage) and credible BV roll-forward.

---

### Sector-Specific Considerations

| Sector | Key issue for BV and P/B | Typical treatment |
|---|---|---|
| Banks/insurers | BV close to economic capital; earnings tied to asset/liability remeasurement | P/B + ROE as primary; adjust for AFS/FVOCI, credit losses |
| Real estate | fair value vs cost model changes BV materially | use NAV-style BV; align measurement basis across peers |
| Software/biotech | major internally generated intangibles expensed | consider capitalising R&D (analyst adjustment) or use alternative anchors |
| Industrials (acquisitive) | goodwill and intangibles inflate BV; impairments are lumpy | analyse tangible BV; scrutinise impairment discipline |

---

### Real-World Example
Scenario: Explain a P/B premium via ROE and residual income rather than narrative alone.

```python
market_cap = 3.6e9
bv0 = 2.0e9
earnings = [260e6, 280e6, 300e6, 320e6, 340e6]
dividends = [80e6, 90e6, 100e6, 110e6, 120e6]
re = 0.10

pb = pb_ratio(market_cap, bv0)
bv_path = roll_forward_book_value(bv0, earnings, dividends)
roe_y1 = roe(earnings[0], 0.5*(bv_path[0] + bv_path[1]))
val = pv_residual_income_model(bv0, earnings, dividends, re, g_ri=0.02)

print(f"P/B: {pb:.2f}x | ROE Y1: {roe_y1:.1%} | RI-based Equity Value: ${val['Equity_Value']/1e9:.2f}bn")
```

Interpretation: If P/B is high but ROE is only marginally above Re, the premium requires strong and persistent abnormal earnings (or book value understatement). If neither holds, treat the premium as fragile.

See also: Chapter 6 (pricing earnings), Chapter 9 (equity statement analysis), Chapter 18 (quality of financial statements).
