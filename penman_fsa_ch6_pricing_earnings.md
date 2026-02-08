# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 6: Accrual Accounting and Valuation — Pricing Earnings (P/E, Earnings Quality, Abnormal Earnings Growth)

### Core Concept
Earnings are central to valuation because they summarise value creation over a period, but accrual accounting means earnings are not the same as cash flow. Pricing earnings well requires separating sustainable operating earnings from transitory items and understanding how accounting choices shift earnings across time. In practice, valuation work uses earnings-based multiples and earnings-based intrinsic value models (abnormal earnings / residual income) as complements and cross-checks to DCF.

---

### Formula/Methodology

#### 1) P/E and earnings yield
```text
P/E = Price per share / Earnings per share
```

Earnings yield (useful for cross-checks):
```text
Earnings Yield = EPS / Price per share = 1 / (P/E)
```

Where:
- EPS should be defined consistently (basic vs diluted, continuing operations vs total, normalised vs reported)

#### 2) Sustainable earnings and normalisation
A practical “normalised earnings” construct:
```text
Normalised Earnings = Reported Earnings
                     - After-tax transitory gains/losses
                     ± Reclassification adjustments (operating vs financing)
                     - Unsustainable accounting benefits
```

Common transitory items to exclude (case-by-case):
- restructuring charges (if non-recurring), one-off litigation, impairment losses (often non-cash but signal economics), asset sale gains, fair-value remeasurement noise

After-tax adjustment:
```text
After-tax Adjustment = Pre-tax Adjustment × (1 - Tax rate)
```

#### 3) Abnormal earnings / residual income (earnings-based intrinsic value)
Abnormal earnings (residual income):
```text
AE_t = Earnings_t - Re × BV_(t-1)
```

Equity value identity:
```text
Equity Value_0 = BV_0 + Σ [ AE_t / (1 + Re)^t ] + PV(continuing AE)
```

A common continuation assumption is that abnormal earnings fade toward zero:
```text
AE_{t+1} = φ × AE_t   where 0 ≤ φ < 1
```

Then (one practical closed form for continuation at time N):
```text
CV_N = AE_{N+1} / (Re - φ)   (requires φ < Re in level terms if treated as growth-like; safer to compute PV by simulation)
```

Practical approach:
- Avoid fragile closed forms; model a fade path for AE explicitly over a few years and set AE → 0.

#### 4) Abnormal earnings growth (AEG) intuition for P/E
A useful way to interpret P/E is that price reflects:
- current capital invested (book value), plus
- the present value of future abnormal earnings (or abnormal earnings growth)

A simplified logic for “earnings are valuable only if they are expected to persist and grow”:
```text
Higher persistence of normalised earnings  => higher P/E
Higher expected growth in abnormal earnings => higher P/E
Higher Re (risk)                           => lower P/E
```

#### 5) Operating vs financing separation (to make earnings comparable)
When comparing earnings multiples across firms, classify:
- Operating items: revenues, operating costs, operating assets
- Financing items: interest income/expense on net debt, financing gains/losses, preference dividends

A common adjustment:
```text
Operating Earnings (after tax) = (EBIT × (1 - Tax rate))
```

---

### Practical Application (How to apply)

#### A) Define the “earnings” you are pricing (do this before using any multiple)
Pick one and stick to it across peers:
- EPS from continuing operations (preferred for comparability)
- Normalised EPS (for cyclical/trough years)
- Operating earnings (EBIAT/NOPAT) for enterprise-multiple logic

Minimum disclosure checklist:
- basic vs diluted shares
- discontinued operations
- exceptional items policy (management vs audited)
- IFRS vs US GAAP differences affecting earnings timing

#### B) Normalise earnings using a repeatable ruleset
Workflow:
1) Start from reported profit attributable to common shareholders
2) Strip out transitory items (document each adjustment and its tax effect)
3) Reclassify financing vs operating items consistently (especially for enterprise-style comparisons)
4) Recompute “normalised EPS” and “normalised ROE”

Rules of thumb:
- Adjust if the item is (i) material, (ii) non-recurring, and (iii) not expected to recur at a similar magnitude
- If the “one-off” recurs every year, it is not a one-off (treat as structural)

#### C) Tie P/E to ROE, payout, and growth (consistency check)
For stable firms, sanity-check P/E against:
- ROE vs Re (value creation)
- payout policy and expected growth
- sustainability of margins and reinvestment needs

If P/E is high but:
- ROE is not sustainably above Re, or
- earnings quality is weak,
then treat the multiple as fragile.

#### D) Use earnings-based intrinsic value as a cross-check to DCF
Do both:
- DCF: cash flow story (reinvestment, cash conversion)
- Residual income: accounting story (book value anchor, earnings persistence)

Reconcile:
- If DCF is high but residual income is low, you likely over-modeled cash flow (e.g., under-modeled reinvestment) or mis-specified discount rate.

---

### Python Implementation
```python
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
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

def after_tax_adjustment(pre_tax_amount: float, tax_rate: float) -> float:
    """After-tax = pre-tax × (1 - tax_rate)."""
    a = _f(pre_tax_amount, "pre_tax_amount")
    t = _f(tax_rate, "tax_rate")
    if t < 0 or t > 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    return float(a * (1.0 - t))

def normalised_earnings(reported_earnings: float,
                        adjustments_pre_tax: Sequence[float],
                        tax_rate: float) -> float:
    """Normalised earnings by removing transitory items (pre-tax list), tax-effected.

    Convention:
    - Positive adjustment_pre_tax = item included in reported earnings that you want to REMOVE (e.g., one-off gain).
    - Negative adjustment_pre_tax = expense included in reported earnings that you want to ADD BACK (e.g., one-off restructuring).
    Thus: Normalised = Reported - AfterTax(sum(adjustments_pre_tax))
    """
    rep = _f(reported_earnings, "reported_earnings")
    if not isinstance(adjustments_pre_tax, (list, tuple, np.ndarray)):
        raise ValueError("adjustments_pre_tax must be a sequence.")
    adj = np.array([_f(x, "adjustment") for x in adjustments_pre_tax], dtype=float)
    at = after_tax_adjustment(float(np.sum(adj)), tax_rate)
    return float(rep - at)

def eps(earnings_to_common: float, weighted_avg_shares: float) -> float:
    """EPS = earnings to common / weighted average shares."""
    e = _f(earnings_to_common, "earnings_to_common")
    s = _f(weighted_avg_shares, "weighted_avg_shares")
    out = safe_divide(e, s, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("weighted_avg_shares must be non-zero.")
    return float(out)

def pe_ratio(price_per_share: float, eps_value: float) -> float:
    """P/E = price / EPS (requires EPS > 0 for interpretability)."""
    p = _f(price_per_share, "price_per_share")
    e = _f(eps_value, "eps_value")
    if abs(e) < 1e-12:
        raise ValueError("EPS must be non-zero.")
    # Allow negative EPS but flag that the ratio is not meaningful for valuation
    return float(p / e)

def earnings_yield(price_per_share: float, eps_value: float) -> float:
    """Earnings yield = EPS / price."""
    p = _f(price_per_share, "price_per_share")
    e = _f(eps_value, "eps_value")
    out = safe_divide(e, p, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("price_per_share must be non-zero.")
    return float(out)

def residual_income(earnings: float, cost_of_equity: float, opening_bv: float) -> float:
    """AE/RI = earnings - Re * opening BV."""
    e = _f(earnings, "earnings")
    re = _f(cost_of_equity, "cost_of_equity")
    bv = _f(opening_bv, "opening_bv")
    if re <= -0.999999:
        raise ValueError("cost_of_equity must be > -100%.")
    return float(e - re * bv)

def pv_abnormal_earnings(bv0: float,
                         earnings: Sequence[float],
                         dividends: Sequence[float],
                         cost_of_equity: float,
                         *,
                         fade_to_zero_years: int = 5) -> Dict[str, float]:
    """Equity value via BV0 + PV(abnormal earnings) with explicit fade to zero.

    Method:
    - Compute AE each year from earnings and opening BV
    - Discount AE at Re
    - After the explicit horizon, assume AE fades linearly to 0 over `fade_to_zero_years`
      and discount those additional AE terms (no perpetuity closed form).

    This is robust vs fragile terminal formulas.

    Raises:
        ValueError: invalid inputs
    """
    b0 = _f(bv0, "bv0")
    re = _f(cost_of_equity, "cost_of_equity")
    if re <= -0.999999:
        raise ValueError("cost_of_equity must be > -100%.")
    if fade_to_zero_years < 0:
        raise ValueError("fade_to_zero_years must be >= 0.")

    if not isinstance(earnings, (list, tuple, np.ndarray)) or not isinstance(dividends, (list, tuple, np.ndarray)):
        raise ValueError("earnings and dividends must be sequences.")
    e = np.array([_f(x, "earnings") for x in earnings], dtype=float)
    d = np.array([_f(x, "dividends") for x in dividends], dtype=float)
    if e.size != d.size:
        raise ValueError("earnings and dividends must have the same length.")
    # roll forward BV using clean surplus (no OCI here; incorporate separately if needed)
    bv = np.empty(e.size + 1, dtype=float)
    bv[0] = b0
    for t in range(1, bv.size):
        bv[t] = bv[t-1] + e[t-1] - d[t-1]

    ae = np.array([residual_income(e[t], re, bv[t]) for t in range(e.size)], dtype=float)

    t = np.arange(1, ae.size + 1, dtype=float)
    pv_ae = float(np.sum(ae / np.power(1.0 + re, t)))

    # Fade years AE after horizon
    pv_fade = 0.0
    if fade_to_zero_years > 0:
        last_ae = float(ae[-1])
        # Linear fade: AE_{N+k} = last_ae * (1 - k/(fade_years+1))
        extra = []
        for k in range(1, fade_to_zero_years + 1):
            extra.append(last_ae * (1.0 - (k / (fade_to_zero_years + 1.0))))
        extra = np.array(extra, dtype=float)
        t2 = np.arange(ae.size + 1, ae.size + extra.size + 1, dtype=float)
        pv_fade = float(np.sum(extra / np.power(1.0 + re, t2)))

    equity_value = float(b0 + pv_ae + pv_fade)
    return {"BV0": float(b0), "PV_AE": pv_ae, "PV_Fade": pv_fade, "Equity_Value": equity_value}

# Example usage: normalise earnings, compute P/E and an abnormal earnings value
company = {
    "price": 24.0,                 # $/share
    "reported_net_income": 420e6,   # $
    "shares": 200e6,               # shares
    "tax_rate": 0.25,
    # adjustments_pre_tax: +gain to remove, -expense to add back
    "adjustments_pre_tax": [120e6, -60e6],  # remove $120m one-off gain; add back $60m one-off restructuring
    "bv0": 2_800e6,                # $ book value at start
    "dividends": [120e6, 130e6, 140e6, 150e6, 160e6],
    "earnings_forecast": [450e6, 470e6, 490e6, 510e6, 530e6],
    "re": 0.095
}

norm_ni = normalised_earnings(company["reported_net_income"], company["adjustments_pre_tax"], company["tax_rate"])
norm_eps = eps(norm_ni, company["shares"])
pe = pe_ratio(company["price"], norm_eps)
ey = earnings_yield(company["price"], norm_eps)

val = pv_abnormal_earnings(company["bv0"], company["earnings_forecast"], company["dividends"], company["re"], fade_to_zero_years=5)

print(f"Normalised EPS: ${norm_eps:.2f} | P/E: {pe:.1f}x | Earnings Yield: {ey:.1%}")
print(f"Equity Value (AE model): ${val['Equity_Value']/1e9:.2f}bn")
```

---

### Valuation Impact
Why this matters:
- Earnings are the most-used valuation input (multiples), but earnings quality determines whether the multiple is meaningful. A clean, repeatable normalisation approach reduces “multiple shopping” and improves defensibility.
- Earnings-based intrinsic value (abnormal earnings / residual income) provides a balance-sheet-anchored alternative to DCF, particularly useful when cash flows are distorted by reinvestment timing.

Impact on multiples:
- Reported EPS inflated by transitory gains leads to artificially low P/E (value trap risk).
- Capitalisation policies and impairment timing can shift earnings materially and distort cross-company P/E comparability.

Impact on DCF inputs:
- Normalised earnings informs sustainable margins, tax rates, and reinvestment assumptions.
- Abnormal earnings frameworks highlight whether high DCF values rely on persistent competitive advantage.

Comparability issues across companies:
- IFRS vs US GAAP treatment of development costs, leases, stock compensation, and impairments can shift earnings.
- Different “adjusted EBITDA” definitions can diverge from audited earnings; reconcile carefully.

Practical adjustments:
```python
def normalise_eps(reported_eps: float, after_tax_adjustments_per_share: float) -> float:
    """Normalised EPS = Reported EPS - after-tax adjustments per share."""
    import numpy as np
    eps_r = float(reported_eps)
    adj = float(after_tax_adjustments_per_share)
    if not np.isfinite(eps_r) or not np.isfinite(adj):
        raise ValueError("Inputs must be finite.")
    return float(eps_r - adj)
```

---

### Quality of Earnings Flags
⚠️ Large “exceptional” items every year (recurring one-offs).  
⚠️ Material gap between earnings growth and operating cash flow growth not explained by working capital seasonality.  
⚠️ Significant earnings uplift driven by accounting changes, reclassifications, or fair-value gains.  
⚠️ EPS accretion from buybacks financed by leverage without operational improvement (higher risk, fragile P/E).  
✅ Normalised earnings track operating drivers (volume, price, margin) and reconcile to cash over the cycle.

---

### Sector-Specific Considerations

| Sector | Key earnings-quality issue | Typical treatment |
|---|---|---|
| Banks | provisions/credit losses and fair value through P&L | normalise through-cycle losses; reconcile OCI/FV movements |
| Insurers | reserve changes and assumption updates | focus on underlying underwriting/operating earnings |
| Software | capitalised development vs expensed R&D; SBC | treat SBC consistently; consider capitalising R&D for comparability |
| Industrials | restructuring and impairments in cycles | normalise over cycle; separate operating vs transitory items |

---

### Real-World Example
Scenario: A company looks cheap on reported P/E due to a one-off gain; normalisation changes the conclusion.

```python
price = 24.0
reported_net_income = 420e6
shares = 200e6
tax_rate = 0.25

# remove $120m pre-tax gain; add back $60m pre-tax restructuring
adjustments_pre_tax = [120e6, -60e6]
norm_ni = normalised_earnings(reported_net_income, adjustments_pre_tax, tax_rate)
norm_eps = eps(norm_ni, shares)
pe_norm = pe_ratio(price, norm_eps)

print(f"Normalised EPS: ${norm_eps:.2f}")
print(f"Normalised P/E: {pe_norm:.1f}x")
```

Interpretation: If “cheapness” disappears after normalisation, the original low P/E was driven by transitory earnings and should not be used to anchor valuation.

See also: Chapter 4 (cash vs accrual and DCF), Chapter 5 (pricing book values), Chapter 18 (quality of financial statements).
