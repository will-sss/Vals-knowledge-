# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 15: Anchoring on the Financial Statements: Simple Forecasting and Simple Valuation (Anchoring, Residual Income, Simple Forecast Rules)

### Core Concept
When full-information forecasting is impractical, valuation can still be disciplined by **anchoring** on current financial statements and using simple, internally consistent forecasting rules. The key is to link price-to-book and price-earnings to **expected profitability (ROE/RNOA)**, **growth in book value / net operating assets**, and **persistence** of abnormal earnings (or residual income), rather than relying on ad hoc “story” forecasts.

---

### Formula/Methodology

#### 1) Residual income (equity perspective)
```text
Residual Income (RI)_t = Earnings_t - (r × Book Value_{t-1})
```
Where:
- Earnings_t = accounting earnings to common equity in period t (after tax, after interest)
- r = cost of equity (as decimal, e.g., 0.10)
- Book Value_{t-1} = beginning-of-period common equity book value

Interpretation:
- RI captures value created beyond the required return on book equity.

#### 2) Clean surplus relation (equity)
```text
Book Value_t = Book Value_{t-1} + Earnings_t - Dividends_t
```
Where:
- Requires “clean” accounting (no direct-to-equity items excluded); in practice, adjust for OCI items if material to common equity changes.

#### 3) Residual income valuation (RIV)
```text
Equity Value_0 = Book Value_0 + Σ_{t=1..T} [ RI_t / (1 + r)^t ] + Terminal Value_RI_T
```

#### 4) Continuing value for residual income (simple perpetuity with persistence)
A common simple closure is **persistence** in residual income:
```text
RI_{t+1} = ω × RI_t
```
Then, the present value of residual income beyond horizon T can be approximated by:
```text
Terminal Value_RI_T = RI_{T+1} / (r - g_RI)
```
Where:
- g_RI is long-run growth rate of residual income (often set to 0 for “fade to zero” when competition erodes abnormal returns), or implied via ω:
```text
If RI grows at rate g_RI: RI_{T+1} = RI_T × (1 + g_RI)
```

Practical choice:
- For mature competitive markets, assume RI fades toward zero (abnormal earnings dissipate).
- For strong moats, use slower fade (higher persistence ω) but cross-check against competitive dynamics and reinvestment needs.

#### 5) Link to ROE and growth (simple forecasting)
Using:
```text
ROE_t = Earnings_t / Average Book Value_t
Retention Ratio (b) = 1 - Payout Ratio
Growth in Book Value (approx) = g_B ≈ ROE × b
```
These provide a simple way to forecast:
- Book value growth from profitability and payout
- Earnings from ROE × book value
- Residual income from earnings and book value

#### 6) Enterprise/operations analogue (optional in this chapter’s spirit)
If you anchor on operations (NOA, OPAT, WACC), you can use a residual-income-like framework:
```text
Residual Operating Income (ReOI)_t = OPAT_t - (WACC × NOA_{t-1})
Enterprise Value_0 ≈ NOA_0 + PV(ReOI_t)
```
Use this when capital structure differences are large and operating separation is clean.

---

### Practical Application (How to apply)

#### A) Build a “simple valuation” workflow in 10 steps
1) Start with current **book value (BV0)** and current **earnings (E1)**.
2) Choose **cost of equity (r)** consistent with your risk view (and cross-check vs implied cost of capital).
3) Compute baseline **RI1 = E1 - r × BV0**.
4) Pick a short explicit horizon (e.g., 3–5 years) for “fade”.
5) Choose a simple **ROE path** (constant ROE, or ROE mean-reverting toward industry).
6) Choose **payout / retention** to forecast BV growth (BV_t = BV_{t-1} + E_t - Div_t).
7) Forecast earnings via ROE × BV.
8) Compute RI each year and discount at r.
9) Close with terminal RI:
   - Fade to zero, or
   - Persistence factor ω, or
   - Long-run g_RI with strong justification.
10) Cross-check implied P/B and P/E against peers and sanity bounds.

#### B) Anchoring rules (practical heuristics)
Use these as “guardrails”:

| Rule | What it prevents | Practical check |
|---|---|---|
| ROE cannot exceed reinvestment-constrained economics indefinitely | unrealistic supernormal profits | compare ROE vs RNOA and competitive intensity |
| Growth without reinvestment is suspicious | free-growth assumption | link growth to retention and incremental investment |
| Residual income should fade in competitive industries | perpetual abnormal returns | impose decay ω < 1 and test sensitivity |
| Payout affects BV growth and future earnings | inconsistent dividend assumptions | reconcile dividends to financing capacity |

#### C) When simple valuation works best
- Stable accounting, meaningful book value, and earnings aligned with value creation (many industrials, services).
- Less reliable where book value is disconnected from economics (early-stage tech, heavy intangibles not capitalised).

---

### Python Implementation
```python
from typing import List, Dict, Any, Optional
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def residual_income(earnings: float, book_value_beg: float, cost_of_equity: float) -> float:
    """Compute residual income: RI = E - r * BV.

    Args:
        earnings: period earnings attributable to common equity ($)
        book_value_beg: beginning-of-period common equity book value ($)
        cost_of_equity: required return on equity (decimal)

    Returns:
        float: residual income ($)

    Raises:
        ValueError: if inputs invalid
    """
    e = _f(earnings, "earnings")
    bv = _f(book_value_beg, "book_value_beg")
    r = _f(cost_of_equity, "cost_of_equity")
    if r <= 0 or r > 0.5:
        raise ValueError("cost_of_equity should be a positive decimal (e.g., 0.10).")
    return float(e - r * bv)

def forecast_simple_riv(
    bv0: float,
    roe_path: List[float],
    payout_ratio_path: List[float],
    cost_of_equity: float,
    terminal_fade: str = "fade_to_zero",
    omega: Optional[float] = None,
    g_ri: Optional[float] = None
) -> Dict[str, Any]:
    """Simple residual income valuation with anchoring on BV and ROE.

    Mechanics:
      - Earnings_t = ROE_t * avg(BV_{t-1}, BV_t) approximately; here we use BV_{t-1} for simplicity.
      - Div_t = payout_ratio_t * Earnings_t
      - BV_t = BV_{t-1} + Earnings_t - Div_t
      - RI_t = Earnings_t - r * BV_{t-1}
      - Value = BV0 + PV(RI_t) + terminal RI value

    Args:
        bv0: current common equity book value ($)
        roe_path: list of ROE assumptions for years 1..T (decimals)
        payout_ratio_path: list of payout ratios for years 1..T (decimals between 0 and 1)
        cost_of_equity: r (decimal)
        terminal_fade: one of {"fade_to_zero","persistence","growth"}
        omega: persistence factor for RI, required if terminal_fade="persistence"
        g_ri: terminal growth for RI, required if terminal_fade="growth"

    Returns:
        dict: schedules and equity value estimate

    Raises:
        ValueError: invalid input shapes or values
    """
    bv = _f(bv0, "bv0")
    r = _f(cost_of_equity, "cost_of_equity")
    if bv < 0:
        raise ValueError("bv0 is negative; check common equity definition or adjustments.")
    if r <= 0 or r > 0.5:
        raise ValueError("cost_of_equity should be a positive decimal (e.g., 0.10).")

    if len(roe_path) == 0:
        raise ValueError("roe_path must have at least one year.")
    if len(roe_path) != len(payout_ratio_path):
        raise ValueError("roe_path and payout_ratio_path must have the same length.")

    T = len(roe_path)

    years = []
    bvs_beg = []
    earnings = []
    dividends = []
    bvs_end = []
    ris = []
    pv_ris = []

    bv_beg = bv
    for t in range(1, T + 1):
        roe = _f(roe_path[t-1], f"roe_year_{t}")
        payout = _f(payout_ratio_path[t-1], f"payout_year_{t}")

        if roe < -1.0 or roe > 1.5:
            raise ValueError(f"roe_year_{t} is implausible; provide as decimal (e.g., 0.12).")
        if payout < 0 or payout > 1:
            raise ValueError(f"payout_year_{t} must be between 0 and 1.")

        e = roe * bv_beg
        d = payout * e
        bv_end = bv_beg + e - d

        ri = residual_income(e, bv_beg, r)
        disc = (1.0 + r) ** t
        pv = ri / disc

        years.append(t)
        bvs_beg.append(bv_beg)
        earnings.append(e)
        dividends.append(d)
        bvs_end.append(bv_end)
        ris.append(ri)
        pv_ris.append(pv)

        bv_beg = bv_end

    # Terminal value on RI beyond T
    if terminal_fade == "fade_to_zero":
        tv = 0.0
    elif terminal_fade == "persistence":
        if omega is None:
            raise ValueError("omega is required for terminal_fade='persistence'.")
        w = _f(omega, "omega")
        if w < 0 or w >= 1:
            raise ValueError("omega should be in [0, 1) for fading persistence.")
        ri_T = ris[-1]
        ri_T1 = w * ri_T
        # Perpetuity of a geometrically declining RI stream:
        # PV at time T = RI_{T+1} / (r - g_RI), with g_RI approximated by (w-1) (negative)
        # Here treat as level growth g_RI = w - 1, so r - g_RI = r - (w-1) = r + 1 - w
        denom = r + 1.0 - w
        if denom <= 0:
            raise ValueError("Terminal denominator non-positive; adjust omega or cost_of_equity.")
        tv_T = ri_T1 / denom
        tv = tv_T / ((1.0 + r) ** T)
    elif terminal_fade == "growth":
        if g_ri is None:
            raise ValueError("g_ri is required for terminal_fade='growth'.")
        g = _f(g_ri, "g_ri")
        if g >= r:
            raise ValueError("g_ri must be less than cost_of_equity for a stable terminal value.")
        ri_T = ris[-1]
        ri_T1 = ri_T * (1.0 + g)
        tv_T = ri_T1 / (r - g)
        tv = tv_T / ((1.0 + r) ** T)
    else:
        raise ValueError("terminal_fade must be one of {'fade_to_zero','persistence','growth'}.")

    equity_value = bv0 + float(np.sum(pv_ris)) + float(tv)

    return {
        "equity_value": float(equity_value),
        "pv_residual_income": float(np.sum(pv_ris)),
        "terminal_value_pv": float(tv),
        "schedule": {
            "year": years,
            "bv_beg": bvs_beg,
            "earnings": earnings,
            "dividends": dividends,
            "bv_end": bvs_end,
            "residual_income": ris,
            "pv_residual_income": pv_ris
        }
    }

# Example usage (numbers in $m)
inputs = {
    "bv0": 2_000.0,
    "roe_path": [0.14, 0.13, 0.12, 0.11, 0.10],       # fade toward 10%
    "payout_path": [0.30, 0.35, 0.40, 0.45, 0.50],    # rising payout
    "r": 0.10
}

out = forecast_simple_riv(
    bv0=inputs["bv0"],
    roe_path=inputs["roe_path"],
    payout_ratio_path=inputs["payout_path"],
    cost_of_equity=inputs["r"],
    terminal_fade="fade_to_zero"  # conservative closure
)

print(f"Equity value: ${out['equity_value']:.0f}m")
print(f"PV(RI): ${out['pv_residual_income']:.0f}m | PV(Terminal): ${out['terminal_value_pv']:.0f}m")
```

---

### Valuation Impact
Why this matters:
- Provides a practical alternative to DCF when cash-flow forecasting is noisy: value can be anchored on book value plus discounted abnormal earnings.
- Helps avoid “growth optimism” by enforcing accounting consistency: growth affects book value, earnings, dividends, and residual income mechanically.

Impact on multiples:
- High P/B should be supported by ROE > r and expected persistence; otherwise it is vulnerable to mean reversion.
- P/E is more interpretable when you anchor on sustainable earnings and reconcile them to book value growth and payout.

Impact on DCF inputs:
- Residual income valuation uses the same discount rate concept (cost of equity) but relies less on long-horizon free cash flow.
- The implied terminal assumptions (fade vs persistence) are direct, transparent, and easy to stress-test.

Comparability issues:
- Different accounting policies (capitalisation, provisions, SBC, impairment timing) change book value and earnings, hence RI.
- Clean surplus violations (OCI, revaluations, FX translation effects) can break the BV–E–D linkage; consider adjustments if material.

Practical adjustments:
```python
def normalize_earnings(reported_earnings: float, one_offs_after_tax: float = 0.0) -> float:
    """Remove one-offs from earnings for sustainable RI forecasting."""
    e = float(reported_earnings)
    adj = float(one_offs_after_tax)
    if not np.isfinite(e) or not np.isfinite(adj):
        raise ValueError("Inputs must be finite.")
    return e - adj
```

---

### Quality of Earnings Flags (for simple forecasting)
⚠️ ROE boosted by one-off gains (asset sales, fair value gains) — not persistent.  
⚠️ Earnings growth driven by working-capital timing or under-provisioning — reversals likely.  
⚠️ Book value inflated/deflated by aggressive capitalisation or impairment lags — distorts RI base.  
⚠️ Large OCI movements (pensions, FX, FVOCI) causing clean-surplus breaks — reconcile equity roll-forward.  
✅ Earnings and book value move consistently with clean-surplus logic; payout and reinvestment are coherent.  
✅ ROE stability supported by operating drivers (margin, turnover) rather than accounting noise.  

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banks/insurers | book value is central; ROE is core | residual income frameworks often fit well; use regulatory capital constraints |
| Asset-light software | book value understated (unrecognised intangibles) | residual income can understate value; adjust for capitalised R&D where feasible |
| Cyclicals | earnings temporarily high/low | anchor on mid-cycle ROE; enforce fade; stress-test persistence |
| Real estate | fair value model affects earnings/book | treat FV changes carefully; consider NAV-based anchors |

---

### Real-World Example
Scenario: Use anchoring to value a firm with stable book value and fading ROE.

```python
company = {
    "bv0": 3_500.0,          # $m
    "r": 0.11,               # cost of equity
    "roe_path": [0.16, 0.14, 0.13, 0.12],   # fading profitability
    "payout_path": [0.25, 0.30, 0.35, 0.40]
}

out = forecast_simple_riv(
    bv0=company["bv0"],
    roe_path=company["roe_path"],
    payout_ratio_path=company["payout_path"],
    cost_of_equity=company["r"],
    terminal_fade="persistence",
    omega=0.6   # residual income decays 40% per year beyond horizon
)

print(f"Anchored equity value: ${out['equity_value']:.0f}m")
```

Interpretation:
- Value is driven by current book value plus discounted abnormal earnings; as ROE fades toward r, residual income shrinks.
- If market price implies materially higher persistence than your ω/ROE path, you have a clear, testable disagreement with the market.

See also: Chapter 14 (enterprise multiples and operating value), Chapter 16 (full-information forecasting), Chapter 18 (quality of financial statements), Chapter 19 (equity risk and return).
