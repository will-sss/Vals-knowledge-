# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 17: Creating Accounting Value and Economic Value (Residual Income, Economic Profit, Value Drivers)

### Core Concept
This chapter separates **accounting value creation** (earnings and book value growth) from **economic value creation** (returns above the required return on capital). For valuation, the key is identifying whether reported performance reflects sustainable economic profits or simply accounting outcomes driven by leverage, accruals, or reinvestment that earns only the cost of capital.

---

### Formula/Methodology

#### 1) Residual income (equity economic value creation)
```text
Residual Income_t (RI_t) = Earnings_t - (r × Book Value_{t-1})
```
Where:
- Earnings_t = comprehensive earnings attributable to common equity (consistent definition through time)
- r = cost of equity (decimal)
- Book Value_{t-1} = beginning-of-period common equity book value

Interpretation:
- RI_t > 0 means equity earns more than required; value is created.
- RI_t = 0 means earnings just cover the capital charge; value is preserved but not created.

Valuation identity:
```text
Equity Value_0 = Book Value_0 + PV(RI_1..∞)
```

#### 2) Economic profit / residual operating income (operations economic value creation)
```text
Residual Operating Income_t (ReOI_t) = OPAT_t - (WACC × NOA_{t-1})
```
Where:
- OPAT_t = operating profit after tax (excludes financing)
- NOA_{t-1} = net operating assets at beginning of period
- WACC = required return on operating capital (decimal)

Equivalent “spread” form:
```text
ReOI_t = NOA_{t-1} × (RNOA_t - WACC)
RNOA_t = OPAT_t / NOA_{t-1}
```

#### 3) Linking accounting growth to economic value
Value is not created by growth alone; it is created when growth is at returns above the charge:
```text
Economic Value Creation_t ≈ Invested Capital_{t-1} × (ROIC_t - WACC)
```
Where:
- Invested Capital is the operating capital base (NOA or an ROIC-consistent invested capital definition)
- ROIC_t is the operating return (decimal)

#### 4) Clean surplus constraint (equity accounting discipline)
```text
Ending Book Value_t = Beginning Book Value_{t-1} + Earnings_t - Dividends_t
```
Use this to reconcile earnings, payouts, and book value growth so residual income valuation is internally consistent.

---

### Practical Application (How to apply)

#### A) Diagnose whether “value creation” is real
Use a value-creation checklist:

| Question | Test | What it means |
|---|---|---|
| Are returns above required return? | ROIC vs WACC, RI > 0 | True economic profit |
| Is profit driven by leverage? | ROE decomposition; financing vs operating split | Riskier, not necessarily value-creating |
| Is growth earning attractive returns? | ΔInvested Capital and incremental ROIC | Growth can destroy value if incremental ROIC < WACC |
| Are profits backed by cash? | CFO vs earnings; accrual build | QoE risk if not cash-backed |
| Do returns persist? | fade analysis; competitive position | Sustainability of value creation |

#### B) Incremental economics (most common valuation failure point)
Avoid assuming the firm earns current ROIC on all future growth. Test incremental returns:
```text
Incremental ROIC_t ≈ ΔNOPAT_t / ΔInvested Capital_{t-1→t}
```
If incremental ROIC trends toward WACC, expect economic profit to fade.

#### C) Bridge between DCF and residual income
Residual income and DCF should reconcile if inputs are consistent. Use this as a model QA check:
- If DCF implies high value but RI analysis shows low/no residual income, you likely have:
  - inconsistent cost of capital,
  - inconsistent book values (clean surplus violated),
  - misclassified operating vs financing items,
  - or terminal assumptions producing hidden supernormal returns.

#### D) Using value creation in comps and multiples
Economic profit helps explain valuation premiums:
- High EV/EBITDA often reflects expectations of sustained ROIC > WACC.
- High P/B often reflects expectations of residual income (earnings above equity charge).
Use these to test market pricing:
```text
If Market Price >> Book Value, market is pricing PV(RI) > 0.
```

---

### Python Implementation
```python
from typing import Dict, Any
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def residual_income(earnings: float, book_value_beg: float, cost_of_equity: float) -> float:
    """Compute equity residual income: RI = Earnings - r * BV_{t-1}."""
    e = _f(earnings, "earnings")
    bv = _f(book_value_beg, "book_value_beg")
    r = _f(cost_of_equity, "cost_of_equity")
    if bv < 0:
        raise ValueError("book_value_beg is negative; residual income may be non-interpretable without adjustments.")
    if r <= 0 or r > 0.6:
        raise ValueError("cost_of_equity must be a positive decimal (e.g., 0.11).")
    return float(e - r * bv)

def residual_operating_income(opat: float, noa_beg: float, wacc: float) -> float:
    """Compute residual operating income: ReOI = OPAT - WACC * NOA_{t-1}."""
    opat = _f(opat, "opat")
    noa = _f(noa_beg, "noa_beg")
    wacc = _f(wacc, "wacc")
    if noa < 0:
        raise ValueError("noa_beg is negative; check operating/financing classification and accumulated losses.")
    if wacc <= 0 or wacc > 0.5:
        raise ValueError("wacc must be a positive decimal (e.g., 0.09).")
    return float(opat - wacc * noa)

def roic_spread_value_creation(invested_capital_beg: float, roic: float, wacc: float) -> float:
    """Economic profit style value creation: IC * (ROIC - WACC)."""
    ic = _f(invested_capital_beg, "invested_capital_beg")
    roic = _f(roic, "roic")
    wacc = _f(wacc, "wacc")
    if ic < 0:
        raise ValueError("invested_capital_beg is negative; check definition and netting.")
    if not (-1.0 < roic < 2.0):
        raise ValueError("roic should be a plausible decimal (e.g., 0.15).")
    if wacc <= 0 or wacc > 0.5:
        raise ValueError("wacc must be a positive decimal.")
    return float(ic * (roic - wacc))

def clean_surplus_check(bv_beg: float, earnings: float, dividends: float, bv_end_reported: float, tol: float = 1e-6) -> Dict[str, float]:
    """Check clean surplus: BV_end = BV_beg + Earnings - Dividends."""
    bv_beg = _f(bv_beg, "bv_beg")
    earnings = _f(earnings, "earnings")
    dividends = _f(dividends, "dividends")
    bv_end_reported = _f(bv_end_reported, "bv_end_reported")
    implied = bv_beg + earnings - dividends
    diff = bv_end_reported - implied
    ok = 1.0 if abs(diff) <= tol else 0.0
    return {"bv_end_implied": float(implied), "diff": float(diff), "ok_flag": float(ok)}

# Example usage (all $m)
example = {
    "earnings": 220.0,
    "bv_beg": 1_500.0,
    "opat": 300.0,
    "noa_beg": 2_000.0,
    "r": 0.11,
    "wacc": 0.09
}

ri = residual_income(example["earnings"], example["bv_beg"], example["r"])
reoi = residual_operating_income(example["opat"], example["noa_beg"], example["wacc"])
econ_profit = roic_spread_value_creation(example["noa_beg"], roic=example["opat"]/example["noa_beg"], wacc=example["wacc"])

print(f"Residual income: ${ri:.1f}m")
print(f"Residual operating income: ${reoi:.1f}m")
print(f"Economic profit (IC*(ROIC-WACC)): ${econ_profit:.1f}m")
```

---

### Valuation Impact
Why this matters:
- Prevents “growth optimism” by forcing an explicit test: **does growth earn more than the required return?**
- Provides a disciplined explanation for valuation premiums/discounts: **PV of future residual income/economic profit**.

Impact on multiples:
- EV/EBITDA and EV/Revenue premiums often embed an expectation of **persistent ROIC > WACC**.
- P/B and P/E can be interpreted via residual income: higher expected RI persistence supports higher P/B.

Impact on DCF inputs:
- Improves terminal value realism: in competitive markets, ROIC often fades toward WACC, constraining terminal assumptions.
- If your DCF implies perpetual ROIC > WACC, you need strong moat evidence or explicit reinvestment/competition logic.

Comparability issues:
- Accounting policy differences (capitalised R&D, leases, provisions, pension classification) shift OPAT/NOA and thus ROIC/RNOA.
- For comparables, normalise operating profit and operating capital definitions consistently.

Practical adjustments:
```python
def normalize_invested_capital(reported_noa: float, add_back_operating_leases: float = 0.0, remove_excess_cash: float = 0.0) -> float:
    """Illustrative normalisation of operating capital for comparability."""
    noa = float(reported_noa)
    leases = float(add_back_operating_leases)
    cash = float(remove_excess_cash)
    if not np.isfinite(noa + leases + cash):
        raise ValueError("Inputs must be finite.")
    return noa + leases - cash
```

---

### Quality of Earnings Flags
⚠️ ROIC appears high because invested capital is understated (capitalised costs missing, operating leases off-balance sheet, aggressive WC netting).  
⚠️ Residual income positive only due to one-offs (disposal gains, tax credits) not repeatable in operating profit.  
⚠️ Rising earnings with rising accruals and weak cash conversion (CFO lagging).  
⚠️ “Value creation” driven by debt-funded buybacks (ROE lift without operating improvement).  
✅ Positive residual income with stable cash conversion and consistent operating definitions over time.  
✅ Incremental ROIC supports growth story (ΔNOPAT/ΔIC stays above WACC).  

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banks/insurers | operating vs financing separation is blurred | prefer equity residual income with clean book value discipline |
| Technology | capitalised dev costs and SBC distort OPAT/NOA | normalise capitalisation policy; treat SBC consistently |
| Retail | leases and working capital drive capital base | capitalise leases for comparability; forecast WC with sales |
| Cyclicals | peak earnings inflate ROIC | use mid-cycle OPAT and capital base; fade residual profits faster |

---

### Real-World Example
Scenario: A company grows revenue fast, but reinvestment needs rise and incremental ROIC is falling. Test whether growth creates value.

```python
# Two-year incremental ROIC illustration
year1 = {"nopat": 180.0, "ic": 1_200.0}
year2 = {"nopat": 200.0, "ic": 1_450.0}
wacc = 0.10

delta_nopat = year2["nopat"] - year1["nopat"]
delta_ic = year2["ic"] - year1["ic"]
if delta_ic <= 0:
    raise ValueError("ΔIC must be positive for incremental ROIC interpretation.")

incr_roic = delta_nopat / delta_ic
econ_profit_incr = delta_ic * (incr_roic - wacc)

print(f"Incremental ROIC: {incr_roic:.1%}")
print(f"Incremental economic profit: ${econ_profit_incr:.1f}m")
```

Interpretation:
- If incremental ROIC < WACC, incremental economic profit is negative: growth is value-destructive unless economics improve.
- Use this to discipline forecasting: fade abnormal returns, revise capital intensity, or revise growth.

See also: Chapter 12 (profitability analysis), Chapter 13 (growth and sustainable earnings), Chapter 14 (enterprise multiples), Chapter 15 (simple forecasting), Chapter 16 (full-information forecasting).
