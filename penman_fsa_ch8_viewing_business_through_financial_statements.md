# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 8: Viewing the Business Through the Financial Statements (Operating vs Financing, Common-Size, Driver Ratios)

### Core Concept
Financial statements can be reorganised into operating and financing components to reveal the true economic drivers of value: operating profitability, operating asset efficiency, and financing structure. This perspective improves valuation quality by making ratios comparable across firms with different leverage and by aligning accounting numbers with enterprise value (operating) versus equity value (financing). In practice, this chapter’s toolkit is the foundation for clean ROIC/ROE decomposition, forecasting, and consistent use of EV multiples.

---

### Formula/Methodology

#### 1) Operating vs financing split (core reclassification)
```text
Net Operating Assets (NOA) = Operating Assets - Operating Liabilities
Net Financial Obligations (NFO) = Financial Liabilities - Financial Assets
```

Balance sheet identity (equity as the residual):
```text
Equity (BV) = NOA - NFO
```

Where:
- Operating items relate to producing and delivering goods/services
- Financial items relate to funding (cash management, debt, interest-bearing items)

Typical classification (high-level):
- Operating assets: receivables, inventory, PP&E used in operations, operating intangibles, goodwill
- Operating liabilities: trade payables, accrued expenses, deferred revenue, provisions (if operational)
- Financial assets: excess cash, marketable securities held for treasury
- Financial liabilities: debt, lease liabilities (interest-bearing), pension deficit (often treated as financing-like), derivatives used for financing/treasury

#### 2) Operating profit after tax and operating profitability
```text
NOPAT = Operating Profit × (1 - Tax rate)
```

Operating return:
```text
RNOA = NOPAT / Average NOA
```

Where:
- Operating Profit is often measured as EBIT (after operating adjustments), excluding net financing income/expense

#### 3) Decomposing operating return into margin and turnover
```text
RNOA = (NOPAT / Revenue) × (Revenue / Average NOA)
```

Components:
- Operating profit margin (after tax): NOPAT / Revenue
- Operating asset turnover: Revenue / Average NOA

#### 4) Linking ROE to operating performance and leverage (operating vs financing lens)
A practical “operating + financing” ROE bridge:
```text
ROE ≈ RNOA + (FLEV × (RNOA - NBC))
```

Where:
```text
FLEV = Average NFO / Average Equity
NBC  = Net Borrowing Cost after tax = (Net Interest Expense × (1 - Tax rate)) / Average NFO
```

Interpretation:
- If RNOA > NBC, financial leverage increases ROE (value positive, but risk rises)
- If RNOA < NBC, leverage destroys ROE (warning sign)

#### 5) Common-size statements (for comparability)
Income statement common-size:
```text
Common-size line item = Line item / Revenue
```

Balance sheet common-size:
```text
Common-size asset/liability = Line item / Total Assets
```

---

### Practical Application (How to apply)

#### A) Reorganise financial statements for valuation (repeatable)
1) **Start with the reported balance sheet** and tag each line as Operating vs Financing.
2) **Compute NOA and NFO** and confirm the identity:
   - BV = NOA - NFO
3) **Rebuild the income statement** into:
   - Operating profit (EBIT / operating items)
   - Net financing income/expense (interest, treasury)
4) **Compute NOPAT and RNOA**, then decompose into:
   - operating margin × operating turnover
5) **Use the reorganised statements for forecasting**:
   - forecast revenue, operating margin, NOA intensity (NOA / revenue) or turnover (revenue / NOA)
   - forecast financing separately (target leverage, interest cost)

Why this matters: you avoid mixing financing effects into operating performance, which is the most common comparability error in valuation.

#### B) Make EV multiples consistent with operating fundamentals
When using EV/EBIT or EV/EBITDA:
- Ensure EBIT/EBITDA is **operating** (exclude financing and one-offs).
- Ensure EV uses **net debt** and other financing-like items consistently (pensions, leases, operating cash classification policy).

#### C) Forecast using “driver ratios” instead of line-by-line noise
A robust forecast structure:
- Revenue growth (volume, price, mix)
- Operating margin (after normalisation)
- NOA intensity (NOA / revenue) or turnover
- Tax rate (cash/structural)
- Financing policy (target NFO / equity and NBC)

This reduces model fragility and improves explainability.

#### D) Reconcile operating vs equity valuation views
- Enterprise valuation focuses on operating cash flows/returns (NOA-driven)
- Equity valuation adds financing structure (NFO-driven)

Use cross-checks:
- If EV growth implies rising NOA intensity without strategic reason, revisit reinvestment assumptions.
- If ROE is high but driven only by leverage (FLEV) with weak RNOA, the equity story is risk-heavy.

---

### Python Implementation
```python
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

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

def compute_noa(operating_assets: float, operating_liabilities: float) -> float:
    """NOA = operating assets - operating liabilities."""
    oa = _f(operating_assets, "operating_assets")
    ol = _f(operating_liabilities, "operating_liabilities")
    return float(oa - ol)

def compute_nfo(financial_liabilities: float, financial_assets: float) -> float:
    """NFO = financial liabilities - financial assets."""
    fl = _f(financial_liabilities, "financial_liabilities")
    fa = _f(financial_assets, "financial_assets")
    return float(fl - fa)

def check_balance_identity(noa: float, nfo: float, equity_bv: float, *, tol: float = 1e-6) -> bool:
    """Checks BV ≈ NOA - NFO within tolerance."""
    noa = _f(noa, "noa"); nfo = _f(nfo, "nfo"); bv = _f(equity_bv, "equity_bv")
    lhs = bv
    rhs = noa - nfo
    return bool(abs(lhs - rhs) <= tol * max(1.0, abs(lhs), abs(rhs)))

def nopat(operating_profit: float, tax_rate: float) -> float:
    """NOPAT = operating profit × (1 - tax_rate)."""
    op = _f(operating_profit, "operating_profit")
    t = _f(tax_rate, "tax_rate")
    if t < 0 or t > 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    return float(op * (1.0 - t))

def rnoa(nopat_value: float, noa_open: float, noa_close: float) -> float:
    """RNOA = NOPAT / average NOA."""
    n = _f(nopat_value, "nopat_value")
    o = _f(noa_open, "noa_open")
    c = _f(noa_close, "noa_close")
    avg_noa = (o + c) / 2.0
    out = safe_divide(n, avg_noa, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Average NOA must be non-zero.")
    return float(out)

def operating_margin(nopat_value: float, revenue: float) -> float:
    """NOPAT margin = NOPAT / revenue."""
    n = _f(nopat_value, "nopat_value")
    r = _f(revenue, "revenue")
    out = safe_divide(n, r, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Revenue must be non-zero.")
    return float(out)

def operating_turnover(revenue: float, noa_open: float, noa_close: float) -> float:
    """Operating turnover = revenue / average NOA."""
    r = _f(revenue, "revenue")
    o = _f(noa_open, "noa_open")
    c = _f(noa_close, "noa_close")
    avg_noa = (o + c) / 2.0
    out = safe_divide(r, avg_noa, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Average NOA must be non-zero.")
    return float(out)

def net_borrowing_cost_after_tax(net_interest_expense: float, tax_rate: float, nfo_open: float, nfo_close: float) -> float:
    """NBC (after tax) = net interest expense × (1 - t) / avg NFO."""
    nie = _f(net_interest_expense, "net_interest_expense")
    t = _f(tax_rate, "tax_rate")
    if t < 0 or t > 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    o = _f(nfo_open, "nfo_open"); c = _f(nfo_close, "nfo_close")
    avg_nfo = (o + c) / 2.0
    out = safe_divide(nie * (1.0 - t), avg_nfo, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Average NFO must be non-zero.")
    return float(out)

def flev(nfo_open: float, nfo_close: float, equity_open: float, equity_close: float) -> float:
    """Financial leverage = avg NFO / avg Equity."""
    o = _f(nfo_open, "nfo_open"); c = _f(nfo_close, "nfo_close")
    eo = _f(equity_open, "equity_open"); ec = _f(equity_close, "equity_close")
    avg_nfo = (o + c) / 2.0
    avg_eq = (eo + ec) / 2.0
    out = safe_divide(avg_nfo, avg_eq, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Average equity must be non-zero.")
    return float(out)

def roe_bridge(rnoa_value: float, flev_value: float, nbc_value: float) -> float:
    """ROE ≈ RNOA + FLEV × (RNOA - NBC)."""
    r = _f(rnoa_value, "rnoa_value")
    fl = _f(flev_value, "flev_value")
    nbc = _f(nbc_value, "nbc_value")
    return float(r + fl * (r - nbc))

# Example usage with realistic numbers
data = {
    "revenue": 5_000.0,                 # $m
    "operating_profit": 750.0,          # $m EBIT (operating)
    "tax_rate": 0.25,
    "operating_assets_open": 6_400.0,   # $m
    "operating_assets_close": 6_900.0,  # $m
    "operating_liab_open": 2_100.0,     # $m
    "operating_liab_close": 2_250.0,    # $m
    "financial_liab_open": 3_000.0,     # $m
    "financial_liab_close": 3_200.0,    # $m
    "financial_assets_open": 400.0,     # $m (excess cash/investments)
    "financial_assets_close": 450.0,    # $m
    "equity_open": 3_100.0,             # $m
    "equity_close": 3_200.0,            # $m
    "net_interest_expense": 120.0       # $m
}

noa_o = compute_noa(data["operating_assets_open"], data["operating_liab_open"])
noa_c = compute_noa(data["operating_assets_close"], data["operating_liab_close"])
nfo_o = compute_nfo(data["financial_liab_open"], data["financial_assets_open"])
nfo_c = compute_nfo(data["financial_liab_close"], data["financial_assets_close"])

# Identity check (approx)
assert check_balance_identity(noa_o, nfo_o, data["equity_open"], tol=1e-3), "Balance identity failed (open)."

n = nopat(data["operating_profit"], data["tax_rate"])
r = rnoa(n, noa_o, noa_c)
m = operating_margin(n, data["revenue"])
t = operating_turnover(data["revenue"], noa_o, noa_c)
nbc = net_borrowing_cost_after_tax(data["net_interest_expense"], data["tax_rate"], nfo_o, nfo_c)
fl = flev(nfo_o, nfo_c, data["equity_open"], data["equity_close"])
roe_est = roe_bridge(r, fl, nbc)

print(f"NOA (avg): ${(noa_o+noa_c)/2:.0f}m | NFO (avg): ${(nfo_o+nfo_c)/2:.0f}m")
print(f"NOPAT: ${n:.0f}m | RNOA: {r:.1%} | Margin: {m:.1%} | Turnover: {t:.2f}x")
print(f"NBC (after tax): {nbc:.1%} | FLEV: {fl:.2f}x | ROE (bridge): {roe_est:.1%}")
```

---

### Valuation Impact
Why this matters:
- Clean operating/financing separation prevents leverage and treasury effects from contaminating operating performance, which is what enterprise value reflects.
- Reorganised statements improve comparability and forecasting by focusing on stable driver ratios (margin, turnover, NOA intensity, financing cost).

Impact on multiples (EV/EBITDA, EV/EBIT, P/E):
- EV multiples should be compared to operating earnings; including financing noise can mis-rank peers.
- P/E comparisons become more meaningful when ROE is decomposed into operating return (RNOA) and leverage effects.

Impact on DCF inputs:
- Operating driver forecast (margin + turnover + reinvestment) is the core of sustainable FCF.
- Financing policy (NFO, NBC) affects equity cash flows and per-share value but should not drive operating value.

Comparability issues across companies:
- Lease accounting, pension classification, and “excess cash” policies change NOA/NFO; document your classification rules.
- IFRS vs US GAAP can shift items between operating and financing sections (e.g., interest paid classification in cash flow).

Practical adjustments:
```python
def classify_excess_cash(total_cash: float, operating_cash_need: float) -> Dict[str, float]:
    """Split cash into operating cash (operating) and excess cash (financing asset)."""
    import numpy as np
    tc = float(total_cash); oc = float(operating_cash_need)
    if not np.isfinite(tc) or not np.isfinite(oc):
        raise ValueError("Inputs must be finite.")
    if tc < 0 or oc < 0:
        raise ValueError("Cash amounts must be >= 0.")
    operating_cash = min(tc, oc)
    excess_cash = max(0.0, tc - oc)
    return {"operating_cash": operating_cash, "excess_cash": excess_cash}
```

---

### Quality of Earnings Flags (Operating/Financing lens)
⚠️ High ROE driven mainly by leverage (high FLEV) while RNOA is mediocre or declining.  
⚠️ Large cash balance treated as “operating” to inflate turnover or mask weak operations (classify excess cash as financing).  
⚠️ Operating profit boosted by reclassifying recurring costs as “non-operating” or “exceptional”.  
⚠️ Material pension deficits or lease liabilities ignored in NFO (understates financing burden).  
✅ Strong RNOA supported by stable margins and improving turnover, with transparent classification and consistent reconciliation.

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Retail | leases dominate capital structure | treat lease liabilities as financing; adjust EBIT/EBITDA consistency |
| Banks | operating vs financing is intertwined | avoid NOA/NFO split; use bank-specific frameworks (net interest margin, CET1) |
| Asset-light services | intangibles and working capital drive NOA | focus on NOA intensity and cash conversion; ensure capitalised costs are treated consistently |
| Utilities | regulated assets and long-lived PP&E | turnover is structurally low; focus on allowed returns and asset base |

---

### Real-World Example
Scenario: Two firms have similar ROE, but one is operating-strong and the other is leverage-driven.

```python
# Firm A: high RNOA, modest leverage
roe_a = roe_bridge(rnoa_value=0.16, flev_value=0.5, nbc_value=0.05)

# Firm B: mediocre RNOA, high leverage
roe_b = roe_bridge(rnoa_value=0.10, flev_value=2.0, nbc_value=0.06)

print(f"Firm A ROE (bridge): {roe_a:.1%}")
print(f"Firm B ROE (bridge): {roe_b:.1%}")
```

Interpretation: Similar ROE can mask very different valuation risk. Firm B’s equity value is more sensitive to funding shocks and downturns because ROE is propped up by leverage rather than operating excellence.

See also: Chapter 6 (pricing earnings), Chapter 12 (profitability analysis), Chapter 14 (linking operations value to P/B and P/E).
