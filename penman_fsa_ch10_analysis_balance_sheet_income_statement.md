# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 10: The Analysis of the Balance Sheet and Income Statement (Operating vs Financing, Recasting, ROCE/ROE)

### Core Concept
Valuation analysis starts by **recasting** the balance sheet and income statement into operating and financing components so profitability and growth are measured on the right base. This clean separation prevents leverage and non-operating items from contaminating operating performance metrics (e.g., margins, ROIC/ROCE), improving comparability across firms and enabling book-value and residual-income style valuation to work reliably.

---

### Formula/Methodology

#### 1) Recasting: operating vs financing (balance sheet)
Core identities:
```text
Operating Assets (OA) = Total Assets - Financial Assets
Operating Liabilities (OL) = Total Liabilities - Financial Liabilities
Net Operating Assets (NOA) = OA - OL

Net Financial Obligations (NFO) = Financial Liabilities - Financial Assets
Book Value of Common Equity (CSE) = NOA - NFO
```

Where (practical classification):
- Financial assets: cash & cash equivalents (excess), marketable securities, short-term investments, derivatives held for trading (often)
- Financial liabilities: interest-bearing debt, lease liabilities (economically debt-like), pension deficits (often treated as financing-like), derivatives used for financing/hedging financing risk (case-specific)
- Operating assets/liabilities: working capital items, PPE, intangibles, provisions tied to operations (case-specific)

Consistency check:
```text
Total Assets - Total Liabilities = Equity
NOA - NFO = Equity
```

#### 2) Recasting: operating vs financing (income statement)
```text
Operating Income (OI) = Revenue - Operating Expenses (incl. operating depreciation/amortisation)
Net Financial Expense (NFE) = Interest Expense - Interest Income (after tax if comparing to OI after tax)

Comprehensive Income ≈ Operating Income - Net Financial Expense + Other Comprehensive/Dirty Surplus (handled separately)
```

Practical “after-tax” alignment:
```text
NOPAT = Operating Income × (1 - Tax Rate)
After-tax NFE = NFE × (1 - Tax Rate)   (approx; refine if tax shields differ)
```

#### 3) Profitability on the right base
Operating return:
```text
RNOA = NOPAT / Average NOA
```

Equity return:
```text
ROE = Net Income to Common / Average Common Equity
```

Link between ROE and operating + financing:
```text
ROE = RNOA + (Financial Leverage × Spread)
Financial Leverage = NFO / Common Equity
Spread = RNOA - After-tax Cost of Debt
```

After-tax cost of debt (approx):
```text
After-tax Cost of Debt = Interest Rate on Debt × (1 - Tax Rate)
Interest Rate on Debt ≈ Interest Expense / Average Financial Liabilities
```

#### 4) Growth decomposition (operating vs financing)
```text
Growth in NOA = (NOA_t - NOA_{t-1}) / NOA_{t-1}
Operating Free Cash Flow (conceptual) relates to changes in NOA and NOPAT:
FCF ≈ NOPAT - ΔNOA
```

This is not a substitute for a full cash flow statement, but a fast cross-check for whether operating growth is consuming cash.

---

### Practical Application (How to apply)

#### A) Recast the balance sheet (step-by-step)
1) Start with reported balance sheet.
2) Classify each line item as:
   - Operating asset / operating liability, or
   - Financial asset / financial liability
3) Compute NOA and NFO.
4) Tie back to equity: NOA - NFO must equal common equity (after adjusting for minority interests / preferred where relevant).

Common classification rules (practical):
- Treat lease liabilities as financial liabilities for comparability and leverage analysis.
- Treat excess cash as financial asset; operating cash can be included in operating assets if clearly required for operations (rare to do precisely—document assumption).
- Treat defined benefit pension deficits as financing-like (because it behaves like debt); treat pension plan assets as financial assets (net them).

#### B) Recast the income statement (step-by-step)
1) Identify interest income/expense and separate financing from operating.
2) Identify “other income/expense” items and decide if operating (recurring) or non-operating (one-off, investing gains).
3) Build NOPAT:
   - OI × (1 - tax rate)
   - Use a consistent tax rate (effective or marginal) depending on use-case; document which.

#### C) Compute and interpret RNOA, ROE, leverage and spread
Use RNOA to judge operating performance independent of leverage.
Use ROE to assess the shareholder result, but diagnose whether:
- ROE is high because operations are strong (high RNOA), or
- ROE is high because leverage is high and spread is positive (riskier).

#### D) Use the recast statements for forecasting and valuation
Forecasting:
- Forecast operating drivers on NOA: sales, margins → NOPAT; reinvestment → ΔNOA
- Forecast financing separately: debt policy → NFO, interest cost

Valuation cross-check:
- If you run an enterprise DCF, the implied equity value should be consistent with NOA/NFO story and any residual income approach based on equity + abnormal earnings.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Tuple
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

def recast_balance_sheet(bs: Dict[str, float]) -> Dict[str, float]:
    """Recast a balance sheet into operating vs financing components.

    Args:
        bs: dict with at least:
            total_assets, total_liabilities,
            financial_assets, financial_liabilities

    Returns:
        dict: OA, OL, NOA, NFO, equity_check

    Raises:
        ValueError: missing or invalid inputs
    """
    ta = _f(bs.get("total_assets"), "total_assets")
    tl = _f(bs.get("total_liabilities"), "total_liabilities")
    fa = _f(bs.get("financial_assets"), "financial_assets")
    fl = _f(bs.get("financial_liabilities"), "financial_liabilities")

    if fa < 0 or fl < 0:
        raise ValueError("financial_assets and financial_liabilities must be >= 0.")

    oa = ta - fa
    ol = tl - fl
    noa = oa - ol
    nfo = fl - fa

    equity_reported = ta - tl
    equity_check = noa - nfo  # should equal equity_reported under consistent classification

    return {
        "OA": float(oa),
        "OL": float(ol),
        "NOA": float(noa),
        "NFO": float(nfo),
        "equity_reported": float(equity_reported),
        "equity_from_recast": float(equity_check),
        "equity_diff": float(equity_check - equity_reported),
    }

def nopat(operating_income: float, tax_rate: float) -> float:
    """NOPAT = Operating income * (1 - tax_rate)."""
    oi = _f(operating_income, "operating_income")
    t = _f(tax_rate, "tax_rate")
    if t < 0 or t > 0.6:
        raise ValueError("tax_rate looks implausible; use decimal (e.g., 0.25).")
    return float(oi * (1.0 - t))

def rnoa(nopat_value: float, noa_open: float, noa_close: float) -> float:
    """RNOA = NOPAT / average NOA."""
    n = _f(nopat_value, "nopat_value")
    o = _f(noa_open, "noa_open")
    c = _f(noa_close, "noa_close")
    avg_noa = (o + c) / 2.0
    out = safe_divide(n, avg_noa, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Average NOA must be non-zero for RNOA.")
    return float(out)

def after_tax_cost_of_debt(interest_expense: float, avg_fin_liab: float, tax_rate: float) -> float:
    """After-tax cost of debt ≈ (interest_expense / avg_fin_liab) * (1 - tax_rate)."""
    ie = _f(interest_expense, "interest_expense")
    afl = _f(avg_fin_liab, "avg_fin_liab")
    t = _f(tax_rate, "tax_rate")
    if afl <= 0:
        raise ValueError("avg_fin_liab must be > 0.")
    if t < 0 or t > 0.6:
        raise ValueError("tax_rate looks implausible; use decimal (e.g., 0.25).")
    pre_tax = ie / afl
    return float(pre_tax * (1.0 - t))

def roe(net_income: float, equity_open: float, equity_close: float) -> float:
    """ROE = net income / average equity."""
    ni = _f(net_income, "net_income")
    e0 = _f(equity_open, "equity_open")
    e1 = _f(equity_close, "equity_close")
    avg_eq = (e0 + e1) / 2.0
    out = safe_divide(ni, avg_eq, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("Average equity must be non-zero for ROE.")
    return float(out)

def roe_decomposition(rnoa_value: float, nfo_open: float, nfo_close: float, equity_open: float, equity_close: float, after_tax_kd: float) -> Dict[str, float]:
    """Decompose ROE into operating return plus leverage * spread."""
    r = _f(rnoa_value, "rnoa_value")
    n0 = _f(nfo_open, "nfo_open")
    n1 = _f(nfo_close, "nfo_close")
    e0 = _f(equity_open, "equity_open")
    e1 = _f(equity_close, "equity_close")
    kd = _f(after_tax_kd, "after_tax_kd")

    avg_nfo = (n0 + n1) / 2.0
    avg_eq = (e0 + e1) / 2.0
    lev = safe_divide(avg_nfo, avg_eq, on_zero=np.nan)
    if lev is None or not np.isfinite(lev):
        raise ValueError("Average equity must be non-zero for leverage.")
    spread = r - kd
    roe_implied = r + lev * spread
    return {"financial_leverage": float(lev), "spread": float(spread), "roe_implied": float(roe_implied)}

# Example usage ($m)
bs_open = {"total_assets": 12_000.0, "total_liabilities": 7_500.0, "financial_assets": 900.0, "financial_liabilities": 3_000.0}
bs_close = {"total_assets": 13_200.0, "total_liabilities": 8_100.0, "financial_assets": 1_000.0, "financial_liabilities": 3_300.0}

rb0 = recast_balance_sheet(bs_open)
rb1 = recast_balance_sheet(bs_close)

tax_rate = 0.25
operating_income = 1_100.0
net_income = 720.0
interest_expense = 180.0

nop = nopat(operating_income, tax_rate)
rnoa_val = rnoa(nop, rb0["NOA"], rb1["NOA"])

equity_open = rb0["equity_reported"]
equity_close = rb1["equity_reported"]
roe_val = roe(net_income, equity_open, equity_close)

avg_fin_liab = (bs_open["financial_liabilities"] + bs_close["financial_liabilities"]) / 2.0
kd_at = after_tax_cost_of_debt(interest_expense, avg_fin_liab, tax_rate)

decomp = roe_decomposition(
    rnoa_value=rnoa_val,
    nfo_open=rb0["NFO"],
    nfo_close=rb1["NFO"],
    equity_open=equity_open,
    equity_close=equity_close,
    after_tax_kd=kd_at
)

print(f"NOA open/close: ${rb0['NOA']:.0f}m / ${rb1['NOA']:.0f}m | NFO open/close: ${rb0['NFO']:.0f}m / ${rb1['NFO']:.0f}m")
print(f"RNOA: {rnoa_val:.1%} | ROE: {roe_val:.1%}")
print(f"After-tax cost of debt: {kd_at:.1%} | Leverage: {decomp['financial_leverage']:.2f}x | Spread: {decomp['spread']:.1%}")
print(f"ROE (implied by decomposition): {decomp['roe_implied']:.1%}")
```

---

### Valuation Impact
Why this matters:
- Operating performance should be measured on **operating capital** (NOA), not muddied by financing structure. This improves the reliability of profitability metrics used to judge competitive position and to forecast cash flows.
- Separating operating and financing clarifies whether returns are generated by the business model or by leverage, which affects risk and cost of capital choices.

Impact on multiples:
- EV/EBITDA and EV/EBIT comparisons are more meaningful when you normalise operating earnings and strip out financing noise.
- P/B and ROE are only interpretable when book value is properly defined (common equity) and earnings are not inflated by one-off financing gains.

Impact on DCF inputs:
- Forecasting reinvestment is cleaner using ΔNOA (operating growth consumes operating capital).
- Financing policy flows into NFO and interest costs, rather than distorting operating margins.

Comparability issues:
- Lease-heavy firms look more leveraged once lease liabilities are treated as financing.
- Firms with large excess cash or investment portfolios need consistent financial asset classification to avoid understating operating returns.

Practical adjustments:
```python
def normalise_operating_income(reported_ebit: float, one_off_operating_items: float = 0.0) -> float:
    """Remove non-recurring operating items to estimate sustainable operating income."""
    import numpy as np
    e = float(reported_ebit); adj = float(one_off_operating_items)
    if not np.isfinite(e) or not np.isfinite(adj):
        raise ValueError("Inputs must be finite.")
    return e - adj
```

---

### Quality of Earnings Flags (recasting-focused)
⚠️ Operating income boosted by gains that are actually financing/investing (asset sales, fair value gains on investments).  
⚠️ Reported ROE high but RNOA mediocre and leverage high (performance driven by balance sheet risk).  
⚠️ Large “other income/expense” and shifting classification year-to-year (comparability breaks).  
⚠️ Persistent working capital build (ΔNOA) while EBITDA is rising—cash conversion deteriorates.  
✅ RNOA improves alongside stable/no worsening leverage; operating margins and NOA turnover are consistent with business economics.

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banks/Insurers | Operating vs financing split is not meaningful the same way | use equity-based valuation; treat interest as operating; focus on ROE, spread metrics, regulatory capital |
| Retail / Airlines | Large leases and working capital swings | treat leases as financing; focus on NOA turnover and lease-adjusted leverage |
| Real estate / Holding companies | Large investment portfolios | classify investments as financial assets; separate property operations vs investments; consider sum-of-parts |
| Asset-light tech | Intangibles and SBC | clarify which intangibles are operating; treat SBC as operating expense + dilution (per-share focus) |

---

### Real-World Example
Scenario: Two firms have identical ROE, but only one has strong operating returns.

```python
firm1 = {"rnoa": 0.16, "after_tax_kd": 0.06, "leverage": 0.5}
firm2 = {"rnoa": 0.10, "after_tax_kd": 0.06, "leverage": 2.0}

def roe_from_components(rnoa: float, kd: float, lev: float) -> float:
    if lev < 0:
        raise ValueError("leverage must be >= 0.")
    return rnoa + lev * (rnoa - kd)

roe1 = roe_from_components(firm1["rnoa"], firm1["after_tax_kd"], firm1["leverage"])
roe2 = roe_from_components(firm2["rnoa"], firm2["after_tax_kd"], firm2["leverage"])

print(f"Firm 1 ROE: {roe1:.1%} (strong operations, moderate leverage)")
print(f"Firm 2 ROE: {roe2:.1%} (weaker operations, high leverage)")
```

Interpretation: Similar ROE can hide very different economics. For valuation and forecasting, anchor on RNOA and the NOA/NFO split, then decide whether leverage is sustainable and appropriately priced in the discount rate.

See also: Chapter 3 (statements in valuation), Chapter 4 (cash vs accrual and DCF), Chapter 9 (equity statement), Chapter 12 (profitability), Chapter 18 (quality of financial statements).
