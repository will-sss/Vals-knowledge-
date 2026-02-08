# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 9: The Analysis of the Statement of Shareholders' Equity (Clean Surplus, Comprehensive Income, Dilution)

### Core Concept
The statement of shareholders’ equity explains how book value changes over time through earnings, dividends, share issues/repurchases, and “other comprehensive income” (OCI) items. For valuation, this statement is the reconciliation tool that links accounting performance (earnings), payout policy, and dilution to intrinsic value models based on residual income and book value. Practically, it helps you spot earnings that bypass the income statement, identify capital raising masking weak performance, and normalise per-share value.

---

### Formula/Methodology

#### 1) Book value roll-forward (equity reconciliation)
```text
Ending Book Value (BV_t) = Beginning Book Value (BV_{t-1})
                          + Comprehensive Income_t
                          - Dividends_t
                          + Net Share Issues_t
                          + Other Owner Transactions_t
```

Where:
- Comprehensive Income = Net Income + OCI
- Net Share Issues = cash proceeds from share issuance (net of buybacks) at issue price; includes employee share plans (often non-cash but dilutive)

#### 2) Clean surplus accounting (core for residual income valuation)
Clean surplus condition:
```text
BV_t = BV_{t-1} + Earnings_t - Dividends_t
```

If the clean surplus condition is violated (because some gains/losses go directly to equity via OCI), adjust:
```text
Clean Surplus Earnings_t = Reported Earnings_t + Dirty Surplus Items_t
Dirty Surplus Items_t = OCI_t + items recognised directly in equity (net of tax)
```

#### 3) Comprehensive income and OCI
```text
Comprehensive Income = Net Income + OCI
```

Common OCI components (typical, not exhaustive):
- FX translation (foreign operations)
- Cash flow hedge reserves
- Fair value changes for certain financial assets (classification-dependent)
- Actuarial gains/losses on defined benefit plans (IFRS: often OCI)
- Revaluation surplus (IFRS revaluation model)

#### 4) Per-share book value and dilution (share count matters)
```text
Book Value per Share (BVPS) = Common Equity (Book Value) / Diluted Shares Outstanding
```

Diluted shares (practical):
```text
Diluted Shares ≈ Basic Shares + In-the-money Options (Treasury Stock Method) + Other Dilutive Instruments
```

Treasury stock method (simplified):
```text
Incremental Shares = Options Outstanding × (1 - Exercise Price / Average Share Price)
```

Constraints:
- If Exercise Price >= Average Price → incremental shares = 0 (not dilutive)

#### 5) Residual income (RI) valuation link (why equity statement matters)
Residual income:
```text
RI_t = Earnings_t - (r × BV_{t-1})
```

Where:
- r = cost of equity (as decimal)
- Earnings should be “clean surplus earnings” when OCI/dirty-surplus items are material

---

### Practical Application (How to apply)

#### A) Build the equity roll-forward and reconcile all changes
1) Pull: beginning and ending equity (common equity attributable to shareholders).
2) Pull: net income, OCI, dividends, share issues, buybacks, and other equity movements.
3) Reconcile:
   - If reconciliation doesn’t tie, identify missing lines (share-based payment reserves, revaluation reserves, translation reserve movements).

Use this to confirm:
- Whether earnings growth translates into book value growth (sustainable value creation)
- Whether book value growth is coming from operations or capital injections

#### B) Convert “dirty surplus” to clean surplus for valuation models
If you use residual income (or any book-value anchored approach):
- Add OCI and direct-to-equity items back into “clean surplus earnings”
- Treat **owner transactions** (dividends, buybacks, issuance) separately

Rule of thumb:
- OCI that is persistent (e.g., recurring pension OCI) must be incorporated into normal earnings capacity.
- OCI that is transitory (e.g., one-off FX translation) should be separated and assessed for recurrence risk.

#### C) Diagnose dilution and capital structure moves
Key checks:
- Net share issuance when earnings are weak can be “life support” financing.
- Buybacks funded by debt can inflate ROE temporarily (equity shrinks) without improving operations.
- Share-based payment can be economically equivalent to paying staff in stock: treat as compensation, not “free equity”.

#### D) Tie equity statement to per-share valuation
Per-share value is sensitive to:
- Share count trajectory (issuance, employee options)
- Buyback price discipline (repurchasing above intrinsic value destroys value)
- Comprehensive income items that affect book value and future earnings (e.g., hedging reserves rolling into P&L)

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

def comprehensive_income(net_income: float, oci: float) -> float:
    """Comprehensive income = net income + OCI."""
    ni = _f(net_income, "net_income")
    o = _f(oci, "oci")
    return float(ni + o)

def equity_rollforward(
    bv_open: float,
    net_income: float,
    oci: float,
    dividends: float,
    net_share_issues: float,
    other_owner_txns: float = 0.0
) -> float:
    """Ending BV = BV_open + comprehensive income - dividends + net share issues + other owner txns."""
    bv = _f(bv_open, "bv_open")
    div = _f(dividends, "dividends")
    nsi = _f(net_share_issues, "net_share_issues")
    oot = _f(other_owner_txns, "other_owner_txns")

    ci = comprehensive_income(net_income, oci)
    return float(bv + ci - div + nsi + oot)

def clean_surplus_earnings(reported_earnings: float, dirty_surplus_items: float) -> float:
    """Clean surplus earnings = reported earnings + dirty surplus items (OCI + direct-to-equity, net of tax)."""
    re = _f(reported_earnings, "reported_earnings")
    ds = _f(dirty_surplus_items, "dirty_surplus_items")
    return float(re + ds)

def residual_income(clean_earnings: float, cost_of_equity: float, bv_open: float) -> float:
    """RI = clean earnings - r * BV_open."""
    e = _f(clean_earnings, "clean_earnings")
    r = _f(cost_of_equity, "cost_of_equity")
    bv = _f(bv_open, "bv_open")
    if r < -0.5 or r > 1.0:
        raise ValueError("cost_of_equity looks implausible; use decimal (e.g., 0.10).")
    return float(e - r * bv)

def treasury_stock_incremental_shares(options_outstanding: float, exercise_price: float, avg_share_price: float) -> float:
    """Incremental shares (treasury stock method, simplified)."""
    opts = _f(options_outstanding, "options_outstanding")
    ex = _f(exercise_price, "exercise_price")
    p = _f(avg_share_price, "avg_share_price")
    if opts < 0:
        raise ValueError("options_outstanding must be >= 0.")
    if ex < 0 or p <= 0:
        raise ValueError("Prices must be positive, exercise_price must be >= 0.")
    if ex >= p:
        return 0.0
    return float(opts * (1.0 - ex / p))

def bvps(common_equity_bv: float, diluted_shares: float) -> float:
    """Book value per share."""
    bv = _f(common_equity_bv, "common_equity_bv")
    sh = _f(diluted_shares, "diluted_shares")
    out = safe_divide(bv, sh, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("diluted_shares must be non-zero.")
    return float(out)

# Example usage (all $m except share counts)
example = {
    "bv_open": 4_800.0,
    "net_income": 620.0,
    "oci": -80.0,
    "dividends": 200.0,
    "net_share_issues": 150.0,  # issuance net of buybacks
    "other_owner_txns": 0.0,
    "bv_close_reported": 5_290.0,  # suppose this is disclosed
    "cost_of_equity": 0.10,
    "basic_shares": 1_000.0,  # million shares
    "options_outstanding": 60.0,
    "exercise_price": 8.0,
    "avg_share_price": 12.0
}

bv_close_calc = equity_rollforward(
    bv_open=example["bv_open"],
    net_income=example["net_income"],
    oci=example["oci"],
    dividends=example["dividends"],
    net_share_issues=example["net_share_issues"],
    other_owner_txns=example["other_owner_txns"]
)

# Clean surplus earnings adjustment (treat OCI as dirty-surplus; add other direct-to-equity if applicable)
clean_e = clean_surplus_earnings(example["net_income"], dirty_surplus_items=example["oci"])
ri = residual_income(clean_e, example["cost_of_equity"], example["bv_open"])

inc_sh = treasury_stock_incremental_shares(
    example["options_outstanding"], example["exercise_price"], example["avg_share_price"]
)
diluted_shares = example["basic_shares"] + inc_sh
bvps_value = bvps(bv_close_calc, diluted_shares)

print(f"Ending BV (calc): ${bv_close_calc:.0f}m | Ending BV (reported): ${example['bv_close_reported']:.0f}m")
print(f"Comprehensive income: ${comprehensive_income(example['net_income'], example['oci']):.0f}m")
print(f"Clean surplus earnings: ${clean_e:.0f}m | Residual income: ${ri:.0f}m")
print(f"Diluted shares: {diluted_shares:.1f}m | BVPS: ${bvps_value:.2f}")
```

---

### Valuation Impact
Why this matters:
- Book value and “clean surplus earnings” are the anchors for residual income valuation and for interpreting P/B and ROE correctly.
- OCI and direct-to-equity items can be economically real but invisible in net income; ignoring them misstates sustainable earnings and risk.

Impact on multiples:
- P/E can look cheap if losses/gains are parked in OCI; adjust earnings capacity using clean-surplus earnings.
- P/B comparisons are distorted by buybacks/issuance and OCI reserves; interpret P/B together with ROE (clean-surplus basis).

Impact on DCF:
- Equity statement items (issuance/buybacks) change per-share value even when enterprise value is unchanged.
- Persistent OCI (e.g., pension actuarial losses) can signal higher future cash needs (contributions) and risk premiums.

Comparability issues:
- Accounting regimes differ in what flows through OCI vs P&L (especially financial instruments, pensions, hedges).
- Some firms present “total comprehensive income” prominently; others bury OCI movements—always reconcile.

Practical adjustments:
```python
def adjust_earnings_for_dirty_surplus(reported_earnings: float, oci: float, other_direct_equity: float = 0.0) -> float:
    """Create clean-surplus earnings for valuation by adding back dirty-surplus items."""
    import numpy as np
    re = float(reported_earnings); o = float(oci); ode = float(other_direct_equity)
    if not np.isfinite(re) or not np.isfinite(o) or not np.isfinite(ode):
        raise ValueError("Inputs must be finite.")
    return re + o + ode
```

---

### Quality of Earnings Flags (Equity statement focused)
⚠️ Large recurring OCI losses (pension, hedging, financial assets) while management highlights “adjusted EPS”.  
⚠️ Book value growth driven mainly by net share issuance rather than earnings (dependency on external capital).  
⚠️ Frequent equity restructurings (special dividends, buybacks, reissues) that obscure performance trends.  
⚠️ Share-based payment treated as “non-cash add-back” without recognising dilution cost.  
✅ Reconciliation ties cleanly; OCI items are explained and their cash implications are discussed; buybacks are disciplined and aligned with intrinsic value.

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banks/Insurers | OCI from AFS/ FVOCI portfolios and hedges can be material | treat OCI as risk-bearing and model capital/ROE impact; watch regulatory capital effects |
| Multinationals | FX translation reserves volatile | separate translation OCI from operating performance; assess if hedging policy stabilises cash flows |
| Industrials with DB pensions | actuarial gains/losses in OCI (IFRS) | incorporate pension deficit as financing-like claim; model contribution cash outflows and risk |
| High-growth tech | heavy SBC and frequent issuance | treat SBC as compensation + dilution; focus on per-share value creation not headline earnings |

---

### Real-World Example
Scenario: Two firms have identical net income, but one has large OCI losses and heavy dilution.

```python
firm_a = {"net_income": 500.0, "oci": 0.0, "net_share_issues": -200.0}      # buybacks
firm_b = {"net_income": 500.0, "oci": -250.0, "net_share_issues": 300.0}    # issuance + OCI losses

ci_a = comprehensive_income(firm_a["net_income"], firm_a["oci"])
ci_b = comprehensive_income(firm_b["net_income"], firm_b["oci"])

clean_a = clean_surplus_earnings(firm_a["net_income"], firm_a["oci"])
clean_b = clean_surplus_earnings(firm_b["net_income"], firm_b["oci"])

print(f"Firm A comprehensive income: ${ci_a:.0f}m | clean earnings: ${clean_a:.0f}m")
print(f"Firm B comprehensive income: ${ci_b:.0f}m | clean earnings: ${clean_b:.0f}m")
print(f"Firm A net share issues: ${firm_a['net_share_issues']:.0f}m (buybacks)")
print(f"Firm B net share issues: ${firm_b['net_share_issues']:.0f}m (issuance)")
```

Interpretation: Firm B’s “same earnings” are lower quality because value is leaking through OCI and per-share ownership is being diluted through issuance. In valuation, adjust earnings to clean surplus and model per-share effects explicitly.

See also: Chapter 5 (pricing book values), Chapter 6 (pricing earnings), Chapter 12 (profitability), Chapter 18 (quality of financial statements).
