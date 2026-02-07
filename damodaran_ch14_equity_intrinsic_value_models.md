# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 14: Equity Intrinsic Value Models (Dividend Discount, FCFE, Residual Income, Two-Stage Models)

### Core Concept
Equity intrinsic valuation models value equity directly by discounting **cash flows to equity** (dividends or FCFE) at the **cost of equity**, or by valuing **excess returns** over the required return (residual income). The practical application is choosing the right equity model based on payout policy and financing needs, and ensuring internal consistency among growth, reinvestment, and returns.

See also: Chapter 12 (terminal value closure), Chapter 15 (firm valuation/WACC/APV), Chapter 16 (equity value per share), Chapter 9 (measuring earnings).

---

### Formula/Methodology

#### 1) Dividend Discount Model (DDM) — stable growth
```text
Equity Value (end of year 0) = Div_1 / (k_e − g)

Where:
Div_1 = expected dividend next period
k_e   = cost of equity (decimal)
g     = stable growth rate in dividends/earnings (decimal)
Constraint: k_e > g
```
When to use:
- Mature firms with stable payout policies where dividends reflect cash returned to shareholders.
Avoid when:
- Dividends are disconnected from capacity to pay (high buybacks, irregular payouts), or firm is transitioning rapidly.

#### 2) FCFE model — stable growth
```text
Equity Value = FCFE_1 / (k_e − g)

Where:
FCFE_1 = expected free cash flow to equity next period
k_e    = cost of equity
g      = stable growth rate in FCFE
Constraint: k_e > g
```
FCFE definition (common operating form):
```text
FCFE = Net Income
     + Depreciation & Amortization
     − Capex
     − ΔNWC
     + Net Debt Issued

Where:
Net Debt Issued = New Debt − Debt Repaid
ΔNWC = change in non-cash working capital
```
Practical note:
- FCFE already accounts for financing; discount at k_e (not WACC).

#### 3) Two-stage equity valuation (high growth then stable)
```text
Equity Value = Σ [FCFE_t / (1 + k_e)^t]  for t=1..n  +  Terminal Equity Value_n / (1 + k_e)^n

Terminal Equity Value_n = FCFE_{n+1} / (k_e − g_stable)
```
Use when the firm is expected to transition from abnormal growth to stable growth.

#### 4) Residual Income (RI) / Excess Return model
Residual income measures value created beyond the required return on beginning book equity.
```text
Residual Income_t = Net Income_t − (k_e × Book Equity_{t−1})

Equity Value = Book Equity_0 + Σ [Residual Income_t / (1 + k_e)^t] + Terminal RI Value
```
Key consistency:
- Book equity must be measured consistently (clean surplus assumption: changes in book equity reflect earnings minus payouts, plus/minus share issuance/repurchase effects).

#### 5) Linking growth to fundamentals (equity perspective)
```text
g ≈ ROE × (1 − Payout Ratio)

Where:
ROE = Net Income / Book Equity
Payout Ratio = Dividends / Net Income (or total payout including buybacks if modeled)
```
Interpretation:
- High growth with high payout requires high ROE or external equity issuance (which affects per-share value).

---

### Python Implementation
```python
from typing import List, Optional, Dict


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def ddm_stable(div_next: float, ke: float, g: float) -> float:
    """Stable-growth DDM: Value = Div1 / (ke - g)."""
    for name, v in {"div_next": div_next, "ke": ke, "g": g}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if ke <= g:
        raise ValueError("Need ke > g for stable-growth DDM.")
    return div_next / (ke - g)


def fcfe(net_income: float, da: float, capex: float, delta_nwc: float, net_debt_issued: float) -> float:
    """
    FCFE = NI + D&A - Capex - ΔNWC + Net Debt Issued.
    """
    for name, v in {
        "net_income": net_income, "da": da, "capex": capex, "delta_nwc": delta_nwc, "net_debt_issued": net_debt_issued
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if capex < 0:
        raise ValueError("capex must be >= 0")
    return net_income + da - capex - delta_nwc + net_debt_issued


def terminal_value_perpetuity(cf_next: float, r: float, g: float) -> float:
    """Generic TV = CF_{n+1}/(r-g)."""
    for name, v in {"cf_next": cf_next, "r": r, "g": g}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if r <= g:
        raise ValueError("Need r > g for a finite perpetuity terminal value.")
    return cf_next / (r - g)


def pv_cash_flows(cash_flows: List[float], r: float) -> float:
    """Present value of cash flows discounted at r."""
    if not _num(r) or r <= -1:
        raise ValueError("r must be numeric and > -100%.")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list.")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values.")
        pv += cf / ((1.0 + r) ** t)
    return pv


def equity_value_two_stage_fcfe(
    fcfe_forecast: List[float],
    ke: float,
    g_stable: float
) -> float:
    """
    Two-stage FCFE model: PV(FCFE_1..n) + PV(Terminal value at n).

    Args:
        fcfe_forecast: list of FCFE for years 1..n.
        ke: cost of equity.
        g_stable: stable growth rate after year n.

    Returns:
        float: equity value at t=0.

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(fcfe_forecast, list) or len(fcfe_forecast) == 0:
        raise ValueError("fcfe_forecast must be a non-empty list")
    if not _num(ke) or ke <= -1:
        raise ValueError("ke must be numeric and > -100%.")
    if not _num(g_stable):
        raise ValueError("g_stable must be numeric")
    n = len(fcfe_forecast)
    for cf in fcfe_forecast:
        if not _num(cf):
            raise ValueError("fcfe_forecast contains non-numeric values")

    # terminal based on FCFE_{n+1} = FCFE_n * (1 + g_stable) (only if year n is near-stable)
    fcfe_n = fcfe_forecast[-1]
    fcfe_n1 = fcfe_n * (1.0 + g_stable)
    tv_n = terminal_value_perpetuity(fcfe_n1, ke, g_stable)

    pv_stage1 = pv_cash_flows(fcfe_forecast, ke)
    pv_tv = tv_n / ((1.0 + ke) ** n)
    return pv_stage1 + pv_tv


def residual_income_equity_value(
    book_equity0: float,
    net_income_forecast: List[float],
    book_equity_begin: List[float],
    ke: float
) -> float:
    """
    Equity value via residual income:
    Value = BE0 + Σ RI_t/(1+ke)^t
    where RI_t = NI_t - ke*BE_{t-1}.

    Args:
        book_equity0: book equity at time 0.
        net_income_forecast: NI for years 1..n.
        book_equity_begin: beginning book equity for years 1..n (i.e., BE_{t-1}).
        ke: cost of equity.

    Returns:
        float: equity value.

    Raises:
        ValueError: invalid inputs.
    """
    if not _num(book_equity0) or book_equity0 < 0:
        raise ValueError("book_equity0 must be numeric and >= 0")
    if not _num(ke) or ke <= -1:
        raise ValueError("ke must be numeric and > -100%.")
    if len(net_income_forecast) == 0 or len(net_income_forecast) != len(book_equity_begin):
        raise ValueError("net_income_forecast and book_equity_begin must have same non-zero length")
    pv_ri = 0.0
    for t, (ni, be_prev) in enumerate(zip(net_income_forecast, book_equity_begin), start=1):
        if not _num(ni) or not _num(be_prev) or be_prev < 0:
            raise ValueError("invalid NI or beginning book equity values")
        ri = ni - ke * be_prev
        pv_ri += ri / ((1.0 + ke) ** t)
    return book_equity0 + pv_ri


# Example usage (all $m)
ke = 0.10
g_stable = 0.03

# Two-stage FCFE: explicit forecast then stable growth
fcfe_years_1_5 = [120, 145, 170, 190, 205]
equity_val_fcfe = equity_value_two_stage_fcfe(fcfe_years_1_5, ke, g_stable)
print(f"Equity value (two-stage FCFE): ${equity_val_fcfe:,.0f}m")

# Stable-growth DDM example
div1 = 50.0
equity_val_ddm = ddm_stable(div1, ke, g_stable)
print(f"Equity value (stable DDM): ${equity_val_ddm:,.0f}m")

# Residual income example (3-year RI block)
be0 = 900.0
ni_forecast = [110.0, 120.0, 125.0]
be_begin = [900.0, 940.0, 980.0]  # beginning book equity each year (illustrative)
equity_val_ri = residual_income_equity_value(be0, ni_forecast, be_begin, ke)
print(f"Equity value (residual income, no terminal RI): ${equity_val_ri:,.0f}m")
```

---

### Valuation Impact
Why this matters:
- Equity models are often more natural for financial firms and dividend-centric businesses, and they avoid firm-value bridges when capital structure is stable and payouts reflect capacity to pay.
- FCFE explicitly incorporates financing; if you instead discount firm cash flows at k_e (or FCFE at WACC), the valuation will be inconsistent.
- Residual income provides a robust cross-check when cash flows are hard to estimate but book equity is meaningful and accounting is clean.

Impact on multiples:
- Stable DDM/FCFE frameworks imply justified P/E levels via k_e and g; use them to sanity-check whether market P/E is consistent with growth and risk.
- Residual income ties to P/B: persistent excess ROE over k_e supports higher P/B multiples.

Impact on DCF inputs:
- Terminal growth and payout assumptions must converge to stable conditions (k_e > g; sustainable ROE; sustainable payout).
- For FCFE, leverage assumptions matter: Net debt issued is a driver; unstable debt policies can distort FCFE and per-share value.

Practical adjustments:
```python
def equity_value_per_share(equity_value: float, shares_outstanding: float, options_value: float = 0.0) -> float:
    """Per-share value with a simple option/dilution deduction."""
    for name, v in {"equity_value": equity_value, "shares_outstanding": shares_outstanding, "options_value": options_value}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be > 0")
    return (equity_value - options_value) / shares_outstanding
```

---

### Quality of Earnings Flags (Equity Model Pitfalls)
⚠️ Dividends used as cash flow when the firm is underpaying dividends relative to capacity (DDM undervalues).  
⚠️ FCFE inflated by temporary debt issuance; leverage cannot rise forever (terminal FCFE must reflect stable leverage).  
⚠️ Using accounting net income without normalisation (one-offs, accounting noise) drives residual income errors.  
⚠️ Book equity distorted by buybacks at prices above/below intrinsic value; clean-surplus assumption breaks without adjustment.  
✅ Green flag: payout/FCFE policies are modelled consistently with reinvestment and target leverage, and residual income cross-check aligns with FCFE value.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banks/insurers | Debt is operating; payouts regulated | Prefer equity models (dividends/FCFE) with regulatory capital constraints. |
| Utilities | Stable payouts | DDM or FCFE can work well; ensure capex needs are reflected in FCFE. |
| Mature consumer | Buybacks dominate payouts | Model total payout (dividends + buybacks) or use FCFE. |
| High growth | No dividends; reinvestment heavy | Use multi-stage FCFE with fade; avoid stable DDM assumptions. |

---

### Practical Comparison Tables

#### 1) Choosing the equity model
| Situation | Preferred model | Why |
|---|---|---|
| Stable dividends | DDM | Dividends reflect distribution policy |
| Dividends not representative | FCFE | Captures true capacity to pay |
| Financial firms | Equity models | Firm value/WACC less meaningful |
| Book value meaningful | Residual income | Strong link to value via ROE − k_e |

#### 2) Consistency checks
| Check | Rule | Fix if fails |
|---|---|---|
| Finite TV | k_e > g | Reduce g or increase k_e (justify) |
| Sustainable payout | payout aligns with reinvestment needs | Model buybacks/debt changes explicitly |
| Stable leverage in terminal | net debt issuance stabilises | Use target debt ratio in terminal FCFE |
| Residual income sanity | ROE relative to k_e drives RI sign | Recast earnings/book equity definitions |

---

### Real-World Example
Scenario: A mature company pays low dividends but conducts buybacks and maintains a target leverage ratio. You value equity using a two-stage FCFE model and compute implied per-share value.

```python
ke = 0.10
g = 0.03
fcfe_forecast = [120, 145, 170, 190, 205]  # $m

equity_value = equity_value_two_stage_fcfe(fcfe_forecast, ke, g)
per_share = equity_value_per_share(equity_value, shares_outstanding=250.0, options_value=150.0)

print(f"Equity value: ${equity_value:,.0f}m")
print(f"Value per share: ${per_share:,.2f}")
```

Interpretation: Equity intrinsic models are most useful when the claim being valued (equity) has cash flows that can be defined cleanly. When dividends are not informative, FCFE is usually more reliable than DDM, provided leverage assumptions converge to a stable long-run policy.
