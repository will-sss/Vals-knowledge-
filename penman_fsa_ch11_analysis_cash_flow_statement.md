# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 11: The Analysis of the Cash Flow Statement (Cash vs Accrual, FCFE/FCFF, Cash Flow Diagnostics)

### Core Concept
The cash flow statement explains **where cash came from and where it went**, but valuation requires mapping cash flows to **value drivers**: operating performance, reinvestment, and financing policy. The core practical skill is to reconcile **earnings to cash**, isolate sustainable cash generation, and detect working-capital or “non-cash” accrual patterns that will unwind and affect future free cash flow.

---

### Formula/Methodology

#### 1) Operating cash flow (CFO) and accruals (diagnostic)
Common reconciliation concept:
```text
CFO ≈ Net Income
      + Non-cash charges (D&A, impairments, provisions)
      ± Changes in working capital (ΔWC)
      ± Other accrual adjustments
```

Accruals proxy (useful QoE check):
```text
Total Accruals = Net Income - CFO
Accruals Ratio = (Net Income - CFO) / Average Total Assets
```

Where:
- CFO = cash flow from operating activities
- Average Total Assets = (Assets_t + Assets_{t-1}) / 2

Interpretation:
- Persistently positive accruals (NI > CFO) can signal aggressive revenue recognition, capitalised costs, or working-capital build.

#### 2) Free cash flow to the firm (FCFF)
Basic build:
```text
FCFF = NOPAT + D&A - Capex - ΔNWC
```

Where:
- NOPAT = EBIT × (1 - Tax rate) (or operating profit after tax)
- Capex = capital expenditures (gross)
- ΔNWC = change in net working capital (operating current assets - operating current liabilities)

Cash flow statement cross-check:
```text
FCFF ≈ CFO - Capex + After-tax Interest Paid   (classification differences exist)
```

#### 3) Free cash flow to equity (FCFE)
Basic build (from FCFF):
```text
FCFE = FCFF - After-tax Interest + Net Borrowing
Net Borrowing = Debt Issued - Debt Repaid
```

Alternative build (equity cash flow lens):
```text
FCFE = CFO - Capex + Net Borrowing
```

#### 4) Cash conversion metrics (diagnostic)
```text
CFO Conversion = CFO / EBITDA   (or CFO / EBIT; define consistently)
FCF Conversion = FCFF / EBITDA  (or FCFF / NOPAT)
```

#### 5) “Cash vs Value” framing
Key separation:
```text
Operating cash flow reflects operating performance + working-capital timing.
Financing cash flows reflect capital structure decisions (not operating value creation).
```

---

### Practical Application (How to apply)

#### A) Build a cash flow bridge (earnings → CFO → FCFF)
1) Start with Net Income and CFO from the statement of cash flows.
2) Compute accruals: NI - CFO (and the accruals ratio).
3) Split working-capital effects:
   - ΔReceivables, ΔInventory, ΔPayables (or the reported “changes in working capital” bucket).
4) Compute FCFF:
   - Prefer **NOPAT + D&A - Capex - ΔNWC**
   - If you only have cash flow statement, approximate **FCFF ≈ CFO - Capex + after-tax interest paid** (note classification differences).
5) Compare FCFF trend to revenue/EBIT growth: is growth “self-funding” or consuming cash?

#### B) Diagnose “good” vs “bad” cash flow
- Healthy pattern: CFO tracks earnings over time; ΔNWC stable; Capex aligned with growth.
- Concern pattern: Earnings rising but CFO flat/negative due to receivables/inventory build; frequent “add-backs” from provisions or impairments; recurring restructuring cash costs.

#### C) Use cash flow statement to validate forecasting assumptions
- Working capital: if ΔNWC is structurally negative (payables financing), you cannot assume it continues indefinitely without supplier strain.
- Capex: reconcile depreciation vs capex; if capex persistently below depreciation, check whether assets are being under-invested (sustainability risk).
- Taxes: if cash taxes are well below book taxes, check for loss utilisation, tax holidays, or deferrals (may reverse).

#### D) Valuation mapping
- Use FCFF for enterprise DCF (discount at WACC).
- Use FCFE for equity DCF (discount at cost of equity), but only if financing policy is stable and forecastable.

---

### Python Implementation
```python
from typing import Dict, Any, Optional
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

def accruals_metrics(net_income: float, cfo: float, assets_open: float, assets_close: float) -> Dict[str, float]:
    """Compute accrual diagnostics.

    Args:
        net_income: Net income for period ($)
        cfo: Cash flow from operations ($)
        assets_open: Total assets at start of period ($)
        assets_close: Total assets at end of period ($)

    Returns:
        dict: total_accruals, accruals_ratio

    Raises:
        ValueError: invalid inputs or zero average assets
    """
    ni = _f(net_income, "net_income")
    c = _f(cfo, "cfo")
    a0 = _f(assets_open, "assets_open")
    a1 = _f(assets_close, "assets_close")
    avg_assets = (a0 + a1) / 2.0
    if abs(avg_assets) < 1e-12:
        raise ValueError("Average assets must be non-zero.")
    total_accruals = ni - c
    accr_ratio = total_accruals / avg_assets
    return {"total_accruals": float(total_accruals), "accruals_ratio": float(accr_ratio)}

def fcff_from_drivers(ebit: float, tax_rate: float, da: float, capex: float, delta_nwc: float) -> float:
    """Compute FCFF from operating drivers.

    FCFF = NOPAT + D&A - Capex - ΔNWC

    Args:
        ebit: Earnings before interest and tax ($)
        tax_rate: tax rate as decimal (e.g., 0.25)
        da: depreciation & amortisation ($)
        capex: capital expenditures ($, positive number)
        delta_nwc: change in net working capital ($, positive means investment/cash outflow)

    Returns:
        float: FCFF ($)

    Raises:
        ValueError: invalid inputs
    """
    e = _f(ebit, "ebit")
    t = _f(tax_rate, "tax_rate")
    if t < 0 or t > 0.6:
        raise ValueError("tax_rate looks implausible; use decimal (e.g., 0.25).")
    d = _f(da, "da")
    cx = _f(capex, "capex")
    dnwc = _f(delta_nwc, "delta_nwc")
    if cx < 0:
        raise ValueError("capex should be a positive cash outflow amount.")
    nopat = e * (1.0 - t)
    return float(nopat + d - cx - dnwc)

def fcfe_from_fcff(fcff: float, interest_expense: float, tax_rate: float, net_borrowing: float) -> float:
    """Compute FCFE from FCFF.

    FCFE = FCFF - After-tax Interest + Net Borrowing

    Args:
        fcff: Free cash flow to firm ($)
        interest_expense: interest paid/expense ($)
        tax_rate: tax rate as decimal
        net_borrowing: debt issued - debt repaid ($)

    Returns:
        float: FCFE ($)

    Raises:
        ValueError: invalid inputs
    """
    f = _f(fcff, "fcff")
    i = _f(interest_expense, "interest_expense")
    t = _f(tax_rate, "tax_rate")
    nb = _f(net_borrowing, "net_borrowing")
    if t < 0 or t > 0.6:
        raise ValueError("tax_rate looks implausible; use decimal (e.g., 0.25).")
    after_tax_interest = i * (1.0 - t)
    return float(f - after_tax_interest + nb)

def cash_conversion(cfo: float, ebitda: float, fcff: float) -> Dict[str, float]:
    """Compute cash conversion ratios."""
    c = _f(cfo, "cfo")
    e = _f(ebitda, "ebitda")
    f = _f(fcff, "fcff")
    cfo_conv = safe_divide(c, e, on_zero=np.nan)
    fcf_conv = safe_divide(f, e, on_zero=np.nan)
    if cfo_conv is None or not np.isfinite(cfo_conv):
        raise ValueError("EBITDA must be non-zero for conversion metrics.")
    if fcf_conv is None or not np.isfinite(fcf_conv):
        raise ValueError("EBITDA must be non-zero for conversion metrics.")
    return {"cfo_to_ebitda": float(cfo_conv), "fcff_to_ebitda": float(fcf_conv)}

# Example usage ($m)
data = {
    "net_income": 720.0,
    "cfo": 540.0,
    "assets_open": 12_000.0,
    "assets_close": 13_200.0,
    "ebit": 1_100.0,
    "tax_rate": 0.25,
    "da": 420.0,
    "capex": 650.0,
    "delta_nwc": 120.0,
    "interest_expense": 180.0,
    "net_borrowing": 90.0,
    "ebitda": 1_520.0
}

acc = accruals_metrics(data["net_income"], data["cfo"], data["assets_open"], data["assets_close"])
fcff = fcff_from_drivers(data["ebit"], data["tax_rate"], data["da"], data["capex"], data["delta_nwc"])
fcfe = fcfe_from_fcff(fcff, data["interest_expense"], data["tax_rate"], data["net_borrowing"])
conv = cash_conversion(data["cfo"], data["ebitda"], fcff)

print(f"Total accruals (NI - CFO): ${acc['total_accruals']:.0f}m | Accruals ratio: {acc['accruals_ratio']:.2%}")
print(f"FCFF: ${fcff:.0f}m | FCFE: ${fcfe:.0f}m")
print(f"CFO/EBITDA: {conv['cfo_to_ebitda']:.1%} | FCFF/EBITDA: {conv['fcff_to_ebitda']:.1%}")
```

---

### Valuation Impact
Why this matters:
- DCF valuation relies on **free cash flow**, not accounting earnings. The cash flow statement is the primary tool for validating whether earnings are translating into cash and whether growth is consuming or generating cash.
- Cash flow diagnostics identify whether a firm is funding growth internally (higher value and resilience) or requiring repeated external financing (higher risk and dilution potential).

Impact on multiples:
- Weak cash conversion often leads to lower EV/EBITDA (market discounts earnings quality).
- Firms with strong, stable FCFF often sustain higher valuation multiples because cash supports reinvestment, dividends, or buybacks.

Impact on DCF inputs:
- Forecast ΔNWC and capex from history and business model; avoid assuming “nice” steady cash flow if the working-capital cycle is volatile.
- Identify temporary tax or working-capital benefits that will reverse and reduce terminal cash flow.

Comparability issues:
- IFRS/US GAAP classification differences (interest paid in operating vs financing; dividends received/paid) can shift CFO/FCF. Normalize definitions across peers.
- Capitalised development costs and other capitalisation policies can inflate CFO by moving cash outflows to investing.

Practical adjustments:
```python
def normalise_cfo_for_working_capital(cfo: float, delta_nwc: float, target_delta_nwc: float = 0.0) -> float:
    """Normalise CFO by replacing actual ΔNWC with a target/steady-state assumption."""
    import numpy as np
    c = float(cfo); dn = float(delta_nwc); td = float(target_delta_nwc)
    if not np.isfinite(c) or not np.isfinite(dn) or not np.isfinite(td):
        raise ValueError("Inputs must be finite.")
    # CFO includes -ΔNWC (an investment reduces CFO). Replace with target:
    return c + dn - td
```

---

### Quality of Earnings Flags (cash flow-focused)
⚠️ Net income rising but CFO consistently lagging (NI > CFO) with increasing receivables or inventory.  
⚠️ “One-off” restructuring charges recur annually and consume cash despite being labelled non-recurring.  
⚠️ Capex persistently below depreciation with declining asset base—earnings may be unsustainable.  
⚠️ CFO inflated by stretched payables (temporary supplier financing) or factoring without clear disclosure.  
⚠️ Investing cash flows dominated by capitalised costs that should be expensed (aggressive capitalisation).  
✅ CFO tracks earnings over a cycle; FCFF positive and stable; working-capital investment scales sensibly with sales growth.

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Software / R&D heavy | capitalised development costs and SBC distort cash/earnings linkage | adjust for capitalised R&D; treat SBC as operating cost + dilution; focus on FCFF and per-share value |
| Retail / distribution | working capital seasonality | use rolling/TTM ΔNWC; normalize for peak inventory cycles |
| Construction / project-based | contract assets/liabilities drive CFO volatility | align cash conversion to project milestones; stress-test ΔNWC and customer advances |
| Financials | CFO/FCF definitions differ materially | use equity cash flow/dividends or residual income approaches; avoid FCFF-style DCF without deep adjustments |

---

### Real-World Example
Scenario: A company reports strong earnings growth, but cash conversion is deteriorating due to receivables growth.

```python
# $m example
company = {
    "net_income": 200.0,
    "cfo": 80.0,
    "assets_open": 2_000.0,
    "assets_close": 2_400.0,
    "ebit": 320.0,
    "tax_rate": 0.25,
    "da": 60.0,
    "capex": 90.0,
    "delta_nwc": 140.0,  # large working capital investment
    "ebitda": 380.0
}

acc = accruals_metrics(company["net_income"], company["cfo"], company["assets_open"], company["assets_close"])
fcff = fcff_from_drivers(company["ebit"], company["tax_rate"], company["da"], company["capex"], company["delta_nwc"])
conv = cash_conversion(company["cfo"], company["ebitda"], fcff)

print(f"Accruals ratio: {acc['accruals_ratio']:.2%} | CFO/EBITDA: {conv['cfo_to_ebitda']:.1%}")
print(f"FCFF: ${fcff:.0f}m (negative indicates growth consuming cash)")
```

Interpretation: High accruals and weak CFO/EBITDA suggest earnings quality risk; in valuation, either reduce near-term cash flows (higher reinvestment) or challenge revenue/collection assumptions and reassess growth sustainability.

See also: Chapter 4 (cash vs accrual and DCF), Chapter 10 (recasting operating vs financing), Chapter 12 (profitability), Chapter 13 (growth and sustainable earnings), Chapter 18 (quality of financial statements).
