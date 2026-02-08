# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 4: Cash Accounting, Accrual Accounting, and Discounted Cash Flow Valuation (FCF, Accrual Drivers, DCF Practicalities)

### Core Concept
Discounted cash flow (DCF) valuation prices expected future cash flows, but reported financial statements are primarily accrual-based, not cash-based. Practical valuation requires reconciling accrual earnings to cash flows, understanding how working capital and investment accounting create timing differences, and using statements to forecast cash flows in a way that is auditable and consistent.

---

### Formula/Methodology

#### 1) Discounted Cash Flow (DCF)
Value of a stream of cash flows discounted at a required return:
```text
PV = Σ [ CF_t / (1 + r)^t ]  for t = 1..N
```

Enterprise value using free cash flow to the firm (FCFF) and WACC:
```text
EV = Σ [ FCFF_t / (1 + WACC)^t ] + TV_N / (1 + WACC)^N
```

Equity value using free cash flow to equity (FCFE) and cost of equity (Re):
```text
Equity Value = Σ [ FCFE_t / (1 + Re)^t ] + TV_N / (1 + Re)^N
```

Where:
- CF_t = cash flow at time t (must match discount rate)
- r = required return consistent with CF risk and claim
- FCFF = cash flow available to all capital providers (debt + equity)
- FCFE = cash flow available to equity holders after debt cash flows
- TV_N = terminal value at horizon N

Terminal value (perpetuity growth, at time N):
```text
TV_N = CF_(N+1) / (r - g)
```

Where:
- CF_(N+1) = cash flow in year N+1
- g = long-run growth rate (must be < r)

#### 2) Cash accounting vs accrual accounting
Cash accounting recognises cash receipts/payments when cash moves.
Accrual accounting recognises revenues when earned and expenses when incurred, regardless of cash timing.

Accrual bridge (generic):
```text
Accrual Earnings = Cash Flow + Accrual Adjustments
```

A common operational bridge for cash flow from operations (CFO):
```text
CFO ≈ Operating Profit After Tax + Non-cash Charges - ΔWorking Capital
```

Free cash flow to the firm:
```text
FCFF = NOPAT + Depreciation & Amortisation - Capex - ΔNWC
```

Where:
- NOPAT = EBIT × (1 - Tax rate) (operating profit after tax)
- ΔNWC = change in net working capital (operating current assets - operating current liabilities)
- Capex = capital expenditure on operating assets

Free cash flow to equity (one practical form):
```text
FCFE = Net Income + D&A - Capex - ΔNWC + Net Borrowing
```

Where:
- Net Borrowing = new debt issued - debt repaid

#### 3) Accruals and working capital mechanics (timing differences)
Working capital investment consumes cash:
```text
If ΔNWC > 0  => cash outflow (reduces FCF)
If ΔNWC < 0  => cash inflow (increases FCF)
```

Key working capital components:
- Accounts receivable: increases reduce cash vs sales (collection lag)
- Inventory: increases reduce cash (build stock)
- Accounts payable / accruals: increases increase cash (payment lag)

---

### Practical Application (How to apply)

#### A) Don’t forecast “cash” in isolation: forecast operating drivers that map to cash
Recommended order:
1) Forecast revenues and operating margins (to get EBIT / NOPAT)
2) Forecast reinvestment needs (Capex and working capital) as functions of scale
3) Derive FCFF/FCFE from statement-consistent bridges

Practical driver ratios (often more stable than levels):
- Capex / Revenue
- Depreciation / Capex (asset age/capital intensity indicator)
- ΔNWC / Revenue (or NWC / Revenue)

#### B) Build consistency checks to prevent DCF “fantasy cash flows”
Checks that should always be performed:
- Revenue growth implies working capital and/or capex needs; a model with high growth but zero reinvestment is inconsistent.
- Depreciation cannot exceed the depreciable asset base dynamics indefinitely (watch unrealistic D&A vs capex).
- If tax rate is materially below normal, assess sustainability (loss utilisation, one-offs).

#### C) Use accruals to diagnose cash flow quality
High accrual earnings with weak cash flow can signal:
- aggressive revenue recognition
- capitalisation of expenses
- working capital build (collection/inventory issues)

Accrual ratio (one practical diagnostic):
```text
Accrual Ratio = (Net Income - CFO) / Average Total Assets
```

Interpretation:
- Higher accrual ratio suggests more earnings not backed by cash (potential quality risk)

#### D) Choose the right cash flow definition and discount rate
Rules:
- Discount FCFF with WACC
- Discount FCFE with cost of equity (Re)
- Don’t mix after-interest cash flows with WACC (mismatch)
- Terminal growth g must be < discount rate

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

def nopat(ebit: float, tax_rate: float) -> float:
    """NOPAT = EBIT × (1 - tax_rate)."""
    e = _f(ebit, "ebit")
    t = _f(tax_rate, "tax_rate")
    if t < 0 or t > 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    return float(e * (1.0 - t))

def fcff(nopat_val: float, da: float, capex: float, delta_nwc: float) -> float:
    """FCFF = NOPAT + D&A - Capex - ΔNWC."""
    n = _f(nopat_val, "nopat")
    d = _f(da, "da")
    c = _f(capex, "capex")
    w = _f(delta_nwc, "delta_nwc")
    return float(n + d - c - w)

def fcfe(net_income: float, da: float, capex: float, delta_nwc: float, net_borrowing: float) -> float:
    """FCFE = Net Income + D&A - Capex - ΔNWC + Net Borrowing."""
    ni = _f(net_income, "net_income")
    d = _f(da, "da")
    c = _f(capex, "capex")
    w = _f(delta_nwc, "delta_nwc")
    nb = _f(net_borrowing, "net_borrowing")
    return float(ni + d - c - w + nb)

def pv_cashflows(cashflows: Sequence[float], rate: float, *, start_t: int = 1) -> float:
    """PV = Σ CF_t / (1+r)^t for end-of-period CFs."""
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

def terminal_value_perpetuity(cf_next: float, rate: float, g: float) -> float:
    """TV = CF_(N+1) / (rate - g) with validation."""
    cf1 = _f(cf_next, "cf_next")
    r = _f(rate, "rate")
    gg = _f(g, "g")
    if r <= -0.999999:
        raise ValueError("rate must be > -100%.")
    if gg >= r:
        raise ValueError("g must be < rate to compute a stable perpetuity.")
    return float(cf1 / (r - gg))

def dcf_value(cashflows: Sequence[float], rate: float, g: float) -> Dict[str, float]:
    """PV of explicit cashflows + perpetuity terminal value at horizon N."""
    pv = pv_cashflows(cashflows, rate, start_t=1)
    cf = np.array([_f(x, "cashflow") for x in cashflows], dtype=float)
    n = int(cf.size)
    cf_next = float(cf[-1] * (1.0 + _f(g, "g")))
    tv = terminal_value_perpetuity(cf_next, rate, _f(g, "g"))
    pv_tv = float(tv / ((1.0 + _f(rate, "rate")) ** n))
    return {"PV_Explicit": float(pv), "TV_N": float(tv), "PV_TV": float(pv_tv), "Total": float(pv + pv_tv)}

def accrual_ratio(net_income: float, cfo: float, avg_total_assets: float) -> float:
    """Accrual Ratio = (Net Income - CFO) / Avg Total Assets."""
    ni = _f(net_income, "net_income")
    cf = _f(cfo, "cfo")
    ata = _f(avg_total_assets, "avg_total_assets")
    out = safe_divide(ni - cf, ata, on_zero=np.nan)
    if out is None or not np.isfinite(out):
        raise ValueError("avg_total_assets must be non-zero.")
    return float(out)

# Example usage: derive FCFF and value it
inputs = {
    "revenue": 1_000_000_000,   # $1,000m
    "ebit": 150_000_000,        # $150m
    "tax_rate": 0.25,
    "da": 40_000_000,           # $40m
    "capex": 60_000_000,        # $60m
    "delta_nwc": 20_000_000     # $20m (working capital investment)
}

n = nopat(inputs["ebit"], inputs["tax_rate"])
fcff_y1 = fcff(n, inputs["da"], inputs["capex"], inputs["delta_nwc"])
print(f"FCFF (Y1): ${fcff_y1/1e6:.1f}m")

# Simple 5-year projection (illustrative): grow FCFF at 6% for 5 years, then terminal at 3%
fcff_series = [fcff_y1 * ((1.06) ** i) for i in range(0, 5)]
val = dcf_value(fcff_series, rate=0.095, g=0.03)
print({k: round(v/1e6, 1) for k, v in val.items()})
```

---

### Valuation Impact
Why this matters:
- DCF depends on cash flows, but forecasting cash flows credibly requires understanding accrual-to-cash timing differences. Accrual statements provide a disciplined structure for forecasting operating performance and reinvestment needs.
- Working capital and capex assumptions often drive valuation more than margins; linking them to revenue (and checking consistency) reduces model risk.

Impact on multiples:
- Weak cash conversion (high accruals, working capital build) can justify a lower multiple even if reported earnings are strong.
- EBITDA-based multiples can overstate cash generation in capex-heavy or working-capital-hungry businesses.

Impact on DCF inputs:
- FCFF must be internally consistent with growth and reinvestment.
- Terminal value sensitivity to g and discount rate requires explicit validation (g < discount rate).

Comparability issues across companies:
- Different working-capital models (subscription vs wholesale) alter cash conversion and therefore sustainable valuation.
- Capitalisation policies affect earnings but not necessarily cash; adjust when comparing profitability or “cash-like” metrics.

Practical adjustments:
```python
def normalise_delta_nwc(delta_nwc: float, revenue: float, target_nwc_to_rev: float, current_nwc_to_rev: float) -> float:
    """Adjust ΔNWC toward a target NWC/Revenue assumption for comparability."""
    import numpy as np
    d = float(delta_nwc); rev = float(revenue)
    if not np.isfinite(d) or not np.isfinite(rev):
        raise ValueError("Inputs must be finite.")
    if abs(rev) < 1e-12:
        raise ValueError("revenue must be non-zero.")
    # If current NWC/Rev is above target, assume working capital releases over time (negative ΔNWC adjustment)
    adjustment = (current_nwc_to_rev - target_nwc_to_rev) * rev
    return float(d + adjustment)
```

---

### Quality of Earnings Flags
⚠️ Reported earnings rising while CFO is flat/declining (possible working capital stress or aggressive accruals).  
⚠️ Persistent positive accrual ratio or sharp increases in receivables/inventory relative to revenue.  
⚠️ Capex suppressed below depreciation during growth (may be under-investing; future capex catch-up risk).  
✅ Stable cash conversion metrics (CFO/EBITDA, ΔNWC/Revenue) and reinvestment assumptions consistent with growth.

---

### Sector-Specific Considerations

| Sector | Key cash vs accrual issue | Typical treatment |
|---|---|---|
| Subscription software | deferred revenue and billing terms drive CFO timing | model billings/collections; track deferred revenue movements |
| Retail/wholesale | inventory cycles dominate cash conversion | forecast inventory days and payables discipline |
| Construction/engineering | contract assets/liabilities create large timing swings | reconcile revenue recognition to cash receipts |
| Capital-intensive utilities | capex timing is central; depreciation may lag investment | capex as driver, not residual; stress-test reinvestment |

---

### Real-World Example
Scenario: Translate accrual operating profit into FCFF, then value using DCF with explicit reinvestment and working capital.

```python
ebit = 150e6
tax_rate = 0.25
da = 40e6
capex = 60e6
delta_nwc = 20e6

n = nopat(ebit, tax_rate)
fcff_y1 = fcff(n, da, capex, delta_nwc)

fcff_series = [fcff_y1 * (1.06 ** i) for i in range(5)]
value = dcf_value(fcff_series, rate=0.095, g=0.03)

print(f"FCFF Y1: ${fcff_y1/1e6:.1f}m")
print(f"Enterprise Value: ${value['Total']/1e9:.2f}bn")
```

Interpretation: If valuation is highly sensitive to ΔNWC and capex assumptions, present a valuation range and justify reinvestment ratios using segment economics and operating model evidence.

See also: Chapter 5 (pricing book values), Chapter 6 (pricing earnings), Chapter 18 (quality of financial statements).
