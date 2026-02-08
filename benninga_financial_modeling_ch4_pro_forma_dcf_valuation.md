# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 4: Pro Forma Analysis and Valuation Based on the Discounted Cash Flow Approach (drivers, FCFF, terminal value, sensitivity)

### Core Concept
A pro forma model converts operating assumptions (revenue growth, margins, working capital, capex) into forecast financial statements and free cash flows. In DCF, enterprise value is the present value of forecast free cash flow to the firm (FCFF) plus terminal value, discounted at WACC. For valuation work, the key skill is building forecasts that are internally consistent (growth ↔ reinvestment ↔ ROIC) and then stress-testing the valuation drivers.

### Formula/Methodology

#### 1) Core operating drivers (high-level)
```text
Revenue_t = Revenue_(t-1) × (1 + growth_t)

EBIT_t = Revenue_t × EBIT margin_t
NOPAT_t = EBIT_t × (1 - tax_rate_t)

Operating working capital_t = (Operating current assets - Non-interest-bearing operating current liabilities)_t
ΔNWC_t = NWC_t - NWC_(t-1)

FCFF_t = NOPAT_t + D&A_t - Capex_t - ΔNWC_t
```

Where:
- EBIT = earnings before interest and taxes (operating)
- NOPAT = net operating profit after tax (unlevered operating profit)
- D&A = depreciation and amortisation (non-cash)
- Capex = capital expenditures (gross)
- ΔNWC = change in net working capital (cash outflow if positive)

#### 2) Enterprise value from FCFF
```text
Enterprise Value (EV) = Σ_{t=1..N} FCFF_t / (1 + WACC)^t  +  TV_N / (1 + WACC)^N
```

#### 3) Terminal value (two practical methods)
Perpetuity growth:
```text
TV_N = FCFF_(N+1) / (WACC - g)
FCFF_(N+1) = FCFF_N × (1 + g)   (common simplification)
```

Exit multiple:
```text
TV_N = Metric_N × ExitMultiple
Metric_N can be EBITDA_N, EBIT_N, Revenue_N, etc.
```

#### 4) Moving from EV to equity value
```text
Equity Value = EV + Non-operating assets - Net debt - Other claims
Value per share = Equity Value / Diluted shares
```

Other claims may include:
- minority interests (NCI)
- pension deficits (if debt-like)
- provisions treated as debt-like (case-by-case)

#### 5) Internal consistency checks
```text
Reinvestment rate_t = (Capex_t - D&A_t + ΔNWC_t) / NOPAT_t
Growth_t (sustainable) ≈ ROIC_t × Reinvestment rate_t

ROIC_t = NOPAT_t / InvestedCapital_t
```

---

### Practical Application (How to build a robust pro forma DCF)

#### Step 1: Build a driver-based forecast (not a plug)
- Revenue: use volume × price where possible; otherwise justify growth path (market growth + share).
- Margins: separate gross margin, operating expenses, and operating leverage.
- Working capital: forecast DSO, DIO, DPO or NWC as % revenue; ensure no “working capital free lunch.”
- Capex and D&A: capex linked to growth/maintenance; D&A linked to asset base.

#### Step 2: Compute FCFF cleanly
Best practice:
- Start from EBIT, not net income (avoid financing effects).
- Use a sustainable cash tax rate; treat losses/NOLs explicitly if material.
- Avoid double-counting: if you capitalise leases (IFRS 16), keep FCFF consistent with lease treatment.

#### Step 3: Terminal value discipline
- Perpetuity growth g must be <= long-run nominal GDP growth in that currency (rule-of-thumb).
- Ensure FCFF_(N+1) reflects a steady-state: stable margins, reinvestment, and ROIC.
- Cross-check implied exit multiple from perpetuity method and compare to market multiples.

#### Step 4: Reconcile to market reality
- Compare implied EV/EBITDA and EV/Revenue to peers.
- Check whether implied ROIC, reinvestment, and growth are plausible for the industry.

#### Step 5: Sensitivity and scenarios
- Minimum: WACC and terminal growth / exit multiple.
- Also test: margin, revenue growth, and reinvestment intensity.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def fcff_from_drivers(
    revenue: float,
    ebit_margin: float,
    tax_rate: float,
    da: float,
    capex: float,
    delta_nwc: float
) -> float:
    """
    Compute FCFF from simple operating drivers.

    Args:
        revenue: forecast revenue (currency)
        ebit_margin: EBIT margin (decimal, e.g., 0.18)
        tax_rate: cash tax rate (0-1)
        da: depreciation & amortisation (currency)
        capex: capital expenditures (currency, positive number)
        delta_nwc: change in net working capital (currency; + is cash outflow)

    Returns:
        float: FCFF (currency)

    Raises:
        ValueError: invalid inputs.
    """
    rev = _num(revenue, "revenue")
    m = _num(ebit_margin, "ebit_margin")
    t = _num(tax_rate, "tax_rate")
    da = _num(da, "da")
    cx = _num(capex, "capex")
    dnwc = _num(delta_nwc, "delta_nwc")

    if rev < 0:
        raise ValueError("revenue must be >= 0.")
    if not (-1.0 <= m <= 1.0):
        raise ValueError("ebit_margin must be between -1 and 1.")
    if not (0.0 <= t <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    if cx < 0:
        raise ValueError("capex must be >= 0.")

    ebit = rev * m
    nopat = ebit * (1.0 - t)
    fcff = nopat + da - cx - dnwc
    return fcff

def terminal_value_perpetuity(fcff_n: float, wacc: float, g: float) -> float:
    """
    Perpetuity growth terminal value using FCFF_(N+1) = FCFF_N * (1+g).

    Raises:
        ValueError: if wacc <= g or inputs invalid.
    """
    f = _num(fcff_n, "fcff_n")
    r = _num(wacc, "wacc")
    g = _num(g, "g")
    if r <= -0.99:
        raise ValueError("wacc is too low.")
    if g <= -0.99:
        raise ValueError("g is too low.")
    if r <= g:
        raise ValueError("wacc must be > g for perpetuity.")
    fcff_next = f * (1.0 + g)
    return fcff_next / (r - g)

def terminal_value_exit_multiple(metric_n: float, exit_multiple: float) -> float:
    """
    Exit multiple terminal value.

    Raises:
        ValueError: if exit_multiple < 0.
    """
    metric = _num(metric_n, "metric_n")
    mult = _num(exit_multiple, "exit_multiple")
    if mult < 0:
        raise ValueError("exit_multiple must be >= 0.")
    return metric * mult

def present_value_cashflows(cashflows: List[float], discount_rate: float) -> float:
    """
    PV of a list of cashflows at a constant discount rate.

    Args:
        cashflows: list of FCFF for years 1..N
        discount_rate: WACC (decimal)

    Returns:
        float: PV

    Raises:
        ValueError: if discount_rate <= -1 or cashflows empty.
    """
    r = _num(discount_rate, "discount_rate")
    if r <= -0.99:
        raise ValueError("discount_rate must be > -0.99.")
    if not cashflows:
        raise ValueError("cashflows must not be empty.")
    pv = 0.0
    for t, cf in enumerate(cashflows, start=1):
        c = _num(cf, f"cashflows[{t}]")
        pv += c / ((1.0 + r) ** t)
    return pv

def enterprise_value_dcf(fcff: List[float], wacc: float, terminal_value: float) -> float:
    """
    Enterprise value from forecast FCFF and a terminal value at year N.

    Raises:
        ValueError: invalid inputs.
    """
    tv = _num(terminal_value, "terminal_value")
    r = _num(wacc, "wacc")
    pv_fcff = present_value_cashflows(fcff, r)
    n = len(fcff)
    pv_tv = tv / ((1.0 + r) ** n)
    return pv_fcff + pv_tv

def equity_value_from_ev(
    enterprise_value: float,
    non_operating_assets: float = 0.0,
    net_debt: float = 0.0,
    other_claims: float = 0.0
) -> float:
    """
    Convert enterprise value to equity value.

    net_debt: positive means debt > cash.
    other_claims: pensions, minorities, provisions treated as debt-like (positive subtracts equity).
    """
    ev = _num(enterprise_value, "enterprise_value")
    noa = _num(non_operating_assets, "non_operating_assets")
    nd = _num(net_debt, "net_debt")
    oc = _num(other_claims, "other_claims")
    return ev + noa - nd - oc

def value_per_share(equity_value: float, diluted_shares: float) -> float:
    """
    Compute equity value per diluted share.

    Raises:
        ValueError: shares <= 0.
    """
    eq = _num(equity_value, "equity_value")
    sh = _num(diluted_shares, "diluted_shares")
    if sh <= 0:
        raise ValueError("diluted_shares must be > 0.")
    return eq / sh

def implied_exit_multiple_from_perpetuity(tv: float, metric_n: float) -> float:
    """
    Cross-check: implied exit multiple = TV / Metric_N

    Raises:
        ValueError: metric_n <= 0.
    """
    tv = _num(tv, "tv")
    m = _num(metric_n, "metric_n")
    if m <= 0:
        raise ValueError("metric_n must be > 0.")
    return tv / m

# Example usage (illustrative)
assumptions = [
    # year, revenue, ebit_margin, tax_rate, da, capex, delta_nwc
    (1, 1_000_000_000, 0.18, 0.25, 60_000_000, 90_000_000, 20_000_000),
    (2, 1_080_000_000, 0.19, 0.25, 65_000_000, 95_000_000, 10_000_000),
    (3, 1_160_000_000, 0.20, 0.25, 70_000_000, 100_000_000, 12_000_000),
    (4, 1_240_000_000, 0.20, 0.25, 75_000_000, 105_000_000, 15_000_000),
    (5, 1_320_000_000, 0.20, 0.25, 80_000_000, 110_000_000, 15_000_000),
]

fcffs = []
ebitdas = []
for (yr, rev, m, t, da, cx, dnwc) in assumptions:
    cf = fcff_from_drivers(rev, m, t, da, cx, dnwc)
    fcffs.append(cf)
    ebit = rev * m
    ebitda = ebit + da
    ebitdas.append(ebitda)

wacc_rate = 0.09
g = 0.03

tv = terminal_value_perpetuity(fcff_n=fcffs[-1], wacc=wacc_rate, g=g)
ev = enterprise_value_dcf(fcff=fcffs, wacc=wacc_rate, terminal_value=tv)

eq = equity_value_from_ev(ev, non_operating_assets=50_000_000, net_debt=600_000_000, other_claims=80_000_000)
vps = value_per_share(eq, diluted_shares=200_000_000)

implied_mult = implied_exit_multiple_from_perpetuity(tv, metric_n=ebitdas[-1])

print(f"Enterprise value: ${ev/1e9:.2f}B")
print(f"Equity value: ${eq/1e9:.2f}B")
print(f"Value per share: ${vps:.2f}")
print(f"Implied EV/EBITDA multiple (terminal year): {implied_mult:.1f}x")
```

---

### Valuation Impact
Why this matters:
- Pro forma modeling is the bridge between narrative assumptions and valuation outputs. Errors in working capital, capex, or terminal assumptions typically dwarf small P&L differences.
- DCF valuation is highly sensitive to WACC and terminal value assumptions; disciplined internal checks prevent “beautiful but wrong” outputs.

Impact on multiples:
- DCF output can be expressed as implied EV/EBITDA and compared to market. If implied multiples are unrealistic, revisit drivers.
- Normalisation (one-offs, leases, SBC) affects EBITDA and therefore implied multiples and terminal exit assumptions.

Impact on DCF inputs:
- Growth must be supported by reinvestment and ROIC; otherwise terminal value can embed impossible economics.
- Taxes and NWC drive cash conversion; ignoring them can inflate value materially.

Comparability issues across companies:
- Different accounting (leases, capitalised development costs, revenue recognition) changes margins and FCFF timing; adjust for comparability.
- Different lifecycle stages: high-growth firms often have temporarily depressed margins and high reinvestment; terminal assumptions must reflect maturation.

Practical adjustments:
```python
def normalize_ebit_for_one_offs(reported_ebit: float, one_off_expense: float = 0.0, one_off_income: float = 0.0) -> float:
    """Remove one-off items from EBIT (expense add-back; income subtract)."""
    e = _num(reported_ebit, "reported_ebit")
    oe = _num(one_off_expense, "one_off_expense")
    oi = _num(one_off_income, "one_off_income")
    if oe < 0 or oi < 0:
        raise ValueError("one-off inputs must be >= 0.")
    return e + oe - oi
```

---

### Quality of Earnings Flags
⚠️ Forecast working capital improves unrealistically (DSO down, DPO up) without operational explanation.  
⚠️ Capex below depreciation for extended periods despite growth (asset base cannot support it long-run).  
⚠️ Terminal value dominates EV with aggressive g or margins (implicit “perpetual outperformance”).  
⚠️ Taxes set to very low effective rates without NOL schedule support.  
⚠️ EBITDA “adjustments” that remove recurring costs (SBC, recurring restructuring, customer success) without justification.  
✅ Forecast ties to operational drivers and yields plausible steady-state ROIC, reinvestment, and margins.

---

### Sector-Specific Considerations

| Sector | Key DCF modeling issue | Typical handling |
|---|---|---|
| SaaS / tech | SBC and deferred revenue dynamics | include dilution separately; model contract liabilities and cash taxes carefully |
| Retail | seasonality and working capital swings | model monthly/quarterly WC or use conservative NWC ratios |
| Industrials | maintenance vs growth capex | split capex; tie depreciation to PP&E and asset lives |
| Utilities | regulated returns and asset base | model RAB and allowed returns; terminal based on regulation assumptions |
| Banks | FCFF not meaningful | use equity-based valuation (dividend discount/residual income) |

---

### Real-World Example
Scenario: You have a 5-year forecast, WACC 9%, terminal growth 3%, net debt and debt-like claims. Compute EV, equity value, value per share, and implied terminal EV/EBITDA.

```python
# Using the example above:
print(f"EV: ${ev/1e9:.2f}B; Equity: ${eq/1e9:.2f}B; VPS: ${vps:.2f}; Terminal implied EV/EBITDA: {implied_mult:.1f}x")
```

Interpretation: Use implied multiple and steady-state reinvestment checks to validate whether the terminal assumptions are market-consistent and economically feasible.

See also: Chapter 3 (WACC) for discount rate; Chapter 6 (leasing) for lease-adjusted cash flows; Chapter 13 (betas) for equity cost inputs; Chapter 22/24 (Monte Carlo) for probabilistic valuation and scenario distributions.
