# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 3: Understanding Financial Statements (Operating vs Financing, Clean Metrics, Accounting Adjustments)

### Core Concept
Valuation depends on **cash flows, growth, and risk**, but the inputs come from financial statements that mix operating and financing items and embed accounting choices. The practical task is to **recast statements** to isolate operating performance (for ROIC/FCFF) and financing structure (for WACC/claims), and to normalise earnings for one-offs and accounting distortions. fileciteturn3file2

See also: Chapter 9 (Measuring Earnings), Chapter 15 (Firm Valuation: WACC/APV), Chapter 16 (Equity Value per Share).

---

### Core Concept: Recast Before You Value
A valuation-ready model needs:
- **Operating income** that reflects the economics of the business (not financing, not accounting one-offs).
- **Invested capital** that matches the operating income definition.
- **Cash flows** (FCFF/FCFE) that are consistent with the discount rate (WACC/k_e).

Workflow:
1) Identify and separate **operating** vs **financing** items.  
2) Normalise earnings (remove one-offs; treat recurring items consistently).  
3) Convert earnings to cash flows using reinvestment (capex, working capital) and taxes.  
4) Create consistency checks (ROIC vs reinvestment, margins vs peers, leverage vs risk).

---

### Formula/Methodology

#### 1) Operating vs financing income
```text
Operating Income (EBIT) = Revenue − Operating Expenses (including D&A if using EBIT)
Net Income = EBIT − Interest Expense +/− Other Non-operating Items − Taxes
```
Practical rule:
- Use **EBIT(1 − tax)** for operating profitability (NOPAT) when valuing the firm (FCFF/WACC).
- Use **Net income** (after interest) when valuing equity directly (FCFE/k_e), but still normalise.

#### 2) NOPAT and ROIC
```text
NOPAT = Adjusted EBIT × (1 − Tax Rate)
ROIC = NOPAT / Invested Capital
```
Where:
- Adjusted EBIT excludes one-offs and reflects operating lease/stock comp treatment consistently.
- Invested Capital is operating assets net of non-interest-bearing operating liabilities (definition must match NOPAT).

#### 3) Free cash flow to the firm (FCFF)
```text
FCFF = NOPAT + Depreciation & Amortization − Capex − ΔNWC
```
Where:
- ΔNWC = change in non-cash working capital (receivables + inventory − payables, etc.)

#### 4) Enterprise to equity bridge (claims)
```text
Equity Value = Enterprise Value − Net Debt − Other Claims
Other Claims may include: minority interest, preferred equity, pension deficit, options, etc.
```

---

### Python Implementation
```python
from typing import Dict, Optional


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def nopat(adjusted_ebit: float, tax_rate: float) -> float:
    """
    NOPAT = Adjusted EBIT * (1 - tax_rate)

    Args:
        adjusted_ebit: operating EBIT after normalisation (currency).
        tax_rate: marginal tax rate (decimal, 0-1).

    Returns:
        float: NOPAT (currency).

    Raises:
        ValueError: invalid inputs.
    """
    if not _num(adjusted_ebit):
        raise ValueError("adjusted_ebit must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    return adjusted_ebit * (1.0 - tax_rate)


def roic(nopat_value: float, invested_capital: float) -> Optional[float]:
    """
    ROIC = NOPAT / Invested Capital

    Returns None if invested_capital <= 0 (not meaningful / indicates definition issue).
    """
    if not _num(nopat_value) or not _num(invested_capital):
        raise ValueError("inputs must be numeric")
    if invested_capital <= 0:
        return None
    return nopat_value / invested_capital


def fcff(nopat_value: float, da: float, capex: float, delta_nwc: float) -> float:
    """
    FCFF = NOPAT + D&A - Capex - ΔNWC

    Args:
        nopat_value: NOPAT (currency).
        da: depreciation & amortization (currency).
        capex: capital expenditures (currency, should be >= 0).
        delta_nwc: change in net working capital (currency; + means cash outflow).

    Returns:
        float: FCFF (currency).

    Raises:
        ValueError: invalid inputs.
    """
    for name, v in {"nopat_value": nopat_value, "da": da, "capex": capex, "delta_nwc": delta_nwc}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if capex < 0:
        raise ValueError("capex must be >= 0")
    return nopat_value + da - capex - delta_nwc


def normalize_ebit(reported_ebit: float,
                   one_off_charges: float = 0.0,
                   one_off_gains: float = 0.0,
                   recurring_addbacks: float = 0.0) -> float:
    """
    Simple EBIT normalization:
    Adjusted EBIT = Reported EBIT + one_off_charges - one_off_gains + recurring_addbacks

    Notes:
    - Charges are assumed to reduce reported EBIT; add back if truly non-recurring.
    - Gains are assumed to increase reported EBIT; remove if non-recurring.
    - Recurring_addbacks used for consistent presentation adjustments (e.g., reclassifications).

    Raises:
        ValueError: invalid inputs.
    """
    for name, v in {
        "reported_ebit": reported_ebit,
        "one_off_charges": one_off_charges,
        "one_off_gains": one_off_gains,
        "recurring_addbacks": recurring_addbacks
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return reported_ebit + one_off_charges - one_off_gains + recurring_addbacks


def bridge_ev_to_equity(ev: float, net_debt: float, other_claims: float = 0.0) -> float:
    """
    Equity Value = EV - Net Debt - Other Claims

    Args:
        ev: enterprise value (currency).
        net_debt: debt - cash (currency).
        other_claims: minorities, pref, pensions, options (currency).

    Returns:
        float: equity value.

    Raises:
        ValueError: invalid inputs.
    """
    for name, v in {"ev": ev, "net_debt": net_debt, "other_claims": other_claims}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return ev - net_debt - other_claims


# Example usage (realistic, $m)
financials = {
    "reported_ebit": 620.0,
    "one_off_restructuring": 55.0,
    "one_off_asset_gain": 20.0,
    "tax_rate": 0.24,
    "invested_capital": 4_800.0,
    "da": 210.0,
    "capex": 260.0,
    "delta_nwc": 45.0
}

adj_ebit = normalize_ebit(
    reported_ebit=financials["reported_ebit"],
    one_off_charges=financials["one_off_restructuring"],
    one_off_gains=financials["one_off_asset_gain"]
)

n = nopat(adj_ebit, financials["tax_rate"])
r = roic(n, financials["invested_capital"])
f = fcff(n, financials["da"], financials["capex"], financials["delta_nwc"])

print(f"Adjusted EBIT: ${adj_ebit:.0f}m")
print(f"NOPAT: ${n:.0f}m")
print(f"ROIC: {r:.1%}" if r is not None else "ROIC: n/a (check invested capital definition)")
print(f"FCFF: ${f:.0f}m")
```

---

### Valuation Impact
Why this matters:
- **Operating vs financing separation** determines whether you should use **FCFF/WACC** (firm value) or **FCFE/k_e** (equity value). Mixing them (e.g., FCFF discounted at k_e) produces wrong values. fileciteturn3file2  
- **Normalised earnings** drive sustainable margins, ROIC, and therefore sustainable value (especially terminal value).  
- **Recast invested capital** anchors reinvestment and growth: high growth without reinvestment or with implausible ROIC is a red flag and should be corrected.

Impact on multiples:
- EV/EBITDA comparisons require consistent EBITDA definitions (leases, stock comp, capitalised costs). Bad comparability leads to mispricing conclusions.  
- P/E is sensitive to financing structure and tax effects; use only when capital structures are comparable or when the analysis explicitly adjusts for them.

Impact on DCF inputs:
- NOPAT margin and reinvestment rates (capex + working capital) are direct drivers of FCFF.  
- Invested capital definition affects ROIC and hence the plausibility of forecast returns and terminal assumptions.

Practical adjustments:
```python
def implied_reinvestment_rate(growth: float, roic_value: float) -> Optional[float]:
    """Reinvestment rate ≈ Growth / ROIC (for stable return models). Returns None if ROIC <= 0."""
    if not _num(growth) or not _num(roic_value):
        raise ValueError("inputs must be numeric")
    if roic_value <= 0:
        return None
    return growth / roic_value
```

---

### Quality of Earnings Flags
⚠️ Large “non-recurring” add-backs every year (recurring non-recurring items).  
⚠️ Revenue recognition pull-forwards or unusual working capital movements boosting current earnings (cash conversion deteriorates).  
⚠️ Capitalisation policies inflate EBIT (e.g., capitalised operating costs) and depress current expenses.  
⚠️ EBITDA focus hides deteriorating reinvestment needs (capex rising, maintenance underfunded).  
⚠️ “Other income” or gains driving earnings rather than core operations.  
✅ Strong green flag: stable relationship among NOPAT, reinvestment, and growth; cash conversion tracks earnings over time.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Software | Stock-based compensation, capitalised dev costs | Decide consistent treatment for SBC (expense vs add-back); adjust EBIT/FCFF accordingly and keep peer definitions consistent. |
| Retail | Leases embedded in operations | Treat leases consistently (lease-adjusted EBITDA/EV) or use consistent reported basis across peers. |
| Banks/insurers | Financing is operating | Use equity-based metrics; avoid FCFF as defined for industrial firms. |
| Energy/resources | Depletion and reserve replacement | Stress maintenance capex vs reported D&A; avoid overstating FCFF by underestimating sustaining capex. |

---

### Practical Comparison Tables

#### 1) Recasting Map: What goes where
| Item | Treat as Operating? | Treat as Financing? | Why it matters |
|---|---|---|---|
| Interest expense/income |  | ✅ | Financing cost, not operating performance (for firm valuation) |
| Operating leases | ✅ (economic operating cost) |  | Must be consistent for multiples and FCFF |
| Pension service cost | ✅ |  | Operating labour cost component |
| Pension interest cost |  | ✅ | Financing component of pension liability |
| One-off restructuring | ✅ (exclude from sustainable) |  | Normalise for steady-state margins |
| Asset sale gains | Exclude |  | Non-operating/non-recurring value drivers |

#### 2) Quick consistency checks (model QA)
| Check | Rule of thumb | What to do if it fails |
|---|---|---|
| Growth vs reinvestment | High growth needs reinvestment | Revisit capex/NWC assumptions |
| ROIC vs margins | ROIC should reconcile with margin + turnover | Recast invested capital and EBIT |
| Earnings vs cash | Over time, cash should track earnings | Investigate working capital, capex, capitalisation policies |
| Peer multiple comparability | Same definitions (EBITDA, debt-like items) | Normalize metrics and enterprise value components |

---

### Real-World Example
Scenario: You have a target’s statements with reported EBIT boosted by an asset sale gain and depressed by a “one-off” restructuring charge. You normalise EBIT, compute NOPAT, ROIC, and FCFF to build a consistent firm DCF baseline.

```python
reported_ebit = 620.0
one_off_charges = 55.0
one_off_gains = 20.0
tax_rate = 0.24

adj_ebit = normalize_ebit(reported_ebit, one_off_charges=one_off_charges, one_off_gains=one_off_gains)
n = nopat(adj_ebit, tax_rate)

invested_capital = 4_800.0
da = 210.0
capex = 260.0
delta_nwc = 45.0

r = roic(n, invested_capital)
f = fcff(n, da, capex, delta_nwc)

print(f"Adjusted EBIT: ${adj_ebit:.0f}m")
print(f"NOPAT: ${n:.0f}m")
print(f"ROIC: {r:.1%}" if r is not None else "ROIC: n/a")
print(f"FCFF: ${f:.0f}m")
```

Interpretation: The normalised operating base (NOPAT/ROIC/FCFF) is the input to (a) the explicit forecast period and (b) terminal value assumptions. If ROIC is implausibly high relative to peers, either the business truly has a moat (pricing power, low capital intensity) or the accounting classification/invested capital definition is wrong and must be fixed before valuation.
