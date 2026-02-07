# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 9: Measuring Earnings (Operating Earnings, Normalisation, Reinvestment Link, Earnings Quality)

### Core Concept
Earnings are the most common “language” for valuation (P/E, EV/EBITDA, margins), but accounting earnings often differ from economic earnings due to one-offs, classification issues, and timing choices. The practical goal is to produce **sustainable operating earnings** that are consistent across firms and time, then map those earnings to **cash flows** and **returns on capital** used in intrinsic valuation.

See also: Chapter 3 (recasting statements), Chapter 12 (terminal value uses sustainable earnings), Chapter 17–20 (multiples depend on consistent earnings definitions).

---

### Formula/Methodology

#### 1) Operating earnings vs net income (classification discipline)
```text
Operating Earnings (EBIT) = Revenue − Operating Expenses
NOPAT = Adjusted EBIT × (1 − Tax Rate)

Net Income = EBIT − Interest Expense +/− Non-operating Items − Taxes
```
Practical rule:
- Use **EBIT/NOPAT** for operating performance and firm valuation (FCFF/WACC).
- Use **Net income** for equity valuation (FCFE/k_e), but still normalise and remove non-operating noise.

#### 2) Normalised (sustainable) earnings
```text
Normalised EBIT = Reported EBIT + Non-recurring Charges − Non-recurring Gains + Classification Adjustments
Normalised Net Income = Reported Net Income + Non-recurring Charges − Non-recurring Gains + Classification Adjustments
```
Examples of “classification adjustments” (pick a consistent policy and apply across peers):
- Stock-based compensation: expense vs add-back with a separate dilution adjustment.
- Operating leases: treat as operating cost (reported) or convert to debt-like and adjust EBITDA/EV consistently.
- Capitalised operating expenses (e.g., dev costs): expense-equivalent adjustment to avoid overstating EBIT.

#### 3) From earnings to cash flows (reinvestment link)
```text
FCFF = NOPAT + Depreciation & Amortization − Capex − ΔNWC
```
Core consistency check:
- High earnings growth without reinvestment is usually not credible unless the business is truly capital-light and scaling through price/mix or working capital release.

#### 4) Earnings-based multiples require consistent numerators and denominators
```text
P/E = Equity Value / Net Income (per-share: Price / EPS)
EV/EBITDA = Enterprise Value / EBITDA
EV/EBIT = Enterprise Value / EBIT
```
Consistency rules:
- If EBITDA is “adjusted,” EV must include all debt-like claims consistently (leases, pensions, minorities, preferred).
- If using EBIT/NOPAT, ensure depreciation/capex intensity and capitalisation policies are comparable (or explicitly adjust).

---

### Python Implementation
```python
from typing import Optional, Dict


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def normalize_earnings(
    reported: float,
    non_recurring_charges: float = 0.0,
    non_recurring_gains: float = 0.0,
    classification_addbacks: float = 0.0
) -> float:
    """
    Generic earnings normalisation.

    Normalised = Reported + Charges - Gains + Classification Addbacks

    Args:
        reported: reported earnings metric (EBIT, EBITDA, Net Income) in currency.
        non_recurring_charges: costs to add back if truly non-recurring (currency, >=0 typical).
        non_recurring_gains: gains to remove if truly non-recurring (currency, >=0 typical).
        classification_addbacks: recurring reclassifications for comparability (currency).

    Returns:
        float: normalised earnings (currency).

    Raises:
        ValueError: invalid input types.
    """
    for name, v in {
        "reported": reported,
        "non_recurring_charges": non_recurring_charges,
        "non_recurring_gains": non_recurring_gains,
        "classification_addbacks": classification_addbacks
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if non_recurring_charges < 0 or non_recurring_gains < 0:
        raise ValueError("non_recurring_charges and non_recurring_gains must be >= 0")
    return reported + non_recurring_charges - non_recurring_gains + classification_addbacks


def nopat(adjusted_ebit: float, tax_rate: float) -> float:
    """NOPAT = Adjusted EBIT * (1 - tax_rate)."""
    if not _num(adjusted_ebit):
        raise ValueError("adjusted_ebit must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    return adjusted_ebit * (1.0 - tax_rate)


def fcff(nopat_value: float, da: float, capex: float, delta_nwc: float) -> float:
    """FCFF = NOPAT + D&A - Capex - ΔNWC."""
    for name, v in {"nopat_value": nopat_value, "da": da, "capex": capex, "delta_nwc": delta_nwc}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if capex < 0:
        raise ValueError("capex must be >= 0")
    return nopat_value + da - capex - delta_nwc


def pe_ratio(equity_value: float, net_income: float) -> Optional[float]:
    """P/E = Equity Value / Net Income. Returns None if net_income <= 0."""
    for name, v in {"equity_value": equity_value, "net_income": net_income}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if net_income <= 0:
        return None
    return equity_value / net_income


def ev_multiple(ev: float, metric: float) -> Optional[float]:
    """EV/Metric. Returns None if metric <= 0."""
    for name, v in {"ev": ev, "metric": metric}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if metric <= 0:
        return None
    return ev / metric


def bridge_ev_to_equity(ev: float, net_debt: float, other_claims: float = 0.0) -> float:
    """Equity = EV - Net Debt - Other Claims."""
    for name, v in {"ev": ev, "net_debt": net_debt, "other_claims": other_claims}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return ev - net_debt - other_claims


# Example usage (all $m)
data = {
    "reported_ebit": 620.0,
    "reported_net_income": 310.0,
    "non_recurring_charges": 55.0,
    "non_recurring_gains": 20.0,
    "tax_rate": 0.24,
    "da": 210.0,
    "capex": 260.0,
    "delta_nwc": 45.0,
    "ev": 6_800.0,
    "net_debt": 900.0,
}

adj_ebit = normalize_earnings(
    data["reported_ebit"],
    non_recurring_charges=data["non_recurring_charges"],
    non_recurring_gains=data["non_recurring_gains"]
)

adj_ni = normalize_earnings(
    data["reported_net_income"],
    non_recurring_charges=data["non_recurring_charges"],
    non_recurring_gains=data["non_recurring_gains"]
)

n = nopat(adj_ebit, data["tax_rate"])
f = fcff(n, data["da"], data["capex"], data["delta_nwc"])

equity_value = bridge_ev_to_equity(data["ev"], data["net_debt"])
pe = pe_ratio(equity_value, adj_ni)
ev_ebit = ev_multiple(data["ev"], adj_ebit)

print(f"Adjusted EBIT: ${adj_ebit:.0f}m")
print(f"Adjusted Net Income: ${adj_ni:.0f}m")
print(f"NOPAT: ${n:.0f}m")
print(f"FCFF: ${f:.0f}m")
print(f"EV/EBIT: {ev_ebit:.1f}x" if ev_ebit is not None else "EV/EBIT: n/a")
print(f"P/E: {pe:.1f}x" if pe is not None else "P/E: n/a (loss-making or near-zero earnings)")
```

---

### Valuation Impact
Why this matters:
- **Terminal value often depends on sustainable earnings** (stable margins and returns). If current earnings include one-offs, valuation will be biased—often materially.
- **Multiples are fragile**: even small definition differences (leases, SBC, capitalised costs) can swing EV/EBITDA or P/E conclusions more than “real” economics.
- **Earnings must reconcile with reinvestment**: high margins with underfunded capex or aggressive working-capital release can inflate earnings but not value.

Impact on multiples:
- Adjusted EBITDA/EBIT supports more comparable EV multiples; adjusted net income supports more comparable P/E and ROE comparisons.
- In loss-making firms, shift away from P/E and EV/EBIT; use EV/Revenue with margin path and unit economics (see Chapter 22).

Impact on DCF inputs:
- Adjusted EBIT drives NOPAT and FCFF directly; changes in normalisation feed into the whole cash-flow forecast.
- Classification choices affect invested capital and ROIC, which constrain sustainable growth and terminal assumptions.

Practical adjustments:
```python
def apply_sbc_policy(adjusted_ebit: float, sbc_expense: float, policy: str = "expense") -> float:
    """
    Simple policy switch for SBC treatment in EBIT:
    - 'expense': keep SBC in EBIT (no add-back)
    - 'addback': add SBC back to EBIT (requires separate dilution adjustment elsewhere)
    """
    if policy not in {"expense", "addback"}:
        raise ValueError("policy must be 'expense' or 'addback'")
    if not _num(adjusted_ebit) or not _num(sbc_expense):
        raise ValueError("inputs must be numeric")
    if sbc_expense < 0:
        raise ValueError("sbc_expense must be >= 0")
    return adjusted_ebit + sbc_expense if policy == "addback" else adjusted_ebit
```

---

### Quality of Earnings Flags
⚠️ Repeated “one-off” restructuring charges over multiple years (suggests ongoing operating issue).  
⚠️ Large gains from asset sales or mark-to-market items included in “operating” results.  
⚠️ EBITDA improving while capex and working-capital needs are rising (earnings not translating to value).  
⚠️ Aggressive capitalisation (dev costs, customer acquisition, maintenance) boosting EBIT in the short run.  
⚠️ Margin expansion without clear driver (pricing power, mix, scale) and without peer corroboration.  
✅ Green flag: consistent earnings definition, clear bridge from EBIT to cash flows, and stable cash conversion over time.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Software | SBC + capitalised development | Decide and document SBC policy; adjust for dev cost capitalisation if peers differ. |
| Retail | Lease expense embedded in opex | Keep treatment consistent across peers; avoid mixing lease-adjusted and reported multiples. |
| Industrials | Cyclical “peak” earnings | Normalise using mid-cycle margins or multi-year averages; avoid valuing on peak EBIT. |
| Banks | Provisioning drives earnings | Use through-the-cycle credit losses; focus on ROE and book value metrics. |

---

### Practical Comparison Tables

#### 1) Common normalisation items (earnings quality)
| Item | Typical adjustment | Why |
|---|---|---|
| Restructuring | Add back if truly non-recurring | Prevent understated sustainable margin |
| Asset sale gains | Remove | Not operating; non-repeatable |
| Litigation/settlements | Case-by-case | Often non-recurring, but check recurrence |
| Impairments | Usually exclude from operating | Non-cash; indicates past overinvestment |
| SBC | Policy choice + disclose | Affects comparability and per-share value |
| Capitalised costs | “Expense-equivalent” adjust | Avoid overstated EBIT/margins |

#### 2) Multiple consistency rules
| Multiple | Numerator must include | Denominator must match | Common pitfall |
|---|---|---|---|
| EV/EBITDA | All debt-like claims in EV | EBITDA definition consistent | Mixing lease-adjusted EBITDA with unadjusted EV |
| EV/EBIT | Same as above | EBIT consistent re: capitalisation | Using EBIT with capitalised opex vs peers expensing |
| P/E | Equity value | Net income consistent | Using P/E across very different leverage/tax regimes |

---

### Real-World Example
Scenario: You are building a peer multiple screen but suspect the target’s EBIT includes non-recurring gains and peers treat SBC differently. You normalise, compute NOPAT and FCFF for DCF consistency, and compute EV/EBIT and P/E on a consistent basis.

```python
reported_ebit = 620.0
reported_net_income = 310.0
one_off_charges = 55.0
one_off_gains = 20.0
tax_rate = 0.24

adj_ebit = normalize_earnings(reported_ebit, one_off_charges, one_off_gains)
adj_net_income = normalize_earnings(reported_net_income, one_off_charges, one_off_gains)

# Apply SBC policy if needed (example: add back $40m SBC)
adj_ebit_sbc = apply_sbc_policy(adj_ebit, sbc_expense=40.0, policy="expense")

n = nopat(adj_ebit_sbc, tax_rate)
f = fcff(n, da=210.0, capex=260.0, delta_nwc=45.0)

ev = 6_800.0
net_debt = 900.0
equity_value = bridge_ev_to_equity(ev, net_debt)

print(f"Adj EBIT (policy): ${adj_ebit_sbc:.0f}m")
print(f"NOPAT: ${n:.0f}m, FCFF: ${f:.0f}m")
print(f"EV/EBIT: {ev_multiple(ev, adj_ebit_sbc):.1f}x")
pe = pe_ratio(equity_value, adj_net_income)
print(f"P/E: {pe:.1f}x" if pe is not None else "P/E: n/a")
```

Interpretation: The value conclusion should not hinge on whether one firm capitalises a cost or labels a charge “non-recurring.” Your job is to align definitions so multiples reflect economics, then cross-check with DCF cash flows built from the same operating base.
