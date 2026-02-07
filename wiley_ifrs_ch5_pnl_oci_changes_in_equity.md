# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 5: Statements of Profit or Loss, OCI, and Changes in Equity (Run-rate earnings, OCI mechanics, equity bridges)

### Core Concept
This chapter focuses on how IFRS presents performance in profit or loss, other comprehensive income (OCI), and the statement of changes in equity. For valuation, the key is to isolate sustainable operating earnings (for multiples and NOPAT) from volatile or non-operating items (fair value gains, disposal gains, FX, one-offs) and to understand equity movements that change per-share value (dividends, buybacks, share issues, share-based payments, OCI reserves).

### Formula/Methodology

#### 1) Earnings measures used in valuation (definition must be consistent)
```text
EBITDA = Revenue - (Operating expenses excluding D&A)
EBIT = EBITDA - Depreciation - Amortization
NOPAT = EBIT × (1 - Tax rate)
```

Where:
- Operating expenses should exclude financing items and non-operating gains/losses based on your policy.

#### 2) OCI and total comprehensive income
```text
Total Comprehensive Income = Profit or Loss + OCI
```

OCI split (conceptual):
```text
OCI = OCI (recyclable to P&L) + OCI (non-recyclable)
```

#### 3) Changes in equity (bridge for per-share analysis)
```text
Equity_end = Equity_begin + Profit_or_loss + OCI + Owner_contributions - Owner_distributions + Other_changes
```

Typical owner contributions/distributions:
- Share issues, share-based payment share issues (equity-settled), capital injections
- Dividends, share buybacks, capital reductions

#### 4) Basic per-share value (valuation output, not IFRS requirement)
```text
Equity Value per Share = Equity Value / Diluted Shares Outstanding
```

---

### Practical Application for Valuation Analysts

#### 1) Build a “run-rate” operating earnings bridge (reported → adjusted)
The income statement often includes items that should not be capitalised into a steady-state multiple:
- Disposal gains/losses
- Fair value movements (investment property, financial instruments, biological assets)
- Major restructuring and litigation
- One-time impairments (note: impairment is non-cash but signals weaker economics)
- FX items (policy-dependent: operating vs financing vs non-operating)

A standard bridge:
```text
Adjusted EBIT = Reported EBIT - Non-operating gains + Non-operating losses - One-off income + One-off costs (if treated as non-recurring)
Adjusted EBITDA = Adjusted EBIT + D&A (consistent D&A definition)
```

#### 2) OCI: when it matters for valuation vs when it is noise
OCI can matter when:
- It contains items that will later recycle into P&L (timing issue)
- It captures economic changes in pension obligations or hedges (risk, cash implications)
- It explains equity movements affecting book value and P/B comparability

OCI is often less relevant when:
- It reflects purely accounting measurement changes with limited cash flow implications (depends on item)

#### 3) Statement of changes in equity: track dilution and owner flows
For per-share valuation integrity, you need:
- Opening shares, issues, buybacks, option exercises, SBC dilution
- Dividends and distributions
- Movements in reserves that may affect distributable capacity (jurisdiction-specific, but equity structure matters)

---

### Python Implementation
```python
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def compute_ebitda(revenue: float, operating_exp_ex_da: float) -> float:
    """
    EBITDA = Revenue - operating expenses excluding D&A.

    Args:
        revenue: total revenue
        operating_exp_ex_da: operating costs excluding depreciation and amortization

    Returns:
        EBITDA

    Raises:
        ValueError: non-finite inputs.
    """
    r = _num(revenue, "revenue")
    o = _num(operating_exp_ex_da, "operating_exp_ex_da")
    return r - o

def compute_ebit(ebitda: float, depreciation: float, amortization: float) -> float:
    """
    EBIT = EBITDA - depreciation - amortization.
    """
    e = _num(ebitda, "ebitda")
    d = _num(depreciation, "depreciation")
    a = _num(amortization, "amortization")
    return e - d - a

def compute_nopat(ebit: float, tax_rate: float) -> float:
    """
    NOPAT = EBIT * (1 - tax_rate)

    Raises:
        ValueError: invalid tax rate.
    """
    e = _num(ebit, "ebit")
    t = _num(tax_rate, "tax_rate")
    if not (0.0 <= t <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    return e * (1.0 - t)

def build_run_rate_bridge(reported_ebit: float, adjustments: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a run-rate EBIT bridge from reported EBIT.

    Convention:
      - adjustment amount is added to reported EBIT to arrive at adjusted EBIT.
      - For a non-operating gain included in reported EBIT, use negative adjustment to remove it.
      - For a one-off cost included in reported EBIT that you want to add back, use positive adjustment.

    Args:
        reported_ebit: EBIT as presented (or mapped policy EBIT)
        adjustments: list of dicts with keys:
            - item: description
            - amount: numeric
            - rationale: short text

    Returns:
        DataFrame with bridge steps and adjusted EBIT.

    Raises:
        ValueError: malformed inputs.
    """
    rep = _num(reported_ebit, "reported_ebit")
    rows = [{"step": "Reported EBIT", "amount": rep, "rationale": "As reported / mapped"}]
    total_adj = 0.0

    for i, adj in enumerate(adjustments):
        if not isinstance(adj, dict):
            raise ValueError(f"Adjustment {i} must be a dict.")
        item = str(adj.get("item", f"adj_{i}"))
        amt = adj.get("amount", None)
        if not isinstance(amt, (int, float)):
            raise ValueError(f"Adjustment '{item}' amount must be numeric.")
        rationale = str(adj.get("rationale", ""))
        total_adj += float(amt)
        rows.append({"step": item, "amount": float(amt), "rationale": rationale})

    adjusted = rep + total_adj
    rows.append({"step": "Adjusted EBIT", "amount": adjusted, "rationale": "Reported + adjustments"})
    return pd.DataFrame(rows)

def equity_bridge(
    equity_begin: float,
    profit_or_loss: float,
    oci: float,
    owner_contrib: float,
    owner_distrib: float,
    other: float = 0.0
) -> Dict[str, float]:
    """
    Equity_end = Equity_begin + Profit_or_loss + OCI + Owner_contrib - Owner_distrib + Other

    Returns:
        dict with equity_end.
    """
    eb = _num(equity_begin, "equity_begin")
    p = _num(profit_or_loss, "profit_or_loss")
    o = _num(oci, "oci")
    c = _num(owner_contrib, "owner_contrib")
    d = _num(owner_distrib, "owner_distrib")
    oth = _num(other, "other")
    end = eb + p + o + c - d + oth
    return {"equity_end": end}

def per_share_value(equity_value: float, diluted_shares: float) -> float:
    """
    Equity value per share.

    Raises:
        ValueError: invalid shares.
    """
    ev = _num(equity_value, "equity_value")
    sh = _num(diluted_shares, "diluted_shares")
    if sh <= 0:
        raise ValueError("diluted_shares must be > 0.")
    return ev / sh

# Example usage
rev = 1_200.0
opex_ex_da = 820.0
dep = 60.0
amort = 20.0
tax_rate = 0.23

ebitda = compute_ebitda(rev, opex_ex_da)
ebit = compute_ebit(ebitda, dep, amort)
nopat = compute_nopat(ebit, tax_rate)

print(f"EBITDA: ${ebitda:.1f}m")
print(f"EBIT: ${ebit:.1f}m")
print(f"NOPAT: ${nopat:.1f}m")

bridge = build_run_rate_bridge(
    reported_ebit=ebit,
    adjustments=[
        {"item": "Remove disposal gain", "amount": -12.0, "rationale": "Non-operating gain included in EBIT"},
        {"item": "Add back restructuring", "amount": 18.0, "rationale": "One-off cost; assess recurrence"},
    ],
)
print(bridge)

eq = equity_bridge(1_900.0, profit_or_loss=120.0, oci=-10.0, owner_contrib=25.0, owner_distrib=40.0)
print(eq)

ps = per_share_value(equity_value=3_850.0, diluted_shares=770.0)
print(f"Equity value per share: ${ps:.2f}")
```

---

### Valuation Impact
Why this matters:
- Multiples are sensitive to how “earnings” are defined; IFRS presentation variability makes reported subtotals unreliable without standardisation.
- Run-rate earnings drive EV/EBITDA, EV/EBIT, and P/E; inconsistent treatment of fair value gains or disposals can lead to systematic overvaluation.
- OCI and equity movements affect book value and per-share metrics; ignoring dilution and owner flows can misstate equity value per share.

Impact on multiples:
- Remove non-operating gains from operating earnings; treat fair value movements consistently across peers.
- If EBITDA excludes certain expenses by presentation, adjust to a consistent definition (e.g., lease and SBC policy choices, where relevant).

Impact on DCF inputs:
- NOPAT is derived from EBIT and taxes; non-operating items in EBIT distort operating cash flow.
- Equity bridge provides a cross-check against retained earnings logic and ensures model consistency.

Practical adjustments:
```python
def adjust_reported_earnings(reported: float, remove_gains: float = 0.0, addback_costs: float = 0.0) -> float:
    """
    Simple earnings normalisation.

    Args:
        reported: reported metric (EBIT/EBITDA)
        remove_gains: non-operating gains included in reported metric
        addback_costs: one-off costs to add back

    Returns:
        adjusted metric
    """
    r = float(reported)
    g = float(remove_gains)
    c = float(addback_costs)
    if not np.isfinite(r) or not np.isfinite(g) or not np.isfinite(c):
        raise ValueError("inputs must be finite.")
    return r - g + c
```

---

### Quality of Earnings Flags
⚠️ Fair value gains in profit or loss boosting “operating” earnings (common in investment property and some financial instruments).  
⚠️ Disposal gains treated as recurring operating income (especially if disposal activity is frequent).  
⚠️ Large “other income/expense” with weak disclosure; high risk of hidden one-offs.  
⚠️ Frequent restructuring charges described as non-recurring every year.  
✅ Clear reconciliation from statutory profit to adjusted earnings with itemised evidence and consistent policy over time.

---

### Sector-Specific Considerations

| Sector | Typical distortion | Typical valuation treatment |
|---|---|---|
| Real Estate / Investment Property | Fair value movements in P&L | Separate NOI/EBITDA from FV movements; consider NAV-based valuation |
| Asset managers / Insurance | OCI and FV reserves can be large | Analyse OCI components; choose appropriate earnings metric and consider AUM-related KPIs |
| Technology / SaaS | Stock-based compensation and “other income” | Decide consistent SBC policy for comps; separate FX/fair value effects |
| Industrials | Restructuring and impairment cycles | Treat restructuring as recurring if frequent; use mid-cycle margins where relevant |

---

### Real-World Example
Scenario: A company reports EBIT of $300m, including $40m fair value gain and $15m disposal gain, plus a $25m restructuring cost. You want adjusted EBIT for EV/EBIT.

```python
reported_ebit = 300.0
fair_value_gain = 40.0
disposal_gain = 15.0
restructuring_cost = 25.0

adjusted_ebit = adjust_reported_earnings(
    reported=reported_ebit,
    remove_gains=fair_value_gain + disposal_gain,
    addback_costs=restructuring_cost
)

ev = 5_100.0
multiple = ev / adjusted_ebit if adjusted_ebit != 0 else float("inf")

print(f"Adjusted EBIT: ${adjusted_ebit:.1f}m")
print(f"EV/Adjusted EBIT: {multiple:.1f}x")
```

Interpretation: Removing gains raises the multiple; adding back truly one-off costs reduces it. The discipline is consistent policy plus evidence.

See also: Chapter 13 (impairment) for non-cash charges and signalling; Chapter 20 (revenue) and Chapter 24 (financial instruments) for items frequently driving P&L volatility.
