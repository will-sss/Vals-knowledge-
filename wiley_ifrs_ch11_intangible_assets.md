# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 11: Intangible Assets (IAS 38, capitalization rules, amortisation vs impairment, software costs)

### Core Concept
IAS 38 sets the rules for recognising and measuring intangible assets, distinguishing between acquired intangibles and internally generated intangibles, and requiring strict criteria for capitalising development costs. For valuation, IAS 38 is central because it affects EBITDA/EBIT comparability (capitalised development costs vs expensed R&D), invested capital (ROIC), and the risk profile of future cash flows (amortisation schedules, impairment triggers, and the credibility of capitalised assets).

### Formula/Methodology

#### 1) Recognition: development vs research (capitalisation gate)
```text
Research costs: expense as incurred (no capitalisation)

Development costs: capitalise only if ALL criteria are met (practical summary):
- Technical feasibility to complete
- Intention to complete and use/sell
- Ability to use/sell
- Probable future economic benefits (market or internal usefulness)
- Adequate resources to complete
- Ability to measure expenditures reliably
```

#### 2) Subsequent measurement
Cost model (typical):
```text
Carrying Amount = Cost - Accumulated Amortisation - Accumulated Impairment
```

Amortisation (finite-lived intangibles):
```text
Annual Amortisation = (Cost - Residual Value) / Useful Life
```

Indefinite-lived intangibles:
```text
No amortisation; test for impairment at least annually (IAS 36 linkage)
```

#### 3) R&D and capitalisation impacts on operating performance
Conceptual bridge:
```text
EBIT (capitalisation policy) = EBIT (expense policy) + Development costs capitalised - Amortisation on capitalised development (current period)
```

#### 4) ROIC comparability (capitalised intangibles as invested capital)
```text
Adjusted Invested Capital = Reported Invested Capital + Capitalised R&D (net of amortisation)
Adjusted NOPAT = NOPAT + (After-tax R&D expense) - (After-tax amortisation of capitalised R&D)
Adjusted ROIC = Adjusted NOPAT / Adjusted Invested Capital
```

---

### Practical Application for Valuation Analysts

#### 1) Normalise R&D / software development for comparability
Two companies can have identical economics but different accounting:
- Company A expenses most development costs → lower EBIT/EBITDA today
- Company B capitalises development costs → higher EBIT/EBITDA today but higher amortisation later

For peer multiples:
- Decide a policy: either (i) treat development costs as operating expense (expense-like), or (ii) capitalise a consistent portion across peers and adjust EBIT/IC to match.

A pragmatic approach for many valuations:
- Use reported numbers for DCF (cash focus) but perform **peer multiple normalisation** when capitalisation practices differ materially.
- For ROIC analysis, consider capitalising R&D for sectors where it is clearly “investment” (software, pharma, engineering).

#### 2) Watch for “capitalised cost inflation”
IAS 38 criteria are judgemental; aggressive capitalisation can be used to boost profits:
- Rising capitalised development as % of total R&D
- Weak linkage to product launches or revenue growth
- Large intangibles with subsequent impairments

#### 3) Amortisation schedules and terminal value
- Heavy amortisation of acquired intangibles can depress EBIT but not cash (non-cash), affecting EV/EBIT multiples.
- For terminal value, ensure reinvestment assumptions reflect ongoing R&D needs (even if capitalised).

#### 4) Internally generated brands and goodwill-like items
IAS 38 generally prohibits recognising internally generated brands, mastheads, publishing titles, customer lists as intangibles. This creates:
- Understatement of internally built intangible value on balance sheet
- Potential misinterpretation of asset-light business capital intensity if you rely on accounting invested capital

---

### Python Implementation
```python
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def straight_line_amortisation(cost: float, residual_value: float, useful_life_years: float) -> float:
    """
    Annual amortisation for finite-lived intangible.

    Raises:
        ValueError: invalid inputs.
    """
    c = _num(cost, "cost")
    r = _num(residual_value, "residual_value")
    life = _num(useful_life_years, "useful_life_years")
    if life <= 0:
        raise ValueError("useful_life_years must be > 0.")
    if r < 0:
        raise ValueError("residual_value must be >= 0.")
    if r > c:
        raise ValueError("residual_value cannot exceed cost.")
    return (c - r) / life

def normalize_ebit_for_capitalised_development(reported_ebit: float, dev_capitalised: float, amort_on_dev: float) -> float:
    """
    Restate EBIT as if development costs were expensed (expense-like policy).

    Logic:
      If company capitalises dev costs, reported EBIT is higher by (dev_capitalised - amort_on_dev).
      To expense-like: subtract that benefit.

    Adjusted EBIT = reported_ebit - (dev_capitalised - amort_on_dev)

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(reported_ebit, "reported_ebit")
    cap = _num(dev_capitalised, "dev_capitalised")
    am = _num(amort_on_dev, "amort_on_dev")
    if cap < 0 or am < 0:
        raise ValueError("dev_capitalised and amort_on_dev must be >= 0.")
    return e - (cap - am)

def capitalise_rd_series(rd_expense: pd.Series, amort_years: int) -> pd.DataFrame:
    """
    Simple capitalisation of R&D: create a notional R&D asset and amortise straight-line.

    Assumptions:
      - Each year's R&D is treated as a separate 'vintage' asset
      - Amortised straight-line over amort_years
      - No residual value

    Args:
        rd_expense: Series indexed by period (ascending)
        amort_years: amortisation horizon (>=1)

    Returns:
        DataFrame with columns: rd_expense, amortisation, rd_asset_end

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(rd_expense, pd.Series):
        raise ValueError("rd_expense must be a pandas Series.")
    if amort_years < 1:
        raise ValueError("amort_years must be >= 1.")
    if not pd.api.types.is_numeric_dtype(rd_expense):
        raise ValueError("rd_expense must be numeric.")

    idx = list(rd_expense.index)
    # Track vintages: list of (remaining_years, annual_amort, carrying_amount)
    vintages = []
    rows = []
    rd_asset = 0.0

    for t in idx:
        spend = float(rd_expense.loc[t])
        if not np.isfinite(spend) or spend < 0:
            raise ValueError("R&D expense must be finite and >= 0.")

        # Add new vintage
        annual_amort = spend / amort_years
        vintages.append({"remaining": amort_years, "annual_amort": annual_amort, "carrying": spend})
        rd_asset += spend

        # Amortise all vintages for the period
        amort = 0.0
        for v in vintages:
            if v["remaining"] > 0:
                amort += v["annual_amort"]
                v["carrying"] -= v["annual_amort"]
                v["remaining"] -= 1

        # Remove fully amortised vintages (numerical tolerance)
        new_vintages = []
        rd_asset = 0.0
        for v in vintages:
            if v["remaining"] > 0 and v["carrying"] > 1e-9:
                new_vintages.append(v)
                rd_asset += max(0.0, v["carrying"])
        vintages = new_vintages

        rows.append({"period": t, "rd_expense": spend, "amortisation": amort, "rd_asset_end": rd_asset})

    return pd.DataFrame(rows).set_index("period")

def adjust_roic_for_capitalised_rd(
    nopat: float,
    invested_capital: float,
    rd_expense: float,
    rd_amortisation: float,
    tax_rate: float
) -> Dict[str, float]:
    """
    Adjust NOPAT and invested capital for capitalised R&D conceptually.

    Adjusted NOPAT = NOPAT + (1-tax)*R&D expense - (1-tax)*R&D amortisation
    Adjusted IC = IC + R&D asset (requires separate calculation; here provide delta = (R&D expense - amortisation) as rough proxy)

    Returns:
        dict with adjusted_nopat and adjusted_roic (requires adjusted_ic input, approximated).

    Note:
        For robust use, compute full R&D asset from history; this is a single-period proxy.

    Raises:
        ValueError: invalid inputs.
    """
    n = _num(nopat, "nopat")
    ic = _num(invested_capital, "invested_capital")
    rd = _num(rd_expense, "rd_expense")
    am = _num(rd_amortisation, "rd_amortisation")
    tr = _num(tax_rate, "tax_rate")
    if not (0.0 <= tr <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    if rd < 0 or am < 0:
        raise ValueError("rd_expense and rd_amortisation must be >= 0.")
    adj_n = n + (1.0 - tr) * rd - (1.0 - tr) * am
    # Single-period proxy for adjusted IC change (placeholder)
    adj_ic = ic + max(0.0, rd - am)
    if adj_ic <= 0:
        raise ValueError("adjusted invested capital must be > 0.")
    return {"adjusted_nopat": adj_n, "adjusted_roic": adj_n / adj_ic, "adjusted_ic_proxy": adj_ic}

# Example usage
amort = straight_line_amortisation(cost=150_000_000, residual_value=0.0, useful_life_years=5)
print(f"Annual amortisation: ${amort/1e6:.1f}M")

reported_ebit = 220.0
dev_cap = 35.0
am_on_dev = 6.0
ebit_expense_like = normalize_ebit_for_capitalised_development(reported_ebit, dev_cap, am_on_dev)
print(f"EBIT (expense-like): ${ebit_expense_like:.1f}m")

rd_series = pd.Series([90.0, 100.0, 115.0, 130.0], index=["2021", "2022", "2023", "2024"])
rd_cap = capitalise_rd_series(rd_series, amort_years=4)
print(rd_cap)

adj = adjust_roic_for_capitalised_rd(nopat=140.0, invested_capital=900.0, rd_expense=130.0, rd_amortisation=95.0, tax_rate=0.25)
print(adj)
```

---

### Valuation Impact
Why this matters:
- Capitalised development costs can **inflate operating profit** and **understate reinvestment** if you ignore the build-up of intangible assets.
- Intangibles often dominate value in software, media, pharma, and services; accounting balance sheets understate internally generated intangibles, distorting ROIC and capital intensity.
- Amortisation vs impairment affects EBIT and multiples; cash flow focus reduces noise, but peer comparisons still require adjustments.

Impact on multiples:
- EV/EBITDA: capitalising development makes EBITDA higher; adjust for policy differences across peers.
- EV/EBIT: acquired intangibles amortisation can depress EBIT; decide whether to use EBITA (policy choice) and stay consistent.
- P/E: affected by amortisation and impairment; distinguish economic from accounting expense.

Impact on DCF inputs:
- R&D and product development are ongoing reinvestment; ensure forecast cash costs reflect reality even if capitalised.
- Terminal margins should incorporate sustainable innovation spend; do not extrapolate temporarily inflated margins from aggressive capitalisation.

Practical adjustments:
```python
def normalize_ebitda_for_capitalised_development(reported_ebitda: float, dev_capitalised: float) -> float:
    """
    Expense-like EBITDA adjustment: subtract development costs that were capitalised.
    Adjusted EBITDA = reported_ebitda - dev_capitalised

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(reported_ebitda, "reported_ebitda")
    cap = _num(dev_capitalised, "dev_capitalised")
    if cap < 0:
        raise ValueError("dev_capitalised must be >= 0.")
    return e - cap
```

---

### Quality of Earnings Flags
⚠️ Capitalised development costs rising faster than revenue without clear product outputs.  
⚠️ Large “internally generated intangible” balances with limited disclosures on criteria and useful lives.  
⚠️ Frequent impairment of capitalised development (suggests aggressive recognition).  
⚠️ EBITDA “improvement” driven by higher capitalisation rather than cash generation.  
✅ Transparent IAS 38 disclosures: capitalised amounts, amortisation methods, useful lives, impairment testing triggers, and reconciliation of carrying amounts.

---

### Sector-Specific Considerations

| Sector | Key IAS 38 issue | Typical valuation treatment |
|---|---|---|
| Software / SaaS | Capitalised software development costs | Normalise EBITDA/EBIT for capitalisation differences; consider capitalising R&D for ROIC comparability |
| Pharma / Biotech | Research expensed; development can be capitalised late-stage | Use pipeline probability-weighting; treat R&D as investment and focus on cash burn and optionality |
| Media / Gaming | Content costs and useful lives | Align amortisation with revenue pattern; consider unit economics and release cycles |
| Industrials / Engineering | Development and tooling | Validate capitalisation criteria; cross-check capex and intangible build with project pipeline |

---

### Real-World Example
Scenario: A SaaS company capitalises $50m of development costs and amortises $12m. You want comparable EV/EBIT and a ROIC view.

```python
reported_ebit = 180.0
dev_capitalised = 50.0
amort_on_dev = 12.0

ebit_expense_like = normalize_ebit_for_capitalised_development(reported_ebit, dev_capitalised, amort_on_dev)
print(f"Expense-like EBIT: ${ebit_expense_like:.1f}m")

# If EV is $3,600m:
ev = 3_600.0
mult_reported = ev / reported_ebit if reported_ebit != 0 else float("inf")
mult_expense_like = ev / ebit_expense_like if ebit_expense_like != 0 else float("inf")

print(f"EV/EBIT reported: {mult_reported:.1f}x")
print(f"EV/EBIT expense-like: {mult_expense_like:.1f}x")
```

Interpretation: Expense-like normalisation usually increases the multiple for aggressive capitalisers; it also improves comparability across SaaS peers.

See also: Chapter 15 (business combinations) for acquired intangibles recognition; Chapter 13 (impairment) for IAS 36 testing; Chapter 7 (IAS 8) for estimate changes in useful lives and amortisation methods.
