# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 3: Presentation of Financial Statements (IAS 1, comparability, OCI, subtotals)

### Core Concept
IAS 1 sets the baseline for how IFRS financial statements are structured and presented so users can compare performance across time and across companies. For valuation, IAS 1 choices affect what sits in EBITDA/operating profit, what gets parked in other comprehensive income (OCI), and whether operating vs financing items are easy to separate for ROIC/FCF modelling.

### Formula/Methodology
```text
Comprehensive Income = Profit or Loss + Other Comprehensive Income (OCI)
```

Where:
- Profit or Loss = income less expenses recognised in profit or loss
- OCI = income/expense items not recognised in profit or loss (as required/permitted by other IFRS)

```text
Equity (end) = Equity (begin) + Total Comprehensive Income + Owner Contributions - Owner Distributions + Other Equity Movements
```

Where:
- Owner contributions/distributions are shown separately from non-owner movements in equity.

### Practical Application for Valuation Analysts

#### 1) Build a valuation-ready operating/financing view (presentation → modelling)
IAS 1 presentation can vary by entity (line items, subtotals, sequencing). For valuation you typically need a standardised view:
- Operating performance: revenue, operating expenses, operating profit, operating taxes (NOPAT proxy)
- Financing: interest, debt, finance costs, pension interest components (where relevant), FX on financing (if separable)
- Non-operating/exceptional: disposal gains, fair value movements, restructuring, litigation provisions (case-by-case)

Use a rules-based “re-org” mapping to standardise statements before you forecast.

### Python Implementation
```python
from typing import Dict, Any

def reorganize_pnl_for_valuation(pnl: Dict[str, float]) -> Dict[str, float]:
    """
    Reclassifies a simple P&L into valuation-friendly buckets (operating vs financing vs non-operating).

    Args:
        pnl: Dict of P&L line items (values in reporting currency).
             Expected keys may include: revenue, cogs, opex, depreciation, amortization,
             interest_expense, interest_income, fx, disposal_gains, fair_value_gains, tax

    Returns:
        Dict[str, float]: standardised buckets and key subtotals:
            - operating_ebitda
            - operating_ebit
            - financing_net
            - non_operating_net
            - profit_before_tax_mapped
            - tax
            - profit_after_tax_mapped

    Raises:
        ValueError: if required inputs are missing or non-numeric.
    """
    if not isinstance(pnl, dict):
        raise ValueError("pnl must be a dict of line_item -> amount")

    def _get(key: str, default: float = 0.0) -> float:
        v = pnl.get(key, default)
        if v is None:
            return default
        if not isinstance(v, (int, float)):
            raise ValueError(f"Line item '{key}' must be numeric (got {type(v)})")
        return float(v)

    revenue = _get("revenue")
    cogs = _get("cogs")
    opex = _get("opex")
    depreciation = _get("depreciation")
    amortization = _get("amortization")

    # Operating (definition may differ by company; keep consistent within your model)
    operating_ebitda = revenue - cogs - opex
    operating_ebit = operating_ebitda - depreciation - amortization

    # Financing
    interest_expense = _get("interest_expense")
    interest_income = _get("interest_income")
    financing_net = interest_income - interest_expense

    # Non-operating (examples; adapt to your policy)
    fx = _get("fx")
    disposal_gains = _get("disposal_gains")
    fair_value_gains = _get("fair_value_gains")
    non_operating_net = fx + disposal_gains + fair_value_gains

    profit_before_tax_mapped = operating_ebit + financing_net + non_operating_net
    tax = _get("tax")
    profit_after_tax_mapped = profit_before_tax_mapped - tax

    return {
        "operating_ebitda": operating_ebitda,
        "operating_ebit": operating_ebit,
        "financing_net": financing_net,
        "non_operating_net": non_operating_net,
        "profit_before_tax_mapped": profit_before_tax_mapped,
        "tax": tax,
        "profit_after_tax_mapped": profit_after_tax_mapped,
    }

# Example usage
pnl_example = {
    "revenue": 1_000.0,
    "cogs": 600.0,
    "opex": 250.0,
    "depreciation": 40.0,
    "amortization": 10.0,
    "interest_expense": 35.0,
    "interest_income": 5.0,
    "fx": -2.0,
    "disposal_gains": 12.0,
    "fair_value_gains": 0.0,
    "tax": 15.0,
}

mapped = reorganize_pnl_for_valuation(pnl_example)
print(mapped)
```

#### 2) OCI: classify “recycling” vs “non-recycling” (valuation relevance)
IAS 1 requires OCI to be split between:
- Items that will not be reclassified to profit or loss, and
- Items that will be reclassified to profit or loss (often called “recycled”).

For valuation:
- Non-recycling OCI often reflects measurement changes that affect equity but not future operating cash flows (e.g., revaluation surplus movements; actuarial gains/losses).
- Recycling OCI may later hit profit or loss and can distort run-rate earnings if you don’t understand the mechanism (e.g., certain FVOCI debt instruments and hedging reserves).

### Python Implementation
```python
from typing import List, Dict, Any

def classify_oci_items(oci_items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates OCI into 'recyclable' and 'non_recyclable' buckets.

    Args:
        oci_items: list of dicts, each with:
            - name: str
            - amount: float (positive or negative)
            - recyclable: bool (True if expected to be reclassified to P&L)

    Returns:
        dict: totals by bucket and total_oci

    Raises:
        ValueError: for malformed inputs.
    """
    if not isinstance(oci_items, list):
        raise ValueError("oci_items must be a list of dicts")

    recyclable_total = 0.0
    non_recyclable_total = 0.0

    for i, item in enumerate(oci_items):
        if not isinstance(item, dict):
            raise ValueError(f"OCI item {i} must be a dict")
        name = item.get("name", f"item_{i}")
        amount = item.get("amount", None)
        recyclable = item.get("recyclable", None)

        if not isinstance(amount, (int, float)):
            raise ValueError(f"OCI item '{name}' amount must be numeric")
        if not isinstance(recyclable, bool):
            raise ValueError(f"OCI item '{name}' recyclable must be boolean")

        if recyclable:
            recyclable_total += float(amount)
        else:
            non_recyclable_total += float(amount)

    total_oci = recyclable_total + non_recyclable_total
    return {
        "oci_recyclable": recyclable_total,
        "oci_non_recyclable": non_recyclable_total,
        "total_oci": total_oci,
    }

# Example usage
oci = [
    {"name": "FX translation (foreign operations)", "amount": -8.0, "recyclable": True},
    {"name": "Cash flow hedge reserve (effective portion)", "amount": 3.0, "recyclable": True},
    {"name": "Actuarial gains/losses (DB pensions)", "amount": -6.0, "recyclable": False},
    {"name": "Revaluation surplus movement", "amount": 10.0, "recyclable": False},
]
print(classify_oci_items(oci))
```

#### 3) Comparatives and reclassifications (trend integrity)
IAS 1 emphasises:
- Comparative information (prior period) for amounts and relevant narrative disclosures.
- Consistency of presentation from period to period; changes require reclassification of comparatives when practicable and disclosure if not.
- In certain retrospective cases, an additional statement of financial position at the beginning of the preceding period is required.

For valuation, this directly affects time-series analytics (growth rates, margins, working capital turns) and can create false trend breaks if reclassifications are ignored.

### Python Implementation
```python
import pandas as pd
from typing import Dict

def apply_reclassification(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Reclassifies line items by mapping old labels to new labels and aggregates duplicates.

    Args:
        df: DataFrame indexed by period with columns as line items (numeric).
        mapping: dict old_name -> new_name

    Returns:
        DataFrame: with reclassified columns and duplicates aggregated.

    Raises:
        ValueError: if df is not a DataFrame or contains non-numeric data.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be a dict old_name -> new_name")

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' must be numeric to reclassify")

    out = df.copy()
    out.columns = [mapping.get(c, c) for c in out.columns]
    out = out.groupby(level=0, axis=1).sum()  # aggregate duplicates
    return out

# Example usage
df = pd.DataFrame(
    {
        "Admin expenses": [100, 110],
        "Selling expenses": [50, 55],
        "Distribution costs": [20, 22],
    },
    index=["2023", "2024"],
)
mapping = {"Distribution costs": "Selling expenses"}  # align presentation
print(apply_reclassification(df, mapping))
```

### Valuation Impact
Why this matters:
- Multiples (EV/EBITDA, EV/EBIT): inconsistent presentation (e.g., “other income”, netting policies, disposal gains) can inflate/deflate EBITDA comparability.
- DCF inputs: forecasting requires stable definitions of operating profit, taxes, and working-capital items; IAS 1 reclassifications can create artificial volatility.
- ROIC/ROCE analysis: to compute invested capital and operating returns, you need a clean split between operating vs financing and a consistent treatment of exceptional/non-operating items.
- Bridge between P&L and equity: OCI classification affects equity movements and can explain why book value changes despite “stable” net income.

Practical adjustments:
```python
def normalize_ebitda(reported_ebitda: float, one_offs: float) -> float:
    """
    Normalises EBITDA by removing one-offs (positive one_offs reduce EBITDA; negative increase).
    Use a consistent sign convention in your model.

    Args:
        reported_ebitda: reported EBITDA.
        one_offs: net one-off impact included in EBITDA (e.g., restructuring costs, disposal gains if included).

    Returns:
        float: adjusted EBITDA

    Raises:
        ValueError: if inputs are not numeric.
    """
    if not isinstance(reported_ebitda, (int, float)) or not isinstance(one_offs, (int, float)):
        raise ValueError("reported_ebitda and one_offs must be numeric")
    return float(reported_ebitda) - float(one_offs)
```

### Quality of Earnings Flags
⚠️ Netting/offsetting hides volatility: recurring gains netted against expenses (or vice versa) can mask operational weakness; request gross disclosures where material.  
⚠️ Disposal gains/losses treated as operating: routine asset sales included in operating profit can contaminate run-rate margins and EV/EBITDA.  
⚠️ Large recyclable OCI swings: significant “recycling” reserves (hedges, FVOCI debt) can foreshadow future P&L impacts (timing noise in earnings).  
✅ Stable operating subtotals with clear note reconciliation: consistent line-item definitions across periods and explicit bridge between statutory and adjusted measures.

### Sector-Specific Considerations

| Sector | Key IAS 1 / presentation issue | Typical valuation treatment |
|---|---|---|
| Technology / SaaS | “Other income” (R&D credits, FX, fair value gains) sometimes blended into operating results | Define operating profit policy; exclude fair value gains; treat FX consistently |
| Banking | Interest income/expense is operating, not financing | Use sector-specific operating framework (NII, provisions, fee income); avoid generic EBITDA |
| Real Estate / Investment Property | Fair value movements may sit in profit or loss | Separate recurring rental NOI from fair value gains; consider NAV-based approaches |
| Insurance / Asset managers | OCI can be material (AOCI / FV movements) | Analyse OCI components; ensure multiples are based on stable earnings measure |

### Real-World Example
Scenario: A company reports “operating profit” including $12m disposal gains. You want a run-rate EV/EBIT multiple.

```python
def adjusted_operating_profit(reported_operating_profit: float, disposal_gains: float) -> float:
    """
    Removes disposal gains from operating profit to estimate run-rate operating profit.

    Args:
        reported_operating_profit: operating profit as presented.
        disposal_gains: gains included within operating profit.

    Returns:
        float: adjusted operating profit

    Raises:
        ValueError: for invalid numeric inputs.
    """
    if not isinstance(reported_operating_profit, (int, float)):
        raise ValueError("reported_operating_profit must be numeric")
    if not isinstance(disposal_gains, (int, float)):
        raise ValueError("disposal_gains must be numeric")
    return float(reported_operating_profit) - float(disposal_gains)

company = {
    "reported_operating_profit": 180.0,  # $m
    "disposal_gains_included": 12.0,     # $m
    "enterprise_value": 2_700.0,         # $m
}

op_adj = adjusted_operating_profit(company["reported_operating_profit"], company["disposal_gains_included"])
ev_to_op = company["enterprise_value"] / op_adj if op_adj != 0 else float("inf")
print(f"Adjusted operating profit: ${op_adj:.1f}m")
print(f"EV / Adjusted operating profit: {ev_to_op:.1f}x")
```

Interpretation: If disposal gains are excluded, operating profit falls, and the multiple rises; peer comparisons should use consistent “run-rate” definitions.

See also: Chapter 7 (Accounting Policies, Changes in Accounting Estimates and Errors) for retrospective changes and errors; Chapter 20 (Revenue) and Chapter 22 (Leases) for items that frequently distort subtotals and trend comparability.
