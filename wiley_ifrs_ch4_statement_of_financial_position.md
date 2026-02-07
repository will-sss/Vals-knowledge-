# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 4: Statement of Financial Position (Working Capital, Net Debt, Invested Capital, Classification)

### Core Concept
The statement of financial position (balance sheet) is the anchor for valuation adjustments: it defines invested capital for ROIC, net debt for EV-to-equity bridges, and the operating vs financing split that drives clean free cash flow. IFRS presentation and classification choices (current/non-current, netting, provisions, lease liabilities, deferred taxes) can materially shift leverage, liquidity ratios, and comparability across peers.

### Formula/Methodology

#### 1) Net debt (for EV → equity bridge)
```text
Net Debt = Interest-bearing Debt + Lease Liabilities + Other Debt-like Items - Cash and Cash Equivalents
Where:
Debt-like items may include: overdrafts treated as financing, pension deficits (case-by-case), contingent consideration debt components, preference shares (if debt-like).
Cash may be adjusted for restricted cash depending on definition and valuation policy.
```

#### 2) Invested capital (for ROIC / economic profit)
```text
Invested Capital = Operating Assets - Non-interest-bearing Operating Liabilities
Where:
Operating Assets commonly include: PPE, intangibles (excluding goodwill sometimes shown separately), working capital items, right-of-use assets.
Non-interest-bearing operating liabilities commonly include: trade payables, accrued expenses, deferred revenue, provisions (operating).
Exclude financing items: interest-bearing debt, lease liabilities, derivatives treated as financing, income tax payable (policy-dependent), pension liabilities (often treated separately).
```

Alternative operational definition used in many valuation models:
```text
Invested Capital = Net Working Capital + Net PPE + Other Operating Assets - Other Operating Liabilities
Where:
Net Working Capital = (Trade Receivables + Inventory + Other Current Operating Assets) - (Trade Payables + Other Current Operating Liabilities)
```

#### 3) Liquidity and leverage ratios (screening and covenant context)
```text
Current Ratio = Current Assets / Current Liabilities
Quick Ratio = (Current Assets - Inventory) / Current Liabilities
Net Debt / EBITDA = Net Debt / EBITDA
```

#### 4) Classification: current vs non-current (analysis implications)
```text
Working Capital = Current Assets - Current Liabilities
Where:
Classification depends on the operating cycle and settlement expectations.
```

---

### Practical Application for Valuation Analysts

#### 1) Build an “operating” balance sheet for ROIC and FCF
IFRS balance sheets are designed for general-purpose reporting, not valuation. For ROIC/FCF:
- Separate **operating** items (working capital, PPE, operating provisions, deferred revenue) from **financing** items (debt, lease liabilities, derivatives financing, pension liabilities if treated as debt-like).
- Identify **non-operating assets** (excess cash, investments not part of operations, assets held for sale, associate investments depending on valuation method).
- Standardise across peers using a mapping and a clear policy note.

#### 2) Netting and offsetting: avoid hidden leverage
Some balances may be presented net (e.g., certain receivable/payable nets, derivatives with offsetting). For leverage comparability:
- Prefer gross exposure where disclosures allow (especially for derivatives and factoring).
- Identify **reverse factoring/supply chain finance** (may sit in trade payables but behave like debt).

#### 3) Provisions and contingencies: classify debt-like vs operating
Provisions (warranties, restructuring, litigation) may be operating or may behave like debt depending on recurrence and link to operations. For EV bridges:
- Keep operating provisions in working capital/invested capital.
- Treat long-term, financing-like obligations separately if they resemble debt (policy-driven, must be consistent).

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

def compute_net_debt(bs: Dict[str, Any], include_leases: bool = True, restricted_cash: float = 0.0) -> float:
    """
    Compute net debt from balance-sheet components.

    Args:
        bs: dict of balance sheet items (currency units consistent).
            Suggested keys:
                - short_term_debt
                - long_term_debt
                - lease_liabilities (optional)
                - cash_and_equivalents
                - other_debt_like (optional)
        include_leases: whether to include lease liabilities in net debt.
        restricted_cash: cash to exclude from cash_and_equivalents (treated as non-available).

    Returns:
        float: net debt

    Raises:
        ValueError: if required keys missing or invalid.
    """
    if not isinstance(bs, dict):
        raise ValueError("bs must be a dict of balance sheet items.")
    st = _num(bs.get("short_term_debt", 0.0), "short_term_debt")
    lt = _num(bs.get("long_term_debt", 0.0), "long_term_debt")
    cash = _num(bs.get("cash_and_equivalents", 0.0), "cash_and_equivalents")
    other = _num(bs.get("other_debt_like", 0.0), "other_debt_like")
    leases = _num(bs.get("lease_liabilities", 0.0), "lease_liabilities") if include_leases else 0.0

    rc = _num(restricted_cash, "restricted_cash")
    if rc < 0:
        raise ValueError("restricted_cash must be >= 0.")
    available_cash = cash - rc
    if available_cash < 0:
        raise ValueError("restricted_cash cannot exceed cash_and_equivalents.")

    return (st + lt + leases + other) - available_cash

def compute_invested_capital(bs: Dict[str, Any]) -> float:
    """
    Compute invested capital using Operating Assets - Non-interest-bearing Operating Liabilities.

    Args:
        bs: dict with operating assets and operating liabilities keys.
            Suggested keys (customize to your policy):
                Operating assets:
                    - trade_receivables
                    - inventory
                    - other_current_operating_assets
                    - ppe
                    - right_of_use_assets
                    - intangible_assets
                    - other_non_current_operating_assets
                Operating liabilities (non-interest-bearing):
                    - trade_payables
                    - accrued_expenses
                    - deferred_revenue
                    - provisions_operating
                    - other_current_operating_liabilities
                    - other_non_current_operating_liabilities

    Returns:
        float: invested capital

    Raises:
        ValueError: malformed input.
    """
    if not isinstance(bs, dict):
        raise ValueError("bs must be a dict.")

    op_assets_keys = [
        "trade_receivables", "inventory", "other_current_operating_assets",
        "ppe", "right_of_use_assets", "intangible_assets", "other_non_current_operating_assets"
    ]
    op_liab_keys = [
        "trade_payables", "accrued_expenses", "deferred_revenue",
        "provisions_operating", "other_current_operating_liabilities", "other_non_current_operating_liabilities"
    ]

    op_assets = sum(_num(bs.get(k, 0.0), k) for k in op_assets_keys)
    op_liabs = sum(_num(bs.get(k, 0.0), k) for k in op_liab_keys)
    return op_assets - op_liabs

def working_capital_metrics(bs: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute basic working-capital measures.

    Raises:
        ValueError: invalid inputs.
    """
    ca = _num(bs.get("current_assets", 0.0), "current_assets")
    cl = _num(bs.get("current_liabilities", 0.0), "current_liabilities")
    inv = _num(bs.get("inventory", 0.0), "inventory")
    wc = ca - cl

    current_ratio = ca / cl if abs(cl) > 1e-12 else float("inf")
    quick_ratio = (ca - inv) / cl if abs(cl) > 1e-12 else float("inf")

    return {"working_capital": wc, "current_ratio": current_ratio, "quick_ratio": quick_ratio}

# Example usage
bs = {
    "short_term_debt": 120_000_000,
    "long_term_debt": 980_000_000,
    "lease_liabilities": 250_000_000,
    "other_debt_like": 50_000_000,
    "cash_and_equivalents": 180_000_000,
    "trade_receivables": 220_000_000,
    "inventory": 160_000_000,
    "other_current_operating_assets": 40_000_000,
    "ppe": 900_000_000,
    "right_of_use_assets": 240_000_000,
    "intangible_assets": 300_000_000,
    "other_non_current_operating_assets": 60_000_000,
    "trade_payables": 210_000_000,
    "accrued_expenses": 70_000_000,
    "deferred_revenue": 90_000_000,
    "provisions_operating": 45_000_000,
    "other_current_operating_liabilities": 35_000_000,
    "other_non_current_operating_liabilities": 25_000_000,
    "current_assets": 600_000_000,
    "current_liabilities": 450_000_000,
}

nd = compute_net_debt(bs, include_leases=True, restricted_cash=20_000_000)
ic = compute_invested_capital(bs)
wc = working_capital_metrics(bs)

print(f"Net debt: ${nd/1e6:.0f}M")
print(f"Invested capital: ${ic/1e6:.0f}M")
print(wc)
```

---

### Valuation Impact
Why this matters:
- **EV bridge accuracy:** net debt definitions drive equity value; misclassifying lease liabilities, restricted cash, or debt-like provisions can materially shift per-share value.
- **ROIC and economic profit:** invested capital depends on consistent operating/financing splits; differences across peers can make ROIC comparisons meaningless unless standardised.
- **Cash flow modelling:** working capital is derived from balance sheet movements; classification changes can distort ΔNWC and FCF.

Impact on multiples:
- Net debt affects EV and therefore EV/EBITDA and EV/EBIT multiples.
- Balance sheet leverage changes can also influence peer selection and risk adjustments.

Impact on DCF inputs:
- Debt-like items affect capital structure assumptions and WACC.
- Operating asset/liability definitions drive reinvestment rates and ROIC-based growth logic.

Practical adjustments:
```python
def adjust_for_restricted_cash(cash: float, restricted_cash: float) -> float:
    """
    Returns available cash used in net debt.

    Args:
        cash: cash and cash equivalents
        restricted_cash: portion not available for debt service / distribution

    Returns:
        float: available cash
    """
    c = float(cash)
    r = float(restricted_cash)
    if not np.isfinite(c) or not np.isfinite(r):
        raise ValueError("cash and restricted_cash must be finite.")
    if r < 0 or r > c:
        raise ValueError("restricted_cash must be between 0 and cash.")
    return c - r
```

---

### Quality of Earnings Flags
⚠️ **Debt-like items hidden in “other liabilities”:** large “other” lines without clear note support can conceal financing obligations.  
⚠️ **Reverse factoring / supply chain finance:** trade payables increasing while operating cash flow improves may indicate reclassification of financing as operating.  
⚠️ **Lease liabilities omitted from leverage narrative:** post-IFRS 16, leverage comparability requires explicit lease treatment.  
✅ **Transparent net debt reconciliation:** clear bridge from gross debt and cash to net debt, with restricted cash and debt-like adjustments disclosed.

---

### Sector-Specific Considerations

| Sector | Key balance sheet issue | Typical valuation treatment |
|---|---|---|
| Technology / SaaS | Deferred revenue is often large | Treat as operating liability in invested capital; analyse cash conversion vs revenue recognition |
| Banking | Balance sheet is “operating” by nature | Use sector valuation methods; avoid generic net debt / invested capital frameworks |
| Real Estate | Investment property dominates assets | Consider NAV approaches; separate investment property FV changes and debt structure |
| Retail / Consumer | Inventory and leases are material | Tight working-capital analysis; include lease liabilities consistently in leverage and EV |

---

### Real-World Example
Scenario: You need to value equity from enterprise value. The company reports $180m cash, but $20m is restricted; leases are $250m.

```python
ev = 4_800_000_000
net_debt = compute_net_debt(bs, include_leases=True, restricted_cash=20_000_000)
equity_value = ev - net_debt
print(f"Equity value: ${equity_value/1e9:.2f}B")
```

Interpretation: Restricted cash increases net debt versus a naive “cash” number; including leases can materially increase EV leverage and reduce equity value.

See also: Chapter 6 (cash flows) for operating vs financing cash classification; Chapter 22 (leases) for lease liability mechanics; Chapter 26 (income taxes) for deferred tax presentation and valuation treatment.
