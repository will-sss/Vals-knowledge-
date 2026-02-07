# McKinsey Valuation - Key Concepts for Valuation

## Chapter 11: Reorganizing the Financial Statements (Operating vs Financing, Invested Capital, NOPAT, Net Debt)

### Core Concept
Valuation requires separating the business (operations) from how it is financed. Reorganising financial statements means (1) reclassifying the income statement into operating vs non-operating/financing items and (2) reclassifying the balance sheet into operating assets/liabilities vs financing items. This produces consistent valuation building blocks: NOPAT, invested capital, ROIC, free cash flow, and the enterprise-to-equity bridge.

See also: Chapter 10 on valuation frameworks (DCF/EP/APV consistency); Chapter 21 on non-operating items, provisions, and reserves; Chapter 16 on moving from enterprise value to value per share.

---

### Core Concept: Operating vs Financing (Classification Rules)
The goal is to construct an “operating P&L” and an “operating balance sheet” that reflect the economics of the business independent of capital structure.

**Operating items (typical):**
- Revenue, COGS, operating costs
- Depreciation and amortisation related to operations
- Operating working capital changes (inventory, trade receivables, trade payables)
- Operating provisions and accruals (if related to operations)

**Financing / non-operating items (typical):**
- Interest income/expense and financing fees
- Debt and debt-like liabilities
- Excess cash, marketable securities (non-operating)
- Pension deficits (often treated as debt-like), leases (debt-like under IFRS 16), derivatives (depending on purpose)
- Gains/losses on non-operating assets and fair value remeasurements not tied to operations

Practical rule: if an item is primarily a funding decision rather than a production/sales decision, classify it as financing.

---

### Formula/Methodology

#### 1) NOPAT (Net Operating Profit After Tax)
```text
NOPAT = Operating EBIT × (1 − Operating Tax Rate)
```
Where:
- Operating EBIT excludes financing/non-operating income/expenses and non-recurring items that are not expected to continue.
- Operating tax rate should reflect a sustainable effective operating tax rate (avoid one-offs).

#### 2) Invested Capital (Operating Invested Capital)
Use a balance-sheet definition that mirrors how the business is funded by equity + interest-bearing claims, after netting non-interest-bearing operating liabilities.

```text
Invested Capital = Operating Assets − Operating Liabilities (non-interest-bearing)
```
Common operational form:
```text
Invested Capital = Net Working Capital + Net PP&E + Other Operating Assets − Other Operating Liabilities
```
Where:
- Net Working Capital (NWC) = Operating current assets − Operating current liabilities (exclude cash, short-term debt, and other financing items)
- Net PP&E includes right-of-use assets if leases treated as operating capital (ensure lease liability handled consistently)

#### 3) ROIC (built from reorganised statements)
```text
ROIC = NOPAT / Average Invested Capital
```

#### 4) Enterprise-to-equity bridge (net debt + debt-like items)
```text
Equity Value = Enterprise Value − Net Debt − Other Debt-like Claims + Non-operating Assets
Net Debt = Interest-bearing Debt − (Cash and Cash Equivalents deemed excess)
```

---

### Python Implementation
```python
from typing import Dict, Optional

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def nopat(operating_ebit: float, operating_tax_rate: float) -> float:
    """
    Compute NOPAT from operating EBIT and operating tax rate.
    """
    if not _num(operating_ebit):
        raise ValueError("operating_ebit must be numeric")
    if not _num(operating_tax_rate) or not (0.0 <= operating_tax_rate < 1.0):
        raise ValueError("operating_tax_rate must be in [0, 1)")
    return operating_ebit * (1.0 - operating_tax_rate)


def invested_capital(operating_assets: float, operating_liabilities: float) -> float:
    """
    Invested capital = operating assets - non-interest-bearing operating liabilities.
    """
    if not _num(operating_assets) or operating_assets < 0:
        raise ValueError("operating_assets must be numeric and >= 0")
    if not _num(operating_liabilities) or operating_liabilities < 0:
        raise ValueError("operating_liabilities must be numeric and >= 0")
    return operating_assets - operating_liabilities


def roic(nopat_value: float, avg_invested_capital: float) -> Optional[float]:
    """
    ROIC = NOPAT / Average Invested Capital.
    Returns None if avg_invested_capital <= 0.
    """
    if not _num(nopat_value):
        raise ValueError("nopat_value must be numeric")
    if not _num(avg_invested_capital):
        raise ValueError("avg_invested_capital must be numeric")
    if avg_invested_capital <= 0:
        return None
    return nopat_value / avg_invested_capital


def net_debt(interest_bearing_debt: float, cash_and_equivalents: float, excess_cash: Optional[float] = None) -> float:
    """
    Net Debt = Interest-bearing Debt - Excess Cash.
    """
    if not _num(interest_bearing_debt) or interest_bearing_debt < 0:
        raise ValueError("interest_bearing_debt must be numeric and >= 0")
    if not _num(cash_and_equivalents) or cash_and_equivalents < 0:
        raise ValueError("cash_and_equivalents must be numeric and >= 0")
    cash_to_net = cash_and_equivalents if excess_cash is None else excess_cash
    if not _num(cash_to_net) or cash_to_net < 0:
        raise ValueError("excess_cash must be numeric and >= 0")
    if cash_to_net > cash_and_equivalents:
        raise ValueError("excess_cash cannot exceed cash_and_equivalents")
    return interest_bearing_debt - cash_to_net


def equity_value_from_ev(ev: float, nd: float, other_debt_like: float = 0.0, non_operating_assets: float = 0.0) -> float:
    """
    Equity = EV - Net Debt - Other debt-like + Non-operating assets.
    """
    for name, v in {"ev": ev, "nd": nd, "other_debt_like": other_debt_like, "non_operating_assets": non_operating_assets}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return ev - nd - other_debt_like + non_operating_assets


# Example usage
example = {
    "operating_ebit": 420_000_000,
    "operating_tax_rate": 0.24,
    "operating_assets": 3_800_000_000,
    "operating_liabilities": 1_250_000_000,
    "avg_invested_capital": 2_450_000_000,
    "debt": 1_600_000_000,
    "cash": 500_000_000,
    "excess_cash": 300_000_000,
    "ev": 7_900_000_000,
    "other_debt_like": 220_000_000,
    "non_operating_assets": 150_000_000
}

n = nopat(example["operating_ebit"], example["operating_tax_rate"])
ic = invested_capital(example["operating_assets"], example["operating_liabilities"])
r = roic(n, example["avg_invested_capital"])
nd = net_debt(example["debt"], example["cash"], example["excess_cash"])
eq = equity_value_from_ev(example["ev"], nd, example["other_debt_like"], example["non_operating_assets"])

print(f"NOPAT: ${n/1e6:.1f}M")
print(f"Invested Capital: ${ic/1e9:.2f}B")
print(f"ROIC: {r:.1%}" if r is not None else "ROIC: n/a")
print(f"Net Debt: ${nd/1e9:.2f}B")
print(f"Equity Value: ${eq/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- ROIC, reinvestment, and free cash flow are meaningless unless operating and financing items are separated consistently.
- Enterprise value reflects the value of operations; equity value requires subtracting net debt and other debt-like claims and adding back non-operating assets.
- Misclassification (e.g., treating operating lease liabilities as operating in one place and financing in another) is a top driver of valuation errors and peer incomparability.

Impact on multiples:
- EV/EBITDA and EV/EBIT must be computed with a consistent enterprise definition (include debt-like items such as leases, pension deficits if treated as debt-like).
- “Adjusted EBITDA” often excludes costs that are economically required; treat add-backs skeptically.

Impact on DCF:
- FCFF requires operating earnings and operating reinvestment; if cash includes operating cash buffers, net only excess cash in the equity bridge.
- Terminal value should be based on steady-state operating earnings and invested capital that reflect ongoing operations, not non-operating balances.

Practical adjustments (normalising statements for peer comparability):
```python
def classify_balance_sheet(items: Dict[str, float], operating_keys: set, financing_keys: set) -> Dict[str, float]:
    """
    Sum items into operating vs financing buckets based on provided key sets.
    """
    if operating_keys & financing_keys:
        raise ValueError("Keys cannot be both operating and financing")
    op = fin = unc = 0.0
    for k, v in items.items():
        if not _num(v):
            raise ValueError(f"Value for {k} must be numeric")
        if k in operating_keys:
            op += v
        elif k in financing_keys:
            fin += v
        else:
            unc += v
    return {"operating": op, "financing": fin, "unclassified": unc}
```

---

### Quality of Earnings Flags (QoE / Reorg Red Flags)
⚠️ Operating EBIT includes one-offs (restructuring, litigation, asset sales) without normalisation.  
⚠️ Interest expense buried in operating costs.  
⚠️ Lease accounting inconsistently treated across companies (ROU assets/liabilities missing in invested capital or EV).  
⚠️ Pension deficits, provisions, or customer advances (deferred revenue) misclassified, distorting invested capital and ROIC.  
⚠️ Cash fully netted against debt without assessing operating cash needs (inflates equity value).  
✅ Transparent reconciliation from reported financials to reorganised operating/financing statements.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Technology / Software | Deferred revenue and capitalised dev costs affect invested capital | Treat deferred revenue as operating liability; apply consistent dev cost treatment across peers |
| Retail | Operating leases are material and affect EV, invested capital, and EBITDA | Treat lease liabilities as debt-like in EV; ensure EBITDA and ROIC are lease-consistent |
| Industrials | Pension deficits and environmental provisions can be large | Often treated as debt-like for EV-to-equity bridge; disclose assumptions |
| Banking | Operating vs financing separation differs | Use equity-based metrics (FCFE, ROE) and regulatory capital frameworks |
| Utilities | Regulated assets and provisions | Treat regulatory assets/liabilities consistently; model decommissioning provisions explicitly |

---

### Real-World Example (Statement Reorg → ROIC → Equity Bridge)

Scenario: Reorganise to compute NOPAT, invested capital, ROIC, and bridge EV to equity.

```python
reported_bs = {
    "trade_receivables": 420_000_000,
    "inventory": 310_000_000,
    "pp&e": 2_100_000_000,
    "intangible_assets": 600_000_000,
    "trade_payables": 380_000_000,
    "accruals": 190_000_000,
    "cash": 450_000_000,
    "debt": 1_300_000_000,
    "lease_liability": 420_000_000,
}

operating_assets = reported_bs["trade_receivables"] + reported_bs["inventory"] + reported_bs["pp&e"] + reported_bs["intangible_assets"]
operating_liabs = reported_bs["trade_payables"] + reported_bs["accruals"]

ic = invested_capital(operating_assets, operating_liabs)

n = nopat(operating_ebit=280_000_000, operating_tax_rate=0.23)
r = roic(n, avg_invested_capital=ic)

ev = 4_900_000_000
nd = net_debt(
    interest_bearing_debt=reported_bs["debt"] + reported_bs["lease_liability"],
    cash_and_equivalents=reported_bs["cash"],
    excess_cash=250_000_000
)
eq = equity_value_from_ev(ev, nd, other_debt_like=0.0, non_operating_assets=0.0)

print(f"Invested Capital: ${ic/1e9:.2f}B")
print(f"ROIC: {r:.1%}" if r is not None else "ROIC: n/a")
print(f"Equity Value: ${eq/1e9:.2f}B")
```

Interpretation:
- ROIC above WACC supports value-creating growth; ROIC below WACC implies growth destroys value unless the business model changes.
- Equity value is sensitive to debt-like definitions and excess cash assumptions; disclose and reconcile to peers.
