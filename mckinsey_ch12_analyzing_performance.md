# McKinsey Valuation - Key Concepts for Valuation

## Chapter 12: Analyzing Performance (ROIC Drivers, Economic Profit, Competitive Diagnostics)

### Core Concept
Performance analysis for valuation is about explaining value creation: how operating profits, capital efficiency, and growth interact to generate returns on invested capital (ROIC) and economic profit (ROIC − WACC spread). The practical goal is to identify which drivers are sustainable, which are cyclical or one-off, and how those drivers should feed into forecast assumptions and peer comparability.

See also: Chapter 11 on reorganising financial statements (NOPAT/invested capital definitions); Chapter 14 on continuing value (steady-state ROIC and growth); Chapter 18 on multiples (implied performance assumptions).

---

### Core Concept: Start with Value Creation Metrics (ROIC and Economic Profit)
ROIC is the core “quality of growth” metric. Two companies can grow revenue at the same rate, but the one that grows while maintaining ROIC above WACC creates more value. Economic profit translates that spread into currency, enabling direct linkage to enterprise value.

### Formula/Methodology
```text
ROIC = NOPAT / Average Invested Capital

Economic Profit (EP) = Invested Capital × (ROIC − WACC)
```

Where:
- NOPAT: Net operating profit after tax (operating profit × (1 − operating tax rate))
- Invested Capital: Operating assets net of non-interest-bearing operating liabilities
- WACC: Weighted average cost of capital (decimal)

### Python Implementation
```python
from typing import Optional, Dict, List

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def roic(nopat: float, avg_invested_capital: float) -> Optional[float]:
    """
    ROIC = NOPAT / Average Invested Capital

    Returns None if avg_invested_capital <= 0 (not meaningful).
    """
    if not _num(nopat):
        raise ValueError("nopat must be numeric")
    if not _num(avg_invested_capital):
        raise ValueError("avg_invested_capital must be numeric")
    if avg_invested_capital <= 0:
        return None
    return nopat / avg_invested_capital


def economic_profit(invested_capital: float, roic_value: float, wacc: float) -> float:
    """
    Economic Profit = Invested Capital × (ROIC − WACC)
    """
    if not _num(invested_capital) or invested_capital < 0:
        raise ValueError("invested_capital must be numeric and >= 0")
    if not _num(roic_value) or not _num(wacc):
        raise ValueError("roic_value and wacc must be numeric")
    return invested_capital * (roic_value - wacc)


# Example usage
nopat_ex = 320_000_000
avg_ic_ex = 2_400_000_000
wacc_ex = 0.09

r = roic(nopat_ex, avg_ic_ex)
ep = economic_profit(avg_ic_ex, r if r is not None else 0.0, wacc_ex)
print(f"ROIC: {r:.1%}" if r is not None else "ROIC: n/a")
print(f"Economic Profit: ${ep/1e6:,.1f}M")
```

### Valuation Impact
Why this matters:
- ROIC determines whether growth is value-creating (ROIC > WACC) or value-destroying (ROIC < WACC).
- EP supports a clean narrative for why the business deserves a premium/discount multiple and how value will evolve as ROIC and growth fade.
- In DCF, ROIC and reinvestment are mechanically linked: high growth requires investment; value depends on the ROIC earned on that investment.

Practical adjustment:
```python
def normalize_nopat(reported_operating_profit: float, tax_rate: float, one_offs: float = 0.0) -> float:
    """
    Normalize NOPAT by removing one-offs from operating profit before tax.

    Args:
        reported_operating_profit: operating profit/EBIT (currency)
        tax_rate: sustainable operating tax rate (decimal in [0,1))
        one_offs: net one-offs included in operating profit (currency). Positive means inflate profit.

    Returns:
        float: normalized NOPAT (currency)
    """
    if not _num(reported_operating_profit) or not _num(one_offs):
        raise ValueError("reported_operating_profit and one_offs must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    normalized_ebit = reported_operating_profit - one_offs
    return normalized_ebit * (1.0 - tax_rate)
```

### Quality of Earnings Flags
⚠️ ROIC uplift driven by temporary margin spikes (pricing power in a shortage, temporary mix) without evidence of persistence.  
⚠️ ROIC inflated by aggressive add-backs or underinvestment (capex below depreciation for long periods).  
⚠️ Invested capital understated (leases, pensions, provisions misclassified), overstating ROIC.  
✅ ROIC improvement aligns with durable operational changes (process, product, channel) and stable reinvestment patterns.

---

### Core Concept: Decompose ROIC to Diagnose Drivers (Operating Margin × Capital Turnover)
This decomposition turns ROIC into levers a business can influence: pricing/costs (margin) and capital efficiency (turnover). It’s useful for comparing peers where headline ROIC hides different business models.

### Formula/Methodology
```text
ROIC = (NOPAT / Revenue) × (Revenue / Invested Capital)

Where:
NOPAT Margin = NOPAT / Revenue
Invested Capital Turnover = Revenue / Invested Capital

So:
ROIC = NOPAT Margin × Capital Turnover
```

### Python Implementation
```python
def nopat_margin(nopat: float, revenue: float) -> Optional[float]:
    """
    NOPAT margin = NOPAT / Revenue.
    Returns None if revenue <= 0.
    """
    if not _num(nopat) or not _num(revenue):
        raise ValueError("nopat and revenue must be numeric")
    if revenue <= 0:
        return None
    return nopat / revenue


def capital_turnover(revenue: float, invested_capital: float) -> Optional[float]:
    """
    Capital turnover = Revenue / Invested Capital.
    Returns None if invested_capital <= 0.
    """
    if not _num(revenue) or not _num(invested_capital):
        raise ValueError("revenue and invested_capital must be numeric")
    if invested_capital <= 0:
        return None
    return revenue / invested_capital


def roic_from_components(nopat: float, revenue: float, invested_capital: float) -> Optional[float]:
    """
    ROIC = (NOPAT/Revenue) × (Revenue/Invested Capital)
    """
    m = nopat_margin(nopat, revenue)
    t = capital_turnover(revenue, invested_capital)
    if m is None or t is None:
        return None
    return m * t


# Example
rev = 2_800_000_000
nopat_ex = 320_000_000
ic = 2_400_000_000

m = nopat_margin(nopat_ex, rev)
t = capital_turnover(rev, ic)
r2 = roic_from_components(nopat_ex, rev, ic)

print(f"NOPAT margin: {m:.1%}" if m is not None else "NOPAT margin: n/a")
print(f"Capital turnover: {t:.2f}x" if t is not None else "Capital turnover: n/a")
print(f"ROIC (components): {r2:.1%}" if r2 is not None else "ROIC: n/a")
```

### Valuation Impact
Why this matters:
- Multiples often price different models differently (high margin/low turnover vs low margin/high turnover). ROIC decomposition explains why.
- When forecasting, separate what is likely to revert (margins) from what may structurally improve (turnover via working capital and fixed-asset efficiency).

Practical adjustments:
```python
def normalize_revenue_for_turnover(revenue: float, one_off_revenue: float = 0.0) -> float:
    """
    Normalize revenue when computing turnover (remove one-off revenue spikes).
    """
    if not _num(revenue) or not _num(one_off_revenue):
        raise ValueError("revenue and one_off_revenue must be numeric")
    norm = revenue - one_off_revenue
    if norm < 0:
        raise ValueError("normalized revenue cannot be negative")
    return norm
```

### Quality of Earnings Flags
⚠️ Margin improvement comes from cutting necessary costs (maintenance, compliance, R&D) that will reappear.  
⚠️ Turnover improvement comes from working capital squeeze that is not repeatable (supplier stretching).  
✅ Turnover improvement supported by structural shifts (inventory systems, digitisation, asset-lighting).

---

### Core Concept: Break Down Capital Efficiency (Working Capital vs Fixed Assets vs Intangibles)
Capital turnover is a composite. Diagnose invested capital by major components: net working capital, net PP&E, and other operating assets (including capitalised intangibles). This reveals where cash is tied up and what can realistically change.

### Formula/Methodology
```text
Invested Capital = Net Working Capital + Net PP&E + Other Operating Assets − Other Operating Liabilities

Net Working Capital (NWC) = (Trade Receivables + Inventory + Other Operating Current Assets) −
                           (Trade Payables + Accruals + Other Operating Current Liabilities)
```

### Python Implementation
```python
def net_working_capital(operating_current_assets: float, operating_current_liabilities: float) -> float:
    """
    NWC = operating current assets - operating current liabilities (non-interest-bearing).
    """
    if not _num(operating_current_assets) or operating_current_assets < 0:
        raise ValueError("operating_current_assets must be numeric and >= 0")
    if not _num(operating_current_liabilities) or operating_current_liabilities < 0:
        raise ValueError("operating_current_liabilities must be numeric and >= 0")
    return operating_current_assets - operating_current_liabilities


def invested_capital_from_parts(nwc: float, net_ppe: float, other_operating_assets_net: float) -> float:
    """
    Simplified invested capital from parts.
    """
    for name, v in {"nwc": nwc, "net_ppe": net_ppe, "other_operating_assets_net": other_operating_assets_net}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if net_ppe < 0:
        raise ValueError("net_ppe must be >= 0")
    return nwc + net_ppe + other_operating_assets_net


# Example
nwc = net_working_capital(operating_current_assets=980_000_000, operating_current_liabilities=740_000_000)
ic = invested_capital_from_parts(nwc=nwc, net_ppe=1_150_000_000, other_operating_assets_net=310_000_000)
print(f"NWC: ${nwc/1e6:,.0f}M")
print(f"Invested capital: ${ic/1e9:.2f}B")
```

### Valuation Impact
Why this matters:
- Reinvestment needs come from how much working capital and fixed/other assets are required to support growth.
- If management claims higher growth without more capital, validate using turnover and component analysis.
- Shifts in capital intensity directly change free cash flow and terminal value assumptions.

### Quality of Earnings Flags
⚠️ Sustainable margins but cash flow weak because NWC structurally rising (customer terms, inventory build).  
⚠️ Asset-light narrative while capitalised costs/intangibles rising (hidden invested capital).  
✅ NWC discipline and stable capex-to-sales ratio consistent with business model.

---

### Practical Benchmarks and Comparison Tables (Use in Peer Review)

#### Performance Driver Comparison Table (Template)
| Driver | Company | Peer Median | What to investigate if outlier |
|---|---:|---:|---|
| ROIC |  |  | Definition consistency, one-offs, capitalised items |
| NOPAT Margin |  |  | Price/mix, cost structure, non-recurring items |
| Capital Turnover |  |  | Working capital terms, asset base, lease/intangible treatment |
| Economic Profit |  |  | Scale effects, spread sustainability, WACC mismatch |
| NWC / Revenue |  |  | Customer concentration, inventory management |
| Capex / Revenue |  |  | Underinvestment risk, growth capex vs maintenance |

#### Diagnostics Checklist (Forecasting Link)
| Observation | Likely cause | Forecast response |
|---|---|---|
| ROIC high due to margin spike | Cyclical pricing / temporary scarcity | Fade margins toward mid-cycle |
| ROIC high due to turnover improvement | Structural working capital / asset efficiency | Allow persistence if supported |
| ROIC declining with growth | Expansion into lower-return segments | Model declining incremental ROIC and/or higher reinvestment |
| EP negative but earnings rising | Capital intensity rising / returns below WACC | Reassess growth plan; tighten terminal assumptions |

---

### Real-World Example (Peer-Comparable ROIC and EP Diagnostics)

Scenario: Two companies show similar revenue growth but different value creation. Compute ROIC drivers and economic profit.

```python
company_a = {"revenue": 3_000_000_000, "nopat": 360_000_000, "ic": 2_000_000_000, "wacc": 0.09}
company_b = {"revenue": 3_000_000_000, "nopat": 240_000_000, "ic": 1_200_000_000, "wacc": 0.09}

def performance_summary(c: Dict[str, float]) -> Dict[str, float]:
    r = roic(c["nopat"], c["ic"])
    m = nopat_margin(c["nopat"], c["revenue"])
    t = capital_turnover(c["revenue"], c["ic"])
    ep = economic_profit(c["ic"], r if r is not None else 0.0, c["wacc"])
    return {
        "roic": r if r is not None else float("nan"),
        "nopat_margin": m if m is not None else float("nan"),
        "turnover": t if t is not None else float("nan"),
        "economic_profit": ep,
    }

a = performance_summary(company_a)
b = performance_summary(company_b)

print("A:", {k: (f"{v:.1%}" if "roic" in k or "margin" in k else (f"{v:.2f}x" if k=="turnover" else f"${v/1e6:,.0f}M")) for k, v in a.items()})
print("B:", {k: (f"{v:.1%}" if "roic" in k or "margin" in k else (f"{v:.2f}x" if k=="turnover" else f"${v/1e6:,.0f}M")) for k, v in b.items()})
```

Interpretation:
- Company A creates more economic profit because ROIC is higher at similar WACC; investigate whether its margin advantage is durable or cyclical.
- Company B may have higher turnover (asset-light) but lower margin; if its margin can improve, value can rise without major capital increases.
- Use these diagnostics to set forecast fades: fade cyclical margins, sustain structural turnover improvements only with evidence.
