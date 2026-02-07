# McKinsey Valuation - Key Concepts for Valuation

## Chapter 18: Using Multiples (Comparable Selection, Multiple Mechanics, Implied Expectations, Normalisation)

### Core Concept
Multiples are shorthand valuations that embed market expectations about growth, profitability, risk, and reinvestment. Using multiples well is not “pick a peer median and apply it” — it requires (1) defining the right metric and enterprise/equity bridge consistently, (2) selecting comparables based on value drivers (not labels), and (3) normalising earnings/cash flows to make like-for-like comparisons.

See also: Chapter 11 on reorganising statements (consistent EV and operating metrics); Chapter 14 on continuing value (DCF implied exit multiple); Chapter 17 on analysing results (reconciliation and implied expectations).

---

### Core Concept: Multiples are a Compact DCF
A multiple is the market’s compressed DCF. A high EV/EBITDA typically reflects some combination of:
- Higher expected growth
- Higher sustainable margins
- Higher ROIC / lower reinvestment needs
- Lower risk (lower discount rate)
- Stronger competitive position

Therefore, the practical job is to explain differences in multiples using these drivers and to adjust for accounting and capital structure differences.

---

### Formula/Methodology

#### 1) Enterprise Value (EV) vs Equity Value (Market Cap)
```text
Enterprise Value (EV) = Equity Value + Net Debt + Other Debt-like Items − Non-operating Assets ± Other Adjustments
```

Common EV multiples:
```text
EV/EBITDA = Enterprise Value / EBITDA
EV/EBIT   = Enterprise Value / EBIT
EV/NOPAT  = Enterprise Value / NOPAT
EV/Revenue = Enterprise Value / Revenue
```

Common equity multiples:
```text
P/E = Equity Value / Net Income
P/B = Equity Value / Book Value of Equity
```

#### 2) Multiple application (basic)
```text
Implied EV = Multiple × Metric
Implied Equity Value = Implied EV − Net Debt − Other Debt-like + Non-operating Assets − NCI − Pref
Value per share = Implied Equity Value / Fully Diluted Shares
```

#### 3) DCF ↔ Multiple cross-check (implied exit multiple)
```text
Implied EV/EBITDA at terminal = Terminal Value (TV_N) / EBITDA_N
```

Use as a reasonableness check: if implied exit multiple is outside mature peer range, revisit terminal assumptions, metric normalisation, or peer selection.

---

### Python Implementation
```python
from typing import Optional, Dict, List, Tuple

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def enterprise_value(equity_value: float,
                     debt_like: float,
                     cash_excess: float = 0.0,
                     other_adjustments: float = 0.0) -> float:
    """
    EV = Equity + Debt-like - Excess cash + other adjustments.

    Args:
        equity_value: market cap (currency, >= 0)
        debt_like: net debt + other debt-like items (currency, can be >= 0)
        cash_excess: excess cash to subtract (currency, >= 0)
        other_adjustments: signed adjustments (currency)

    Returns:
        float: EV (currency)
    """
    if not _num(equity_value) or equity_value < 0:
        raise ValueError("equity_value must be numeric and >= 0")
    if not _num(debt_like):
        raise ValueError("debt_like must be numeric")
    if not _num(cash_excess) or cash_excess < 0:
        raise ValueError("cash_excess must be numeric and >= 0")
    if not _num(other_adjustments):
        raise ValueError("other_adjustments must be numeric")
    return equity_value + debt_like - cash_excess + other_adjustments


def multiple(value: float, metric: float) -> Optional[float]:
    """Compute value/metric multiple. Returns None if metric <= 0."""
    if not _num(value) or not _num(metric):
        raise ValueError("value and metric must be numeric")
    if metric <= 0:
        return None
    return value / metric


def implied_value_from_multiple(metric: float, mult: float) -> float:
    """Implied value = metric * multiple."""
    if not _num(metric) or not _num(mult):
        raise ValueError("metric and mult must be numeric")
    if mult < 0:
        raise ValueError("mult must be >= 0")
    return metric * mult


def equity_from_ev(ev: float,
                   net_debt: float,
                   other_debt_like: float = 0.0,
                   non_operating_assets: float = 0.0,
                   non_controlling_interests: float = 0.0,
                   preferred_equity: float = 0.0) -> float:
    """Equity = EV - net debt - other debt-like - NCI - pref + non-op assets."""
    for name, v in {
        "ev": ev, "net_debt": net_debt, "other_debt_like": other_debt_like,
        "non_operating_assets": non_operating_assets, "non_controlling_interests": non_controlling_interests,
        "preferred_equity": preferred_equity
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    for name, v in {
        "other_debt_like": other_debt_like, "non_operating_assets": non_operating_assets,
        "non_controlling_interests": non_controlling_interests, "preferred_equity": preferred_equity
    }.items():
        if v < 0:
            raise ValueError(f"{name} must be >= 0")
    return ev - net_debt - other_debt_like - non_controlling_interests - preferred_equity + non_operating_assets


def value_per_share(equity_value: float, diluted_shares: int) -> float:
    if not _num(equity_value):
        raise ValueError("equity_value must be numeric")
    if not isinstance(diluted_shares, int) or diluted_shares <= 0:
        raise ValueError("diluted_shares must be int > 0")
    return equity_value / diluted_shares


def normalize_ebitda(reported_ebitda: float, one_offs: float = 0.0, ifrs16_rent_addback: float = 0.0) -> float:
    """
    Normalise EBITDA by removing one-offs and aligning lease treatment.

    Args:
        reported_ebitda: reported EBITDA (currency)
        one_offs: net non-recurring EBITDA included (currency). Positive means EBITDA inflated.
        ifrs16_rent_addback: if comparing to pre-IFRS16 peers, adjust for rent/lease expense differences (currency).

    Returns:
        float: normalized EBITDA
    """
    if not _num(reported_ebitda) or not _num(one_offs) or not _num(ifrs16_rent_addback):
        raise ValueError("inputs must be numeric")
    return reported_ebitda - one_offs + ifrs16_rent_addback


# Example usage: compute EV/EBITDA and implied per-share value from a chosen multiple
equity_mkt = 6_200_000_000
net_debt = 1_450_000_000
leases = 520_000_000  # treat as debt-like for EV and bridge
excess_cash = 300_000_000

ev = enterprise_value(equity_mkt, debt_like=net_debt + leases, cash_excess=excess_cash)

reported_ebitda = 860_000_000
ebitda_norm = normalize_ebitda(reported_ebitda, one_offs=40_000_000)

ev_ebitda = multiple(ev, ebitda_norm)
print(f"EV: ${ev/1e9:.2f}B")
print(f"Normalized EBITDA: ${ebitda_norm/1e6:.0f}M")
print(f"EV/EBITDA: {ev_ebitda:.1f}x" if ev_ebitda else "EV/EBITDA: n/a")

# Apply peer multiple to derive implied equity value and per-share value
peer_multiple = 10.0
implied_ev = implied_value_from_multiple(ebitda_norm, peer_multiple)
implied_equity = equity_from_ev(implied_ev, net_debt=net_debt, other_debt_like=leases, non_operating_assets=0.0)
vps = value_per_share(implied_equity, diluted_shares=540_000_000)
print(f"Implied EV: ${implied_ev/1e9:.2f}B")
print(f"Implied equity: ${implied_equity/1e9:.2f}B")
print(f"Implied value per share: ${vps:.2f}")
```

---

### Valuation Impact
Why this matters:
- Multiples are the most common market cross-check for DCF outputs and are often the primary approach in transaction settings.
- Correct peer selection and metric normalisation can matter more than the choice between EV/EBITDA vs EV/EBIT vs P/E.
- Multiples provide implied expectations: if you know the multiple and the current metric, you can infer what the market must believe about growth, margin, and risk.

Impact on multiples:
- EV multiples require consistent enterprise definitions and debt-like items; equity multiples require consistent share counts and net income definitions.
- Accounting differences (IFRS 16, capitalised costs, revenue recognition) can change EBITDA, EBIT, and net income and must be normalised.

Impact on DCF:
- DCF implies a terminal multiple. If it is inconsistent with peers, adjust terminal assumptions or explicitly explain why the business deserves a structural premium/discount.

Practical adjustments (multiple-based valuation with bridge consistency):
```python
def multiple_valuation_per_share(metric: float, mult: float,
                                net_debt: float, other_debt_like: float,
                                diluted_shares: int,
                                non_operating_assets: float = 0.0) -> float:
    """
    Apply an EV multiple to a metric, bridge to equity, and compute per-share value.
    """
    implied_ev = implied_value_from_multiple(metric, mult)
    implied_equity = equity_from_ev(implied_ev, net_debt=net_debt, other_debt_like=other_debt_like, non_operating_assets=non_operating_assets)
    return value_per_share(implied_equity, diluted_shares)
```

---

### Quality of Earnings Flags (Multiples and Metric Integrity)
⚠️ “Adjusted EBITDA” includes aggressive add-backs (ongoing costs labelled as one-off).  
⚠️ EBITDA inflated by IFRS 16 lease accounting relative to peers (lease inconsistency).  
⚠️ EBIT/EBITDA excludes required reinvestment (maintenance capex) in capex-heavy sectors (multiple overstates value).  
⚠️ Net income distorted by one-off tax or fair value items; P/E becomes meaningless without normalisation.  
⚠️ EV excludes debt-like items (leases, pensions) while peers include them (apples-to-oranges).  
✅ Clear reconciliation from reported metric → normalised metric and from market cap → EV with consistent definitions.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Software / SaaS | High growth, high reinvestment in opex, stock comp | Use EV/Revenue and EV/ARR alongside EV/EBITDA; normalise stock comp and capitalised dev costs |
| Retail / airlines | Leases drive EV and EBITDA | Lease-consistent EV/EBITDA; consider EV/EBITDAR for comparability |
| Capital-intensive industrials | EBITDA ignores maintenance capex differences | Use EV/EBIT or EV/NOPAT; cross-check with ROIC and reinvestment needs |
| Banks/insurers | EV multiples not meaningful | Use P/TB, P/E, ROE-based frameworks |
| Utilities / infrastructure | Regulated returns, stable cash flows | Use EV/EBITDA with regulated peer set; validate implied WACC and allowed returns |

---

### Practical Comparison Tables

#### 1) Comparable Selection Checklist (Beyond Industry Labels)
| Dimension | What to match | Why it matters |
|---|---|---|
| Growth profile | Revenue growth, unit economics | Drives multiple premiums/discounts |
| Profitability | Margin level and stability | Higher sustainable margin often → higher multiple |
| ROIC / capital intensity | ROIC, turnover, capex-to-sales | Determines value of growth and cash conversion |
| Risk | Leverage, cyclicality, customer concentration | Drives discount rate and market multiple |
| Accounting | Leases, revenue recognition, capitalised costs | Prevents mechanical multiple distortions |

#### 2) Metric Normalisation Checklist
| Item | Affects | Typical fix |
|---|---|---|
| One-offs (restructuring, litigation) | EBITDA/EBIT/NOPAT | Remove from metric; document evidence |
| IFRS 16 leases | EBITDA, EV definition | Use lease-consistent EV and metric (or adjust to EV/EBITDAR) |
| Stock-based compensation | EBITDA (if added back), comparability | Treat as real cost; avoid add-back without rationale |
| Capitalised development / software costs | EBITDA/EBIT, invested capital | Align policy across peers or adjust ROIC/metrics |
| Pension expense/deficit | EV and profitability | Treat deficit as debt-like if material; normalise pension expense |

---

### Real-World Example (Peer Multiple → Per Share + Implied Expectations)

Scenario: Apply a peer EV/EBITDA multiple to a normalised metric, bridge to equity, and compare implied multiple to your DCF terminal multiple.

```python
# Inputs
ebitda_reported = 860_000_000
ebitda_norm = normalize_ebitda(ebitda_reported, one_offs=40_000_000)

peer_mult = 10.0
net_debt = 1_450_000_000
leases = 520_000_000
diluted_shares = 540_000_000

vps = multiple_valuation_per_share(
    metric=ebitda_norm, mult=peer_mult,
    net_debt=net_debt, other_debt_like=leases,
    diluted_shares=diluted_shares
)

print(f"Multiple-based value per share: ${vps:.2f}")

# Cross-check: compare against a DCF implied exit multiple
tv_n = 8_900_000_000
ebitda_n = 900_000_000
implied_exit = multiple(tv_n, ebitda_n)
print(f"DCF implied terminal EV/EBITDA: {implied_exit:.1f}x" if implied_exit else "DCF implied: n/a")
```

Interpretation:
- If the multiple-based per-share value is far above DCF, your DCF may be conservative or the peer multiple may embed stronger growth/ROIC assumptions than your forecast.
- If DCF implied exit multiple is far above peer range, revisit terminal assumptions (growth, margins, reinvestment) or justify a structural premium (moat, durability, risk).
