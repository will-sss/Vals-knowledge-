# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 14: The Value of Operations and the Evaluation of Enterprise Price-to-Book Ratios and Price-Earnings Ratios (Enterprise P/B, Enterprise P/E, Operating vs Financing)

### Core Concept
Equity multiples can be misleading when firms differ in leverage, non-operating assets, or financing structure. This chapter reframes valuation around **operations**: separate operating assets and operating earnings from financing items, then evaluate **enterprise price-to-book** and **enterprise P/E** using operating book value and operating earnings for cleaner cross-company comparability.

---

### Formula/Methodology

#### 1) Enterprise value (operating perspective)
```text
Enterprise Value (EV) = Market Value of Equity + Net Debt
Net Debt = Interest-bearing Debt - Cash and Cash Equivalents
```

Notes:
- Treat excess cash as non-operating (remove from operations).
- Ensure consistent definitions across companies (include leases/pensions if treated as debt in your framework).

#### 2) Operating book value / Net Operating Assets (NOA)
```text
Net Operating Assets (NOA) = Operating Assets - Operating Liabilities
```

Operating liabilities = non-interest-bearing liabilities (e.g., trade payables, accrued expenses), excluding interest-bearing debt.

#### 3) Enterprise price-to-book (operating)
```text
Enterprise P/B (Operating) = EV / NOA
```

Interpretation:
- EV/NOA is “price per $ of operating capital employed”.
- Useful when earnings are volatile but the capital base is informative.

#### 4) Operating earnings (after tax) and operating P/E
```text
OPAT (or NOPAT) = EBIT × (1 - Operating Tax Rate)

Enterprise P/E (Operating) ≈ EV / OPAT
```

Where:
- Use operating tax rate aligned with operating profit (avoid distorted effective tax rates driven by one-offs).
- For banks/insurers, operating/financing separation is different; use sector-specific multiples (often P/TBV, P/E).

#### 5) ROCE / RNOA (link multiple ↔ fundamentals)
```text
ROCE (RNOA) = OPAT / Average NOA
```

High-level link:
- EV/NOA tends to rise with (ROCE - WACC) and expected growth/persistence.
- EV/OPAT tends to rise with (expected growth) and (persistence of operating earnings), and fall with higher risk/WACC.

#### 6) Enterprise-to-equity bridge (to value per share)
```text
Equity Value = EV + Non-operating Assets - Net Debt Adjustments
Value per Share = Equity Value / Diluted Shares
```

Typical non-operating assets:
- Excess cash
- Marketable securities / investments not integral to operations
- Assets held for sale (case-by-case)

---

### Practical Application (How to apply)

#### A) Reorganise statements (minimum for clean multiples)
1) Identify financing items:
   - Interest-bearing debt (incl. leases if you capitalise them)
   - Cash and cash equivalents
   - Interest income/expense
2) Identify operating items:
   - Working capital assets and liabilities
   - PP&E, intangibles used in operations
   - Operating provisions (careful: may be quasi-financing if long-dated)
3) Compute:
   - NOA
   - Net debt
   - OPAT (from EBIT with operating tax)

Deliverable output for comps:

| Item | Symbol | How to compute | Common pitfalls |
|---|---:|---|---|
| Market cap | E | Share price × diluted shares | use diluted if SBC meaningful |
| Net debt | ND | Debt - cash | define “cash” consistently (excess vs required) |
| Enterprise value | EV | E + ND | include pension deficit/leases if treated as debt |
| Operating book | NOA | Op assets - op liabs | misclassifying provisions, deferred taxes |
| Operating earnings | OPAT | EBIT × (1 - t_op) | distortions from one-off tax rates |

#### B) Decide which multiple to lead with
Use a decision rule:

| Condition | Lead multiple | Why |
|---|---|---|
| Stable margins; good tax comparability | EV/OPAT or EV/EBIT | closest to operating earnings power |
| Significant leverage differences across peers | EV-based multiples | strips financing effects |
| Earnings temporarily depressed or volatile | EV/NOA (and ROCE) | capital base + profitability lens |
| Business with large non-operating assets | EV/OPAT + explicit non-op adjustments | prevents “cheap” P/E illusion |

#### C) Cross-check multiples with profitability
If EV/NOA is high but ROCE is mediocre:
- Market is pricing in improvement (margin expansion, turnover gains) or growth.
- Validate via operational drivers; otherwise treat as overvaluation risk.

If EV/OPAT is high:
- Check persistence (recurrence) of OPAT and reinvestment needs.
- Separate transitory operating items and normalise.

---

### Python Implementation
```python
from typing import Dict, Any
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def compute_net_debt(debt: float, cash: float) -> float:
    """Net debt = interest-bearing debt - cash.

    Args:
        debt: gross interest-bearing debt ($)
        cash: cash and cash equivalents ($)

    Returns:
        float: net debt ($)

    Raises:
        ValueError: invalid inputs
    """
    d = _f(debt, "debt")
    c = _f(cash, "cash")
    return float(d - c)

def compute_ev(market_cap: float, net_debt: float) -> float:
    """Enterprise value = market cap + net debt."""
    mc = _f(market_cap, "market_cap")
    nd = _f(net_debt, "net_debt")
    return float(mc + nd)

def compute_noa(operating_assets: float, operating_liabilities: float) -> float:
    """NOA = operating assets - operating liabilities."""
    oa = _f(operating_assets, "operating_assets")
    ol = _f(operating_liabilities, "operating_liabilities")
    return float(oa - ol)

def compute_opat(ebit: float, operating_tax_rate: float) -> float:
    """OPAT = EBIT × (1 - t_op).

    Raises if tax rate outside plausible bounds.
    """
    e = _f(ebit, "ebit")
    t = _f(operating_tax_rate, "operating_tax_rate")
    if t < 0 or t > 0.6:
        raise ValueError("operating_tax_rate looks implausible; provide as decimal (e.g., 0.25).")
    return float(e * (1.0 - t))

def safe_multiple(numerator: float, denominator: float, name: str) -> float:
    """Compute numerator/denominator with divide-by-zero protection."""
    n = _f(numerator, f"{name}_numerator")
    d = _f(denominator, f"{name}_denominator")
    if abs(d) < 1e-12:
        raise ValueError(f"{name}: denominator is zero/too small.")
    return float(n / d)

def enterprise_pb(ev: float, noa: float) -> float:
    """EV/NOA (enterprise price-to-book)."""
    return safe_multiple(ev, noa, "EV_NOA")

def enterprise_pe(ev: float, opat: float) -> float:
    """EV/OPAT (enterprise P/E on operating earnings)."""
    return safe_multiple(ev, opat, "EV_OPAT")

def roce(opat: float, avg_noa: float) -> float:
    """ROCE (RNOA) = OPAT / avg NOA."""
    return safe_multiple(opat, avg_noa, "OPAT_AVG_NOA")

# Example usage ($m)
company = {
    "market_cap": 8_500.0,
    "debt": 2_200.0,
    "cash": 700.0,
    "operating_assets": 6_400.0,
    "operating_liabilities": 1_900.0,
    "ebit": 920.0,
    "t_op": 0.25,
    "avg_noa": 4_300.0
}

nd = compute_net_debt(company["debt"], company["cash"])
ev = compute_ev(company["market_cap"], nd)
noa = compute_noa(company["operating_assets"], company["operating_liabilities"])
opat = compute_opat(company["ebit"], company["t_op"])

ev_noa = enterprise_pb(ev, noa)
ev_opat = enterprise_pe(ev, opat)
rnoa = roce(opat, company["avg_noa"])

print(f"Net debt: ${nd:.0f}m | EV: ${ev:.0f}m")
print(f"NOA: ${noa:.0f}m | OPAT: ${opat:.0f}m")
print(f"EV/NOA: {ev_noa:.2f}x | EV/OPAT: {ev_opat:.2f}x | ROCE: {rnoa:.1%}")
```

---

### Valuation Impact
Why this matters:
- Enterprise multiples (EV/OPAT, EV/NOA) reduce distortion from leverage and capital structure, improving peer comparability.
- EV/NOA combined with ROCE makes “value creation” visible: high EV/NOA should be justified by ROCE above WACC and/or strong growth with reinvestment discipline.

Impact on multiples:
- Reported P/E can look “cheap” if a firm has excess cash or significant non-operating gains boosting earnings.
- EV-based multiples avoid double-counting cash in both numerator (market cap) and denominator (earnings including interest income).

Impact on DCF inputs:
- Operating reorganisation aligns forecast drivers (revenue, margins, reinvestment) with FCFF.
- NOA and ROCE-based logic can be used as cross-checks on reinvestment and terminal value assumptions.

Comparability issues:
- Different definitions of cash (required vs excess), treatment of leases, pensions, and provisions shift net debt and EV.
- Differences in capitalised intangibles or development costs affect NOA and therefore EV/NOA.

Practical adjustments:
```python
def adjust_ev_for_pension_and_leases(ev: float, pension_deficit: float = 0.0, lease_liability: float = 0.0) -> float:
    """Add quasi-debt items to EV if your framework treats them as financing."""
    import numpy as np
    e = float(ev)
    p = float(pension_deficit)
    l = float(lease_liability)
    if not (np.isfinite(e) and np.isfinite(p) and np.isfinite(l)):
        raise ValueError("Inputs must be finite.")
    return e + p + l
```

---

### Quality of Earnings / Multiple-Use Flags
⚠️ EV/OPAT computed with EBIT that includes “other income” from disposals or fair value gains.  
⚠️ “Cheap” EV/EBITDA driven by under-provisioning or capitalised costs (inflated EBITDA).  
⚠️ Net debt understated by excluding leases, pensions, supplier financing, or long-dated provisions treated as operating.  
⚠️ NOA artificially low from aggressive working-capital management (e.g., stretching payables) that may reverse.  
✅ Stable ROCE above WACC with consistent reinvestment to support growth.  
✅ Clear separation of operating vs non-operating items in disclosures; consistent classification over time.  

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Utilities / infrastructure | large regulated asset base; leverage differences | EV/NOA + ROCE vs allowed returns; consider regulated asset base metrics |
| Asset-light tech | capitalised dev / SBC affects NOA and earnings | standardise capitalisation policies; adjust for SBC where relevant |
| Retail | lease treatment drives net debt and EV | include lease liabilities; use EV/EBITDAR where appropriate |
| Financials | operating vs financing separation breaks down | use P/TBV, P/E, ROE and credit quality ratios instead |

---

### Real-World Example
Scenario: Two firms have identical reported P/E, but one holds significant excess cash and the other is levered.

```python
a = {"market_cap": 5_000, "debt": 1_000, "cash": 1_200, "ebit": 420, "t_op": 0.25, "op_assets": 3_500, "op_liabs": 900}
b = {"market_cap": 5_000, "debt": 2_000, "cash": 200,  "ebit": 420, "t_op": 0.25, "op_assets": 3_500, "op_liabs": 900}

def compute_summary(x):
    nd = compute_net_debt(x["debt"], x["cash"])
    ev = compute_ev(x["market_cap"], nd)
    noa = compute_noa(x["op_assets"], x["op_liabs"])
    opat = compute_opat(x["ebit"], x["t_op"])
    return enterprise_pb(ev, noa), enterprise_pe(ev, opat)

ev_noa_a, ev_opat_a = compute_summary(a)
ev_noa_b, ev_opat_b = compute_summary(b)

print(f"Company A EV/NOA={ev_noa_a:.2f}x, EV/OPAT={ev_opat_a:.2f}x")
print(f"Company B EV/NOA={ev_noa_b:.2f}x, EV/OPAT={ev_opat_b:.2f}x")
```

Interpretation: Even if equity P/E is similar, EV-based multiples can diverge materially once cash and leverage differ—giving a cleaner basis for peer valuation and for triangulating DCF results.

See also: Chapter 10 (operating vs financing classification), Chapter 12 (profitability and ROCE), Chapter 13 (growth and sustainability), Chapter 18 (quality of financial statements), Chapter 20 (credit risk).
