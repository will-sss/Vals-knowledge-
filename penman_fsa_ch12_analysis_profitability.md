# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 12: The Analysis of Profitability (ROCE/ROE Decomposition, Operating vs Financing, Sustainable Profitability)

### Core Concept
Profitability analysis separates **operating performance** from **financing effects** so valuation can focus on value creation in operations. The practical goal is to decompose returns (ROCE/ROE) into **margins, turnover, leverage, and financing spread**, identify what is **sustainable**, and translate profitability drivers into forward cash flows or residual income.

---

### Formula/Methodology

#### 1) Operating profitability: Return on operating assets / capital (ROCE / RNOA)
```text
ROCE (or RNOA) = Operating Profit After Tax / Net Operating Assets
```

Where:
- Operating Profit After Tax (OPAT) ≈ EBIT × (1 - Tax rate) (use operating tax rate if available)
- Net Operating Assets (NOA) = Operating Assets - Operating Liabilities  
  (exclude interest-bearing debt/cash equivalents used for financing)

Useful decompositions:
```text
ROCE = Operating Margin × Operating Asset Turnover

Operating Margin = OPAT / Revenue
Operating Asset Turnover = Revenue / NOA
```

#### 2) Equity profitability: Return on equity (ROE)
```text
ROE = Net Income / Average Common Equity
```

Penman-style linkage to operations + financing:
```text
ROE ≈ ROCE + (Financial Leverage × (ROCE - After-tax Cost of Debt))
```

Where:
- Financial Leverage (FLEV) = Net Financial Obligations / Common Equity
- Net Financial Obligations (NFO) = Interest-bearing Debt - Financial Assets (excess cash/investments)
- After-tax Cost of Debt (Rd_after) = Interest Rate × (1 - Tax rate)

Financing spread:
```text
Financing Spread = ROCE - Rd_after
```

Interpretation:
- If ROCE > Rd_after, leverage increases ROE (positive spread).
- If ROCE < Rd_after, leverage destroys equity profitability.

#### 3) Dupont (earnings-based) decomposition (useful when NOA/NFO split unavailable)
```text
ROE = Net Profit Margin × Asset Turnover × Equity Multiplier

Net Profit Margin = Net Income / Revenue
Asset Turnover = Revenue / Average Total Assets
Equity Multiplier = Average Total Assets / Average Equity
```

#### 4) Sustainable profitability (diagnostic)
```text
Sustainable ROCE/ROE focuses on recurring operating profit and normalised capital base.
```

Common normalisations:
- Remove one-offs (restructuring, impairments, gains/losses on disposals).
- Normalise margins for cyclicality and input cost spikes.
- Replace “lumpy” working capital with steady-state ΔNWC assumptions.

---

### Practical Application (How to apply)

#### A) Recast into operating vs financing (minimum viable)
1) Identify operating profit (EBIT) and compute OPAT using an operating tax rate.
2) Estimate NOA:
   - Operating assets: PPE, intangibles used in operations, working capital assets.
   - Operating liabilities: trade payables, accruals, deferred revenue (contract liabilities), provisions (operating).
   - Exclude: interest-bearing debt, pension deficits treated as financing (case-by-case), excess cash.
3) Compute ROCE (OPAT / NOA) using average NOA over the period.

#### B) Decompose ROCE into drivers
- Margin vs turnover: determine whether profitability comes from pricing/power (margin) or capital efficiency (turnover).
- Track across time and peers to identify structural advantages vs temporary tailwinds.

#### C) Link to valuation models
- In residual income / abnormal earnings models, profitability above required return drives value.
- In DCF, ROCE interacts with growth to determine reinvestment needs:
  - Higher ROCE for a given growth rate generally implies lower required reinvestment (higher FCFF).

Rule-of-thumb relationship:
```text
Reinvestment Rate ≈ Growth / ROCE   (using consistent definitions)
```

#### D) Stress test sustainability
- If ROCE is high due to temporarily low capital base (underinvestment) or working-capital squeeze, forecast should fade.
- If ROCE is boosted by capitalised costs (R&D/dev) or aggressive revenue recognition, adjust and reassess.

---

### Python Implementation
```python
from typing import Dict, Any, Optional
import numpy as np

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def safe_divide(n: float, d: float, *, on_zero: Optional[float] = None) -> Optional[float]:
    n = _f(n, "numerator")
    d = _f(d, "denominator")
    if abs(d) < 1e-12:
        return on_zero
    return float(n / d)

def opat(ebit: float, tax_rate: float) -> float:
    """Operating profit after tax (OPAT) from EBIT.

    Args:
        ebit: Earnings before interest and tax ($)
        tax_rate: decimal (e.g., 0.25)

    Returns:
        float: OPAT ($)

    Raises:
        ValueError: invalid inputs
    """
    e = _f(ebit, "ebit")
    t = _f(tax_rate, "tax_rate")
    if t < 0 or t > 0.6:
        raise ValueError("tax_rate looks implausible; use decimal (e.g., 0.25).")
    return float(e * (1.0 - t))

def net_operating_assets(operating_assets: float, operating_liabilities: float) -> float:
    """Compute NOA = Operating Assets - Operating Liabilities."""
    oa = _f(operating_assets, "operating_assets")
    ol = _f(operating_liabilities, "operating_liabilities")
    return float(oa - ol)

def roce(opat_value: float, noa_open: float, noa_close: float) -> float:
    """Compute ROCE (RNOA) using average NOA."""
    o = _f(opat_value, "opat")
    n0 = _f(noa_open, "noa_open")
    n1 = _f(noa_close, "noa_close")
    avg_noa = (n0 + n1) / 2.0
    r = safe_divide(o, avg_noa, on_zero=None)
    if r is None:
        raise ValueError("Average NOA must be non-zero.")
    return float(r)

def roce_decomposition(opat_value: float, revenue: float, noa_avg: float) -> Dict[str, float]:
    """Decompose ROCE into margin and turnover."""
    o = _f(opat_value, "opat")
    rev = _f(revenue, "revenue")
    noa = _f(noa_avg, "noa_avg")
    if abs(rev) < 1e-12:
        raise ValueError("revenue must be non-zero.")
    if abs(noa) < 1e-12:
        raise ValueError("noa_avg must be non-zero.")
    margin = o / rev
    turnover = rev / noa
    return {"operating_margin": float(margin), "operating_turnover": float(turnover), "roce": float(margin * turnover)}

def net_financial_obligations(interest_bearing_debt: float, financial_assets: float) -> float:
    """NFO = Debt - Financial Assets (excess cash/investments)."""
    d = _f(interest_bearing_debt, "interest_bearing_debt")
    fa = _f(financial_assets, "financial_assets")
    return float(d - fa)

def financial_leverage(nfo: float, equity: float) -> float:
    """FLEV = NFO / Equity."""
    n = _f(nfo, "nfo")
    e = _f(equity, "equity")
    lev = safe_divide(n, e, on_zero=None)
    if lev is None:
        raise ValueError("equity must be non-zero.")
    return float(lev)

def roe_from_ops_and_financing(roce_value: float, rd_pre_tax: float, tax_rate: float, flev: float) -> Dict[str, float]:
    """Approximate ROE from ROCE, after-tax cost of debt, and leverage."""
    r = _f(roce_value, "roce")
    rd = _f(rd_pre_tax, "rd_pre_tax")
    t = _f(tax_rate, "tax_rate")
    l = _f(flev, "flev")
    if t < 0 or t > 0.6:
        raise ValueError("tax_rate looks implausible; use decimal.")
    rd_after = rd * (1.0 - t)
    spread = r - rd_after
    roe_approx = r + l * spread
    return {"rd_after_tax": float(rd_after), "financing_spread": float(spread), "roe_approx": float(roe_approx)}

# Example usage ($m)
example = {
    "revenue": 5_000.0,
    "ebit": 900.0,
    "tax_rate": 0.25,
    "noa_open": 3_200.0,
    "noa_close": 3_600.0,
    "debt": 1_800.0,
    "financial_assets": 300.0,
    "equity": 2_100.0,
    "rd_pre_tax": 0.06
}

op = opat(example["ebit"], example["tax_rate"])
r = roce(op, example["noa_open"], example["noa_close"])
noa_avg = (example["noa_open"] + example["noa_close"]) / 2.0
dec = roce_decomposition(op, example["revenue"], noa_avg)

nfo = net_financial_obligations(example["debt"], example["financial_assets"])
lev = financial_leverage(nfo, example["equity"])
roe_link = roe_from_ops_and_financing(r, example["rd_pre_tax"], example["tax_rate"], lev)

print(f"OPAT: ${op:.0f}m | ROCE: {r:.1%}")
print(f"Operating margin: {dec['operating_margin']:.1%} | Turnover: {dec['operating_turnover']:.2f}x")
print(f"FLEV: {lev:.2f}x | After-tax Rd: {roe_link['rd_after_tax']:.1%} | Spread: {roe_link['financing_spread']:.1%}")
print(f"ROE (approx): {roe_link['roe_approx']:.1%}")
```

---

### Valuation Impact
Why this matters:
- Profitability determines whether growth creates value. High ROCE with disciplined reinvestment generally supports higher enterprise value because incremental growth produces cash.
- Separating operating and financing profitability prevents false conclusions (e.g., a high ROE driven purely by leverage rather than operating strength).

Impact on multiples:
- Higher and more sustainable operating profitability often supports higher EV/EBIT and EV/EBITDA.
- If ROE is high due to leverage and a narrow/negative spread, equity multiples can compress due to higher risk.

Impact on DCF inputs:
- ROCE helps translate growth into reinvestment: higher ROCE implies less capital required per dollar of growth, raising FCFF.
- Profitability analysis informs fade assumptions (how quickly returns revert to industry norms).

Comparability issues:
- Different capitalisation policies (development costs, leases, IFRS 16, pension classification) change the capital base (NOA) and margins.
- One-offs in operating profit distort margins; normalise across peers.

Practical adjustments:
```python
def normalise_operating_profit(reported_op: float, one_offs_after_tax: float) -> float:
    """Remove after-tax one-offs to get a sustainable operating profit."""
    import numpy as np
    rop = float(reported_op)
    adj = float(one_offs_after_tax)
    if not np.isfinite(rop) or not np.isfinite(adj):
        raise ValueError("Inputs must be finite.")
    return rop - adj
```

---

### Quality of Earnings Flags (profitability-focused)
⚠️ ROCE improving while capex is below depreciation (profitability may be boosted by underinvestment).  
⚠️ Margin expansion driven by capitalised costs (software dev, customer acquisition) rather than true efficiency.  
⚠️ ROE high but financing spread negative (ROCE < Rd_after), indicating leverage is value-destructive.  
⚠️ Large swings in provisions/impairments included in “operating” profit in ways that smooth earnings.  
✅ ROCE improvement accompanied by stable/healthy cash conversion and consistent asset turnover.  
✅ Profitability supported by structural drivers (pricing power, scale economies, efficient working capital).

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Capital-light services | understated operating assets (off-balance sheet intangibles) inflate ROCE | interpret ROCE with caution; supplement with unit economics and cash conversion |
| Software | capitalised development + SBC distort margin and capital base | treat SBC as operating cost + dilution; adjust capitalised dev to maintain comparability |
| Industrials | cyclicality drives temporary margin/turnover swings | use mid-cycle margins and normalised turnover; stress test fixed-cost leverage |
| Retail | leases (IFRS 16) inflate assets and change operating profit presentation | ensure consistent lease treatment across comparables before comparing ROCE/ROA |

---

### Real-World Example
Scenario: Two firms have similar ROE, but one earns it through high ROCE (operations) while the other relies on leverage.

```python
firm_A = {"roce": 0.18, "rd_pre_tax": 0.06, "tax_rate": 0.25, "flev": 0.5}
firm_B = {"roce": 0.10, "rd_pre_tax": 0.07, "tax_rate": 0.25, "flev": 2.5}

a = roe_from_ops_and_financing(firm_A["roce"], firm_A["rd_pre_tax"], firm_A["tax_rate"], firm_A["flev"])
b = roe_from_ops_and_financing(firm_B["roce"], firm_B["rd_pre_tax"], firm_B["tax_rate"], firm_B["flev"])

print(f"Firm A ROE approx: {a['roe_approx']:.1%} | Spread: {a['financing_spread']:.1%}")
print(f"Firm B ROE approx: {b['roe_approx']:.1%} | Spread: {b['financing_spread']:.1%}")
```

Interpretation: Firm A’s ROE is driven by operating value creation (high ROCE and positive spread), typically more sustainable and supportive of higher multiples. Firm B’s ROE depends on high leverage with a thin/negative spread, increasing fragility and lowering valuation resilience.

See also: Chapter 10 (recasting operating vs financing), Chapter 11 (cash flow diagnostics), Chapter 13 (growth and sustainable earnings), Chapter 18 (quality of financial statements), Chapter 19 (equity risk and return).
