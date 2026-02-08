# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 3: Calculating the Weighted Average Cost of Capital (WACC) (CAPM, cost of debt, target weights, WACC in DCF)

### Core Concept
WACC is the blended required return on a firm’s operating assets, weighted by the market values of equity and debt (and other financing claims if material). It is the discount rate for free cash flow to the firm (FCFF) in enterprise DCF and a key driver of valuation sensitivity: small changes in WACC can cause large changes in enterprise value, especially via terminal value.

### Formula/Methodology

#### 1) WACC (core)
```text
WACC = (E / V) × Re + (D / V) × Rd × (1 - Tc)

Where:
E = Market value of equity
D = Market value of interest-bearing debt (net of cash only if you use net-debt weights consistently; usually use gross debt for weights)
V = Total capital = E + D (plus other financing claims if included)
Re = Cost of equity
Rd = Pre-tax cost of debt (market yield / current borrowing rate)
Tc = Marginal corporate tax rate (use a sustainable marginal rate, not one-off effective tax)
```

#### 2) Cost of equity (CAPM baseline)
```text
Re = Rf + βL × ERP + (optional) size premium + (optional) country risk premium

Where:
Rf = risk-free rate matched to currency and duration
βL = levered beta (equity beta)
ERP = equity risk premium for that market/currency
```

#### 3) Levered beta from unlevered beta (Hamada-style; common in practice)
```text
βL = βU × (1 + (1 - Tc) × (D / E))

Where:
βU = unlevered (asset) beta
D/E = target debt-to-equity ratio (market values)
Tc = marginal tax rate
```

#### 4) Cost of debt (rating/spread or yield-based)
```text
Rd = Rf + Credit spread

Credit spread can be estimated from:
- observed bond yields
- synthetic rating (interest coverage → rating → spread)
- CDS spreads (if available)
```

#### 5) Tax shield treatment (practical)
```text
Use after-tax cost of debt in WACC only if:
- you are discounting pre-tax operating cash flows (FCFF) and
- the business is expected to be a tax-paying entity long-term.

If persistent losses / NOLs are expected, the effective tax shield may be delayed or reduced.
```

---

### Practical Application (Step-by-step)

1) Choose currency and valuation date
- Risk-free rate, ERP, and credit spreads must all be in the same currency.
- Use a valuation-date-consistent market inputs set.

2) Estimate target capital structure
- Prefer long-run target weights over current “temporary” balance sheet positioning.
- Use market values (equity market cap; debt at market value if observable).

3) Estimate cost of equity (Re)
- Start with CAPM: Rf + βL×ERP
- Compute βL from peer betas if the firm’s beta is unstable; use industry/peer medians.

4) Estimate cost of debt (Rd)
- If traded debt: use yield to maturity of representative debt.
- If not: synthetic rating → spread; validate with banking facility margins.

5) Compute WACC and sanity-check
- Compare to peer WACCs and implied valuation multiples.
- Stress test WACC ± 50–150 bps to see valuation sensitivity.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def cost_of_equity_capm(rf: float, beta: float, erp: float,
                        size_premium: float = 0.0,
                        country_risk_premium: float = 0.0) -> float:
    """CAPM-style cost of equity.

    Args:
        rf: risk-free rate (decimal, e.g., 0.04)
        beta: levered beta
        erp: equity risk premium (decimal)
        size_premium: optional add-on (decimal)
        country_risk_premium: optional add-on (decimal)

    Returns:
        float: cost of equity (decimal)

    Raises:
        ValueError: if inputs are not finite.
    """
    rf = _num(rf, "rf")
    beta = _num(beta, "beta")
    erp = _num(erp, "erp")
    sp = _num(size_premium, "size_premium")
    crp = _num(country_risk_premium, "country_risk_premium")
    return rf + beta * erp + sp + crp

def levered_beta(beta_u: float, debt: float, equity: float, tax_rate: float) -> float:
    """Hamada-style levered beta.

    Args:
        beta_u: unlevered beta
        debt: market value of debt
        equity: market value of equity
        tax_rate: marginal tax rate (0-1)

    Returns:
        float: levered beta

    Raises:
        ValueError: if equity <= 0, or tax rate outside [0,1].
    """
    beta_u = _num(beta_u, "beta_u")
    d = _num(debt, "debt")
    e = _num(equity, "equity")
    t = _num(tax_rate, "tax_rate")
    if e <= 0:
        raise ValueError("equity must be > 0.")
    if d < 0:
        raise ValueError("debt must be >= 0.")
    if not (0.0 <= t <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    return beta_u * (1.0 + (1.0 - t) * (d / e))

def cost_of_debt(rf: float, credit_spread: float) -> float:
    """Pre-tax cost of debt as rf + spread."""
    rf = _num(rf, "rf")
    spread = _num(credit_spread, "credit_spread")
    if spread < 0:
        raise ValueError("credit_spread must be >= 0.")
    return rf + spread

def wacc(equity: float, debt: float, re: float, rd: float, tax_rate: float,
         other_capital: float = 0.0, rother: Optional[float] = None) -> float:
    """Compute WACC including optional other capital claims.

    Args:
        equity: market value of equity (currency)
        debt: market value of interest-bearing debt (currency)
        re: cost of equity (decimal)
        rd: pre-tax cost of debt (decimal)
        tax_rate: marginal tax rate (0-1)
        other_capital: optional other financing claims (preferred equity, minorities treated as capital)
        rother: required return on other capital (decimal); if None and other_capital>0 -> error

    Returns:
        float: WACC (decimal)

    Raises:
        ValueError: invalid weights or inputs.
    """
    e = _num(equity, "equity")
    d = _num(debt, "debt")
    re = _num(re, "re")
    rd = _num(rd, "rd")
    t = _num(tax_rate, "tax_rate")
    oc = _num(other_capital, "other_capital")
    if e < 0 or d < 0 or oc < 0:
        raise ValueError("capital values must be >= 0.")
    if not (0.0 <= t <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    if oc > 0:
        if rother is None:
            raise ValueError("rother must be provided when other_capital > 0.")
        ro = _num(rother, "rother")
    else:
        ro = 0.0

    v = e + d + oc
    if v <= 0:
        raise ValueError("Total capital V must be > 0.")
    we = e / v
    wd = d / v
    woc = oc / v

    return we * re + wd * rd * (1.0 - t) + woc * ro

def implied_wacc_sensitivity(tv_weight: float, wacc: float, delta_bps: float) -> float:
    """Quick sensitivity proxy: how much discount rate move matters when TV dominates.

    Args:
        tv_weight: fraction of EV from terminal value (0-1)
        wacc: base WACC (decimal)
        delta_bps: change in WACC in basis points (+100 = +1%)

    Returns:
        float: adjusted WACC (decimal)

    Raises:
        ValueError: invalid inputs.
    """
    tvw = _num(tv_weight, "tv_weight")
    base = _num(wacc, "wacc")
    dbps = _num(delta_bps, "delta_bps")
    if not (0.0 <= tvw <= 1.0):
        raise ValueError("tv_weight must be between 0 and 1.")
    return base + dbps / 10_000.0

# Example usage (numbers illustrative)
inputs = {
    "equity": 4_500_000_000,  # $4.5B market cap
    "debt": 1_500_000_000,    # $1.5B debt at market
    "tax_rate": 0.25,
    "rf": 0.04,
    "erp": 0.05,
    "beta_u": 0.85,
    "credit_spread": 0.025
}

beta_l = levered_beta(inputs["beta_u"], inputs["debt"], inputs["equity"], inputs["tax_rate"])
re = cost_of_equity_capm(inputs["rf"], beta_l, inputs["erp"])
rd = cost_of_debt(inputs["rf"], inputs["credit_spread"])
w = wacc(inputs["equity"], inputs["debt"], re, rd, inputs["tax_rate"])

print(f"Levered beta: {beta_l:.2f}")
print(f"Cost of equity (Re): {re:.2%}")
print(f"Cost of debt (Rd): {rd:.2%}")
print(f"WACC: {w:.2%}")
```

---

### Valuation Impact
Why this matters:
- WACC is the discount rate for FCFF; it is often the single biggest driver of enterprise value through terminal value.
- Mis-specifying capital structure weights, risk-free rate currency, or credit spreads can produce systematically biased valuations.

Impact on multiples:
- Higher WACC generally implies lower sustainable EV/EBITDA and EV/FCF multiples (all else equal).
- Peer multiple spreads often reflect differences in perceived risk (which also shows up in WACC).

Impact on DCF inputs:
- WACC directly discounts FCFF and shapes terminal value (via continuing value).
- WACC interacts with growth assumptions: higher growth must be supported by reinvestment and risk-appropriate discounting.

Comparability issues across companies:
- Different leverage levels, tax rates, and segment risk profiles yield different WACCs.
- Ensure consistent treatment of leases, pensions, and hybrid instruments across peers when defining debt.

Practical adjustments:
```python
def debt_like_adjustment(gross_debt: float, lease_liability: float = 0.0, pension_deficit: float = 0.0) -> float:
    """Build a debt-like figure for valuation bridge and, optionally, weights.

    Args:
        gross_debt: interest-bearing debt
        lease_liability: IFRS 16 lease liability (if treated as debt-like)
        pension_deficit: net pension deficit (if treated as debt-like)

    Returns:
        float: adjusted debt-like amount
    """
    gd = _num(gross_debt, "gross_debt")
    ll = _num(lease_liability, "lease_liability")
    pd = _num(pension_deficit, "pension_deficit")
    if gd < 0 or ll < 0 or pd < 0:
        raise ValueError("inputs must be >= 0.")
    return gd + ll + pd
```

---

### Quality of Earnings Flags (WACC-related)
⚠️ Using effective tax rate distorted by one-offs (tax credits, settlements) instead of sustainable marginal rate.  
⚠️ Mixing currencies (e.g., USD cash flows discounted with GBP risk-free/ERP).  
⚠️ Using book-value weights for E and D instead of market values.  
⚠️ Credit spread inconsistent with observed borrowing rates or covenants (understates risk).  
✅ Cross-checks against peer implied WACC, and reconciliation of inputs to observable market data.

---

### Sector-Specific Considerations

| Sector | Key WACC issue | Typical handling |
|---|---|---|
| Utilities | regulated returns and leverage | use target regulatory gearing; validate with allowed returns |
| Banks | WACC not used for enterprise FCFF | prefer equity-based valuation (cost of equity, residual income) |
| Real estate | debt and property risk, lease structures | consider property-level discount rates; treat leases consistently |
| High-growth tech | beta instability and SBC dilution | peer/industry beta; scenario WACC; keep share dilution separate |

---

### Real-World Example
Scenario: DCF is producing enterprise value using FCFF. Analyst needs WACC and then to show how a 100 bps move changes discounting assumptions.

```python
base_wacc = w
wacc_up = implied_wacc_sensitivity(tv_weight=0.70, wacc=base_wacc, delta_bps=100)
wacc_down = implied_wacc_sensitivity(tv_weight=0.70, wacc=base_wacc, delta_bps=-100)

print(f"Base WACC: {base_wacc:.2%}")
print(f"WACC +100 bps: {wacc_up:.2%}")
print(f"WACC -100 bps: {wacc_down:.2%}")
```

Interpretation: If terminal value is a large share of EV, modest WACC changes can materially change valuation; always show a sensitivity table.

See also: Chapter 4 (DCF valuation) for FCFF discounting; Chapter 13 (betas) for estimating β; Chapter 7 (duration) and Chapter 8 (term structure) for rate curve inputs used in risk-free selection.
