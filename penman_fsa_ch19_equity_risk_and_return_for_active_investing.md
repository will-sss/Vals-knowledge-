# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 19: Analysis of Equity Risk and Return for Active Investing (Equity Risk, Beta, Fundamentals)

### Core Concept
Equity valuation depends on the required return investors demand for bearing risk, and on how firm fundamentals map into that risk. Practical equity-risk analysis connects **business risk (operating leverage, cyclicality, financial leverage)** to expected returns, and translates those into **discount rates, equity risk premiums, and valuation haircuts** (e.g., higher terminal discounting, lower multiples).

---

### Formula/Methodology

#### 1) Holding period return (HPR) and expected return
```text
HPR = (P1 - P0 + D1) / P0
```
Where:
- P0 = price at start
- P1 = price at end
- D1 = dividends received during period

Expected return (conceptual):
```text
E[R] = E[(P1 - P0 + D1) / P0]
```

#### 2) Excess return and equity risk premium (ERP)
```text
Excess Return = Re - Rf
ERP = E[Rm] - Rf
```
Where:
- Re = required return on equity
- Rf = risk-free rate
- Rm = market return

#### 3) CAPM required return
```text
Re = Rf + βe × (ERP)
```
Where:
- βe = equity beta (systematic risk relative to market)

#### 4) Levered vs unlevered beta (risk decomposition)
```text
βa ≈ (E/V) × βe + (D/V) × βd
```
If debt beta βd is small (common approximation for investment-grade debt), then:
```text
βa ≈ (E/V) × βe
```

A commonly used practical unlevering/relevering step (with assumptions):
```text
βu = βl / (1 + (1 - Tc) × (D/E))
βl = βu × (1 + (1 - Tc) × (D/E))
```
Where:
- βu = unlevered beta (asset/business risk)
- βl = levered beta (equity risk)
- Tc = tax rate (decimal)
- D/E = market value debt-to-equity

#### 5) Equity risk from accounting fundamentals (practical diagnostics)
No single “closed-form” accounting beta exists in practice; instead use **drivers that predict higher required returns**:
```text
Financial Leverage = Net Debt / Equity (or Debt / Equity)
Operating Leverage (proxy) = Fixed Cost Share or ΔEBIT / ΔSales sensitivity
Earnings Quality Risk = Accruals intensity and cash conversion instability
```
Use these as inputs to:
- select peer betas,
- justify a valuation discount (higher Re),
- stress scenarios.

---

### Practical Application (How to apply)

#### A) Build the equity discount rate for valuation (decision rules)
1) Set **Rf** (match currency and duration to cash flows; use consistent risk-free curve).  
2) Choose **ERP** (consistent with region and market).  
3) Estimate **β**:
   - Prefer **bottom-up beta**: peer unlever → average → relever to target capital structure.
   - Use regression beta as a sense check only (noise, window sensitivity, thin trading).  
4) Compute **Re** and apply:
   - in DCF for equity cash flows (FCFE discounting), or
   - in WACC (for FCFF discounting).  

#### B) Bottom-up beta workflow (valuation-ready)
| Step | Action | Output |
|---|---|---|
| 1 | Choose comparable listed peers | Peer set |
| 2 | Pull levered betas and capital structures | βl, D/E |
| 3 | Unlever each peer | βu (peer) |
| 4 | Take median/trimmed mean βu | βu (target) |
| 5 | Relever using target D/E and Tc | βl (target) |
| 6 | Compute Re and run sensitivities | Re grid |

#### C) Stress-test risk through fundamentals (active-investing lens)
Risk shows up as valuation fragility:
- High leverage + volatile margins → larger downside convexity → higher required return.
- Aggressive accruals + weak cash conversion → risk of earnings reversal → lower multiple and higher Re.
- Cyclical demand and high fixed costs → higher beta and more variable terminal value.

Use a simple “risk overlay” approach:
- Base Re from CAPM/bottom-up beta.
- Add a structured increment only when justified by **observable risks** (e.g., governance, illiquidity, key-customer concentration) and apply it consistently in scenarios rather than ad hoc.

---

### Python Implementation
```python
from typing import Sequence, Dict, Any, Optional
import numpy as np
import pandas as pd

def _f(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def capm_cost_of_equity(rf: float, beta: float, erp: float) -> float:
    """CAPM cost of equity: Re = Rf + beta * ERP.

    Args:
        rf: Risk-free rate (decimal, e.g., 0.04 for 4%).
        beta: Equity beta.
        erp: Equity risk premium (decimal).
    Returns:
        float: Required return on equity (decimal).
    Raises:
        ValueError: for non-finite inputs.
    """
    rf = _f(rf, "rf")
    beta = _f(beta, "beta")
    erp = _f(erp, "erp")
    return float(rf + beta * erp)

def unlever_beta(beta_levered: float, debt_to_equity: float, tax_rate: float) -> float:
    """Unlever beta using: beta_u = beta_l / (1 + (1 - Tc) * (D/E)).

    Args:
        beta_levered: Levered beta.
        debt_to_equity: D/E using market values (decimal, e.g., 0.5 for 50%).
        tax_rate: Corporate tax rate (decimal).
    Returns:
        float: Unlevered beta.
    Raises:
        ValueError: if denominator invalid or inputs out of range.
    """
    bl = _f(beta_levered, "beta_levered")
    de = _f(debt_to_equity, "debt_to_equity")
    tc = _f(tax_rate, "tax_rate")
    if tc < 0 or tc > 0.6:
        raise ValueError("tax_rate must be between 0 and 0.6 (decimal).")
    if de < 0:
        raise ValueError("debt_to_equity must be >= 0.")
    denom = 1.0 + (1.0 - tc) * de
    if denom <= 0:
        raise ValueError("Invalid denominator in unlevering formula.")
    return float(bl / denom)

def relever_beta(beta_unlevered: float, debt_to_equity: float, tax_rate: float) -> float:
    """Relever beta using: beta_l = beta_u * (1 + (1 - Tc) * (D/E))."""
    bu = _f(beta_unlevered, "beta_unlevered")
    de = _f(debt_to_equity, "debt_to_equity")
    tc = _f(tax_rate, "tax_rate")
    if tc < 0 or tc > 0.6:
        raise ValueError("tax_rate must be between 0 and 0.6 (decimal).")
    if de < 0:
        raise ValueError("debt_to_equity must be >= 0.")
    return float(bu * (1.0 + (1.0 - tc) * de))

def bottom_up_beta(peers: pd.DataFrame, tax_rate: float, target_de: float, trim: float = 0.1) -> Dict[str, float]:
    """Compute bottom-up beta from peer set.

    peers columns required:
        - beta_levered
        - debt_to_equity (market D/E)
    Args:
        peers: DataFrame of peer inputs.
        tax_rate: Corporate tax rate (decimal).
        target_de: Target D/E for relevering (decimal).
        trim: fraction to trim from each tail (0 to <0.5).
    Returns:
        dict: {beta_unlevered, beta_levered_target, n_peers_used}
    """
    if not {"beta_levered", "debt_to_equity"}.issubset(peers.columns):
        raise ValueError("peers must have columns: beta_levered, debt_to_equity")
    tc = _f(tax_rate, "tax_rate")
    tde = _f(target_de, "target_de")
    if trim < 0 or trim >= 0.5:
        raise ValueError("trim must be in [0, 0.5).")
    if tde < 0:
        raise ValueError("target_de must be >= 0.")

    df = peers.copy()
    df["beta_u"] = df.apply(lambda r: unlever_beta(r["beta_levered"], r["debt_to_equity"], tc), axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta_u"])
    if df.empty:
        raise ValueError("No valid peers after cleaning.")

    # Trimmed mean for robustness
    bu_sorted = np.sort(df["beta_u"].to_numpy())
    n = len(bu_sorted)
    k = int(np.floor(trim * n))
    bu_use = bu_sorted[k:n-k] if n - 2*k >= 1 else bu_sorted
    bu = float(np.mean(bu_use))

    bl_target = relever_beta(bu, tde, tc)
    return {"beta_unlevered": bu, "beta_levered_target": bl_target, "n_peers_used": float(len(df))}

# Example usage
peer_inputs = pd.DataFrame(
    {
        "beta_levered": [1.10, 0.95, 1.30, 0.80],
        "debt_to_equity": [0.40, 0.20, 0.60, 0.10],
    }
)

out = bottom_up_beta(peer_inputs, tax_rate=0.25, target_de=0.35, trim=0.1)
re = capm_cost_of_equity(rf=0.04, beta=out["beta_levered_target"], erp=0.05)

print(f"Bottom-up unlevered beta: {out['beta_unlevered']:.2f}")
print(f"Target levered beta: {out['beta_levered_target']:.2f}")
print(f"Cost of equity (CAPM): {re:.2%}")
```

---

### Valuation Impact
Why this matters:
- The required return on equity drives **discounting** and therefore present value; small changes in Re materially change value for long-duration cash flows.
- Beta and risk assessment drive **terminal value sensitivity**: higher risk compresses multiples and increases the discount rate.

Impact on multiples:
- Higher perceived risk → lower P/E and EV/EBITDA (higher discounting of future earnings).
- Comparing multiples across peers requires consistent risk framing (beta, leverage, cyclicality).

Impact on DCF inputs:
- Re affects FCFE discounting directly; via WACC it affects FCFF discounting.
- Risk assessment affects scenario weights (downside probability), not just point discount rates.

Comparability issues:
- Regression betas are noisy for thinly traded stocks and short histories; bottom-up betas are more stable for valuation.
- Capital structure differences matter: relevering aligns risk to the target financing mix.

Practical adjustments:
```python
def risk_overlay(re_base: float, increment_bps: float) -> float:
    """Apply a transparent risk overlay (use sparingly; justify with evidence).

    Args:
        re_base: Base cost of equity (decimal).
        increment_bps: Increment in basis points (e.g., 150 = +1.50%).
    Returns:
        float: Adjusted cost of equity (decimal).
    """
    re_base = float(re_base)
    inc = float(increment_bps)
    if not np.isfinite(re_base + inc):
        raise ValueError("Inputs must be finite.")
    if inc < -500 or inc > 1500:
        raise ValueError("increment_bps outside reasonable bounds.")
    return re_base + inc / 10_000.0
```

---

### Quality of Earnings / Risk Flags (equity-risk lens)
⚠️ High leverage with volatile operating margins (equity becomes a thin residual claim).  
⚠️ Large operating leverage (high fixed-cost base; earnings swing more than sales).  
⚠️ Weak cash conversion and rising accruals (higher probability of earnings reversal).  
⚠️ Customer/supplier concentration, key-person risk, or contract roll-over cliffs.  
⚠️ Frequent “adjusted” earnings add-backs with shifting definitions (opacity increases risk).  
✅ Stable margins and cash conversion across cycles; conservative accounting.  
✅ Transparent disclosures on risks and consistent segment reporting.

---

### Sector-Specific Considerations

| Sector | Key Risk Driver | Practical Beta / Re approach |
|---|---|---|
| Cyclicals (industrials, autos) | demand cyclicality + fixed costs | use long-run peer set; consider mid-cycle margins and higher beta |
| Tech / growth | long duration of cash flows | Re sensitivity is high; use scenario-based risk rather than arbitrary overlays |
| Utilities | regulation and leverage | beta often lower; focus on allowed returns and regulatory resets |
| Banks | leverage embedded in business model | avoid simple D/E relevering; use sector-specific risk metrics and regulation constraints |

---

### Real-World Example
Scenario: Value impact of a 100 bps change in cost of equity for a long-duration equity cash flow.

```python
def pv_perpetuity(cash_flow: float, discount_rate: float, growth_rate: float = 0.0) -> float:
    """Present value of a growing perpetuity: CF1 / (r - g)."""
    cf = float(cash_flow)
    r = float(discount_rate)
    g = float(growth_rate)
    if not np.isfinite(cf + r + g):
        raise ValueError("Inputs must be finite.")
    if r <= g:
        raise ValueError("discount_rate must be > growth_rate.")
    return cf / (r - g)

cf1 = 50.0  # $50m next year equity cash flow
g = 0.02

pv_low = pv_perpetuity(cf1, discount_rate=0.09, growth_rate=g)
pv_high = pv_perpetuity(cf1, discount_rate=0.10, growth_rate=g)

print(f"PV at 9%:  ${pv_low:.1f}m")
print(f"PV at 10%: ${pv_high:.1f}m")
print(f"Value change: {(pv_high/pv_low - 1):.1%}")
```

Interpretation:
- A +1.0% increase in Re can reduce value materially, especially when terminal value dominates.
- Use this sensitivity to:
  - calibrate scenario ranges for beta/ERP,
  - justify conservative terminal assumptions for high-risk firms.

See also: Chapter 12 (profitability analysis), Chapter 13 (growth & sustainable earnings), Chapter 17 (economic vs accounting value), Chapter 18 (earnings quality and accruals).
