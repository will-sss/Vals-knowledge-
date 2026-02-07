# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 12: Closure in Valuation — Estimating Terminal Value (Stable Growth, Reinvestment, ROIC, Exit Multiples Cross-Checks)

### Core Concept
Terminal value (TV) captures value beyond the explicit forecast horizon and often represents the majority of intrinsic value. The practical job is to force **closure**: in the terminal period, assumptions about growth, reinvestment, and returns must be internally consistent and constrained by macro reality (long-run growth cannot exceed the economy/currency trend indefinitely).

See also: Chapter 9 (measuring sustainable earnings), Chapter 15 (WACC/APV), Chapter 16 (equity value per share), Chapter 17 (relative valuation cross-checks).

---

### Formula/Methodology

#### 1) Perpetuity (stable growth) terminal value — firm (FCFF/WACC)
```text
Terminal Value (end of year n) = FCFF_{n+1} / (WACC − g)

Where:
FCFF_{n+1} = FCFF in year n+1 (first year after explicit forecast)
WACC       = weighted average cost of capital (decimal)
g          = stable growth rate (decimal)
Constraint: WACC > g
```
Implementation note:
- Use **FCFF_{n+1} = FCFF_n × (1 + g)** only if the terminal-year FCFF is already in a stable state.
- Otherwise, compute FCFF_{n+1} from terminal NOPAT and reinvestment (recommended below).

#### 2) Perpetuity (stable growth) terminal value — equity (FCFE/k_e)
```text
Terminal Equity Value (end of year n) = FCFE_{n+1} / (k_e − g)

Where:
FCFE_{n+1} = FCFE in year n+1
k_e        = cost of equity (decimal)
g          = stable growth rate (decimal)
Constraint: k_e > g
```

#### 3) Stable growth requires reinvestment — tie growth to ROIC
A key closure relationship for firm valuation:
```text
Reinvestment Rate = g / ROIC

Where:
Reinvestment Rate = Reinvestment / NOPAT (decimal)
g                 = stable growth rate (decimal)
ROIC              = return on invested capital in steady state (decimal)
Constraint: ROIC > 0
```
Then terminal FCFF can be built from terminal NOPAT:
```text
FCFF_{n+1} = NOPAT_{n+1} × (1 − Reinvestment Rate)
          = NOPAT_{n+1} × (1 − g/ROIC)
```
Interpretation:
- Higher stable growth requires higher reinvestment unless ROIC is very high.
- If you assume high g with low reinvestment, you are implicitly assuming “free growth” — usually not credible.

#### 4) Stable growth constraints (practical guardrails)
```text
g should be <= long-run nominal growth rate of the economy/currency
For mature currencies, g typically in low single digits (nominal) unless strong justification.
```
Also, terminal operating assumptions should converge toward sustainable levels:
- margins normalize (competition)
- ROIC tends toward cost of capital unless durable moat is justified
- reinvestment stabilizes (capex and working capital intensity)

#### 5) Exit multiple terminal value (as a cross-check, not a substitute)
```text
Terminal Value (exit multiple) = Terminal Metric × Exit Multiple

Examples:
TV = EBITDA_n × (EV/EBITDA exit multiple)
TV = EBIT_n × (EV/EBIT exit multiple)
```
Cross-check logic:
- Exit multiple implies a perpetuity-equivalent g given WACC and terminal cash flow conversion.
- If the implied g is nonsensical, the exit multiple is not consistent with fundamentals.

---

### Python Implementation
```python
from typing import Optional


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def terminal_value_perpetuity(cash_flow_next: float, discount_rate: float, growth_rate: float) -> float:
    """
    Generic perpetuity TV: TV = CF_{n+1} / (r - g).

    Args:
        cash_flow_next: CF in year n+1 (currency).
        discount_rate: r as decimal.
        growth_rate: g as decimal.

    Returns:
        float: terminal value at end of year n.

    Raises:
        ValueError: if r <= g, invalid inputs.
    """
    for name, v in {"cash_flow_next": cash_flow_next, "discount_rate": discount_rate, "growth_rate": growth_rate}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if discount_rate <= growth_rate:
        raise ValueError("discount_rate must be greater than growth_rate (r > g)")
    return cash_flow_next / (discount_rate - growth_rate)


def reinvestment_rate_from_g_roic(growth_rate: float, roic: float) -> Optional[float]:
    """Reinvestment Rate = g / ROIC. Returns None if ROIC <= 0."""
    if not _num(growth_rate) or not _num(roic):
        raise ValueError("inputs must be numeric")
    if roic <= 0:
        return None
    return growth_rate / roic


def terminal_fcff_from_nopat(nopat_next: float, growth_rate: float, roic: float) -> Optional[float]:
    """
    FCFF_{n+1} = NOPAT_{n+1} * (1 - g/ROIC). Returns None if ROIC <= 0 or reinvestment > 1.
    """
    if not _num(nopat_next) or not _num(growth_rate) or not _num(roic):
        raise ValueError("inputs must be numeric")
    rr = reinvestment_rate_from_g_roic(growth_rate, roic)
    if rr is None:
        return None
    if rr > 1:
        return None  # implies reinvestment exceeds NOPAT (not stable without external financing effects)
    return nopat_next * (1.0 - rr)


def implied_exit_multiple_from_perpetuity(tv: float, terminal_metric: float) -> Optional[float]:
    """Implied exit multiple = TV / metric. Returns None if metric <= 0."""
    if not _num(tv) or not _num(terminal_metric):
        raise ValueError("inputs must be numeric")
    if terminal_metric <= 0:
        return None
    return tv / terminal_metric


def implied_g_from_exit_multiple(cash_flow_next: float, discount_rate: float, tv: float) -> Optional[float]:
    """
    From TV = CF_{n+1} / (r - g) => g = r - CF_{n+1}/TV. Returns None if TV<=0.
    """
    if not _num(cash_flow_next) or not _num(discount_rate) or not _num(tv):
        raise ValueError("inputs must be numeric")
    if tv <= 0:
        return None
    return discount_rate - (cash_flow_next / tv)


# Example usage (all $m)
WACC = 0.09
g = 0.03
ROIC_terminal = 0.14

# Build terminal cash flow from terminal operating profitability (recommended closure)
NOPAT_n1 = 520.0
FCFF_n1 = terminal_fcff_from_nopat(NOPAT_n1, g, ROIC_terminal)
if FCFF_n1 is None:
    raise ValueError("Terminal FCFF could not be computed; check ROIC and g assumptions.")

TV_perp = terminal_value_perpetuity(FCFF_n1, WACC, g)

# Cross-check: implied exit multiple from a terminal EBITDA metric
EBITDA_n = 800.0
exit_multiple_implied = implied_exit_multiple_from_perpetuity(TV_perp, EBITDA_n)

# If you choose an exit multiple instead, back out implied g
TV_exit = EBITDA_n * 10.0  # 10.0x EV/EBITDA exit multiple
g_implied = implied_g_from_exit_multiple(cash_flow_next=FCFF_n1, discount_rate=WACC, tv=TV_exit)

print(f"Terminal reinvestment rate (g/ROIC): {reinvestment_rate_from_g_roic(g, ROIC_terminal):.1%}")
print(f"Terminal FCFF (n+1): ${FCFF_n1:,.0f}m")
print(f"Terminal Value (perpetuity): ${TV_perp:,.0f}m")
print(f"Implied EV/EBITDA from perpetuity TV: {exit_multiple_implied:.1f}x" if exit_multiple_implied else "n/a")
print(f"Implied g from 10.0x exit multiple TV: {g_implied:.1%}" if g_implied is not None else "n/a")
```

---

### Valuation Impact
Why this matters:
- TV can drive most of enterprise value; weak closure (inconsistent g, ROIC, reinvestment) is the fastest way to generate an unjustifiable valuation.
- Stable growth links to reinvestment: assuming high g without reinvestment inflates FCFF and TV.
- Exit multiples should be consistent with fundamentals; otherwise they are “market mood” inputs disguised as valuation.

Impact on multiples:
- Perpetuity TV implies an “exit multiple” embedded in your DCF. If the implied multiple is far from market/peer reality, your terminal assumptions likely conflict with observed pricing.
- Conversely, an exit multiple implies a perpetuity-equivalent g; if implied g is implausible (too high vs macro), the exit multiple is not defensible.

Impact on DCF inputs:
- Use sustainable margins and ROIC in terminal year; fade to stable levels rather than freezing a peak year.
- Ensure WACC > g (firm) and k_e > g (equity) and that g is consistent with the valuation currency’s long-run nominal growth.

Practical adjustments:
```python
def validate_terminal_assumptions(discount_rate: float, growth_rate: float, roic: float) -> None:
    """Hard checks for terminal closure."""
    if discount_rate <= growth_rate:
        raise ValueError("Need r > g for a finite perpetuity terminal value.")
    if roic <= 0:
        raise ValueError("ROIC must be > 0 for reinvestment closure.")
    rr = growth_rate / roic
    if rr > 1:
        raise ValueError("Reinvestment rate > 100% is not stable; reduce g or increase ROIC (with justification).")
```

---

### Quality of Earnings Flags (Terminal Value Risk)
⚠️ Terminal margins assume “peak” profitability persists with no competitive fade.  
⚠️ Terminal growth exceeds the long-run nominal growth of the valuation currency/economy.  
⚠️ Terminal ROIC is far above cost of capital with no moat evidence; implies perpetual abnormal profits.  
⚠️ Terminal reinvestment is inconsistent with growth (high g with low reinvestment).  
⚠️ DCF value is dominated by TV but the terminal block is not sensitivity-tested.

✅ Green flag: TV built from terminal NOPAT and reinvestment closure (g/ROIC), with implied exit multiple cross-checked against peer multiples and a sensitivity grid (WACC × g).

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Consumer staples | Stable long-run economics | Perpetuity TV often appropriate; g near long-run nominal growth. |
| Cyclicals | Mean reversion | Terminal margins/ROIC should be mid-cycle, not peak; use normalised earnings. |
| High-growth tech | Long transition to stable | Use multi-stage fade to stable margins/ROIC; avoid forcing early stable growth. |
| Commodities | Finite resources | Perpetuity may be inappropriate; model depletion or long-run replacement economics; consider shorter TV horizon. |
| Financials | Equity-focused cash flows | Prefer equity intrinsic models (dividends/FCFE) for TV logic; ensure k_e > g. |

---

### Practical Comparison Tables

#### 1) Terminal value method selection
| Situation | Better TV method | Why |
|---|---|---|
| Mature, stable business | Perpetuity (stable growth) | Fundamentals can be captured with sustainable g and margins |
| Business priced on market multiples | Perpetuity + exit multiple cross-check | Avoid anchoring solely to market mood |
| Cyclical business | Perpetuity with mid-cycle normalisation | Peak/through-cycle differences dominate TV |
| Finite-life assets | Finite horizon or liquidation value | Perpetuity assumption breaks down |

#### 2) Terminal closure checks (must-pass)
| Check | Rule | Fix if fails |
|---|---|---|
| Finite value | r > g | Lower g or increase r (justify) |
| Macro consistency | g <= long-run nominal growth | Reduce g or switch currency/real model |
| Reinvestment closure | Reinvestment = g/ROIC | Adjust ROIC fade, reinvestment, or g |
| Economic plausibility | ROIC fades toward WACC unless moat | Add fade, remove peak economics |

---

### Real-World Example
Scenario: A DCF for an industrial company shows 70% of EV from terminal value. You rebuild TV using NOPAT and reinvestment closure and cross-check implied EV/EBITDA vs a chosen exit multiple.

```python
WACC = 0.09
g = 0.03
ROIC_terminal = 0.12

NOPAT_n1 = 500.0
FCFF_n1 = terminal_fcff_from_nopat(NOPAT_n1, g, ROIC_terminal)
validate_terminal_assumptions(WACC, g, ROIC_terminal)

TV_perp = terminal_value_perpetuity(FCFF_n1, WACC, g)

EBITDA_n = 780.0
implied_exit = implied_exit_multiple_from_perpetuity(TV_perp, EBITDA_n)

TV_exit = EBITDA_n * 9.0
g_from_exit = implied_g_from_exit_multiple(FCFF_n1, WACC, TV_exit)

print(f"Terminal FCFF: ${FCFF_n1:,.0f}m")
print(f"Perpetuity TV: ${TV_perp:,.0f}m")
print(f"Implied exit multiple: {implied_exit:.1f}x")
print(f"Exit-multiple TV (9.0x): ${TV_exit:,.0f}m -> implied g: {g_from_exit:.1%}")
```

Interpretation: If implied exit multiple is far above peers without superior terminal ROIC/g justification, the perpetuity block is too optimistic. If a chosen exit multiple implies an implausible g, the exit multiple is not consistent with the cash flow conversion and discount rate.
