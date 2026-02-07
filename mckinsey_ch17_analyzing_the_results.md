# McKinsey Valuation - Key Concepts for Valuation

## Chapter 17: Analyzing the Results (Value Drivers, Sensitivities, Implied Expectations, Reconciliation)

### Core Concept
After building a valuation, the critical work is diagnosing what the model is actually saying: which assumptions create value, how sensitive value is to those assumptions, and whether the implied expectations are plausible relative to history, peers, and market pricing. This chapter is about turning a spreadsheet output into an investment-grade judgement using driver decomposition, scenario/sensitivity analysis, and reconciliation to multiples and market values.

See also: Chapter 14 on continuing value (terminal sensitivity); Chapter 15 on WACC (discount-rate sensitivity); Chapter 18 on multiples (market cross-checks).

---

### Core Concept: Decompose Value into Drivers (Growth, ROIC, and Competitive Fade)
A valuation is a structured set of assumptions about:
- **Profitability** (margins → NOPAT)
- **Capital intensity** (reinvestment → invested capital and working capital)
- **Growth** (revenue/volume/pricing)
- **Risk** (WACC)
- **Long-run economics** (terminal g and steady-state ROIC)

The objective is to identify which assumptions are both (1) material to value and (2) defensible given competitive position.

---

### Formula/Methodology

#### 1) Enterprise Value from DCF
```text
Enterprise Value = PV(FCFF in explicit forecast) + PV(Terminal Value)
```
Where:
- PV uses WACC as the discount rate for FCFF

#### 2) Terminal Value share (diagnostic)
```text
Terminal Value Share = PV(Terminal Value) / Enterprise Value
```

#### 3) Sensitivity (one-variable / two-variable)
```text
Sensitivity to input X = (Value_high − Value_low) / (X_high − X_low)
```
Use practical grids for:
- WACC vs g (terminal value sensitivity)
- WACC vs margin
- Growth vs reinvestment intensity (turnover)

#### 4) Implied expectations (reverse engineering)
Common reverse checks:
- Implied exit multiple from perpetuity TV:
```text
Implied EV/EBITDA = TV_Gordon / EBITDA_N
```
- Implied ROIC / reinvestment needed for growth:
```text
Reinvestment Rate ≈ g / ROIC   (steady-state simplification)
```
- Implied “value of growth” logic:
```text
If ROIC ≈ WACC, incremental growth adds little value (value ~ no-growth)
If ROIC > WACC, growth adds value proportional to spread and scale
```

---

### Python Implementation
```python
from typing import List, Dict, Tuple, Optional

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def pv(value: float, rate: float, t: int) -> float:
    """Present value of a single cash flow."""
    if not _num(value):
        raise ValueError("value must be numeric and not NaN")
    if not _num(rate) or rate <= 0:
        raise ValueError("rate must be numeric and > 0")
    if not isinstance(t, int) or t < 0:
        raise ValueError("t must be an int >= 0")
    return value / ((1.0 + rate) ** t)


def pv_cashflows(cashflows: List[float], rate: float, t0: int = 1) -> float:
    """
    PV of a list of cash flows occurring at t=t0..t0+len-1.
    """
    if not isinstance(cashflows, list) or len(cashflows) == 0:
        raise ValueError("cashflows must be a non-empty list")
    if not _num(rate) or rate <= 0:
        raise ValueError("rate must be numeric and > 0")
    if not isinstance(t0, int) or t0 < 0:
        raise ValueError("t0 must be an int >= 0")
    total = 0.0
    for i, cf in enumerate(cashflows):
        if not _num(cf):
            raise ValueError("cashflow values must be numeric and not NaN")
        total += pv(cf, rate, t0 + i)
    return total


def terminal_value_gordon(fcff_n: float, wacc: float, g: float) -> float:
    """TV_N = FCFF_{N+1} / (WACC - g), where FCFF_{N+1} = FCFF_N*(1+g)."""
    if not _num(fcff_n) or not _num(wacc) or not _num(g):
        raise ValueError("inputs must be numeric and not NaN")
    if wacc <= 0:
        raise ValueError("wacc must be > 0")
    if wacc <= g:
        raise ValueError("wacc must be greater than g")
    fcff_n1 = fcff_n * (1.0 + g)
    return fcff_n1 / (wacc - g)


def enterprise_value_dcf(fcff: List[float], wacc: float, g: float) -> Dict[str, float]:
    """
    Compute EV as PV(explicit FCFF) + PV(TV).
    Assumes fcff list is years 1..N and terminal value is at year N.
    """
    if not isinstance(fcff, list) or len(fcff) == 0:
        raise ValueError("fcff must be a non-empty list")
    n = len(fcff)
    pv_explicit = pv_cashflows(fcff, wacc, t0=1)
    tv_n = terminal_value_gordon(fcff[-1], wacc, g)
    pv_tv = pv(tv_n, wacc, n)
    ev = pv_explicit + pv_tv
    share = pv_tv / ev if ev > 0 else 0.0
    return {"pv_explicit": pv_explicit, "tv_n": tv_n, "pv_tv": pv_tv, "ev": ev, "tv_share": share}


def sensitivity_grid_wacc_g(fcff: List[float], wacc_list: List[float], g_list: List[float]) -> List[Tuple[float, float, float]]:
    """Return (wacc, g, ev) for valid combinations."""
    out = []
    for w in wacc_list:
        for g in g_list:
            if w <= g:
                continue
            ev = enterprise_value_dcf(fcff, w, g)["ev"]
            out.append((w, g, ev))
    return out


def implied_exit_multiple(tv_n: float, ebitda_n: float) -> Optional[float]:
    """Implied EV/EBITDA = TV_N / EBITDA_N. Returns None if EBITDA_N <= 0."""
    if not _num(tv_n) or not _num(ebitda_n):
        raise ValueError("inputs must be numeric")
    if ebitda_n <= 0:
        return None
    return tv_n / ebitda_n


# Example usage
fcff = [280_000_000, 330_000_000, 380_000_000, 430_000_000, 480_000_000]  # years 1..5
wacc = 0.09
g = 0.03

res = enterprise_value_dcf(fcff, wacc, g)
print(f"EV: ${res['ev']/1e9:.2f}B")
print(f"PV(explicit): ${res['pv_explicit']/1e9:.2f}B")
print(f"PV(TV): ${res['pv_tv']/1e9:.2f}B (TV share {res['tv_share']:.1%})")

ebitda_n = 760_000_000
im_mult = implied_exit_multiple(res["tv_n"], ebitda_n)
print(f"Implied EV/EBITDA at terminal: {im_mult:.1f}x" if im_mult else "Implied multiple: n/a")
```

---

### Valuation Impact
Why this matters:
- Many valuation “errors” are not spreadsheet errors but interpretation errors: failing to see that value is coming from an aggressive terminal assumption, an implausibly high ROIC spread, or a low WACC.
- Investor and IC-style review requires articulating which assumptions drive value, how realistic they are, and what must be true for the valuation to be achieved.

Impact on multiples:
- DCF implies an exit multiple. If it is inconsistent with mature peers, either terminal assumptions are wrong or the peer multiple is not comparable.
- If EV/EBITDA implied is high, check for (a) high g, (b) high terminal margins, (c) low reinvestment intensity, or (d) low WACC.

Impact on DCF inputs:
- If value is highly sensitive to WACC or g, widen scenario ranges and avoid over-precision.
- If value is highly sensitive to margin, reassess competitive position and cost structure realism.

Practical adjustments:
```python
def normalize_value_inputs(base: Dict[str, float], overrides: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of base inputs with overrides applied (simple scenario builder)."""
    out = dict(base)
    for k, v in overrides.items():
        if not _num(v):
            raise ValueError("override values must be numeric")
        out[k] = v
    return out
```

---

### Quality of Earnings / Model Review Flags
⚠️ Value is dominated by terminal value share with a short explicit forecast (TV share extremely high without justification).  
⚠️ Terminal assumptions imply an exit multiple far above mature peers with no moat explanation.  
⚠️ Forecast margins rise while reinvestment falls (inconsistent with competitive pressure and growth economics).  
⚠️ Working capital or capex assumptions embed one-time benefits as if permanent.  
⚠️ WACC set below reasonable peer range (Rf/ERP/beta/spread not supportable).  
✅ Key drivers are explicitly tied to competitive position evidence and validated with cross-checks.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Cyclicals | Mid-cycle normalisation | Build scenarios (peak/mid/trough), evaluate through-cycle value and terminal assumptions |
| High growth / platforms | Fade and reinvestment realism | Use explicit fade paths for growth and margins; validate reinvestment (R&D/S&M) not understated |
| Asset-heavy (manufacturing, utilities) | Capital intensity dominates | Stress test capex and working capital assumptions; validate terminal turnover |
| Retail / airlines | Leases and fixed costs | Ensure lease-consistent EV, EBITDA, and debt-like items in bridge |
| Financials | DCF/EV less applicable | Use equity-based frameworks and reconcile to P/TB, residual income, ROE sustainability |

---

### Practical Review Tables (Use in Valuation Write-Up)

#### 1) Driver Summary (What is the valuation assuming?)
| Driver | Forecast assumption | Terminal assumption | Evidence to support | Key risk |
|---|---|---|---|---|
| Revenue growth |  |  |  |  |
| Operating/NOPAT margin |  |  |  |  |
| Reinvestment intensity |  |  |  |  |
| ROIC vs WACC spread |  |  |  |  |
| WACC |  |  |  |  |
| Terminal g |  |  |  |  |

#### 2) Sensitivity Grid (Template)
|  | g low | g base | g high |
|---|---:|---:|---:|
| WACC low |  |  |  |
| WACC base |  |  |  |
| WACC high |  |  |  |

---

### Real-World Example (Sensitivity Grid + Narrative)

Scenario: Create a WACC vs g sensitivity grid and interpret where risk is concentrated.

```python
fcff = [280_000_000, 330_000_000, 380_000_000, 430_000_000, 480_000_000]
wacc_list = [0.08, 0.09, 0.10]
g_list = [0.02, 0.03, 0.04]

grid = sensitivity_grid_wacc_g(fcff, wacc_list, g_list)
for w, g, ev in grid:
    print(f"WACC={w:.1%}, g={g:.1%} -> EV=${ev/1e9:.2f}B")
```

Interpretation:
- If EV swings materially across small changes in WACC or g, the valuation is terminal-driven; extend the explicit forecast and tighten steady-state assumptions.
- If EV is more sensitive to WACC than g, focus on capital structure, beta selection, and credit spread evidence.
- If EV is more sensitive to g than WACC, validate long-run growth economics via ROIC and reinvestment consistency checks.
