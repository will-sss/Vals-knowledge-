# McKinsey Valuation - Key Concepts for Valuation

## Chapter 14: Estimating Continuing Value (Terminal Value, Steady-State Normalisation, Cross-Checks)

### Core Concept
Continuing value (terminal value) often drives the majority of enterprise value in a DCF, so small assumption errors can overwhelm the entire valuation. The job is to model a defensible steady state: growth consistent with reinvestment economics, margins and capital intensity consistent with competition, and discounting consistent with risk.

See also: Chapter 9 on forecasting performance (driver-based forecasts and fade); Chapter 12 on performance/ROIC drivers; Chapter 18 on multiples (market cross-checks).

---

### Core Concept: What Continuing Value Represents (A Steady-State Business)
Continuing value is the present value of cash flows beyond the explicit forecast, assuming the company reaches a stable competitive position. “Stable” does not mean no change; it means the business evolves at a sustainable rate with normalised profitability and reinvestment.

Key modelling discipline:
- Normalise the terminal-year cash flow (remove one-offs, cycle peaks/troughs, non-recurring working capital swings).
- Ensure long-run growth is economically feasible (cannot exceed the economy indefinitely; must be supported by reinvestment at plausible ROIC).
- Ensure discounting matches the cash flow definition (FCFF with WACC; FCFE with cost of equity).

---

### Formula/Methodology

#### 1) Perpetuity Growth (Gordon Growth) Method
```text
Terminal Value at year N (TV_N) = FCFF_{N+1} / (WACC − g)

Where:
FCFF_{N+1} = FCFF_N × (1 + g)
g = long-run growth rate (decimal)
```

Constraints (practical):
- g must be < WACC (otherwise TV explodes or becomes negative/infinite).
- g should be consistent with long-run nominal GDP growth for nominal models, unless there is an explicit and defensible share-gain story.

#### 2) Exit Multiple Method
```text
TV_N = Terminal Metric_N × Exit Multiple

Example:
TV_N = EBITDA_N × (EV/EBITDA)_exit
```
Exit multiples should reflect:
- A steady-state company (not a peak/trough year).
- A peer set with similar growth, ROIC, and risk at maturity.
- Consistent accounting/definitions (IFRS 16, capitalised costs, non-recurring add-backs).

#### 3) Value Driver (ROIC/Reinvestment Consistency) Identity (Sanity Check)
```text
Reinvestment Rate ≈ g / ROIC   (simplified, steady ROIC assumption)

Economic Profit view:
Enterprise Value = Invested Capital_0 + PV(Economic Profit)
```

Use this as a consistency check: high g requires either very high ROIC (low reinvestment rate) or substantial reinvestment that will reduce FCFF.

---

### Python Implementation
```python
from typing import Optional

def terminal_value_gordon(fcff_n: float, wacc: float, g: float, use_fcff_n_plus_1: bool = True) -> float:
    """
    Compute terminal value using the Gordon growth method.

    Args:
        fcff_n: FCFF in year N (currency).
        wacc: Discount rate for FCFF (decimal, > 0).
        g: Long-run growth rate (decimal).
        use_fcff_n_plus_1: If True, uses FCFF_{N+1} = FCFF_N * (1 + g).
                           If False, assumes fcff_n already represents FCFF_{N+1}.

    Returns:
        float: Terminal value at year N (currency).

    Raises:
        ValueError: If wacc <= g, wacc <= 0, or inputs not finite.
    """
    for name, v in {"fcff_n": fcff_n, "wacc": wacc, "g": g}.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must be numeric and not NaN")
    if wacc <= 0:
        raise ValueError("wacc must be > 0")
    if wacc <= g:
        raise ValueError("wacc must be greater than g for Gordon growth")
    fcff = fcff_n * (1.0 + g) if use_fcff_n_plus_1 else fcff_n
    return fcff / (wacc - g)


def terminal_value_exit_multiple(metric_n: float, exit_multiple: float) -> float:
    """
    Compute terminal value using an exit multiple.

    Args:
        metric_n: Terminal-year metric (e.g., EBITDA_N) in currency.
        exit_multiple: Exit multiple (e.g., 8.0 for 8x). Must be >= 0.

    Returns:
        float: Terminal value at year N (currency).

    Raises:
        ValueError: If inputs invalid.
    """
    for name, v in {"metric_n": metric_n, "exit_multiple": exit_multiple}.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must be numeric and not NaN")
    if exit_multiple < 0:
        raise ValueError("exit_multiple must be >= 0")
    return metric_n * exit_multiple


def implied_exit_multiple_from_gordon(fcff_n: float, ebitda_n: float, wacc: float, g: float) -> Optional[float]:
    """
    Cross-check: implied EV/EBITDA exit multiple from a Gordon terminal value.

    Returns None if EBITDA_N <= 0 (not meaningful for EV/EBITDA).
    """
    if not isinstance(ebitda_n, (int, float)) or ebitda_n != ebitda_n:
        raise ValueError("ebitda_n must be numeric and not NaN")
    if ebitda_n <= 0:
        return None
    tv = terminal_value_gordon(fcff_n, wacc, g, use_fcff_n_plus_1=True)
    return tv / ebitda_n


def reinvestment_rate_required(g: float, roic: float) -> Optional[float]:
    """
    Simplified steady-state link: Reinvestment Rate ≈ g / ROIC.

    Returns None if roic <= 0.
    """
    for name, v in {"g": g, "roic": roic}.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must be numeric and not NaN")
    if roic <= 0:
        return None
    return g / roic


# Example usage
fcff_n = 520_000_000
wacc = 0.09
g = 0.03
tv_gordon = terminal_value_gordon(fcff_n, wacc, g)

ebitda_n = 820_000_000
implied_multiple = implied_exit_multiple_from_gordon(fcff_n, ebitda_n, wacc, g)

print(f"TV (Gordon): ${tv_gordon/1e9:.2f}B")
print(f"Implied EV/EBITDA exit multiple: {implied_multiple:.1f}x" if implied_multiple else "Implied multiple: n/a")
print(f"Reinvestment rate (if ROIC=12%): {reinvestment_rate_required(g, 0.12):.1%}")
```

---

### Valuation Impact
Why this matters:
- Terminal value often contributes the majority of EV; a 0.5–1.0pp shift in g or a 0.5–1.0pp shift in WACC can move value materially.
- A terminal value method can be mathematically correct but economically wrong if the terminal-year cash flow is not normalised or if g is inconsistent with reinvestment/ROIC.
- Using exit multiples without linking to fundamentals risks importing market mispricing or peer-set definition errors into your DCF.

Impact on multiples:
- Your DCF implies an exit multiple. If the implied multiple is far above steady-state peers, the model is likely assuming overly high terminal growth, overly high terminal margins, or underestimating reinvestment/capital intensity.
- Conversely, an implied multiple far below peers can indicate overly conservative terminal assumptions or misclassification of debt-like items in EV.

Impact on DCF inputs:
- g must be consistent with reinvestment needs and ROIC; high g with low ROIC forces high reinvestment and reduces FCFF.
- Terminal-year margins should reflect competitive dynamics (often fading toward industry norms in the long run).

Practical adjustments (normalising terminal-year cash flow):
```python
def normalize_terminal_fcff(fcff_n: float,
                            cyclical_adjustment: float = 0.0,
                            one_offs: float = 0.0,
                            working_capital_reversal: float = 0.0) -> float:
    """
    Normalize terminal-year FCFF by removing non-recurring items.

    Args:
        fcff_n: reported/modelled FCFF in year N (currency)
        cyclical_adjustment: adjustment to move from peak/trough to mid-cycle (currency)
        one_offs: non-recurring cash flows embedded in FCFF (currency). Positive means FCFF is inflated.
        working_capital_reversal: remove temporary WC release/build effects (currency). Positive means FCFF inflated by WC release.

    Returns:
        float: normalized FCFF (currency)
    """
    for name, v in {"fcff_n": fcff_n, "cyclical_adjustment": cyclical_adjustment, "one_offs": one_offs, "working_capital_reversal": working_capital_reversal}.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must be numeric and not NaN")
    return fcff_n - one_offs - working_capital_reversal + cyclical_adjustment
```

---

### Quality of Earnings Flags (Terminal Value Red Flags)
⚠️ Terminal year is a peak margin year (pricing spike, temporary mix) with no fade to steady state.  
⚠️ Terminal capex materially below depreciation for long periods (underinvestment masquerading as high FCFF).  
⚠️ Terminal working capital assumes perpetual releases (e.g., receivables shrinking as % of sales indefinitely).  
⚠️ g is set above WACC, or close to WACC, producing unstable/implausible terminal values.  
⚠️ Exit multiple chosen from a high-growth peer set while terminal assumptions imply maturity/low growth.  
✅ Terminal assumptions explicitly describe a steady state: stable margins, stable reinvestment intensity, and conservative growth.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Cyclicals (industrials, materials) | Terminal year can land at peak/trough | Use mid-cycle margins and mid-cycle reinvestment; avoid terminal value based on peak earnings |
| Consumer staples | Mature growth and stable margins | Lower g; focus on ROIC stability and small reinvestment needs |
| Software / platforms | Growth fades; reinvestment often in opex | Ensure terminal margins and reinvestment reflect ongoing spend; avoid margin expansion forever |
| Retail / airlines | Leases and fixed costs affect EBITDA comparability | Ensure lease-consistent metrics; be careful with EV/EBITDA exit multiples |
| Regulated utilities | Growth linked to rate base and allowed returns | Tie g to rate base growth and allowed ROIC; use stable reinvestment rate consistent with regulation |

---

### Practical Cross-Checks (Use in Valuation Review)

#### 1) Implied Multiple Cross-Check (Gordon ↔ Exit Multiple)
```text
Implied EV/EBITDA = TV_Gordon / EBITDA_N

If implied multiple is outside a reasonable steady-state peer range:
- Check g, WACC
- Check terminal margins
- Check reinvestment assumptions and working capital stability
- Check definition consistency (leases, pensions, debt-like items)
```

#### 2) Growth vs Reinvestment Consistency Check
```text
Reinvestment Rate ≈ g / ROIC

If g is high and ROIC is modest, reinvestment rate can exceed 100% (implausible).
That signals the model is inconsistent or requires large acquisitions/capital deployment.
```

#### 3) Terminal Value Share of EV
```text
Terminal Value Share = PV(Terminal Value) / Enterprise Value

High shares are common, but extreme values signal that explicit forecast is too short
or terminal assumptions are doing too much work.
```

### Python Implementation (terminal share)
```python
def pv(value: float, rate: float, t: int) -> float:
    if not isinstance(value, (int, float)) or value != value:
        raise ValueError("value must be numeric and not NaN")
    if not isinstance(rate, (int, float)) or rate != rate or rate <= 0:
        raise ValueError("rate must be numeric and > 0")
    if not isinstance(t, int) or t < 0:
        raise ValueError("t must be an integer >= 0")
    return value / ((1.0 + rate) ** t)


def terminal_value_share(pv_explicit: float, tv_n: float, wacc: float, n: int) -> float:
    """
    Terminal value share of enterprise value: PV(TV) / (PV(explicit) + PV(TV))
    """
    if not isinstance(pv_explicit, (int, float)) or pv_explicit != pv_explicit or pv_explicit < 0:
        raise ValueError("pv_explicit must be numeric and >= 0")
    pv_tv = pv(tv_n, wacc, n)
    total = pv_explicit + pv_tv
    if total <= 0:
        raise ValueError("Total PV must be > 0")
    return pv_tv / total


# Example
pv_explicit = 2_900_000_000
tv_n = terminal_value_gordon(520_000_000, 0.09, 0.03)
share = terminal_value_share(pv_explicit, tv_n, 0.09, 5)
print(f"Terminal value share: {share:.1%}")
```

---

### Real-World Example (Build Both Terminal Methods + Cross-Check)

Scenario: You forecast 5 years of FCFF. Compute terminal value via Gordon growth and via exit multiple, then reconcile differences using implied multiples and steady-state assumptions.

```python
# Inputs (terminal year N = 5)
fcff_n = 520_000_000
ebitda_n = 820_000_000
wacc = 0.09
g = 0.03
exit_multiple = 9.0

tv_g = terminal_value_gordon(fcff_n, wacc, g, use_fcff_n_plus_1=True)
tv_m = terminal_value_exit_multiple(ebitda_n, exit_multiple)

implied_mult = implied_exit_multiple_from_gordon(fcff_n, ebitda_n, wacc, g)

print(f"TV (Gordon): ${tv_g/1e9:.2f}B")
print(f"TV (Exit Multiple): ${tv_m/1e9:.2f}B")
print(f"Implied multiple from Gordon: {implied_mult:.1f}x" if implied_mult else "Implied multiple: n/a")

# Normalise terminal FCFF and recompute
fcff_norm = normalize_terminal_fcff(fcff_n, one_offs=30_000_000, working_capital_reversal=20_000_000)
tv_g_norm = terminal_value_gordon(fcff_norm, wacc, g, use_fcff_n_plus_1=True)
print(f"TV (Gordon, normalized FCFF): ${tv_g_norm/1e9:.2f}B")
```

Interpretation:
- If exit multiple TV is much higher than Gordon TV, the exit multiple may implicitly embed higher growth or lower risk than your DCF assumes.
- If Gordon TV implies an exit multiple far above mature peers, your g may be too high, WACC too low, or terminal FCFF too optimistic.
- Normalising terminal FCFF (one-offs, working capital timing) often resolves gaps between methods.
