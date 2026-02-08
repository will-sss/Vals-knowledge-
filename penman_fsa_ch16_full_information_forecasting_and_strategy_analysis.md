# Financial Statement Analysis and Security Valuation (Penman, 5th ed., 2013) - Key Concepts for Valuation

## Chapter 16: Full-Information Forecasting, Valuation, and Business Strategy Analysis (Full-Information Forecasting, Strategy Drivers, Value Decomposition)

### Core Concept
Full-information forecasting ties valuation directly to **business strategy and operating drivers**: how the firm earns profits (margins), how efficiently it uses capital (turnover), and how it finances and reinvests (growth and leverage). The objective is to move from “simple anchors” to forecasts that are **internally consistent**, **economically plausible**, and **linked to competitive advantage**—so your valuation narrative and model reconcile.

---

### Formula/Methodology

#### 1) Value frameworks (equity and operations)
Equity (residual income):
```text
Equity Value_0 = Book Value_0 + PV(Residual Income)
Residual Income_t = Earnings_t - (r × Book Value_{t-1})
```
Operations (residual operating income / economic profit style):
```text
Enterprise Value_0 ≈ Net Operating Assets_0 + PV(Residual Operating Income)
Residual Operating Income_t = OPAT_t - (WACC × NOA_{t-1})
```
Where:
- OPAT = Operating profit after tax (after normalising and excluding financing items)
- NOA = Net operating assets (operating assets minus operating liabilities)
- r = cost of equity; WACC = weighted average cost of capital

Practical use:
- Use equity framing when book value and clean surplus are reliable and capital structure is stable.
- Use operations framing when leverage is changing or when you want a clean separation of operating vs financing.

#### 2) Strategy-to-numbers driver tree (operations)
At the operating level, profitability is driven by margin and turnover:
```text
RNOA = OPAT / NOA
Operating Profit Margin (OPM) = OPAT / Revenue
Net Operating Asset Turnover (NOAT) = Revenue / NOA

So: RNOA = OPM × NOAT
```
Where:
- RNOA = return on net operating assets (operating ROIC analogue)
- OPM captures pricing power/cost structure (strategy, competition)
- NOAT captures capital efficiency (asset intensity, working capital discipline)

#### 3) Growth requires investment (reinvestment link)
A core full-information discipline is linking growth to incremental investment:
```text
ΔNOA_t = NOA_t - NOA_{t-1}
Reinvestment Rate_t ≈ ΔNOA_t / OPAT_t
```
If you forecast revenue growth, you must forecast the operating capital needed to support it (working capital + fixed assets + intangibles where relevant).

#### 4) Persistence and fade of abnormal returns
Abnormal profitability is expected to mean-revert under competition:
```text
Residual Operating Income_{t+1} = ω × Residual Operating Income_t
```
Where:
- ω (omega) is a persistence factor between 0 and 1 (lower = faster fade)

You can implement fade via:
- fading margins to industry,
- fading turnover to sustainable levels,
- fading RNOA directly,
- or fading residual operating income.

#### 5) Terminal value consistency checks
If you close with a continuing value, ensure terminal assumptions are feasible:
```text
Continuing Value (operations) at T = FCF_{T+1} / (WACC - g)
```
Consistency checks:
- g must be below WACC and consistent with reinvestment needs
- implied terminal RNOA should converge to cost of capital (no perpetual supernormal returns unless justified)

---

### Practical Application (How to apply)

#### A) Full-information forecasting workflow
1) Reorganise statements into operating vs financing (NOA, NFO, OPAT, NFE) and build a clean operating P&L.
2) Build the operating driver tree:
   - forecast revenue,
   - forecast operating margin (pricing power, cost drivers),
   - forecast turnover / operating capital intensity (WC %, capex intensity, intangible capitalisation policy).
3) Convert operating drivers into NOA and OPAT paths.
4) Compute RNOA and residual operating income each period.
5) Apply fade logic for abnormal returns:
   - competitive erosion,
   - maturation,
   - mean reversion,
   - capacity constraints.
6) Close with a continuing value consistent with mature economics (RNOA → WACC, stable reinvestment).
7) Cross-check:
   - implied multiples (EV/EBITDA, EV/Revenue),
   - implied terminal margins/turnover,
   - implied capital intensity,
   - accounting consistency (clean surplus / operating-financing separation).

#### B) “Strategy analysis” mapping (qual → quant)
Use this table to force a translation from narrative to forecast inputs:

| Strategy claim | Model implication | What to forecast | Evidence to look for |
|---|---|---|---|
| Pricing power / differentiated product | higher sustainable margin, slower fade | OPM level + fade rate | gross margin stability, churn, NPS, competitive structure |
| Scale economies | margin expansion with growth | OPM increases as revenue grows | fixed cost leverage, unit economics |
| Platform / network effects | persistence in abnormal returns | ω higher (slower fade), higher terminal margins | retention, cohort economics |
| Capital-light model | higher turnover, lower reinvestment | higher NOAT, lower ΔNOA per revenue | WC %, capex/revenue, opex capitalization policy |
| Regulated returns | cap on abnormal returns | RNOA anchored near allowed return | regulatory filings, allowed ROE |

#### C) Common modelling mistakes this chapter helps avoid
- Forecasting growth without matching investment (NOA stays flat while revenue grows).
- Leaving margins permanently above industry without explaining barriers to entry.
- Using terminal growth with unrealistic capital intensity or persistent abnormal returns.
- Mixing operating and financing items (interest, pension finance costs) into OPAT.

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

def safe_div(n: float, d: float, name: str) -> float:
    n = _f(n, f"{name}_numerator")
    d = _f(d, f"{name}_denominator")
    if d == 0:
        raise ValueError(f"{name}: denominator is zero.")
    return float(n / d)

def compute_operating_drivers(opat: float, revenue: float, noa_beg: float) -> Dict[str, float]:
    """Compute operating drivers used in full-information forecasting.

    Args:
        opat: operating profit after tax ($)
        revenue: revenue ($)
        noa_beg: beginning-of-period net operating assets ($)

    Returns:
        dict: OPM, NOAT, RNOA

    Raises:
        ValueError: invalid inputs / zero denominators
    """
    opat = _f(opat, "opat")
    revenue = _f(revenue, "revenue")
    noa_beg = _f(noa_beg, "noa_beg")

    if revenue == 0:
        raise ValueError("revenue is zero; cannot compute margin or turnover.")
    if noa_beg == 0:
        raise ValueError("noa_beg is zero; cannot compute turnover or RNOA.")

    opm = opat / revenue
    noat = revenue / noa_beg
    rnoa = opat / noa_beg
    return {"opm": float(opm), "noat": float(noat), "rnoa": float(rnoa)}

def residual_operating_income(opat: float, noa_beg: float, wacc: float) -> float:
    """Residual operating income: ReOI = OPAT - WACC * NOA_{t-1}."""
    opat = _f(opat, "opat")
    noa_beg = _f(noa_beg, "noa_beg")
    wacc = _f(wacc, "wacc")
    if wacc <= 0 or wacc > 0.5:
        raise ValueError("wacc should be a positive decimal (e.g., 0.09).")
    return float(opat - wacc * noa_beg)

def fade_value(current: float, target: float, year: int, total_years: int, method: str = "linear") -> float:
    """Fade a metric from current to target over an explicit horizon."""
    current = _f(current, "current")
    target = _f(target, "target")
    year = int(year)
    total_years = int(total_years)
    if total_years <= 0:
        raise ValueError("total_years must be > 0.")
    if year < 1 or year > total_years:
        raise ValueError("year must be in 1..total_years.")
    if method == "linear":
        w = year / total_years
        return float((1 - w) * current + w * target)
    elif method == "exp":
        # Exponential fade; stronger early decay
        k = 3.0 / total_years
        w = 1 - np.exp(-k * year)
        return float((1 - w) * current + w * target)
    else:
        raise ValueError("method must be 'linear' or 'exp'.")

def forecast_full_information(
    revenue0: float,
    noa0: float,
    opm0: float,
    noat0: float,
    revenue_growth: list,
    wacc: float,
    horizon_years: int = 5,
    terminal_g: float = 0.02,
    target_opm: Optional[float] = None,
    target_noat: Optional[float] = None,
    fade_method: str = "linear"
) -> Dict[str, Any]:
    """Full-information forecast using margin + turnover, producing OPAT, NOA, ReOI, and a continuing value.

    Model:
      - Revenue_t = Revenue_{t-1} * (1 + g_t)
      - OPM_t fades toward target_opm (if provided)
      - NOAT_t fades toward target_noat (if provided)
      - OPAT_t = OPM_t * Revenue_t
      - NOA_{t-1} derived from NOAT_{t-1}: NOA_{t-1} = Revenue_{t-1} / NOAT_{t-1}
      - Residual Operating Income_t = OPAT_t - WACC * NOA_{t-1}
      - Continuing Value at T based on FCF approximation:
          FCF_{T+1} ≈ OPAT_{T+1} - ΔNOA_{T+1}

    Args:
        revenue0: starting revenue ($)
        noa0: starting net operating assets ($)
        opm0: starting operating profit margin (decimal)
        noat0: starting NOA turnover (Revenue/NOA)
        revenue_growth: list of annual growth rates for years 1..horizon
        wacc: discount rate for operations (decimal)
        horizon_years: forecast horizon length
        terminal_g: terminal growth rate (decimal)
        target_opm: optional mature margin (decimal)
        target_noat: optional mature turnover (decimal)
        fade_method: 'linear' or 'exp'

    Returns:
        dict with schedules and enterprise value estimate (operations framing)

    Raises:
        ValueError: invalid inputs or unstable terminal assumptions
    """
    revenue0 = _f(revenue0, "revenue0")
    noa0 = _f(noa0, "noa0")
    opm0 = _f(opm0, "opm0")
    noat0 = _f(noat0, "noat0")
    wacc = _f(wacc, "wacc")
    terminal_g = _f(terminal_g, "terminal_g")

    if horizon_years <= 0:
        raise ValueError("horizon_years must be > 0.")
    if len(revenue_growth) != horizon_years:
        raise ValueError("revenue_growth length must equal horizon_years.")
    if wacc <= 0 or wacc > 0.5:
        raise ValueError("wacc should be a positive decimal (e.g., 0.09).")
    if terminal_g < -0.05 or terminal_g >= wacc:
        raise ValueError("terminal_g must be < wacc and within a reasonable range.")

    # If not provided, assume fade to current (no fade)
    if target_opm is None:
        target_opm = opm0
    if target_noat is None:
        target_noat = noat0

    # Basic sanity
    if noat0 <= 0 or target_noat <= 0:
        raise ValueError("NOAT must be positive.")
    if opm0 < -1 or opm0 > 1:
        raise ValueError("opm0 should be a decimal margin (e.g., 0.15).")

    schedule = []
    pv_reoi = 0.0

    rev_prev = revenue0
    noat_prev = noat0

    for t in range(1, horizon_years + 1):
        g = _f(revenue_growth[t-1], f"g_year_{t}")
        rev = rev_prev * (1.0 + g)

        opm = fade_value(opm0, target_opm, t, horizon_years, method=fade_method)
        noat = fade_value(noat0, target_noat, t, horizon_years, method=fade_method)

        # Operating assets needed implied by turnover
        noa_beg = rev_prev / noat_prev
        opat = opm * rev
        reoi = residual_operating_income(opat, noa_beg, wacc)

        disc = (1.0 + wacc) ** t
        pv_reoi += reoi / disc

        # Update for next step
        schedule.append({
            "year": t,
            "revenue": float(rev),
            "opm": float(opm),
            "noat": float(noat),
            "noa_beg": float(noa_beg),
            "opat": float(opat),
            "reoi": float(reoi),
            "pv_reoi": float(reoi / disc),
        })

        rev_prev = rev
        noat_prev = noat

    # Terminal continuing value (simple, using FCF approximation)
    # Compute year T+1 from terminal growth and terminal drivers
    rev_T = schedule[-1]["revenue"]
    opm_T = schedule[-1]["opm"]
    noat_T = schedule[-1]["noat"]

    rev_T1 = rev_T * (1.0 + terminal_g)
    opat_T1 = opm_T * rev_T1

    # NOA_T implied by turnover at end of horizon
    noa_T = rev_T / noat_T
    noa_T1 = rev_T1 / noat_T  # assume terminal turnover stable

    delta_noa_T1 = noa_T1 - noa_T
    fcf_T1 = opat_T1 - delta_noa_T1

    cv_T = fcf_T1 / (wacc - terminal_g)
    cv_pv = cv_T / ((1.0 + wacc) ** horizon_years)

    enterprise_value = noa0 + pv_reoi + cv_pv

    return {
        "enterprise_value": float(enterprise_value),
        "pv_residual_operating_income": float(pv_reoi),
        "continuing_value_pv": float(cv_pv),
        "schedule": schedule,
        "terminal": {
            "rev_T1": float(rev_T1),
            "opat_T1": float(opat_T1),
            "noa_T": float(noa_T),
            "noa_T1": float(noa_T1),
            "delta_noa_T1": float(delta_noa_T1),
            "fcf_T1": float(fcf_T1),
            "cv_T": float(cv_T),
        }
    }

# Example usage (numbers in $m)
example = forecast_full_information(
    revenue0=5_000.0,
    noa0=3_000.0,
    opm0=0.14,
    noat0=5_000.0/3_000.0,
    revenue_growth=[0.08, 0.07, 0.06, 0.05, 0.04],
    wacc=0.09,
    horizon_years=5,
    terminal_g=0.025,
    target_opm=0.12,
    target_noat=1.8,
    fade_method="linear"
)

print(f"Enterprise value: ${example['enterprise_value']:.0f}m")
print(f"PV(ReOI): ${example['pv_residual_operating_income']:.0f}m | PV(CV): ${example['continuing_value_pv']:.0f}m")
```

---

### Valuation Impact
Why this matters:
- Forces forecasts to be **driver-based** (margin, turnover, reinvestment) and therefore harder to “hand-wave,” improving auditability.
- Produces a defensible narrative: the valuation is the mathematical expression of strategy and competitive position.

Impact on multiples:
- Higher sustainable OPM and/or higher NOAT supports higher EV/Revenue and EV/EBITDA, but only if supported by reinvestment needs and competitive persistence.
- Full-information forecasts help explain why a company should trade at a premium/discount to peers.

Impact on DCF inputs:
- Supports more robust FCF forecasts by linking growth to ΔNOA (reinvestment).
- Tightens terminal value discipline: terminal margins/turnover and reinvestment must be coherent with terminal growth.

Comparability issues:
- Different accounting (capitalised development costs, leases, provisions, pension classification) affects OPAT/NOA and thus RNOA and residual operating income.
- Always normalise accounting to maintain comparability when using operating driver models across firms.

Practical adjustments:
```python
def normalize_opat(reported_opat: float, after_tax_adjustments: float = 0.0) -> float:
    """Normalise OPAT for non-recurring items (after tax)."""
    x = float(reported_opat)
    adj = float(after_tax_adjustments)
    if not np.isfinite(x) or not np.isfinite(adj):
        raise ValueError("Inputs must be finite.")
    return x - adj
```

---

### Quality of Earnings Flags (full-information forecasting)
⚠️ Margin expansion assumed without identifiable operating leverage or pricing power.  
⚠️ Turnover improves mechanically but no evidence of asset-light shift (capex, WC, leases) in disclosures.  
⚠️ Growth driven by capitalising expenses (R&D, software) rather than cash economics.  
⚠️ Forecasts ignore working capital build and capacity capex despite strong revenue growth.  
✅ Forecast margins/turnover consistent with segment disclosures, unit economics, and historical reinvestment patterns.  
✅ Fade assumptions consistent with market structure (entry, switching costs, regulation).  

---

### Sector-Specific Considerations

| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS / subscriptions | deferred revenue and capitalised costs affect NOA | normalise operating assets; treat capitalised dev costs consistently |
| Retail / distribution | working capital dominates investment | model WC % of sales explicitly; turnover is critical |
| Industrials | capacity capex and cyclicality | tie turnover and margins to utilisation; use mid-cycle fade |
| Utilities | regulated returns | anchor margins/returns near allowed returns; terminal stability key |

---

### Real-World Example
Scenario: A firm claims a durable moat. Translate that into slower fade of margins and stable turnover, then test implied value.

```python
moat_case = forecast_full_information(
    revenue0=2_500.0,
    noa0=1_200.0,
    opm0=0.18,
    noat0=2_500.0/1_200.0,
    revenue_growth=[0.10, 0.09, 0.08, 0.07, 0.06],
    wacc=0.095,
    horizon_years=5,
    terminal_g=0.03,
    target_opm=0.16,     # moat supports higher mature margin
    target_noat=2.0,
    fade_method="exp"    # slower early fade
)

print(f"Moat-case EV: ${moat_case['enterprise_value']:.0f}m")
```

Interpretation:
- If the value is highly sensitive to terminal_g or target_opm, the “moat” assumption is doing most of the work—document evidence and stress-test.
- Cross-check implied EV/Revenue and EV/EBITDA against peers; premiums must map back to persistent margin/turnover advantages.

See also: Chapter 15 (simple forecasting anchors), Chapter 12 (profitability analysis), Chapter 13 (growth and sustainable earnings), Chapter 18 (quality of financial statements).
