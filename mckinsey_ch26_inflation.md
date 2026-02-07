# McKinsey Valuation - Key Concepts for Valuation

## Chapter 26: Inflation (Nominal vs Real, Pricing Power, Working Capital, Discount Rates, Terminal Consistency)

### Core Concept
Inflation changes valuation by affecting nominal growth, margins, working capital needs, capex, depreciation, and discount rates. The practical requirement is internal consistency: nominal cash flows must be discounted with nominal WACC, and real cash flows with real WACC. Inflation also tests competitive position via pricing power—companies that can pass through costs sustain real margins and real ROIC, while others experience margin compression and value destruction.

See also: Chapter 9 on growth (drivers and pricing/volume split); Chapter 15 on cost of capital (nominal vs real rates); Chapter 14 on continuing value (terminal g vs inflation).

---

### Core Concept: Separate “Real Growth” from “Inflation”
When forecasting:
- **Nominal revenue growth = real volume growth + price inflation (plus mix).**
- Margin stability depends on pass-through and cost structure.
- Working capital and capex often scale with nominal revenue, not real volume, so inflation can increase reinvestment needs even if real growth is flat.

Practical objective:
1) Explicitly model inflation assumptions (general and industry-specific).  
2) Translate into price, cost, and reinvestment drivers.  
3) Validate discount rate and terminal assumptions for consistency.

---

### Formula/Methodology

#### 1) Nominal vs Real Relationship (Fisher approximation)
```text
(1 + nominal_rate) ≈ (1 + real_rate) × (1 + inflation)

real_rate ≈ (1 + nominal_rate) / (1 + inflation) − 1
```

Apply to:
- Risk-free rate, cost of debt, cost of equity, WACC
- Terminal growth (g): nominal g = real g + long-run inflation (approx.)

#### 2) Revenue decomposition
```text
Revenue_t = Revenue_{t-1} × (1 + price_inflation_t) × (1 + real_volume_growth_t) × (1 + mix_effect_t)
```

#### 3) Working capital under inflation
If working capital scales with sales:
```text
NWC_t ≈ NWC_ratio × Revenue_t
ΔNWC_t ≈ NWC_ratio × (Revenue_t − Revenue_{t-1})
```

Inflation increases nominal revenue → increases nominal NWC investment even if real volume is flat.

#### 4) Terminal consistency
```text
If cash flows are nominal:
- WACC must be nominal
- Terminal growth g must be nominal and < WACC

If cash flows are real:
- WACC must be real
- Terminal growth g must be real and < WACC_real
```

---

### Python Implementation
```python
from typing import List, Optional

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def real_from_nominal(nominal: float, inflation: float) -> float:
    """
    Convert nominal rate to real rate.
    real = (1+nominal)/(1+inflation) - 1
    """
    if not _num(nominal) or not _num(inflation):
        raise ValueError("nominal and inflation must be numeric")
    if inflation <= -0.99:
        raise ValueError("inflation is implausibly low (<= -99%); check input")
    return (1.0 + nominal) / (1.0 + inflation) - 1.0


def nominal_from_real(real: float, inflation: float) -> float:
    """
    Convert real rate to nominal rate.
    nominal = (1+real)*(1+inflation) - 1
    """
    if not _num(real) or not _num(inflation):
        raise ValueError("real and inflation must be numeric")
    if inflation <= -0.99:
        raise ValueError("inflation is implausibly low (<= -99%); check input")
    return (1.0 + real) * (1.0 + inflation) - 1.0


def revenue_forecast(base_revenue: float,
                     price_inflation: List[float],
                     real_volume_growth: List[float],
                     mix: Optional[List[float]] = None) -> List[float]:
    """
    Forecast revenue with explicit price inflation and real volume growth.

    Revenue_t = Revenue_{t-1} * (1+pi) * (1+g_real) * (1+mix)

    Args:
        base_revenue: revenue at t=0 (currency, >= 0)
        price_inflation: list of annual price inflation rates (decimal)
        real_volume_growth: list of annual real volume growth rates (decimal), same length as price_inflation
        mix: optional list of mix effects (decimal), same length; if None, treated as 0.

    Returns:
        list of revenues for years 1..N

    Raises:
        ValueError: input validation errors.
    """
    if not _num(base_revenue) or base_revenue < 0:
        raise ValueError("base_revenue must be numeric and >= 0")
    if not isinstance(price_inflation, list) or not isinstance(real_volume_growth, list):
        raise ValueError("price_inflation and real_volume_growth must be lists")
    if len(price_inflation) == 0 or len(price_inflation) != len(real_volume_growth):
        raise ValueError("price_inflation and real_volume_growth must be same non-zero length")
    if mix is None:
        mix = [0.0] * len(price_inflation)
    if not isinstance(mix, list) or len(mix) != len(price_inflation):
        raise ValueError("mix must be None or a list with same length as price_inflation")

    revs = []
    r = base_revenue
    for i, (pi, gr, mx) in enumerate(zip(price_inflation, real_volume_growth, mix), start=1):
        for name, v in {"price_inflation": pi, "real_volume_growth": gr, "mix": mx}.items():
            if not _num(v):
                raise ValueError(f"Year {i} {name} must be numeric")
            if v <= -0.99:
                raise ValueError(f"Year {i} {name} is implausibly low (<= -99%)")
        r = r * (1.0 + pi) * (1.0 + gr) * (1.0 + mx)
        revs.append(r)
    return revs


def nwc_investment(revenues: List[float], nwc_ratio: float, base_revenue: float) -> List[float]:
    """
    Estimate ΔNWC each year assuming NWC = ratio * revenue.

    Args:
        revenues: forecast revenues years 1..N (currency)
        nwc_ratio: NWC as % of revenue (decimal)
        base_revenue: revenue at t=0 (currency)

    Returns:
        list of ΔNWC for years 1..N (currency)
    """
    if not isinstance(revenues, list) or len(revenues) == 0:
        raise ValueError("revenues must be a non-empty list")
    if not _num(nwc_ratio):
        raise ValueError("nwc_ratio must be numeric")
    if not _num(base_revenue) or base_revenue < 0:
        raise ValueError("base_revenue must be numeric and >= 0")
    prev = base_revenue
    out = []
    for i, r in enumerate(revenues, start=1):
        if not _num(r) or r < 0:
            raise ValueError(f"Revenue year {i} must be numeric and >= 0")
        out.append(nwc_ratio * (r - prev))
        prev = r
    return out


# Example usage: keep real volume flat but inflation rises
base_rev = 2_000_000_000
pi = [0.03, 0.04, 0.03, 0.025, 0.02]
g_real = [0.00, 0.00, 0.00, 0.00, 0.00]

revs = revenue_forecast(base_rev, pi, g_real)
d_nwc = nwc_investment(revs, nwc_ratio=0.12, base_revenue=base_rev)

print("Revenues ($B):", [round(x/1e9, 3) for x in revs])
print("ΔNWC ($M):", [round(x/1e6, 1) for x in d_nwc])

# Convert nominal WACC to real WACC using long-run inflation
nom_wacc = 0.10
long_run_infl = 0.02
real_wacc = real_from_nominal(nom_wacc, long_run_infl)
print(f"Nominal WACC: {nom_wacc:.1%} -> Real WACC: {real_wacc:.1%}")
```

---

### Valuation Impact
Why this matters:
- Inflation can inflate nominal revenue and EBITDA while simultaneously increasing working capital, capex, and discount rates; headline growth can be misleading.
- Pricing power becomes a core valuation driver: the ability to pass through inflation determines real margin stability and long-run ROIC.
- Terminal value is highly sensitive to the relationship between WACC and g; inflation assumptions flow into both.

Impact on multiples:
- EV/EBITDA can look “cheaper” or “richer” in inflationary periods if EBITDA rises nominally but interest rates/WACC rise too; compare real economics and normalise for cyclical inflation shocks.
- When comparing peers across currencies/inflation regimes, ensure consistent nominal frameworks or convert to real comparisons.

Impact on DCF inputs:
- Use nominal cash flows with nominal WACC. If you model in real terms, remove inflation from drivers and use real discount rates.
- Avoid embedding high near-term inflation into terminal growth without evidence of sustained long-run inflation and pricing power.

Practical adjustments:
```python
def check_nominal_real_consistency(cashflow_basis: str, discount_rate_is_nominal: bool) -> None:
    """
    Guardrail to prevent mixing nominal and real.
    cashflow_basis: 'nominal' or 'real'
    """
    if cashflow_basis not in {"nominal", "real"}:
        raise ValueError("cashflow_basis must be 'nominal' or 'real'")
    if cashflow_basis == "nominal" and not discount_rate_is_nominal:
        raise ValueError("Inconsistency: nominal cash flows require nominal discount rate")
    if cashflow_basis == "real" and discount_rate_is_nominal:
        raise ValueError("Inconsistency: real cash flows require real discount rate")
```

---

### Quality of Earnings Flags (Inflation / Pricing Power)
⚠️ Revenue growth is primarily price-driven but volume is declining (risk of demand destruction).  
⚠️ Gross margin compresses while management claims “full pass-through” (pricing power overstated).  
⚠️ Working capital expands materially but is explained as “temporary” without evidence (cash conversion risk).  
⚠️ Capex and maintenance costs rise with inflation but forecasts hold capex/reinvestment ratios flat (FCFF overstated).  
⚠️ Inventory accounting effects (LIFO/FIFO differences where applicable) obscure true margin trend.  
✅ Stable real margins and stable cash conversion over inflation cycles indicate durable pricing power and operational discipline.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Consumer staples | Strong pricing power, volume elasticity | Model price/volume separately; validate elasticity and competitor response |
| Retail / discretionary | Elastic demand and promotions | Stress-test volume declines and margin impacts; model markdowns |
| Industrials | Input cost pass-through lag | Model margin lag and working capital; use scenario cases |
| Utilities / regulated | Inflation indexation in tariffs | Reflect regulatory pass-through mechanisms and timing |
| Software / subscriptions | Pricing resets and churn | Model renewal uplift vs churn; avoid assuming full inflation pass-through |

---

### Practical Comparison Tables

#### 1) Inflation Consistency Checklist
| Area | Check | Why |
|---|---|---|
| Cash flow basis | Nominal vs real clearly defined | Prevents rate mismatches |
| WACC | Nominal WACC used with nominal cash flows | Avoids valuation bias |
| Terminal growth | g reflects long-run inflation + real growth | Prevents unrealistic TV |
| Margins | Pass-through assumptions evidenced | Pricing power validation |
| Reinvestment | NWC and capex scale with nominal growth | Avoid FCFF overstatement |

#### 2) Forecast Driver Decomposition
| Driver | Real component | Inflation component | Evidence |
|---|---:|---:|---|
| Revenue | Volume growth | Price inflation |  |
| COGS | Real efficiency | Input inflation |  |
| Opex | Productivity | Wage/rent inflation |  |
| Capex | Capacity needs | Replacement cost inflation |  |

---

### Real-World Example (Nominal Growth, Real Flat: Why FCFF Can Still Fall)

Scenario: A distributor with flat real volumes sees nominal revenue rise with inflation. Working capital increases with revenue, consuming cash. The DCF must reflect higher ΔNWC and potentially higher discount rates.

```python
base_rev = 2_000_000_000
pi = [0.04, 0.04, 0.03, 0.02, 0.02]
g_real = [0.00, 0.00, 0.00, 0.00, 0.00]

revs = revenue_forecast(base_rev, pi, g_real)
d_nwc = nwc_investment(revs, nwc_ratio=0.15, base_revenue=base_rev)

# Simple FCFF illustration: assume NOPAT margin stable at 6%, D&A 2%, capex 3% of revenue
nopat_margin = 0.06
da_ratio = 0.02
capex_ratio = 0.03

fcff = []
for r, dnwc in zip(revs, d_nwc):
    nopat_t = r * nopat_margin
    da_t = r * da_ratio
    capex_t = r * capex_ratio
    fcff.append(nopat_t + da_t - capex_t - dnwc)

print("FCFF ($M):", [round(x/1e6, 1) for x in fcff])
```

Interpretation:
- Even with stable margins, inflation-driven revenue growth can increase reinvestment (ΔNWC) and reduce FCFF.
- If discount rates rise with inflation, valuation can fall despite higher nominal EBITDA—driver decomposition and consistency checks prevent misleading conclusions.
