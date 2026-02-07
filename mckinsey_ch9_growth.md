# McKinsey Valuation - Key Concepts for Valuation

## Chapter 9: Growth (Value-Creating Growth, Growth Decomposition, Sustainability)

### Core Concept
Growth increases value only when the incremental investments required to support that growth earn returns above the cost of capital (ROIC > WACC). For high-ROIC businesses, changes in revenue (scale) can drive shareholder returns more than small changes in ROIC; for low-ROIC businesses, improving ROIC typically creates more value than pursuing growth. fileciteturn1file17

See also: Chapter 8 on ROIC persistence/decay; Chapter 31 on M&A value creation and acquisition premia. fileciteturn1file11

---

### Core Concept: When Growth Creates Value
Growth is not inherently good. It is value-creating only when the company can invest to grow at returns above its cost of capital, and when the growth mechanism is defensible (i.e., hard for competitors/customers to retaliate against). fileciteturn1file17

### Formula/Methodology
```text
Value created by growth depends on the spread (ROIC - WACC) and the scale of new invested capital required.

Economic Profit (EP) = Invested Capital × (ROIC - WACC)
Incremental Economic Profit ≈ ΔInvested Capital × (ROIC_new - WACC)
```
Where:
- Invested Capital (IC): Operating capital employed
- ROIC: Return on invested capital (decimal)
- WACC: Weighted average cost of capital (decimal)
- ΔInvested Capital: additional operating capital required to support growth

### Python Implementation
```python
from typing import Optional

def economic_profit(invested_capital: float, roic: float, wacc: float) -> float:
    """
    Calculate economic profit (value created in-period from a ROIC-WACC spread).

    Args:
        invested_capital: Operating invested capital (currency, e.g., $)
        roic: Return on invested capital (decimal, e.g., 0.12 for 12%)
        wacc: Weighted average cost of capital (decimal, e.g., 0.09 for 9%)

    Returns:
        float: Economic profit in currency units.

    Raises:
        ValueError: If invested_capital is negative or rates are not finite.
    """
    if invested_capital < 0:
        raise ValueError("invested_capital must be >= 0")
    for name, r in {"roic": roic, "wacc": wacc}.items():
        if r is None or not isinstance(r, (int, float)):
            raise ValueError(f"{name} must be a number")
        if r != r:  # NaN
            raise ValueError(f"{name} must be finite (not NaN)")
    return invested_capital * (roic - wacc)


def incremental_economic_profit(delta_invested_capital: float, roic_new: float, wacc: float) -> float:
    """
    Approximate incremental value creation from growth-related investment.

    Args:
        delta_invested_capital: Incremental invested capital required (currency, e.g., $)
        roic_new: ROIC on incremental capital (decimal)
        wacc: WACC (decimal)

    Returns:
        float: Incremental economic profit (currency).

    Raises:
        ValueError: If delta_invested_capital is negative.
    """
    if delta_invested_capital < 0:
        raise ValueError("delta_invested_capital must be >= 0")
    return economic_profit(delta_invested_capital, roic_new, wacc)


# Example usage
company = {"ic": 2_000_000_000, "roic": 0.13, "wacc": 0.09}  # $2.0B IC
ep = economic_profit(company["ic"], company["roic"], company["wacc"])
print(f"Economic profit: ${ep/1e6:,.1f}M")
```

### Valuation Impact
Why this matters:
- **DCF**: Growth assumptions must be consistent with the reinvestment needed and the expected ROIC on that reinvestment. High growth with ROIC ≤ WACC destroys value even if earnings rise.
- **Multiples**: “Growth at any cost” can expand revenue but compress EV/EBITDA or P/E if margins fall and capital intensity rises.
- **Terminal value**: Overstated long-run growth is a common valuation error; evidence suggests growth decays quickly and converges toward modest rates for most firms. fileciteturn1file11

Practical adjustments:
```python
def normalize_growth_for_value(growth_rate: float, roic: float, wacc: float) -> float:
    """
    Heuristic: penalize growth assumptions if ROIC does not exceed WACC.

    Returns an "effective growth" you can use for sanity checks (not a valuation model).
    """
    if growth_rate < -1.0:
        raise ValueError("growth_rate is implausibly low (< -100%)")
    spread = roic - wacc
    if spread <= 0:
        return 0.0  # treat value-destructive growth as no value contribution
    return growth_rate
```

### Quality of Earnings Flags
⚠️ Revenue growth accompanied by **declining ROIC** (especially below WACC) suggests value-destructive expansion. fileciteturn1file17  
⚠️ Growth driven primarily by **price increases** or **market share grabs in mature markets** may be competed away (retaliation risk). fileciteturn1file5  
✅ Growth driven by **market expansion** or **new product categories** is more defensible and can create more durable value. fileciteturn1file1  

---

### Core Concept: Decomposing Revenue Growth (Portfolio Momentum, Share, M&A)
A practical forecasting starting point is to decompose revenue growth into three components: (1) portfolio momentum (market growth exposure), (2) market share performance, and (3) M&A (inorganic growth). For large companies, portfolio momentum tends to be the dominant driver; market share gains are typically least important long-term. fileciteturn1file17

### Formula/Methodology
```text
Total Revenue Growth ≈ Organic Growth + Net M&A Contribution

Organic Growth ≈ Portfolio Momentum + Market Share Performance

Approx. decomposition (if you have segment data):
Portfolio Momentum = Σ (prior-year revenue_i × market_growth_i) / total_prior_revenue
Market Share Performance = Organic Growth - Portfolio Momentum
Net M&A Contribution = Total Growth - Organic Growth
```
Where:
- market_growth_i is growth of the market/segment i (volume + price, or real/nominal as used)
- revenue_i is the company revenue in segment i

### Python Implementation
```python
from typing import Dict, List, Optional

def cagr(start: float, end: float, years: float) -> float:
    """
    Compute compound annual growth rate.

    Args:
        start: Starting value (currency or units), must be > 0
        end: Ending value (currency or units), must be > 0
        years: Number of years, must be > 0

    Returns:
        float: CAGR as decimal.

    Raises:
        ValueError: For invalid inputs (non-positive start/end/years).
    """
    if start <= 0 or end <= 0:
        raise ValueError("start and end must be > 0 for CAGR")
    if years <= 0:
        raise ValueError("years must be > 0")
    return (end / start) ** (1 / years) - 1


def portfolio_momentum(segments: List[Dict[str, float]]) -> float:
    """
    Calculate portfolio momentum using segment weights and market growth.

    Args:
        segments: list of dicts, each containing:
            - 'rev_prior': prior-period revenue for segment (currency)
            - 'market_g': market growth rate for segment (decimal)

    Returns:
        float: Weighted-average market growth for the portfolio (decimal).

    Raises:
        ValueError: If revenues are negative or total revenue is zero.
    """
    total = 0.0
    weighted = 0.0
    for s in segments:
        rev = float(s.get("rev_prior", 0.0))
        g = float(s.get("market_g", 0.0))
        if rev < 0:
            raise ValueError("rev_prior must be >= 0 for all segments")
        total += rev
        weighted += rev * g
    if total <= 0:
        raise ValueError("Total prior revenue must be > 0")
    return weighted / total


def growth_decomposition(total_g: float, organic_g: float, portfolio_mom: float) -> Dict[str, float]:
    """
    Decompose revenue growth into portfolio momentum, share, and M&A.

    Args:
        total_g: Total reported revenue growth (decimal)
        organic_g: Organic growth (decimal) - excluding M&A where possible
        portfolio_mom: Portfolio momentum (decimal)

    Returns:
        dict: {'portfolio_momentum':..., 'market_share':..., 'net_m&a':...}

    Raises:
        ValueError: If any input is NaN.
    """
    for name, r in {"total_g": total_g, "organic_g": organic_g, "portfolio_mom": portfolio_mom}.items():
        if r != r:
            raise ValueError(f"{name} must be finite (not NaN)")
    return {
        "portfolio_momentum": portfolio_mom,
        "market_share": organic_g - portfolio_mom,
        "net_m&a": total_g - organic_g,
    }


# Example usage (segment-based)
segments = [
    {"rev_prior": 600_000_000, "market_g": 0.08},  # Segment A
    {"rev_prior": 400_000_000, "market_g": 0.02},  # Segment B
]
pm = portfolio_momentum(segments)
decomp = growth_decomposition(total_g=0.10, organic_g=0.06, portfolio_mom=pm)
print({k: f"{v:.1%}" for k, v in decomp.items()})
```

### Valuation Impact
Why this matters:
- **Forecast credibility**: Segment-level “portfolio momentum” reduces hand-waving and forces growth to tie to addressable market exposure (and highlights shrinking segments requiring divestment or reinvention). fileciteturn1file0
- **Comparability**: Two companies with the same top-line growth can have different value creation depending on whether growth comes from market tailwinds, price/share grabs, or acquisitions at a premium. fileciteturn1file5
- **M&A**: Inorganic growth often yields lower value creation than organic growth because buyers pay stand-alone value plus a takeover premium. fileciteturn1file1

Practical adjustments (normalizing for M&A “purchased growth”):
```python
def organic_revenue(reported_revenue: float, acquired_revenue: float, divested_revenue: float) -> float:
    """
    Simplified normalization for M&A effects when only revenue deltas are available.
    """
    if reported_revenue < 0:
        raise ValueError("reported_revenue must be >= 0")
    if acquired_revenue < 0 or divested_revenue < 0:
        raise ValueError("acquired_revenue/divested_revenue must be >= 0")
    return reported_revenue - acquired_revenue + divested_revenue
```

### Quality of Earnings Flags
⚠️ “Organic” growth that relies on non-recurring pricing actions or channel stuffing (watch working capital spikes vs revenue).  
⚠️ Growth via acquisitions with integration charges excluded from adjusted EBITDA (double-counting “synergies” in valuation). fileciteturn1file1  
✅ Segment disclosures that clearly reconcile reported to organic growth (FX, M&A, price/volume) improve forecast reliability.

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology / Software | Mix shift + new product categories can sustain higher growth longer, but competition can rapidly compress returns as markets mature | Model cohort/mix drivers; fade growth and margins as competitive intensity rises |
| Industrials | Growth often tied to cycles and replacement demand | Use mid-cycle demand and avoid extrapolating peak order books |
| Oil & Gas / Commodities | Growth heavily driven by price cycles and capex intensity | Separate volume vs price; stress-test under price decks; avoid “structural” growth assumptions in cyclical periods fileciteturn1file4 |
| Consumer Staples | Mature markets; share gains usually provoke retaliation | Conservative growth; focus more on ROIC and stability fileciteturn1file4 |

---

### Core Concept: A Hierarchy of Growth Scenarios (Value Creation vs Retaliation Risk)
Growth that takes revenue from “distant” or unaware competitors (market expansion/new categories) tends to be more durable than growth that directly harms incumbents (share grabs) or customers (price hikes), because retaliation is weaker. fileciteturn1file5

### Formula/Methodology
```text
Conceptual ranking (highest value potential to lowest), driven by who "loses" and how easily they retaliate:

1) Create entirely new product category (innovation)
2) Expand consumption / sell more to existing customers (category growth)
3) Enter fast-growing segments / geographies (portfolio momentum)
4) Bolt-on acquisitions (if price disciplined)
5) Market share gains (especially in mature markets)
6) Price increases (if easily substituted)
```
(Use as a qualitative checklist, not a numeric model.) fileciteturn1file1

### Python Implementation
```python
from dataclasses import dataclass
from typing import Literal

GrowthType = Literal[
    "new_category_innovation",
    "increase_usage_existing_customers",
    "enter_fast_growing_segments",
    "bolt_on_acquisition",
    "market_share_gains",
    "price_increases",
]

@dataclass(frozen=True)
class GrowthAssessment:
    growth_type: GrowthType
    retaliation_risk: str  # 'low'/'medium'/'high'
    typical_value_potential: str  # 'high'/'medium'/'low'


def assess_growth_type(growth_type: GrowthType) -> GrowthAssessment:
    """
    Simple rules-of-thumb mapping based on retaliation dynamics discussed in Chapter 9.
    """
    mapping = {
        "new_category_innovation": GrowthAssessment(growth_type, "low", "high"),
        "increase_usage_existing_customers": GrowthAssessment(growth_type, "medium", "high"),
        "enter_fast_growing_segments": GrowthAssessment(growth_type, "medium", "medium"),
        "bolt_on_acquisition": GrowthAssessment(growth_type, "medium", "medium"),
        "market_share_gains": GrowthAssessment(growth_type, "high", "low"),
        "price_increases": GrowthAssessment(growth_type, "high", "low"),
    }
    if growth_type not in mapping:
        raise ValueError(f"Unknown growth_type: {growth_type}")
    return mapping[growth_type]


print(assess_growth_type("market_share_gains"))
```

### Valuation Impact
Why this matters:
- **Moat and fade assumptions**: High-growth forecasts must align with a growth mechanism that can persist without being competed away.
- **Multiple selection**: A “high growth” peer set is not appropriate if growth is driven by price or temporary share grabs; use peers aligned to the same growth mechanism.
- **M&A modeling**: If growth is primarily inorganic, valuation must explicitly model acquisition pricing and incremental ROIC (often diluted by premia). fileciteturn1file1

### Quality of Earnings Flags
⚠️ “Growth” plans that rely on aggressive pricing while customer churn rises.  
⚠️ Share gains in mature markets paired with margin sacrifice and competitor reactions. fileciteturn1file5  
✅ Innovation-backed growth accompanied by stable/improving ROIC. fileciteturn1file1

---

### Core Concept: Sustaining Growth Is Hard (Growth Decay and Forecast Discipline)
Empirical evidence shows that high growth decays quickly and converges toward modest rates for most companies. In long-horizon valuation work, model growth “fade” and avoid extrapolating recent high growth indefinitely. fileciteturn1file11

### Formula/Methodology
```text
Revenue Growth_t = (Revenue_t / Revenue_{t-1}) - 1

A practical “fade” approach:
- Explicit near-term forecast (years 1–5) based on drivers
- Transition period (years 6–10) fading toward a sustainable rate (often low single digits)
- Continuing value assumes steady-state growth consistent with macro + reinvestment economics

Real Growth ≈ Nominal Growth - Inflation (approximation)
```
Where:
- Real vs nominal must match your discount rate convention (real vs nominal).

### Python Implementation
```python
from typing import List, Optional

def yoy_growth(series: List[float]) -> List[Optional[float]]:
    """
    Compute year-over-year growth for a series.

    Args:
        series: List of revenue values (>= 0)

    Returns:
        List[Optional[float]]: Growth rates with None for first observation and for zero denominators.

    Raises:
        ValueError: If series has fewer than 2 points or contains negative values.
    """
    if len(series) < 2:
        raise ValueError("series must contain at least 2 values")
    if any(x < 0 for x in series):
        raise ValueError("series must not contain negative values")
    out: List[Optional[float]] = [None]
    for prev, curr in zip(series[:-1], series[1:]):
        if prev == 0:
            out.append(None)
        else:
            out.append(curr / prev - 1)
    return out


def linear_fade(start_g: float, end_g: float, n_years: int) -> List[float]:
    """
    Create a linear fade path from start growth to end growth.

    Args:
        start_g: Growth in first year of fade (decimal)
        end_g: Terminal growth at end of fade (decimal)
        n_years: Number of years in fade, must be >= 1

    Returns:
        List[float]: Growth rates for each year of fade.

    Raises:
        ValueError: If n_years < 1 or rates are not finite.
    """
    if n_years < 1:
        raise ValueError("n_years must be >= 1")
    if start_g != start_g or end_g != end_g:
        raise ValueError("start_g and end_g must be finite")
    if n_years == 1:
        return [end_g]
    step = (end_g - start_g) / (n_years - 1)
    return [start_g + i * step for i in range(n_years)]


fade = linear_fade(0.12, 0.03, 6)
print([f"{g:.1%}" for g in fade])
```

### Valuation Impact
Why this matters:
- **Terminal value sensitivity**: A 1–2pp change in steady-state growth materially shifts continuing value; evidence supports conservative long-run growth for most firms. fileciteturn1file11
- **Forecast bias**: Growth expectations in guidance/analyst work can be optimistic; enforce driver-based forecasts and explicit fade. fileciteturn1file4
- **Cross-check with ROIC stability**: Growth decays faster than ROIC; do not assume that sustained high ROIC implies sustained high growth. fileciteturn1file7

Practical adjustments (sanity checks):
```python
def sustainable_growth_check(long_run_growth: float, real_gdp: float, inflation: float, is_nominal: bool = True) -> bool:
    """
    Sanity check: long-run company growth should generally not exceed long-run economy growth
    unless there is a clear share gain/market expansion story.

    Args:
        long_run_growth: assumed long-run growth (decimal)
        real_gdp: long-run real GDP growth (decimal)
        inflation: long-run inflation (decimal)
        is_nominal: if True, compare to nominal GDP ~ real_gdp + inflation

    Returns:
        bool: True if within a conservative macro-consistent band, else False.
    """
    macro = real_gdp + inflation if is_nominal else real_gdp
    return long_run_growth <= macro + 0.01  # +1pp buffer
```

### Quality of Earnings Flags
⚠️ Revenue growth supported by accruals (AR growth faster than revenue) or channel stuffing (inventory build).  
⚠️ Growth reported without clear separation of FX and M&A effects (hard to assess organic trajectory). fileciteturn1file15  
✅ Segment-level evidence of portfolio shifts into growing segments/geographies. fileciteturn1file0  

---

### Practical Benchmarks and Decision Tables

#### Empirical “Growth Reality” Benchmarks (Forecast Discipline)
Long-run analysis of large U.S.-based nonfinancial companies (real growth, smoothed with rolling averages) shows: median real revenue growth around mid-single digits; at least a quarter of companies shrinking in real terms in many years; and rapid decay of high growth toward ~5% and below over time. fileciteturn1file15

| Practical rule | How to apply in a model | What to flag |
|---|---|---|
| High growth decays quickly | Fade high near-term growth toward modest steady-state over ~5–10 years | Terminal growth that locks in high near-term growth fileciteturn1file11 |
| Segment portfolio matters | Forecast bottom-up by segment market growth × exposure | Top-down growth without segment evidence fileciteturn1file0 |
| Organic > acquisition-led (often) | Model acquisition pricing/premia explicitly | Revenue growth from acquisitions with ROIC dilution fileciteturn1file1 |

#### Growth Mechanism Checklist (Use in valuation review)
| Mechanism | Typical “loser” | Retaliation ability | Value potential (directional) |
|---|---|---:|---:|
| New category innovation | Existing solutions | Low initially | High fileciteturn1file1 |
| Expand usage / category growth | Distant industries | Low/Medium | High fileciteturn1file1 |
| Portfolio momentum | Distant industries | Low | Medium/High fileciteturn1file5 |
| Bolt-on M&A | Depends on deal price | Medium | Medium fileciteturn1file1 |
| Share gains (mature markets) | Direct competitors | High | Low fileciteturn1file5 |
| Price increases | Customers | High | Low fileciteturn1file1 |

---

### Real-World Example (Driver-Based Growth + Value Creation)

Scenario: A company is forecasting 12% revenue growth for 5 years. Check whether growth is value-creating and apply a fade to steady-state.

```python
def reinvestment_rate_from_growth(growth: float, roic: float) -> float:
    """
    Approximate reinvestment rate needed to support growth when ROIC is stable.
    Reinvestment Rate ≈ Growth / ROIC (simplified; use with caution).

    Args:
        growth: expected growth in NOPAT (decimal)
        roic: ROIC (decimal)

    Returns:
        float: reinvestment rate (decimal).

    Raises:
        ValueError: If roic <= 0 or growth < -1.
    """
    if roic <= 0:
        raise ValueError("roic must be > 0")
    if growth < -1:
        raise ValueError("growth is implausibly low (< -100%)")
    return growth / roic


company_data = {
    "ic": 1_500_000_000,   # $1.5B invested capital
    "roic": 0.14,          # 14%
    "wacc": 0.09,          # 9%
    "growth_near": 0.12,   # 12% near-term
    "growth_terminal": 0.03
}

ep_now = economic_profit(company_data["ic"], company_data["roic"], company_data["wacc"])
reinv = reinvestment_rate_from_growth(company_data["growth_near"], company_data["roic"])
fade_path = linear_fade(company_data["growth_near"], company_data["growth_terminal"], 6)

print(f"Economic profit (current): ${ep_now/1e6:,.1f}M")
print(f"Implied reinvestment rate (simplified): {reinv:.1%}")
print("Fade path:", [f"{g:.1%}" for g in fade_path])
```

Interpretation:
- Positive economic profit confirms ROIC exceeds WACC, so growth can create value if incremental projects maintain ROIC above WACC.
- If implied reinvestment is extreme (e.g., >100%), the forecast likely requires major capital deployment, acquisitions, or working-capital build—stress-test feasibility.
- The fade path operationalizes the empirical insight that sustained high growth is rare and should transition to modest steady-state rates. fileciteturn1file11
