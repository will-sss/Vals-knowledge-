# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 28: Operating Segments (IFRS 8, CODM view, segment KPIs, valuation by parts)

### Core Concept
IFRS 8 requires entities to disclose operating segments using the “management approach”: segments are reported based on internal reporting reviewed by the chief operating decision maker (CODM). For valuation, segment disclosures are essential for understanding business mix, drivers of ROIC and growth, and for building valuation-by-parts (sum-of-the-parts) models using segment-specific multiples and risk assumptions.

### Formula/Methodology

#### 1) Segment identification (management approach)
```text
Operating segment = a component that:
- engages in business activities generating revenues and incurring expenses,
- has results reviewed regularly by the CODM to allocate resources and assess performance,
- has discrete financial information available.

Reportable segments if they meet any quantitative threshold:
- Revenue >= 10% of combined segment revenue (external + intersegment)
- Profit or loss >= 10% of the greater of (sum of profits of profitable segments) or (sum of losses of loss-making segments)
- Assets >= 10% of combined segment assets

Entity must report enough segments so that external users understand the business.
```

#### 2) Segment measure reconciliation
```text
IFRS 8 allows segment profit/asset measures to be those used internally (may be non-IFRS measures).
Required reconciliations:
- Total segment revenues to entity revenue
- Total segment profit/loss to entity profit/loss before tax (or other specified line)
- Total segment assets to entity assets (if segment assets reported)
- Material items (e.g., interest, depreciation, non-cash, unusual items) if provided to CODM
```

#### 3) Geographic and major customer disclosures
```text
Even with one segment, disclose:
- revenues from external customers by product/service and geography (if practicable)
- non-current assets by geography (if practicable)
- major customers representing >= 10% of revenue (and which segments they relate to)
```

#### 4) Valuation-by-parts framework using segments
```text
Enterprise Value (SOTP) = Σ Segment EV_i + Non-operating assets - Net debt - Other claims

Segment EV_i:
- multiple-based: EV_i = Metric_i × Multiple_i (e.g., EBITDA_i × EV/EBITDA_i)
- DCF-based: EV_i = PV(FCFF_i) using segment-specific WACC and growth assumptions

Consistency:
- segment metrics must align with selected multiple basis (e.g., IFRS 16 effects, SBC, depreciation policy)
- allocate corporate costs consistently (or keep as “unallocated” with its own treatment)
```

---

### Practical Application for Valuation Analysts

#### 1) Build a segment map that matches CODM disclosures
Action steps:
- Extract reported segment names, revenue, profit measure, assets, capex (if disclosed), and narrative drivers.
- Identify “unallocated” items (HQ costs, corporate FX, central R&D, restructuring) and treat separately.

Analyst outputs:
- Segment bridge table (segment totals → group totals) to verify completeness and spot unusual allocations.
- Segment KPI set per segment (growth, margin, asset intensity).

#### 2) Normalise segment metrics before applying multiples
Common issues:
- Segment profit measure may be “adjusted EBIT”, “underlying EBITDA”, etc.
- Corporate allocations may change year to year (comparability risk).
Actions:
- Rebuild segment EBITDA/EBIT from disclosed components if possible.
- Apply consistent adjustments across segments (e.g., IFRS 16 lease add-back/capitalisation, SBC).
- Where segment capex is not disclosed, proxy using segment assets or management commentary; disclose assumptions.

#### 3) Use segments to avoid “blended multiple” errors
Blended group multiples can misprice conglomerates:
- High-growth segment might deserve higher EV/EBITDA than mature segment
- Loss-making segments may require revenue multiples or DCF
Actions:
- Apply segment-specific comp sets and multiples.
- Cross-check implied segment multiples against market peers.

#### 4) Segment disclosures as a quality-of-earnings lens
Watch:
- Shifts of costs into “unallocated”
- Segment boundary changes (“re-segmentation”) that rebase history
- Large intersegment revenues with unclear pricing policy
Actions:
- Track restated comparatives and ensure consistent trend analysis.
- Ask whether intersegment pricing is arm’s length and stable.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def segment_thresholds(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IFRS 8 10% quantitative thresholds to flag reportable segments.

    Expected columns:
        - 'segment'
        - 'revenue_total' (external + intersegment)
        - 'profit' (segment profit measure; can be negative)
        - 'assets' (optional but recommended)

    Returns:
        DataFrame with boolean flags: rev_10pct, profitloss_10pct, assets_10pct, reportable_any

    Raises:
        ValueError: missing required columns or invalid values.
    """
    req = {"segment", "revenue_total", "profit"}
    missing = req - set(segments.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = segments.copy()

    # Validate numerics
    if (df["revenue_total"] < 0).any():
        raise ValueError("revenue_total must be >= 0.")
    # profit can be negative; assets can be missing
    total_rev = df["revenue_total"].sum()
    if total_rev <= 0:
        raise ValueError("Total revenue_total must be > 0.")

    df["rev_10pct"] = df["revenue_total"] >= 0.10 * total_rev

    profits = df.loc[df["profit"] > 0, "profit"].sum()
    losses = df.loc[df["profit"] < 0, "profit"].abs().sum()
    base = max(profits, losses)
    # If base is zero, profit/loss test is not meaningful; set False
    if base > 0:
        df["profitloss_10pct"] = df["profit"].abs() >= 0.10 * base
    else:
        df["profitloss_10pct"] = False

    if "assets" in df.columns:
        if (df["assets"] < 0).any():
            raise ValueError("assets must be >= 0.")
        total_assets = df["assets"].sum()
        df["assets_10pct"] = df["assets"] >= 0.10 * total_assets if total_assets > 0 else False
    else:
        df["assets_10pct"] = False

    df["reportable_any"] = df[["rev_10pct", "profitloss_10pct", "assets_10pct"]].any(axis=1)
    return df

def segment_sotp_valuation(
    segments: List[Dict[str, Any]],
    multiple_field: str = "ev_multiple",
    metric_field: str = "metric",
    non_operating_assets: float = 0.0,
    net_debt: float = 0.0,
    other_claims: float = 0.0
) -> Dict[str, Any]:
    """
    Compute sum-of-the-parts equity value using segment EV = metric × multiple.

    Args:
        segments: list of dicts, each with:
            - 'name'
            - metric_field (e.g., EBITDA)
            - multiple_field (e.g., EV/EBITDA)
        multiple_field: key for multiple
        metric_field: key for metric
        non_operating_assets: add-back (currency)
        net_debt: subtract (currency; positive means debt > cash)
        other_claims: subtract (currency; pensions, minorities, provisions if treated as debt-like)

    Returns:
        dict with segment EV breakdown, enterprise value, equity value

    Raises:
        ValueError: missing fields, invalid numbers.
    """
    noa = _num(non_operating_assets, "non_operating_assets")
    nd = _num(net_debt, "net_debt")
    oc = _num(other_claims, "other_claims")

    seg_out = []
    total_ev = 0.0
    for i, s in enumerate(segments):
        if "name" not in s:
            raise ValueError(f"Segment {i} missing 'name'.")
        if metric_field not in s or multiple_field not in s:
            raise ValueError(f"Segment {s.get('name')} missing '{metric_field}' or '{multiple_field}'.")
        m = _num(s[metric_field], f"{s['name']}.{metric_field}")
        mult = _num(s[multiple_field], f"{s['name']}.{multiple_field}")
        if mult < 0:
            raise ValueError("Multiples must be >= 0.")
        ev = m * mult
        seg_out.append({"segment": s["name"], "metric": m, "multiple": mult, "segment_ev": ev})
        total_ev += ev

    enterprise_value = total_ev + noa
    equity_value = enterprise_value - nd - oc

    return {
        "segments": seg_out,
        "segments_total_ev": total_ev,
        "non_operating_assets": noa,
        "enterprise_value": enterprise_value,
        "net_debt": nd,
        "other_claims": oc,
        "equity_value": equity_value
    }

# Example usage
df = pd.DataFrame([
    {"segment": "Software", "revenue_total": 1_800, "profit": 420, "assets": 900},
    {"segment": "Hardware", "revenue_total": 1_200, "profit": 120, "assets": 1_400},
    {"segment": "Services", "revenue_total": 500, "profit": -30, "assets": 300},
    {"segment": "Corporate", "revenue_total": 0, "profit": -80, "assets": 200},
])

flags = segment_thresholds(df)
print(flags[["segment", "rev_10pct", "profitloss_10pct", "assets_10pct", "reportable_any"]])

sotp = segment_sotp_valuation(
    segments=[
        {"name": "Software", "metric": 450_000_000, "ev_multiple": 18.0},
        {"name": "Hardware", "metric": 220_000_000, "ev_multiple": 9.0},
        {"name": "Services", "metric": 80_000_000, "ev_multiple": 10.0},
    ],
    non_operating_assets=150_000_000,
    net_debt=1_200_000_000,
    other_claims=250_000_000
)
print(f"Equity value (SOTP): ${sotp['equity_value']/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- Segment data enables more accurate valuation when businesses have different risk/return profiles, growth rates, and capital intensity.
- It helps avoid misleading conclusions from group-level averages (blended margins and blended multiples).
- Segment disclosures help validate forecast assumptions and identify hidden concentration risks (single customers, geographies).

Impact on multiples:
- EV/EBITDA: segment-specific multiples often vary widely; use segment EBITDA definitions consistently.
- P/E: corporate/unallocated costs can distort group earnings; isolate segment contributions.

Impact on DCF inputs:
- Use segment-specific growth/margin trajectories and potentially different discount rates when risk differs materially.
- Segment reinvestment needs (capex and working capital) can vary; avoid applying a single group reinvestment rate.

Comparability issues:
- IFRS 8 uses management measures, which may differ across firms; reconcile and normalise.
- Segment boundary changes can rebase history and create artificial growth/margin trends.

Practical adjustments:
```python
def allocate_corporate_costs(segments_revenue: Dict[str, float], corporate_cost: float) -> Dict[str, float]:
    """
    Allocate corporate costs by revenue share (simple allocator).

    Args:
        segments_revenue: dict of segment -> revenue
        corporate_cost: total corporate cost (positive number to allocate)

    Returns:
        dict segment -> allocated cost

    Raises:
        ValueError: invalid inputs.
    """
    cc = _num(corporate_cost, "corporate_cost")
    if cc < 0:
        raise ValueError("corporate_cost must be >= 0.")
    total = sum(_num(v, f"revenue[{k}]") for k, v in segments_revenue.items())
    if total <= 0:
        raise ValueError("Total segment revenue must be > 0.")
    return {k: cc * (v / total) for k, v in segments_revenue.items()}
```

---

### Quality of Earnings Flags
⚠️ Frequent changes in segment definitions without clear restatement of comparatives (trend distortion).  
⚠️ Growing “unallocated” costs that depress group earnings while segments look stable (cost shifting).  
⚠️ Significant intersegment revenues with unclear pricing (margin management risk).  
⚠️ Segment profit measures exclude recurring costs (e.g., SBC, central R&D) without transparent reconciliation.  
✅ Clear reconciliations, stable segment definitions, and disclosure of segment assets/capex to assess capital intensity.

---

### Sector-Specific Considerations

| Sector | Segment disclosure focus | Typical valuation handling |
|---|---|---|
| Conglomerates | business line multiples | SOTP with separate comp sets; corporate cost treatment explicit |
| Banks | geography/product segments and risk-weighted assets | segment ROE and credit costs; avoid EV/EBITDA; use P/B and residual income |
| Energy | upstream/midstream/downstream | different commodity exposure; segment DCFs or separate multiples |
| Software groups | ARR/NRR by product lines | segment growth + margin; revenue multiple by cohort if EBITDA negative |

---

### Real-World Example
Scenario: Group reports 3 segments. You want a quick SOTP and then compute implied group EV/EBITDA to cross-check.

```python
segments = [
    {"name": "A", "metric": 300_000_000, "ev_multiple": 12.0},
    {"name": "B", "metric": 150_000_000, "ev_multiple": 8.0},
    {"name": "C", "metric": 50_000_000, "ev_multiple": 6.0},
]
sotp = segment_sotp_valuation(segments, non_operating_assets=100_000_000, net_debt=900_000_000, other_claims=0)

group_ev = sotp["enterprise_value"]
group_ebitda = sum(s["metric"] for s in segments)
implied_ev_ebitda = group_ev / group_ebitda if group_ebitda > 0 else float("nan")

print(f"Group EV: ${group_ev/1e9:.2f}B")
print(f"Implied group EV/EBITDA: {implied_ev_ebitda:.1f}x")
```

Interpretation: If the implied group multiple is far from observed peer multiples, revisit segment multiples, corporate adjustments, and net debt/claims.

See also: Chapter 18 (provisions) for unallocated corporate items; Chapter 24 (financial instruments) if segment profit includes FVTPL movements; Chapter 20 (revenue) for segment revenue recognition differences; Chapter 16 (equity) for minority interests that matter in SOTP bridges.
