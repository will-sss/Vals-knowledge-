# McKinsey Valuation - Key Concepts for Valuation

## Chapter 19: Valuation by Parts (SOTP, Segment DCF, Conglomerate Discount, Capital Allocation)

### Core Concept
Valuation by parts (sum-of-the-parts, SOTP) values a company as the sum of its distinct businesses and assets, each using the most appropriate method (segment DCF, multiples, asset value), then reconciles to equity value. It is most useful when a firm has segments with meaningfully different growth, ROIC, risk, or capital intensity, or when non-operating assets (investments, excess cash, stakes) are material.

See also: Chapter 16 on EV-to-equity bridge (claims and non-operating assets); Chapter 18 on multiples (peer-based valuation); Chapter 31 on M&A and Chapter 32 on divestitures (structural value realisation).

---

### Core Concept: When SOTP is the Right Tool
Use SOTP when:
- Segments have different economics (e.g., high-growth software + mature services + cyclical industrial).
- Different valuation approaches are appropriate (e.g., regulated utility vs marketplace platform).
- There are material non-operating holdings (minority investments, real estate, IP, tax assets).
- Strategic actions (break-up, carve-out, spin) are plausible and segment transparency exists.

Avoid SOTP when:
- Segment reporting is too aggregated to support independent valuation.
- Shared costs, transfer pricing, and interdependencies dominate and cannot be allocated credibly.
- Capital structure and cash flows are centrally managed and cannot be segmented reliably (unless you explicitly model allocation rules).

---

### Formula/Methodology

#### 1) Sum-of-the-Parts: Enterprise Value
```text
SOTP Enterprise Value = Σ (EV_segment_i) + EV_non-operating assets (measured at fair value) − Segment-level net debt (if valued at equity)
```

Common practice:
- Value each segment on an enterprise basis (DCF on FCFF or EV multiple).
- Add non-operating assets separately (investments, excess cash, surplus property).
- Apply holdco adjustments (central costs, taxes, and debt-like items not allocated to segments).
- Subtract net debt and other debt-like claims at group level to get equity value.

#### 2) Equity Value Bridge (Group-level)
```text
Equity Value = SOTP Enterprise Value
            − Net Debt (group)
            − Other Debt-like Claims
            − Non-controlling Interests
            − Preferred Equity
            + Non-operating Assets (if not already included)
            − Holdco Adjustments (central costs, taxes, pension, provisions, etc.)
```

#### 3) Conglomerate Discount (diagnostic, not a “plug”)
```text
Conglomerate Discount = 1 − (Market EV / SOTP EV)
```
Where:
- Market EV is market-based enterprise value
- SOTP EV is analyst’s SOTP enterprise value (consistent definitions)

Use this to frame the “gap” and what would need to happen (divestitures, re-rating, improved disclosure) rather than forcing a discount mechanically without a rationale.

---

### Python Implementation
```python
from typing import Dict, Optional

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def sotp_enterprise_value(segment_evs: Dict[str, float],
                          non_operating_assets: float = 0.0,
                          holdco_adjustments_ev: float = 0.0) -> float:
    """
    Compute SOTP enterprise value.

    Args:
        segment_evs: dict of {segment_name: EV_segment} (currency)
        non_operating_assets: assets not in segment EVs to add (currency, >= 0)
        holdco_adjustments_ev: signed EV-level adjustments (currency). Negative reduces EV (e.g., PV of central costs).

    Returns:
        float: SOTP enterprise value (currency)

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(segment_evs, dict) or len(segment_evs) == 0:
        raise ValueError("segment_evs must be a non-empty dict")
    total = 0.0
    for k, v in segment_evs.items():
        if not isinstance(k, str) or k.strip() == "":
            raise ValueError("segment name must be a non-empty string")
        if not _num(v):
            raise ValueError(f"EV for segment '{k}' must be numeric and not NaN")
        total += v
    if not _num(non_operating_assets) or non_operating_assets < 0:
        raise ValueError("non_operating_assets must be numeric and >= 0")
    if not _num(holdco_adjustments_ev):
        raise ValueError("holdco_adjustments_ev must be numeric")
    return total + non_operating_assets + holdco_adjustments_ev


def equity_value_from_sotp(sotp_ev: float,
                           net_debt: float,
                           other_debt_like: float = 0.0,
                           non_controlling_interests: float = 0.0,
                           preferred_equity: float = 0.0,
                           additional_non_op_assets: float = 0.0) -> float:
    """
    Bridge from SOTP EV to equity value.

    Args:
        sotp_ev: enterprise value from SOTP (currency)
        net_debt: group net debt (currency)
        other_debt_like: leases/pensions/provisions treated as debt-like (currency, >= 0)
        non_controlling_interests: NCI (currency, >= 0)
        preferred_equity: preferred claims (currency, >= 0)
        additional_non_op_assets: non-op assets not included in sotp_ev (currency, >= 0)

    Returns:
        float: equity value (currency)
    """
    for name, v in {"sotp_ev": sotp_ev, "net_debt": net_debt}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric and not NaN")
    for name, v in {
        "other_debt_like": other_debt_like,
        "non_controlling_interests": non_controlling_interests,
        "preferred_equity": preferred_equity,
        "additional_non_op_assets": additional_non_op_assets
    }.items():
        if not _num(v) or v < 0:
            raise ValueError(f"{name} must be numeric and >= 0")
    return sotp_ev - net_debt - other_debt_like - non_controlling_interests - preferred_equity + additional_non_op_assets


def pv_central_costs(annual_cost: float, wacc: float, g: float = 0.0) -> float:
    """
    PV of perpetual central costs (holdco drag) using Gordon growth.

    PV = Cost_{1} / (WACC - g), where Cost_{1} = annual_cost*(1+g)
    Returns a NEGATIVE adjustment to EV (value-reducing).
    """
    if not _num(annual_cost) or annual_cost < 0:
        raise ValueError("annual_cost must be numeric and >= 0")
    if not _num(wacc) or wacc <= 0:
        raise ValueError("wacc must be numeric and > 0")
    if not _num(g) or g < 0:
        raise ValueError("g must be numeric and >= 0")
    if wacc <= g:
        raise ValueError("wacc must be greater than g")
    cost_1 = annual_cost * (1.0 + g)
    return - cost_1 / (wacc - g)


def conglomerate_discount(market_ev: float, sotp_ev: float) -> Optional[float]:
    """
    Conglomerate discount = 1 - (Market EV / SOTP EV). Returns None if sotp_ev <= 0.
    """
    if not _num(market_ev) or not _num(sotp_ev):
        raise ValueError("inputs must be numeric")
    if sotp_ev <= 0:
        return None
    return 1.0 - (market_ev / sotp_ev)


# Example usage
segment_evs = {
    "Software": 9_200_000_000,
    "Services": 3_100_000_000,
    "Industrial": 2_600_000_000
}

non_op_assets = 650_000_000  # e.g., equity stakes, surplus property, excess cash not in segments
holdco_drag = pv_central_costs(annual_cost=60_000_000, wacc=0.09, g=0.02)  # negative EV adjustment

sotp_ev = sotp_enterprise_value(segment_evs, non_operating_assets=non_op_assets, holdco_adjustments_ev=holdco_drag)

net_debt = 2_400_000_000
leases = 700_000_000
pension_deficit = 260_000_000
nci = 120_000_000

equity = equity_value_from_sotp(sotp_ev, net_debt=net_debt, other_debt_like=leases + pension_deficit, non_controlling_interests=nci)

market_ev = 12_000_000_000
disc = conglomerate_discount(market_ev, sotp_ev)

print(f"SOTP EV: ${sotp_ev/1e9:.2f}B")
print(f"Equity value (SOTP): ${equity/1e9:.2f}B")
print(f"Conglomerate discount: {disc:.1%}" if disc is not None else "Conglomerate discount: n/a")
```

---

### Valuation Impact
Why this matters:
- SOTP isolates where value is created/destroyed across segments (ROIC spreads, growth prospects, reinvestment needs).
- It is the valuation framework most aligned with real-world strategic actions (spin-offs, disposals, break-ups, carve-outs).
- It prevents “averaging errors” where a single group multiple misprices a mix of high-quality and low-quality businesses.

Impact on multiples:
- Group EV/EBITDA can be misleading if segment mixes differ; SOTP allows segment-specific multiples consistent with segment economics.
- Conglomerates can trade at discounts/premiums based on capital allocation, transparency, and complexity; quantify the gap but justify any discount with an action path and evidence.

Impact on DCF:
- Segment DCFs require segment-specific WACC and terminal assumptions if risks differ materially.
- Central costs and holdco items must be valued explicitly (PV of central costs, taxes, trapped cash) rather than ignored.

Practical adjustments:
```python
def allocate_central_costs(segments: Dict[str, float], total_central_cost: float, weights: Dict[str, float]) -> Dict[str, float]:
    """
    Allocate central costs across segments using provided weights (e.g., revenue or EBITDA shares).

    Returns:
        dict: {segment: allocated_cost}
    """
    if not isinstance(segments, dict) or len(segments) == 0:
        raise ValueError("segments must be a non-empty dict")
    if not _num(total_central_cost) or total_central_cost < 0:
        raise ValueError("total_central_cost must be numeric and >= 0")
    if not isinstance(weights, dict) or len(weights) != len(segments):
        raise ValueError("weights must be a dict with same keys as segments")
    w_sum = 0.0
    for k in segments.keys():
        if k not in weights:
            raise ValueError(f"Missing weight for segment {k}")
        if not _num(weights[k]) or weights[k] < 0:
            raise ValueError("weights must be numeric and >= 0")
        w_sum += weights[k]
    if w_sum <= 0:
        raise ValueError("sum of weights must be > 0")
    return {k: total_central_cost * (weights[k] / w_sum) for k in segments.keys()}
```

---

### Quality of Earnings / SOTP Red Flags
⚠️ Segment EBITDA is not comparable due to inconsistent allocation of central costs or transfer pricing.  
⚠️ Segment capex/working capital omitted, leading to overstated segment FCFF and EV.  
⚠️ Non-operating assets double-counted (included in segment EV and added again).  
⚠️ Debt-like items ignored (leases/pensions/provisions) while using EV-based segment multiples.  
⚠️ Using a “conglomerate discount” as a plug without explaining the mechanism (why it exists and how it closes).  
✅ Transparent reconciliation: segment EVs + non-op assets + holdco adjustments → SOTP EV → equity bridge.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Energy / utilities groups | Regulated vs merchant risk differs | Use segment-specific WACC and terminal assumptions; avoid single group multiple |
| Media conglomerates | IP/library assets and ad-cycle vs subscription | Value streaming/subscription separately from linear/ad businesses; model different fades |
| Industrials | Cyclical vs aftermarket/service mix | Separate mid-cycle industrial vs stable services; apply different multiples and normalisation |
| Financial + non-financial mix | EV methods differ | Value financial segments on equity frameworks (P/TB, residual income) and non-financial on EV/DCF |
| Holding companies | Portfolio stakes and holdco costs dominate | Value stakes at fair value, subtract holdco costs and taxes explicitly |

---

### Practical Comparison Tables

#### 1) SOTP Build Template (Enterprise Basis)
| Component | Valuation method | Key metric | Value (EV) |
|---|---|---:|---:|
| Segment A | DCF / EV multiple |  |  |
| Segment B | DCF / EV multiple |  |  |
| Segment C | DCF / EV multiple |  |  |
| Non-operating assets | Mark-to-market |  |  |
| Holdco adjustments | PV of central costs / taxes |  |  |
| **SOTP Enterprise Value** |  |  |  |

#### 2) Bridge to Equity (Group Level)
| Item | Treatment | Amount |
|---|---|---:|
| SOTP EV | Sum of EVs |  |
| Net debt | Subtract |  |
| Other debt-like (leases/pensions) | Subtract |  |
| Non-controlling interests | Subtract |  |
| Preferred equity | Subtract |  |
| Additional non-op assets | Add |  |
| **Equity value** |  |  |

---

### Real-World Example (Segment Multiples + Holdco Drag + Equity Bridge)

Scenario: A diversified company has three segments with different trading multiples. Apply segment multiples to normalised EBITDA, include a central cost drag, add non-operating assets, and bridge to equity value.

```python
# Segment metrics (EBITDA) and multiples
segments = {
    "Software": {"ebitda": 620_000_000, "mult": 15.0},
    "Services": {"ebitda": 410_000_000, "mult": 9.0},
    "Industrial": {"ebitda": 330_000_000, "mult": 8.0}
}

segment_evs = {k: v["ebitda"] * v["mult"] for k, v in segments.items()}

non_op_assets = 650_000_000
holdco_drag = pv_central_costs(annual_cost=60_000_000, wacc=0.09, g=0.02)

sotp_ev = sotp_enterprise_value(segment_evs, non_operating_assets=non_op_assets, holdco_adjustments_ev=holdco_drag)

equity = equity_value_from_sotp(
    sotp_ev,
    net_debt=2_400_000_000,
    other_debt_like=700_000_000 + 260_000_000,
    non_controlling_interests=120_000_000
)

print(f"SOTP EV: ${sotp_ev/1e9:.2f}B")
print(f"Equity value (SOTP): ${equity/1e9:.2f}B")
```

Interpretation:
- Segment-level valuation reveals which business drives most of the value and whether the group multiple is “averaging away” a premium segment.
- Central costs and holdco structure are explicit value drags; quantify them rather than burying them in segment margins.
- Reconcile SOTP equity value to market cap to identify the implied discount/premium and what actions would close the gap.
