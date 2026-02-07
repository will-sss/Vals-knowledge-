# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 14: Consolidations, Joint Arrangements, Associates and Separate Financial Statements (IFRS 10/11/12, IAS 28, control vs influence, EV/EBITDA comparability)

### Core Concept
This chapter covers when and how to consolidate subsidiaries (control), account for joint arrangements (joint operations vs joint ventures), and account for associates (significant influence) using the equity method. For valuation, these rules materially affect reported revenue/EBITDA, net debt, invested capital, and segment comparability—especially when peers differ in whether earnings are consolidated or presented as “share of profit.”

### Formula/Methodology

#### 1) Control test (IFRS 10) — practical summary
```text
Investor controls an investee if ALL are present:
1) Power over investee (relevant activities)
2) Exposure or rights to variable returns
3) Ability to use power to affect returns
```

#### 2) Consolidation mechanics (subsidiaries)
```text
Consolidated financials include 100% of subsidiary assets, liabilities, income, and expenses
Non-Controlling Interests (NCI) presented in equity and profit attribution
Intercompany balances and transactions eliminated on consolidation
```

#### 3) Equity method (associates and joint ventures under IFRS 11/IAS 28)
```text
Carrying Amount_end = Carrying Amount_begin + Share of Profit/Loss - Dividends received ± Other comprehensive income movements
Income statement line: “Share of profit/(loss) of associates and joint ventures” (below operating profit in many presentations)
```

#### 4) Joint arrangements (IFRS 11)
```text
Joint operation: recognise share of assets, liabilities, revenues, expenses (line-by-line, proportionate to rights/obligations)
Joint venture: equity method (single line investment + single line share of profit)
```

#### 5) Enterprise value comparability bridge (common analyst adjustments)
```text
Reported EBITDA may exclude equity-accounted investees.
For peer EV/EBITDA comparisons, you may:
A) Value associates/JVs separately and add to EV (valuation by parts), OR
B) “Gross up” EBITDA (and EV) to include proportional EBITDA and proportional net debt (only if data is available and consistent).
```

---

### Practical Application for Valuation Analysts

#### 1) Identify consolidation perimeter before using multiples
Checklist:
- Is the target reporting IFRS 10 consolidated subsidiaries? (usually yes)
- Are material profit contributors equity-accounted (associates/JVs)?
- Do peers consolidate similar activities differently (e.g., control vs significant influence)?
Action:
- Build a perimeter map: Subsidiaries (fully consolidated) vs JVs vs Associates vs structured entities.

#### 2) Multiples: avoid mixing consolidated EBITDA with “below the line” earnings
Common pitfall:
- Company A: core operations in JV → EBITDA low, share of profit high
- Company B: same operations consolidated → EBITDA higher
If you apply the same EV/EBITDA multiple to both without adjustment, you misprice one.

Preferred approaches:
- **Valuation by parts**: value the JV/associate stake separately (using its own multiples or DCF) and add to enterprise value.
- **Proportionate consolidation adjustment (analyst view)**: estimate proportional EBITDA and net debt of equity-accounted entities and adjust both numerator and denominator consistently (only if disclosures allow).

#### 3) Net debt and EV bridge: treat investments consistently
Equity-accounted investments are usually non-operating assets from the consolidated EV perspective unless you gross them up.
- If you keep consolidated EBITDA (excluding JV EBITDA), then treat JV investment as a non-operating asset: add its value to EV (or subtract from equity bridge depending on method).
- If you gross up EBITDA for proportional JV EBITDA, then you must also gross up net debt for proportional JV net debt.

#### 4) NCI: include in enterprise value, but be consistent with earnings base
Enterprise value typically includes NCI because consolidated EBITDA includes 100% of subsidiary EBITDA.
```text
EV = Market cap + Net debt + NCI + Preferred equity + Other adjustments - Non-operating assets
```
If you instead value only the parent’s share of EBITDA, then NCI treatment changes. In practice, keep consolidated EBITDA and include NCI in EV.

#### 5) Intercompany eliminations and segment KPIs
Consolidation removes internal sales and balances. For valuation:
- Use external revenue and margin
- Watch for changes in consolidation perimeter that create “growth” that is just scope change

#### 6) Structured entities and deconsolidation risk
If control is based on contractual arrangements (special purpose vehicles), changes in facts can lead to deconsolidation.
Valuation impact:
- Step-changes in EBITDA, leverage, and cash flows
- Treat as a risk factor in covenant and downside cases

---

### Python Implementation
```python
from typing import Dict, Any, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def enterprise_value(market_cap: float, net_debt: float, nci: float = 0.0, preferred_equity: float = 0.0,
                    other_adjustments: float = 0.0, non_operating_assets: float = 0.0) -> float:
    """
    Compute enterprise value with standard IFRS-consistent adjustments.

    EV = Market cap + Net debt + NCI + Preferred equity + Other adjustments - Non-operating assets

    Raises:
        ValueError: invalid inputs.
    """
    mc = _num(market_cap, "market_cap")
    nd = _num(net_debt, "net_debt")
    n = _num(nci, "nci")
    pe = _num(preferred_equity, "preferred_equity")
    oa = _num(other_adjustments, "other_adjustments")
    noa = _num(non_operating_assets, "non_operating_assets")
    return mc + nd + n + pe + oa - noa

def equity_method_carrying_amount(begin: float, share_of_profit: float, dividends: float, oci_movement: float = 0.0) -> float:
    """
    Equity method roll-forward.

    End = Begin + Share of profit - Dividends ± OCI movements

    Raises:
        ValueError: invalid inputs (dividends negative).
    """
    b = _num(begin, "begin")
    sop = _num(share_of_profit, "share_of_profit")
    div = _num(dividends, "dividends")
    oci = _num(oci_movement, "oci_movement")
    if div < 0:
        raise ValueError("dividends must be >= 0.")
    return b + sop - div + oci

def value_by_parts_ev(core_ev: float, investee_equity_value: float, ownership_pct: float, investee_net_debt: float = 0.0) -> float:
    """
    Value by parts: add value of stake in an associate/JV to enterprise value.

    Stake enterprise value proxy:
      stake_value = ownership_pct * (investee_equity_value + investee_net_debt)

    Args:
        core_ev: EV from consolidated operations (excluding JV EBITDA)
        investee_equity_value: equity value of the investee
        ownership_pct: ownership percentage (0..1)
        investee_net_debt: net debt of investee (add to get investee EV)

    Returns:
        float: total EV (core + stake EV)

    Raises:
        ValueError: invalid ownership.
    """
    cev = _num(core_ev, "core_ev")
    eqv = _num(investee_equity_value, "investee_equity_value")
    pct = _num(ownership_pct, "ownership_pct")
    nd = _num(investee_net_debt, "investee_net_debt")
    if not (0.0 <= pct <= 1.0):
        raise ValueError("ownership_pct must be between 0 and 1.")
    stake_ev = pct * (eqv + nd)
    return cev + stake_ev

def gross_up_for_equity_accounted_entities(reported_ebitda: float, reported_net_debt: float,
                                          prop_ebitda: float, prop_net_debt: float) -> Dict[str, float]:
    """
    Analyst gross-up (proportionate) for equity-accounted entities.

    Adjusted EBITDA = reported EBITDA + proportional EBITDA
    Adjusted Net debt = reported net debt + proportional net debt

    Use only if:
      - proportional metrics are available and consistent
      - you will compare to peers on the same basis

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(reported_ebitda, "reported_ebitda")
    nd = _num(reported_net_debt, "reported_net_debt")
    pe = _num(prop_ebitda, "prop_ebitda")
    pnd = _num(prop_net_debt, "prop_net_debt")
    return {"adj_ebitda": e + pe, "adj_net_debt": nd + pnd}

# Example usage
ev = enterprise_value(market_cap=5_200, net_debt=1_150, nci=180, preferred_equity=0, other_adjustments=40, non_operating_assets=120)
print(f"EV: ${ev:.0f}m")

eq_end = equity_method_carrying_amount(begin=620, share_of_profit=85, dividends=30, oci_movement=-5)
print(f"Equity-accounted investment carrying amount (end): ${eq_end:.0f}m")

total_ev = value_by_parts_ev(core_ev=6_450, investee_equity_value=2_000, ownership_pct=0.4, investee_net_debt=500)
print(f"Total EV (value by parts): ${total_ev:.0f}m")

grossed = gross_up_for_equity_accounted_entities(reported_ebitda=820, reported_net_debt=1_150, prop_ebitda=110, prop_net_debt=260)
print(grossed)
```

---

### Valuation Impact
Why this matters:
- Consolidation vs equity method can change EBITDA, leverage, and ROIC without any change in economics. If you do not adjust, your EV/EBITDA and credit metrics can be systematically biased.
- NCI must be handled consistently: consolidated EBITDA includes 100% of subsidiary EBITDA, so EV should include NCI.
- Joint arrangements can drive major “hidden” operating exposure (especially infrastructure, energy, property development); valuation by parts often produces the cleanest comparables.

Impact on multiples:
- EV/EBITDA: ensure EBITDA perimeter matches EV perimeter (include NCI; treat associates/JVs consistently).
- P/E and ROE: equity-accounted earnings flow below operating profit; be explicit whether your metric includes “share of profit” and whether the denominator includes the investee value.

Impact on DCF inputs:
- If you model consolidated free cash flows excluding JV distributions, then add JV stake value separately.
- If JV cash distributions are material, model them explicitly as non-operating cash inflows or incorporate into valuation by parts.

Practical adjustments:
```python
def adjust_ev_for_nci(ev_excluding_nci: float, nci: float) -> float:
    """
    Add NCI to EV when using consolidated EBITDA.

    Raises:
        ValueError: invalid inputs.
    """
    base = _num(ev_excluding_nci, "ev_excluding_nci")
    n = _num(nci, "nci")
    return base + n
```

---

### Quality of Earnings Flags
⚠️ Large share-of-profit from associates/JVs with limited disclosure of underlying EBITDA, debt, and cash distributions (hard to value and compare).  
⚠️ Frequent changes in consolidation perimeter creating “growth” that is actually scope change.  
⚠️ Significant off-balance-sheet obligations in unconsolidated entities (guarantees, commitments) not reflected in net debt.  
⚠️ NCI swings driven by acquisitions/disposals without clear bridge.  
✅ Clear IFRS 12 disclosures: summarised financials for material associates/JVs, ownership %, restrictions on cash transfers, commitments and contingencies.

---

### Sector-Specific Considerations

| Sector | Typical structure | Typical valuation treatment |
|---|---|---|
| Infrastructure / Energy | JVs and associates common | Valuation by parts; gross-up only with strong disclosures; include commitments |
| Real estate development | Joint ventures and SPVs | Use project-level valuations; adjust for held-for-sale and JV perimeter |
| Financial services | Structured entities and consolidation judgments | Stress consolidation risk; focus on capital, guarantees, and look-through exposures |
| Consumer / Retail | Usually consolidated subsidiaries | Watch for franchise/associate models; ensure NCI handled in EV |

---

### Real-World Example
Scenario: A company has $700m EV based on consolidated operations (excluding JV EBITDA). It owns 45% of a JV valued at $1,200m equity value with $300m net debt.

```python
core_ev = 700.0
jv_equity = 1200.0
jv_net_debt = 300.0
pct = 0.45

total_ev = value_by_parts_ev(core_ev, jv_equity, pct, jv_net_debt)
print(f"Total EV (core + JV stake): ${total_ev:.0f}m")
```

Interpretation: Treating the JV stake explicitly prevents understatement of enterprise value when consolidated EBITDA excludes JV operating earnings.

See also: Chapter 15 (business combinations) for acquisition accounting and goodwill; Chapter 28 (segments) for perimeter disclosure; Chapter 29 (related parties) for JV/associate governance and transactions; Chapter 25 (fair value) for measuring investments.
