# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 21: Government Grants (IAS 20, grant income classification, EBITDA comparability, cash flow timing)

### Core Concept
IAS 20 governs recognition and presentation of government grants and other government assistance, generally recognising grants when there is reasonable assurance that conditions will be complied with and the grant will be received. For valuation, grants can materially affect reported margins (often embedded within operating profit), distort comparables across jurisdictions, and create non-recurring earnings uplift. Correct handling requires separating sustainable operating economics from policy-driven support and mapping grant accounting to cash.

### Formula/Methodology

#### 1) Recognition threshold (IAS 20)
```text
Recognise a government grant only when:
1) Reasonable assurance the entity will comply with grant conditions, AND
2) Reasonable assurance the grant will be received

If not met: do not recognise (disclose where relevant).
```

#### 2) Grants related to income (P&L support)
```text
Recognise in profit or loss over the periods in which the entity recognises the related costs the grant is intended to compensate.

Presentation options (policy choice):
A) Present as “other income” (gross)
B) Deduct from related expense (net)

Key: the economics are the same; classification affects EBITDA and margin comparability.
```

#### 3) Grants related to assets (capex support)
```text
Two common presentations:
A) Deferred income approach:
   - Recognise grant as deferred income (liability)
   - Recognise in P&L systematically over the asset’s useful life (often straight-line)
B) Deduction approach:
   - Deduct grant from carrying amount of the asset
   - Lower depreciation expense over the asset life

Depreciation interaction (straight-line):
Annual depreciation (deduction approach) = (Cost - Grant) / Useful life
Annual grant income (deferred income approach) ≈ Grant / Useful life
```

#### 4) Repayable grants
```text
If grant becomes repayable (conditions breached):
- Income grants: recognise repayment in P&L (often as increase in expense / reduction of grant income)
- Asset grants: adjust deferred income or asset carrying amount and depreciation prospectively (depending on approach)
```

---

### Practical Application for Valuation Analysts

#### 1) Identify whether grants are sustainable or transitory
Common grant types:
- Wage subsidies / training grants (income-related)
- Energy/green subsidies, R&D credits (can be recurring but policy-dependent)
- Capex grants (asset-related; usually project-specific)
Analyst actions:
- Split grants into “core recurring” vs “non-core / policy” buckets.
- Check whether grant is contractually multi-year or one-off.
- Benchmark grant dependency: grant income as % of EBITDA or % of operating profit.

#### 2) Standardise presentation for peer comparability
Because IAS 20 allows both gross and net presentation, peers may report:
- Higher EBITDA (if grant netted against expense)
- Higher “other income” (if gross)
Practical normalization:
- Reconstruct an “EBITDA before grants” and “EBITDA after grants” view.
- For EV/EBITDA comps, prefer either:
  - include grants consistently across all comps if sustainable and operating in nature, or
  - exclude grants for a conservative “unsubsidised” operating multiple.

#### 3) Map accounting to cash flow
Grants may be received upfront or over time:
- Upfront cash with deferred income recognised in P&L later (asset grants)
- Accrued grant receivable if earned but not received
Valuation actions:
- Model cash receipts in operating or investing cash flows based on facts (disclosure often shows classification).
- Avoid double counting: if FCF is built from EBITDA and working capital, ensure grant receivable movements are captured.

#### 4) Watch for conditionality and clawbacks
Grants often have conditions (employment targets, capex completion, performance metrics).
Valuation actions:
- Check contingent liabilities/repayment risk (see IAS 37 chapter).
- Stress test scenarios where grant is reduced or repaid.

---

### Python Implementation
```python
from typing import Any, Dict, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def asset_grant_deferred_income_schedule(grant: float, useful_life_years: float) -> Dict[str, float]:
    """
    Straight-line recognition of an asset-related grant using deferred income approach.

    Args:
        grant: grant amount received (cash) or receivable
        useful_life_years: asset useful life in years

    Returns:
        dict: {'annual_grant_income': ..., 'annual_deferred_income_release': ...}

    Raises:
        ValueError: invalid inputs.
    """
    g = _num(grant, "grant")
    life = _num(useful_life_years, "useful_life_years")
    if g < 0:
        raise ValueError("grant must be >= 0.")
    if life <= 0:
        raise ValueError("useful_life_years must be > 0.")
    annual = g / life
    return {"annual_grant_income": annual, "annual_deferred_income_release": annual}

def asset_grant_deduction_depreciation(asset_cost: float, grant: float, useful_life_years: float) -> float:
    """
    Annual straight-line depreciation when the grant is deducted from the asset carrying amount.

    Args:
        asset_cost: gross capex cost
        grant: grant amount
        useful_life_years: asset life

    Returns:
        float: annual depreciation expense

    Raises:
        ValueError: invalid inputs or grant > cost.
    """
    cost = _num(asset_cost, "asset_cost")
    g = _num(grant, "grant")
    life = _num(useful_life_years, "useful_life_years")
    if cost <= 0:
        raise ValueError("asset_cost must be > 0.")
    if g < 0:
        raise ValueError("grant must be >= 0.")
    if g > cost:
        raise ValueError("grant cannot exceed asset_cost.")
    if life <= 0:
        raise ValueError("useful_life_years must be > 0.")
    return (cost - g) / life

def normalize_ebitda_for_grants(reported_ebitda: float, grant_income_included: float, policy: str = "exclude") -> float:
    """
    Create a comparable EBITDA metric by optionally removing grant effects.

    Args:
        reported_ebitda: as reported
        grant_income_included: grant benefit included in EBITDA (positive if increases EBITDA)
        policy: 'exclude' to remove grants, 'include' to keep as reported

    Returns:
        float: normalized EBITDA

    Raises:
        ValueError: invalid policy.
    """
    e = _num(reported_ebitda, "reported_ebitda")
    g = _num(grant_income_included, "grant_income_included")
    if policy not in {"exclude", "include"}:
        raise ValueError("policy must be 'exclude' or 'include'.")
    return e - g if policy == "exclude" else e

def grant_dependency(grant_income: float, ebitda: float) -> float:
    """
    Grant dependency ratio = grant income / EBITDA.

    Returns:
        float ratio (can exceed 1 if EBITDA small)

    Raises:
        ValueError: EBITDA = 0.
    """
    g = _num(grant_income, "grant_income")
    e = _num(ebitda, "ebitda")
    if e == 0:
        raise ValueError("ebitda must be non-zero to compute dependency.")
    return g / e

# Example usage
asset_cost = 100_000_000
grant = 20_000_000
life = 10

sched = asset_grant_deferred_income_schedule(grant, life)
dep = asset_grant_deduction_depreciation(asset_cost, grant, life)

print(f"Annual grant income (deferred income): ${sched['annual_grant_income']/1e6:.1f}M")
print(f"Annual depreciation (deduction approach): ${dep/1e6:.1f}M")

reported_ebitda = 150_000_000
grant_in_ebitda = 12_000_000  # e.g., wage subsidy netted in opex
norm_ebitda = normalize_ebitda_for_grants(reported_ebitda, grant_in_ebitda, policy="exclude")
dep_ratio = grant_dependency(grant_in_ebitda, reported_ebitda)
print(f"Normalized EBITDA (exclude grants): ${norm_ebitda/1e6:.1f}M")
print(f"Grant dependency: {dep_ratio:.1%}")
```

---

### Valuation Impact
Why this matters:
- Grants can inflate EBITDA and margins without reflecting sustainable competitive advantage; ignoring this can overvalue the business using multiples or DCF.
- Presentation choices (gross vs net) can make peers appear more or less profitable.
- Asset grants can reduce capex burden and change reinvestment rates, affecting ROIC and FCF.

Impact on multiples:
- EV/EBITDA: decide whether to include grants; apply consistently across comps and disclose policy.
- EV/Revenue: typically less impacted unless grants affect net/gross presentation indirectly.

Impact on DCF inputs:
- Forecast operating margins with and without grants (base vs downside case).
- Model grant cash receipts explicitly where material; treat as operating/investing based on facts and consistency.
- Consider terminal value sustainability: many grants are time-limited.

Practical adjustments:
```python
def valuation_adjustment_for_grants(enterprise_value: float, annual_grant_income: float, multiple: float, assume_non_recurring: bool = True) -> float:
    """
    Simple EV adjustment if you decide grant income is non-recurring and embedded in EBITDA multiple valuation.

    Args:
        enterprise_value: EV implied from reported EBITDA * multiple
        annual_grant_income: grant benefit embedded in EBITDA
        multiple: EV/EBITDA multiple used
        assume_non_recurring: if True, remove grant * multiple from EV

    Returns:
        float: adjusted EV

    Raises:
        ValueError: multiple negative.
    """
    ev = _num(enterprise_value, "enterprise_value")
    g = _num(annual_grant_income, "annual_grant_income")
    m = _num(multiple, "multiple")
    if m < 0:
        raise ValueError("multiple must be >= 0.")
    if not assume_non_recurring:
        return ev
    return ev - (g * m)
```

---

### Quality of Earnings Flags
⚠️ Grant income materially boosts EBITDA but is described as “operating performance” without disclosure.  
⚠️ Aggressive recognition before conditions are clearly met (recognition threshold not satisfied).  
⚠️ Large grant receivable build-up with slow collections (cash risk).  
⚠️ Repeated grant clawbacks/repayments or covenant breaches triggering repayment.  
✅ Clear disclosures of grant terms, conditions, timing, and consistent presentation policy over time.

---

### Sector-Specific Considerations

| Sector | Typical grant type | Key valuation handling |
|---|---|---|
| Energy / Renewables | feed-in tariffs, capex subsidies | Model policy risk; scenario-based margin/IRR; treat as potentially non-terminal |
| Manufacturing | capex grants, training/wage subsidies | Separate project-specific from recurring; normalise EBITDA for comparables |
| R&D / Tech | R&D incentives, innovation grants | Evaluate persistence and eligibility; ensure not double-counted with capitalised dev costs |
| Agriculture | subsidy income | Often recurring but policy-driven; consider long-run sustainability and stress tests |

---

### Real-World Example
Scenario: Company reports $220m EBITDA and receives $30m annual wage subsidy that is netted in operating expenses. You’re using a 9.0x EV/EBITDA multiple.

```python
reported_ebitda = 220_000_000
grant_benefit = 30_000_000
multiple = 9.0

ev_reported = reported_ebitda * multiple
ev_adj = valuation_adjustment_for_grants(ev_reported, grant_benefit, multiple, assume_non_recurring=True)

print(f"EV (reported): ${ev_reported/1e9:.2f}B")
print(f"EV (exclude non-recurring grants): ${ev_adj/1e9:.2f}B")
```

Interpretation: If the grant is likely to roll off, removing it avoids capitalising policy support into terminal enterprise value.

See also: Chapter 6 (cash flows) for grant cash classification; Chapter 7 (accounting policies) for presentation consistency; Chapter 18 (provisions) for grant repayment/clawback contingencies; Chapter 20 (revenue) for customer incentives treated as consideration payable to customers.
