# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 15: Business Combinations (IFRS 3, purchase price allocation, goodwill, synergies, valuation adjustments)

### Core Concept
IFRS 3 requires acquisitions that meet the definition of a business combination to be accounted for using the acquisition method: recognise identifiable assets acquired and liabilities assumed at fair value, recognise goodwill (or bargain purchase), and expense most acquisition-related costs. For valuation, IFRS 3 drives major comparability issues (EBIT/EBITDA impacted by amortisation of acquired intangibles, step-ups, and fair value adjustments), affects invested capital and ROIC, and creates quality-of-earnings signals about acquisition discipline.

### Formula/Methodology

#### 1) Acquisition method overview
```text
Step 1: Identify the acquirer
Step 2: Determine acquisition date
Step 3: Recognise and measure identifiable assets acquired and liabilities assumed at fair value
Step 4: Recognise and measure goodwill or gain from bargain purchase
```

#### 2) Goodwill calculation
```text
Goodwill = Consideration Transferred
         + NCI (at FV or proportionate share, policy choice)
         + Fair Value of previously held equity interest (if step acquisition)
         - Fair Value of identifiable net assets acquired

Where:
Identifiable Net Assets = FV(assets acquired) - FV(liabilities assumed)
```

#### 3) Consideration transferred (typical components)
```text
Consideration = Cash + FV of equity issued + FV of contingent consideration + FV of other consideration
Acquisition-related costs (deal fees): expensed (not part of consideration)
```

#### 4) Purchase price allocation (PPA) mechanics
```text
Allocate purchase consideration to identifiable assets/liabilities at fair value:
- Tangible assets: PPE, inventory step-up
- Intangibles: customer relationships, technology, brands, contracts, order backlog
- Liabilities: provisions, deferred tax arising from FV step-ups
Residual = goodwill
```

#### 5) Contingent consideration
```text
Initial recognition: FV at acquisition date (liability or equity classification)
Subsequent:
- If liability-classified: remeasure to FV through profit or loss (earnings volatility)
- If equity-classified: not remeasured (settlement within equity)
```

#### 6) Post-acquisition P&L impacts (valuation-relevant)
```text
Acquired intangible amortisation: reduces EBIT (non-cash)
Inventory step-up unwind: increases COGS temporarily post-deal
Deferred tax effects: arise from FV step-ups (temporary differences)
Goodwill: not amortised; tested for impairment under IAS 36
```

---

### Practical Application for Valuation Analysts

#### 1) Multiples: separate “operating performance” from acquisition accounting noise
Key comparability challenge:
- Company with frequent acquisitions will have more acquired-intangible amortisation and fair value remeasurements.
Actions:
- For EV/EBIT multiples, consider EBITA (EBIT before acquired intangibles amortisation) as an *analyst metric*, but remain consistent across peers.
- For EV/EBITDA, acquired intangibles amortisation is below EBITDA, but inventory step-up unwind can depress EBITDA indirectly via COGS.

Practical checklist for acquisitions:
- Identify acquired intangible amortisation and add back for “cash earnings” comparability (where appropriate).
- Identify inventory fair value step-up amortisation/unwind in COGS and adjust if material and short-lived.
- Identify contingent consideration FV remeasurement and exclude from underlying earnings if not recurring.

#### 2) Build a PPA bridge for ROIC and invested capital analysis
Acquisitions can make ROIC look worse or better mechanically:
- Invested capital increases via recognised intangibles and goodwill
- NOPAT decreases via acquired intangible amortisation
For ROIC, consider:
- ROIC including intangibles (economic capital employed)
- ROIC excluding goodwill (management efficiency on identifiable capital) as a diagnostic, not a replacement

#### 3) Deal rationale vs economic value creation
IFRS 3 captures accounting goodwill, not “economic goodwill.”
Valuation relevance:
- Accounting goodwill impairment is a lagging indicator of value destruction.
- Frequent impairments or large goodwill balances relative to enterprise value are red flags for acquisition discipline.

#### 4) Contingent consideration: treat like debt-like item when liability-classified
For enterprise value bridges:
- Liability-classified contingent consideration is debt-like (add to net debt adjustments).
For earnings:
- FV remeasurement flows through P&L; treat as non-operating/one-off unless part of core business model.

#### 5) Distinguish asset acquisitions vs business combinations
If a transaction is an asset acquisition (not IFRS 3):
- Costs are capitalised, and asset is recorded at cost allocated to components.
This changes comparability and can affect future depreciation/amortisation patterns.

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

def goodwill(
    consideration: float,
    fv_identifiable_assets: float,
    fv_liabilities: float,
    nci: float = 0.0,
    fv_prev_interest: float = 0.0
) -> float:
    """
    Compute goodwill under IFRS 3.

    Goodwill = Consideration + NCI + FV previously held interest - FV net identifiable assets
    FV net identifiable assets = FV assets - FV liabilities

    Returns:
        float: goodwill (can be negative -> bargain purchase)

    Raises:
        ValueError: invalid inputs.
    """
    cons = _num(consideration, "consideration")
    assets = _num(fv_identifiable_assets, "fv_identifiable_assets")
    liab = _num(fv_liabilities, "fv_liabilities")
    n = _num(nci, "nci")
    prev = _num(fv_prev_interest, "fv_prev_interest")
    net_assets = assets - liab
    return cons + n + prev - net_assets

def acquired_intangibles_amortisation(total_intangibles: float, weighted_avg_life_years: float, residual_value: float = 0.0) -> float:
    """
    Straight-line amortisation for acquired finite-lived intangibles (simplified).

    Args:
        total_intangibles: gross FV of finite-lived acquired intangibles
        weighted_avg_life_years: weighted average useful life in years
        residual_value: assumed residual value

    Returns:
        float: annual amortisation

    Raises:
        ValueError: invalid inputs.
    """
    t = _num(total_intangibles, "total_intangibles")
    life = _num(weighted_avg_life_years, "weighted_avg_life_years")
    rv = _num(residual_value, "residual_value")
    if t < 0:
        raise ValueError("total_intangibles must be >= 0.")
    if life <= 0:
        raise ValueError("weighted_avg_life_years must be > 0.")
    if rv < 0:
        raise ValueError("residual_value must be >= 0.")
    if rv > t:
        raise ValueError("residual_value cannot exceed total_intangibles.")
    return (t - rv) / life

def normalize_ebit_for_acquisition_effects(
    reported_ebit: float,
    acquired_intangibles_amort: float = 0.0,
    inventory_stepup_unwind: float = 0.0,
    contingent_consideration_fv_pnl: float = 0.0
) -> float:
    """
    Produce an 'underlying' EBIT proxy by removing common acquisition accounting noise.

    Adjusted EBIT = reported EBIT
                  + acquired intangibles amortisation (add back)
                  + inventory step-up unwind (add back if included in COGS and deemed non-recurring)
                  - contingent consideration FV gains (remove gains) + FV losses (add back) via sign convention

    Args:
        contingent_consideration_fv_pnl:
            Use positive for gains included in EBIT; negative for losses included in EBIT.

    Returns:
        float: adjusted EBIT

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(reported_ebit, "reported_ebit")
    a = _num(acquired_intangibles_amort, "acquired_intangibles_amort")
    inv = _num(inventory_stepup_unwind, "inventory_stepup_unwind")
    cc = _num(contingent_consideration_fv_pnl, "contingent_consideration_fv_pnl")
    if a < 0 or inv < 0:
        raise ValueError("acquired_intangibles_amort and inventory_stepup_unwind must be >= 0.")
    # Remove CC FV gains (subtract positive), add back losses (subtract negative -> add)
    return e + a + inv - cc

def debt_like_adjustment_for_contingent_consideration(liability_fv: float) -> float:
    """
    Treat liability-classified contingent consideration as debt-like for EV bridge.

    Returns:
        float: debt-like amount to add to net debt adjustments

    Raises:
        ValueError: liability_fv negative.
    """
    v = _num(liability_fv, "liability_fv")
    if v < 0:
        raise ValueError("liability_fv must be >= 0.")
    return v

# Example usage
gw = goodwill(
    consideration=1_250_000_000,
    fv_identifiable_assets=1_100_000_000,
    fv_liabilities=420_000_000,
    nci=80_000_000,
    fv_prev_interest=0.0
)
print(f"Goodwill: ${gw/1e6:.0f}M")

amort = acquired_intangibles_amortisation(total_intangibles=320_000_000, weighted_avg_life_years=8)
print(f"Acquired intangibles amortisation: ${amort/1e6:.1f}M")

adj_ebit = normalize_ebit_for_acquisition_effects(
    reported_ebit=210.0,
    acquired_intangibles_amort=40.0,
    inventory_stepup_unwind=12.0,
    contingent_consideration_fv_pnl=+6.0  # gain included in EBIT
)
print(f"Adjusted EBIT: ${adj_ebit:.1f}m")

debt_like = debt_like_adjustment_for_contingent_consideration(liability_fv=95_000_000)
print(f"Debt-like adjustment (contingent consideration): ${debt_like/1e6:.0f}M")
```

---

### Valuation Impact
Why this matters:
- Acquisition accounting can materially change EBIT/ROIC without changing cash generation. If you value on multiples, you need comparability adjustments.
- Goodwill levels and impairment patterns inform acquisition discipline and the sustainability of growth-by-M&A strategies.
- Contingent consideration introduces both earnings volatility and debt-like claims that should be captured in enterprise value.

Impact on multiples:
- EV/EBIT: acquired intangible amortisation depresses EBIT; EBITA-style adjustments can improve comparability if applied consistently.
- EV/EBITDA: inventory step-up unwind can depress EBITDA temporarily; adjust if material and clearly acquisition-related.
- P/E: impacted by amortisation and FV remeasurements; “adjusted EPS” claims should be reconciled and audited for consistency.

Impact on DCF inputs:
- Synergies should be included only if achievable and consistent with integration costs and timing.
- Reinvestment needs may rise post-deal (integration capex, working capital).
- Terminal value should not assume perpetual acquisition-driven growth without evidence.

Practical adjustments:
```python
def valuation_adjustment_for_acquired_intangibles(ev: float, ebit: float, acquired_intangibles_amort: float) -> float:
    """
    Example: compute EV/EBIT and EV/EBITA style multiples for comparability.

    Returns:
        float: EV/EBITA (EBIT + acquired intangibles amort)

    Raises:
        ValueError: division by zero.
    """
    E = _num(ev, "ev")
    e = _num(ebit, "ebit")
    a = _num(acquired_intangibles_amort, "acquired_intangibles_amort")
    if a < 0:
        raise ValueError("acquired_intangibles_amort must be >= 0.")
    denom = e + a
    if denom == 0:
        raise ValueError("EBITA denominator is zero.")
    return E / denom
```

---

### Quality of Earnings Flags
⚠️ Large recurring “adjusted” add-backs tied to acquisitions (integration costs, restructuring, inventory step-ups) that persist for years.  
⚠️ Contingent consideration FV gains used to boost EBIT/EPS.  
⚠️ Goodwill-to-equity or goodwill-to-EV unusually high without strong cash conversion.  
⚠️ Frequent goodwill impairment (signals overpayment or weak integration).  
⚠️ Aggressive useful lives for acquired intangibles (artificially low amortisation).  
✅ Clear PPA disclosures: major intangible categories, useful lives, deferred tax impacts, and reconciliation of adjustments.

---

### Sector-Specific Considerations

| Sector | Common IFRS 3 PPA features | Typical valuation handling |
|---|---|---|
| Software / Tech | Customer relationships, technology, trade names; high acquired intangibles | Use EBITA/FCF focus; normalise for acquired intangible amortisation where comparing to organic peers |
| Consumer / Brands | Brands, customer lists; long useful lives | Scrutinise brand valuation inputs; watch impairment vs brand strength evidence |
| Industrials | Order backlog, customer contracts; inventory step-ups | Adjust EBITDA for step-up unwind; assess cyclical effects on impairment risk |
| Financial services | Deposit intangibles, client relationships; regulatory capital constraints | Ensure EV/earnings metrics align with capital and accounting treatment |

---

### Real-World Example
Scenario: You are valuing a serial acquirer on EV/EBIT. Reported EBIT is $300m but includes $60m acquired-intangibles amortisation and $15m inventory step-up unwind. EV is $6.0bn.

```python
ev = 6_000.0
reported_ebit = 300.0
acq_amort = 60.0
inv_stepup = 15.0

adj_ebit = normalize_ebit_for_acquisition_effects(reported_ebit, acquired_intangibles_amort=acq_amort, inventory_stepup_unwind=inv_stepup)
mult_reported = ev / reported_ebit if reported_ebit != 0 else float("inf")
mult_adj = ev / adj_ebit if adj_ebit != 0 else float("inf")

print(f"EV/EBIT reported: {mult_reported:.1f}x")
print(f"EV/EBIT adjusted: {mult_adj:.1f}x")
```

Interpretation: Adjusted EBIT provides a closer proxy to cash earnings; use it consistently across peers and triangulate with DCF.

See also: Chapter 13 (impairment) for goodwill testing; Chapter 25 (fair value) for measurement of consideration and acquired assets; Chapter 26 (income taxes) for deferred taxes arising from PPA.
