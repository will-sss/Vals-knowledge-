# McKinsey Valuation - Key Concepts for Valuation

## Chapter 24: Measuring Performance in Capital-Light Businesses (Intangibles, Operating Capital, ROIC Adjustments, Unit Economics)

### Core Concept
Capital-light businesses can look “high ROIC” on paper because many value-creating investments (software development, data, brand, customer acquisition) are expensed rather than capitalised. Measuring performance requires restating operating profit and invested capital to reflect the economics of intangible investment, so ROIC, growth, and free cash flow are comparable across business models and over time.

See also: Chapter 11 on reorganising statements (operating vs financing); Chapter 12 on analysing performance (ROIC drivers); Chapter 18 on multiples (EV/Revenue vs EV/EBITDA in intangible-heavy sectors).

---

### Core Concept: Why Standard ROIC Breaks in Capital-Light Models
In many digital, service, and platform businesses:
- **Operating assets are understated** because internally developed intangibles are expensed (R&D, product, data, S&M).
- **Operating profit is understated** in growth phases because the “investment” is running through opex.
- Reported ROIC can be **too high** (low denominator) or **too volatile** (investment spikes hit profit, not capital).

Practical fix: treat selected intangible investments as capital expenditures (capitalise and amortise) to build an “economic” NOPAT and invested capital.

---

### Formula/Methodology

#### 1) Reported ROIC (baseline)
```text
ROIC_reported = NOPAT_reported / Average Invested Capital_reported
```

#### 2) Economic ROIC with capitalised intangibles
Capitalise a chosen set of intangible expenses (e.g., R&D, a portion of S&M for customer acquisition) over an assumed useful life.

```text
Intangible Asset_t = Σ_{k=0..L-1} Intangible Spend_{t-k} × (1 − k/L)
```
Where:
- L = useful life (years) for the intangible investment (e.g., 3–5 years for software/R&D, 2–4 for customer acquisition, context-dependent)
- Spend_{t-k} = qualifying expense in year t-k
- (1 − k/L) is straight-line “remaining” fraction under straight-line amortisation

Economic operating profit:
```text
EBIT_econ = EBIT_reported + Intangible Spend_t − Amortisation_t
NOPAT_econ = EBIT_econ × (1 − Operating Tax Rate)
```

Economic invested capital:
```text
Invested Capital_econ = Invested Capital_reported + Intangible Asset_t
ROIC_econ = NOPAT_econ / Average Invested Capital_econ
```

#### 3) Reinvestment in capital-light businesses
For economic consistency, treat qualifying intangible spend like capex:
```text
Reinvestment_econ = Capex + ΔNWC + Intangible Spend_t − Amortisation_t (if modelling FCFF from NOPAT)
```
Practical modelling choice:
- Either adjust NOPAT and invested capital (ROIC), or adjust FCFF (cash flow). Avoid double counting by keeping one coherent framework.

---

### Python Implementation
```python
from typing import List, Optional, Dict

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def build_intangible_asset(spend_history: List[float], life_years: int) -> float:
    """
    Compute the current intangible asset balance under straight-line amortisation.

    Intangible Asset = sum over last L years of spend * remaining fraction

    Args:
        spend_history: list of annual qualifying spend, most recent last.
                      Example: [spend_t-L+1, ..., spend_t]
        life_years: useful life L (int >= 1)

    Returns:
        float: intangible asset balance (currency)

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(spend_history, list) or len(spend_history) == 0:
        raise ValueError("spend_history must be a non-empty list")
    if not isinstance(life_years, int) or life_years < 1:
        raise ValueError("life_years must be an int >= 1")

    # use only last L years
    hist = spend_history[-life_years:]
    if len(hist) < life_years:
        # allow shorter history, but results will understate asset for early years
        pass

    asset = 0.0
    # k=0 is most recent year spend (remaining fraction 1.0)
    for k, spend in enumerate(reversed(hist)):
        if not _num(spend) or spend < 0:
            raise ValueError("spend values must be numeric and >= 0")
        remaining = 1.0 - (k / life_years)
        if remaining < 0:
            remaining = 0.0
        asset += spend * remaining
    return asset


def intangible_amortisation(current_asset: float, life_years: int) -> float:
    """
    Straight-line amortisation proxy = current_asset / life_years.
    (Approximation; exact amortisation depends on vintage schedule.)
    """
    if not _num(current_asset) or current_asset < 0:
        raise ValueError("current_asset must be numeric and >= 0")
    if not isinstance(life_years, int) or life_years < 1:
        raise ValueError("life_years must be an int >= 1")
    return current_asset / life_years


def economic_ebit(reported_ebit: float, intangible_spend_t: float, amort_t: float) -> float:
    """
    EBIT_econ = EBIT_reported + spend_t - amort_t.
    """
    for name, v in {"reported_ebit": reported_ebit, "intangible_spend_t": intangible_spend_t, "amort_t": amort_t}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if intangible_spend_t < 0 or amort_t < 0:
        raise ValueError("intangible_spend_t and amort_t must be >= 0")
    return reported_ebit + intangible_spend_t - amort_t


def nopat(ebit: float, tax_rate: float) -> float:
    if not _num(ebit):
        raise ValueError("ebit must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0, 1)")
    return ebit * (1.0 - tax_rate)


def roic(nopat_value: float, avg_invested_capital: float) -> Optional[float]:
    if not _num(nopat_value) or not _num(avg_invested_capital):
        raise ValueError("inputs must be numeric")
    if avg_invested_capital <= 0:
        return None
    return nopat_value / avg_invested_capital


def economic_roic(
    reported_ebit: float,
    reported_avg_invested_capital: float,
    tax_rate: float,
    spend_history: List[float],
    life_years: int
) -> Dict[str, Optional[float]]:
    """
    Compute economic ROIC by capitalising intangibles.

    Returns:
        dict: ebit_econ, nopat_econ, intangible_asset, roic_econ
    """
    intangible_asset = build_intangible_asset(spend_history, life_years)
    amort_t = intangible_amortisation(intangible_asset, life_years)
    spend_t = spend_history[-1]

    ebit_econ = economic_ebit(reported_ebit, spend_t, amort_t)
    nopat_econ = nopat(ebit_econ, tax_rate)

    avg_ic_econ = reported_avg_invested_capital + intangible_asset
    roic_econ = roic(nopat_econ, avg_ic_econ)

    return {
        "ebit_econ": ebit_econ,
        "nopat_econ": nopat_econ,
        "intangible_asset": intangible_asset,
        "roic_econ": roic_econ
    }


# Example usage (capital-light software-like model)
reported = {
    "ebit": 180_000_000,
    "avg_invested_capital": 400_000_000,  # reported operating capital looks low
    "tax_rate": 0.24,
    "qualifying_spend_history": [120_000_000, 140_000_000, 170_000_000, 200_000_000],  # e.g., R&D + product opex
    "life_years": 4
}

res = economic_roic(
    reported_ebit=reported["ebit"],
    reported_avg_invested_capital=reported["avg_invested_capital"],
    tax_rate=reported["tax_rate"],
    spend_history=reported["qualifying_spend_history"],
    life_years=reported["life_years"]
)

print(f"Intangible asset (econ): ${res['intangible_asset']/1e6:.0f}M")
print(f"Economic EBIT: ${res['ebit_econ']/1e6:.0f}M")
print(f"Economic ROIC: {res['roic_econ']:.1%}" if res["roic_econ"] is not None else "Economic ROIC: n/a")
```

---

### Valuation Impact
Why this matters:
- Capitalising intangibles can materially change ROIC and the assessed “quality” of growth. Many capital-light businesses are actually capital-intensive in intangibles.
- Improves comparability across peers with different accounting policies (expense vs capitalise) and across lifecycle stages (growth vs maturity).
- Helps align multiples with fundamentals: EV/EBITDA may look high simply because EBITDA is depressed by expensed growth investment (or inflated if add-backs are aggressive).

Impact on multiples:
- EV/Revenue and EV/ARR are often used when earnings are depressed by reinvestment; economic ROIC helps judge whether a premium multiple is justified.
- If economic ROIC is near WACC, growth is less valuable and high revenue multiples may be fragile.

Impact on DCF inputs:
- If you capitalise intangibles in performance analysis, ensure the DCF reinvestment assumptions reflect the same economics (don’t treat R&D as “free” in terminal).
- Terminal value should assume a sustainable relationship between growth and reinvestment, including intangible investment needed to maintain competitive position.

Practical adjustments:
```python
def economic_reinvestment(capex: float, delta_nwc: float, intangible_spend: float, intangible_amort: float) -> float:
    """Economic reinvestment including intangible investment."""
    for name, v in {"capex": capex, "delta_nwc": delta_nwc, "intangible_spend": intangible_spend, "intangible_amort": intangible_amort}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if capex < 0 or intangible_spend < 0 or intangible_amort < 0:
        raise ValueError("capex, intangible_spend, intangible_amort must be >= 0")
    return capex + delta_nwc + intangible_spend - intangible_amort
```

---

### Quality of Earnings Flags (Capital-Light / Intangibles)
⚠️ Large “adjusted EBITDA” add-backs for stock-based compensation or customer acquisition that are clearly recurring.  
⚠️ Rapid growth with declining R&D/S&M ratios without a credible efficiency driver (possible underinvestment).  
⚠️ Capitalisation policies change over time (software dev costs, commissions) causing artificial margin/ROIC shifts.  
⚠️ High ROIC driven mainly by a tiny invested-capital base (understated intangibles), not true economic advantage.  
✅ Cohort/unit economics show sustainable payback (CAC payback, retention, LTV/CAC) consistent with reinvestment needs.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Software / SaaS | R&D and S&M are growth investments | Consider capitalising a portion of R&D and/or CAC; validate via cohort metrics |
| Professional services | Human capital is primary “asset” | ROIC can be misleading; focus on utilisation, pricing power, retention, and cash conversion |
| Marketplaces | Network effects + spend cycles | Capitalise selective platform build spend with caution; model fade and reinvestment explicitly |
| Consumer brands | Brand investment expensed | Consider whether sustained advertising is maintenance; treat as required ongoing cost |
| Media / content | Content library economics | Capitalise content costs where appropriate; align amortisation with consumption/release patterns |

---

### Practical Comparison Tables

#### 1) What to Capitalise (Decision Rules)
| Spend type | Capitalise? | Why | Typical useful life |
|---|---|---|---|
| Product/R&D (core platform build) | Often yes (for analysis) | Creates multi-year benefit | 3–5 years |
| Sales commissions (CAC) | Sometimes | Multi-year customer benefit | 2–4 years |
| Brand advertising (maintenance) | Usually no | Ongoing to sustain revenue | n/a |
| One-off launch spend | Case-by-case | If benefit persists | 1–3 years |
| Customer support opex | No | Period cost | n/a |

#### 2) Interpretation Guide (Economic ROIC)
| Observation | Likely interpretation | What to do |
|---|---|---|
| ROIC_reported ≫ ROIC_econ | ROIC inflated by understated capital | Use ROIC_econ for durability and growth value |
| ROIC_econ stable, margins volatile | Accounting timing noise | Focus on unit economics and retention |
| ROIC_econ < WACC | Growth may destroy value | Re-check reinvestment assumptions and competitive fade |
| ROIC_econ rises with scale | Economies of scale / moat | Validate with cohort/payback and pricing power |

---

### Real-World Example (Economic ROIC + Valuation Interpretation)

Scenario: A software company reports high ROIC because invested capital excludes internally built software and data. After capitalising R&D-like spend over 4 years, economic ROIC declines, changing the valuation narrative.

```python
reported_ebit = 180_000_000
tax_rate = 0.24
reported_avg_ic = 400_000_000

spend_history = [120_000_000, 140_000_000, 170_000_000, 200_000_000]  # qualifying intangible spend
life = 4

res = economic_roic(reported_ebit, reported_avg_ic, tax_rate, spend_history, life)

# Compare to a hypothetical WACC
wacc = 0.10
roic_econ = res["roic_econ"] or 0.0

spread = roic_econ - wacc
print(f"Economic ROIC: {roic_econ:.1%}")
print(f"ROIC - WACC spread: {spread:.1%}")

# Simple decision framing (not a full valuation):
if roic_econ <= wacc:
    print("Growth value is limited unless ROIC improves or reinvestment needs fall.")
else:
    print("Growth likely creates value; focus on durability of spread and reinvestment efficiency.")
```

Interpretation:
- If economic ROIC drops materially, part of the “premium” multiple may be accounting-driven. The right question becomes whether intangible reinvestment is producing durable competitive advantage (high retention, pricing power, improving unit economics) rather than relying on reported ROIC alone.
