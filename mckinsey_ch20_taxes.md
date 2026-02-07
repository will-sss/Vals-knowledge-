# McKinsey Valuation - Key Concepts for Valuation

## Chapter 20: Taxes (Cash Taxes, NOPAT Taxes, Deferred Taxes, Valuation Consistency, Tax Shields)

### Core Concept
Taxes affect valuation through operating cash flows (cash taxes), after-tax operating profit (NOPAT), discount rates (tax shield in WACC/APV), and balance-sheet items (deferred taxes). The practical objective is consistency: (1) forecast taxes in a way that matches the cash flow definition being discounted, (2) avoid double-counting tax benefits (especially interest tax shields), and (3) treat deferred taxes and tax attributes (NOLs, credits) as operating or non-operating depending on how they affect future cash taxes.

See also: Chapter 11 on reorganising statements (operating taxes vs financing); Chapter 15 on WACC (tax rate in WACC); Chapter 33 on capital structure (interest tax shield dynamics); Chapter 21 on provisions/reserves (tax timing and uncertain positions).

---

### Core Concept: Three “Tax Rates” You Must Distinguish
1) **Statutory rate**: the legal corporate tax rate.  
2) **Effective tax rate (ETR)**: tax expense / pre-tax income (often distorted by one-offs, mix, deferred taxes).  
3) **Cash tax rate**: cash taxes paid / taxable income proxy (what drives FCFF).  

In valuation, use:
- **Operating tax rate** for NOPAT (sustainable, excludes non-operating and one-offs).
- **Cash taxes** for FCFF (explicitly forecast cash taxes, incorporating timing differences and tax attributes).

---

### Formula/Methodology

#### 1) NOPAT-based operating taxes (for ROIC and economic profit)
```text
NOPAT = Operating EBIT × (1 − Operating Tax Rate)
```

Where:
- Operating tax rate is a sustainable rate on operating profit (not necessarily GAAP ETR).

#### 2) FCFF tax treatment (cash taxes)
In FCFF models, taxes should be **cash taxes** paid on operating income (after interest if you’re modeling taxable income at the firm level), but in a standard FCFF/WACC framework you generally:
- Compute operating profit taxes without deducting interest (to keep financing separate)
- Include the interest tax shield through WACC’s (1 − Tc) adjustment on Rd, or use APV explicitly

Practical FCFF setup:
```text
FCFF = NOPAT
     + Depreciation & Amortisation
     − Capex
     − ΔNWC
     − Other Operating Investments
```
Cash tax adjustments are embedded in how you define NOPAT and operating tax rate, and whether you explicitly model cash taxes vs accounting tax expense.

#### 3) Interest tax shield (avoid double-counting)
WACC framework embeds the tax shield via:
```text
WACC = (E/V) × Re + (D/V) × Rd × (1 − Tc)
```
APV framework values it explicitly:
```text
APV = Unlevered Value + PV(Interest Tax Shields) − PV(Financial Distress Costs)
```

#### 4) Deferred taxes (timing differences)
Deferred taxes arise when accounting income differs from taxable income due to timing differences (e.g., accelerated tax depreciation vs straight-line book). Practical valuation choices:
- **Most FCFF models use cash taxes**, making deferred tax expense less relevant for valuation, but deferred tax balances may indicate future cash tax effects.
- Treat deferred tax liabilities (DTLs) as a **timing liability**: not “debt” mechanically, but can reduce future cash taxes if reversing is likely and forecastable.

---

### Python Implementation
```python
from typing import Optional, Dict

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def nopat(operating_ebit: float, operating_tax_rate: float) -> float:
    """NOPAT = Operating EBIT * (1 - tax_rate)."""
    if not _num(operating_ebit):
        raise ValueError("operating_ebit must be numeric")
    if not _num(operating_tax_rate) or not (0.0 <= operating_tax_rate < 1.0):
        raise ValueError("operating_tax_rate must be in [0, 1)")
    return operating_ebit * (1.0 - operating_tax_rate)


def fcff(nopat_value: float, da: float, capex: float, delta_nwc: float, other_operating_investments: float = 0.0) -> float:
    """
    FCFF = NOPAT + D&A - Capex - ΔNWC - other operating investments.
    Sign convention: capex and delta_nwc are positive if they are cash outflows.
    """
    for name, v in {"nopat_value": nopat_value, "da": da, "capex": capex, "delta_nwc": delta_nwc, "other_operating_investments": other_operating_investments}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return nopat_value + da - capex - delta_nwc - other_operating_investments


def interest_tax_shield(interest_expense: float, tax_rate: float) -> float:
    """Interest tax shield = interest * tax_rate."""
    if not _num(interest_expense):
        raise ValueError("interest_expense must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0, 1)")
    # If interest is negative (net interest income), the "shield" is negative (a tax cost)
    return interest_expense * tax_rate


def pv(value: float, rate: float, t: int) -> float:
    if not _num(value):
        raise ValueError("value must be numeric")
    if not _num(rate) or rate <= 0:
        raise ValueError("rate must be numeric and > 0")
    if not isinstance(t, int) or t < 0:
        raise ValueError("t must be int >= 0")
    return value / ((1.0 + rate) ** t)


def pv_tax_shields(shields: Dict[int, float], discount_rate: float) -> float:
    """PV of tax shields indexed by year (t -> shield)."""
    if not isinstance(shields, dict) or len(shields) == 0:
        raise ValueError("shields must be a non-empty dict")
    if not _num(discount_rate) or discount_rate <= 0:
        raise ValueError("discount_rate must be numeric and > 0")
    total = 0.0
    for t, s in shields.items():
        if not isinstance(t, int) or t < 0:
            raise ValueError("year keys must be int >= 0")
        if not _num(s):
            raise ValueError("shield values must be numeric")
        total += pv(s, discount_rate, t)
    return total


def cash_tax_from_taxable_income(taxable_income: float, tax_rate: float, nols_available: float = 0.0) -> Dict[str, float]:
    """
    Simple cash tax model with NOL usage.

    Args:
        taxable_income: taxable income before NOLs (currency)
        tax_rate: statutory/marginal tax rate (decimal)
        nols_available: NOL balance available to offset taxable income (currency)

    Returns:
        dict with keys:
            taxable_after_nol, nol_used, cash_tax

    Raises:
        ValueError: invalid inputs.
    """
    for name, v in {"taxable_income": taxable_income, "tax_rate": tax_rate, "nols_available": nols_available}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0, 1)")
    if nols_available < 0:
        raise ValueError("nols_available must be >= 0")

    nol_used = 0.0
    taxable_after = taxable_income
    if taxable_income > 0 and nols_available > 0:
        nol_used = min(taxable_income, nols_available)
        taxable_after = taxable_income - nol_used

    cash_tax = max(0.0, taxable_after) * tax_rate
    return {"taxable_after_nol": taxable_after, "nol_used": nol_used, "cash_tax": cash_tax}


# Example usage
operating_ebit = 500_000_000
op_tax = 0.24
n = nopat(operating_ebit, op_tax)

fcff_val = fcff(nopat_value=n, da=180_000_000, capex=220_000_000, delta_nwc=40_000_000)
print(f"NOPAT: ${n/1e6:.0f}M")
print(f"FCFF: ${fcff_val/1e6:.0f}M")

# NOL example
tax_model = cash_tax_from_taxable_income(taxable_income=180_000_000, tax_rate=0.24, nols_available=120_000_000)
print(f"Cash tax: ${tax_model['cash_tax']/1e6:.1f}M (NOL used ${tax_model['nol_used']/1e6:.1f}M)")
```

---

### Valuation Impact
Why this matters:
- Taxes directly affect free cash flow and terminal value. Overstating tax benefits (e.g., using an artificially low ETR) can inflate value materially.
- The choice between WACC and APV determines how tax shields are captured; inconsistency leads to double-counting or omission.
- Deferred taxes and tax attributes can be real value drivers if they reduce future cash taxes, but only when their usage is credible and forecastable.

Impact on multiples:
- P/E and EV/EBIT depend on after-tax earnings; one-off tax items distort comparability.
- NOPAT-based multiples (EV/NOPAT) require consistent operating tax rates across peers; differences in jurisdiction mix can drive “structural” multiple differences.

Impact on DCF inputs:
- Use a sustainable operating tax rate for NOPAT; model cash taxes explicitly if timing differences or NOLs are material.
- In terminal value, ensure taxes reflect steady-state (no perpetual NOL shelter unless justified).

Practical adjustments:
```python
def normalize_operating_tax_rate(reported_tax_expense: float,
                                pretax_income: float,
                                one_off_tax_items: float = 0.0,
                                min_rate: float = 0.0,
                                max_rate: float = 0.35) -> Optional[float]:
    """
    Derive a cleaned effective tax rate and clamp to a reasonable range.

    Args:
        reported_tax_expense: total tax expense (currency)
        pretax_income: reported pre-tax income (currency)
        one_off_tax_items: remove one-offs from tax expense (currency). Positive means tax expense inflated.
        min_rate, max_rate: clamp range

    Returns:
        normalized tax rate or None if pretax_income <= 0
    """
    if not _num(reported_tax_expense) or not _num(pretax_income) or not _num(one_off_tax_items):
        raise ValueError("inputs must be numeric")
    if pretax_income <= 0:
        return None
    raw = (reported_tax_expense - one_off_tax_items) / pretax_income
    # clamp
    if not _num(min_rate) or not _num(max_rate) or min_rate < 0 or max_rate <= min_rate:
        raise ValueError("invalid clamp range")
    return max(min_rate, min(max_rate, raw))
```

---

### Quality of Earnings Flags (Tax Red Flags)
⚠️ Using a low reported ETR driven by one-offs (settlements, valuation allowance releases, one-time credits) as a steady-state assumption.  
⚠️ Assuming indefinite cash tax shelter from NOLs without modelling usage limits and profitability requirements.  
⚠️ Double-counting tax shields: (1) using after-interest taxes in FCFF and (2) also applying (1 − Tc) in WACC.  
⚠️ Treating deferred tax liabilities as “free financing” without assessing reversal timing and business continuity.  
⚠️ Large uncertain tax positions or aggressive transfer pricing not reflected in risk/sensitivity.  
✅ Clear separation: operating tax rate for NOPAT, cash taxes for FCFF, and explicit treatment of tax attributes.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Multinationals | Jurisdiction mix drives structural ETR | Build a blended long-run rate; avoid extrapolating temporary mix benefits |
| Capital intensive | Accelerated tax depreciation | Model cash taxes vs book taxes; reflect timing differences (DTL build/reversal) |
| Financials | Different taxable base and regulation | Use sector-specific tax metrics; avoid applying generic operating NOPAT logic blindly |
| Oil & gas / mining | Special levies and ring-fencing | Model statutory + supplemental taxes explicitly; sensitise commodity price effects |
| Real estate | REIT/pass-through structures | Ensure the correct tax regime is applied (entity vs investor-level) |

---

### Practical Comparison Tables

#### 1) Tax Modelling Choice Map
| Situation | Recommended approach | Why |
|---|---|---|
| Stable company, limited timing differences | Use operating tax rate on EBIT for NOPAT; WACC captures tax shield | Simplicity and consistency |
| Material NOLs/credits | Model cash taxes with explicit NOL roll-forward | Value can be meaningful and time-bound |
| Highly levered / changing leverage | Consider APV with explicit tax shields | Avoid WACC weight instability and double-counting |
| Material deferred tax movements | Model cash taxes; use DTA/DTL as diagnostic | Accounting taxes can mislead valuation |

#### 2) Common Double-Count Pitfalls
| Error | Symptom | Fix |
|---|---|---|
| Taxing after interest in FCFF and also using (1 − Tc) in WACC | EV too high vs peers | Compute taxes on operating profit (pre-interest) in FCFF |
| Treating NOLs as perpetual | Terminal cash taxes unrealistically low | Apply NOL usage limits and expiry; revert to steady-state rate |
| Using GAAP ETR including one-offs | Volatile tax rates in model | Normalise and justify sustainable operating tax rate |

---

### Real-World Example (Cash Taxes with NOLs + DCF Consistency)

Scenario: A business has NOLs that reduce cash taxes for several years. Model cash taxes for the explicit period, but ensure terminal taxes revert to a sustainable rate once NOLs are used.

```python
# Year-by-year taxable income forecast (simplified)
taxable_income_forecast = [120_000_000, 180_000_000, 240_000_000, 300_000_000, 320_000_000]
tax_rate = 0.24
nols = 250_000_000

cash_taxes = []
nol_balance = nols

for ti in taxable_income_forecast:
    res = cash_tax_from_taxable_income(ti, tax_rate, nols_available=nol_balance)
    cash_taxes.append(res["cash_tax"])
    nol_balance = max(0.0, nol_balance - res["nol_used"])

print("Cash taxes by year ($M):", [round(x/1e6, 1) for x in cash_taxes])
print(f"Ending NOL balance: ${nol_balance/1e6:.1f}M")

# Terminal: assume no remaining NOLs, use sustainable tax rate on operating EBIT
terminal_operating_ebit = 520_000_000
terminal_nopat = nopat(terminal_operating_ebit, operating_tax_rate=tax_rate)
print(f"Terminal NOPAT (steady-state taxes): ${terminal_nopat/1e6:.0f}M")
```

Interpretation:
- NOLs can increase near-term FCFF and EV, but the benefit is finite and depends on future taxable income.
- Terminal assumptions must not embed “free” taxes indefinitely unless there is a credible structural exemption.
