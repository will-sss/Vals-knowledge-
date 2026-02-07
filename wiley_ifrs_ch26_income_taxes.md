# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 26: Income Taxes (IAS 12, deferred taxes, effective tax rate, valuation adjustments)

### Core Concept
IAS 12 explains how to recognise current and deferred income taxes arising from taxable profit and temporary differences between accounting and tax bases. For valuation, taxes are a primary driver of after-tax cash flows, NOPAT, and ROIC. Deferred tax assets (DTAs) and deferred tax liabilities (DTLs) also affect enterprise-to-equity bridges and comparability (especially after business combinations, fair value remeasurements, and asset revaluations).

### Formula/Methodology

#### 1) Current tax
```text
Current tax expense = Taxable profit × Applicable tax rate
(Current tax payable recognises the amount due to tax authorities)

Where:
Taxable profit is based on tax rules, not accounting profit.
```

#### 2) Temporary differences and deferred tax
```text
Temporary difference = Carrying amount (accounting) - Tax base

Deferred tax liability (DTL): taxable temporary difference (future taxable amounts)
Deferred tax asset (DTA): deductible temporary difference (future deductible amounts) or tax losses/credits

Deferred tax = Temporary difference × Enacted tax rate expected when reversal occurs
```

#### 3) Effective tax rate (ETR)
```text
ETR = Total tax expense / Profit before tax

Where:
Total tax expense = current tax + deferred tax (± adjustments)
```

#### 4) Recognising DTAs (probability test)
```text
Recognise DTA only to the extent it is probable that future taxable profits will be available
against which the deductible temporary differences or tax losses can be utilised.

Practical evidence:
- forecast taxable profits
- history of profitability
- reversal pattern of temporary differences
- tax planning opportunities (where realistic and supportable)
```

#### 5) Deferred tax on business combinations and fair value movements
```text
If assets/liabilities are fair-valued on acquisition or revaluation:
- Tax base may not step up (depending on jurisdiction)
- Creates temporary differences -> DTL often recognised

Common example:
Intangible recognised in acquisition (carrying amount > tax base)
DTL = (Carrying amount - Tax base) × tax rate
```

#### 6) Tax rate changes
```text
Deferred tax balances are remeasured when enacted tax rates change.
The effect goes to P&L or OCI depending on where the underlying item was recognised.
```

---

### Practical Application for Valuation Analysts

#### 1) Use the right tax rate in DCF
Common approaches:
- Use a steady-state cash tax rate (long-run effective cash taxes) rather than accounting ETR if there are large deferred movements.
- If modelling FCFF via NOPAT:
  - NOPAT = EBIT × (1 - tax rate)
  - tax rate should reflect operating taxes, excluding non-recurring items and unusual jurisdictions.

Analyst checks:
- Reconcile ETR vs cash tax paid (cash flow statement).
- Identify drivers of differences: loss utilisation, tax credits, withholding taxes, deferred items.

#### 2) Avoid double counting taxes in EV-to-equity bridge
DTLs/DTAs can be treated as:
- Debt-like adjustments (often for DTLs that behave like long-dated financing), or
- Working capital/non-operating items depending on nature and expected reversal.

Practical rule of thumb:
- DTLs arising from accelerated depreciation may act like quasi-permanent financing (often treated as debt-like but with judgement).
- DTLs arising from asset revaluations or business combinations may reverse on disposal or amortisation; include but consider timing.
- DTAs only valuable if usable (probable taxable profits). Apply a haircut if utilisation is uncertain.

#### 3) ROIC and invested capital consistency
If you include deferred taxes in invested capital, do it consistently across companies and periods.
Two common practices:
- Exclude deferred taxes from invested capital (focus on operating assets net of operating liabilities).
- Include net deferred tax liabilities as part of invested capital (treat like financing).
Pick one, explain, and apply consistently.

#### 4) Quality of earnings: watch deferred tax “noise”
Deferred tax can move due to:
- tax rate changes
- recognition/reversal of DTAs
- one-off restructuring and provisions
These can distort tax expense and net income in a given year.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, List, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def current_tax(taxable_profit: float, tax_rate: float) -> float:
    """
    Compute current tax expense based on taxable profit.

    Args:
        taxable_profit: taxable profit (currency)
        tax_rate: applicable statutory rate (decimal, 0-1)

    Returns:
        float: current tax expense (currency)

    Raises:
        ValueError: invalid inputs.
    """
    tp = _num(taxable_profit, "taxable_profit")
    tr = _num(tax_rate, "tax_rate")
    if not (0.0 <= tr <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    return tp * tr

def deferred_tax_from_temp_diff(carrying_amount: float, tax_base: float, tax_rate: float) -> float:
    """
    Compute deferred tax balance from a temporary difference.

    DTL positive when carrying_amount > tax_base (taxable temp diff)
    DTA negative when carrying_amount < tax_base (deductible temp diff)

    Args:
        carrying_amount: accounting carrying amount
        tax_base: tax base
        tax_rate: enacted rate expected on reversal (decimal, 0-1)

    Returns:
        float: deferred tax (positive=DTL, negative=DTA)

    Raises:
        ValueError: invalid tax_rate.
    """
    ca = _num(carrying_amount, "carrying_amount")
    tb = _num(tax_base, "tax_base")
    tr = _num(tax_rate, "tax_rate")
    if not (0.0 <= tr <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    temp_diff = ca - tb
    return temp_diff * tr

def effective_tax_rate(total_tax_expense: float, profit_before_tax: float) -> float:
    """
    Compute effective tax rate.

    Args:
        total_tax_expense: current + deferred tax expense (currency)
        profit_before_tax: PBT (currency)

    Returns:
        float: effective tax rate (decimal)

    Raises:
        ValueError: if profit_before_tax is zero.
    """
    t = _num(total_tax_expense, "total_tax_expense")
    pbt = _num(profit_before_tax, "profit_before_tax")
    if pbt == 0:
        raise ValueError("profit_before_tax must be non-zero.")
    return t / pbt

def dta_valuation(dta_gross: float, probability_of_utilization: float, haircut: float = 0.0) -> float:
    """
    Estimate 'usable' value of a deferred tax asset for valuation purposes.

    Args:
        dta_gross: recognised or disclosed DTA amount (currency)
        probability_of_utilization: analyst estimate 0-1 of utilisation likelihood
        haircut: additional conservatism 0-1 (e.g., 0.2 for 20% haircut)

    Returns:
        float: adjusted DTA value

    Raises:
        ValueError: invalid probabilities.
    """
    dta = _num(dta_gross, "dta_gross")
    p = _num(probability_of_utilization, "probability_of_utilization")
    h = _num(haircut, "haircut")
    if dta < 0:
        raise ValueError("dta_gross must be >= 0 (use absolute amount).")
    if not (0.0 <= p <= 1.0):
        raise ValueError("probability_of_utilization must be between 0 and 1.")
    if not (0.0 <= h < 1.0):
        raise ValueError("haircut must be in [0, 1).")
    return dta * p * (1.0 - h)

def adjust_equity_for_deferred_taxes(equity_value: float, net_dtl: float, include_dtl_as_debt_like: bool = True) -> float:
    """
    Adjust equity value for net deferred tax liabilities/assets in an EV-to-equity bridge context.

    Args:
        equity_value: starting equity value (currency)
        net_dtl: net deferred tax liability (positive) or asset (negative)
        include_dtl_as_debt_like: if True, subtract DTL and add DTA

    Returns:
        float: adjusted equity value

    Raises:
        ValueError: none.
    """
    eq = _num(equity_value, "equity_value")
    nd = _num(net_dtl, "net_dtl")
    if include_dtl_as_debt_like:
        return eq - nd
    return eq

# Example usage
pbt = 300_000_000
tax_exp = 75_000_000
etr = effective_tax_rate(tax_exp, pbt)
print(f"ETR: {etr:.1%}")

dtl = deferred_tax_from_temp_diff(carrying_amount=500_000_000, tax_base=300_000_000, tax_rate=0.25)
print(f"DTL from temp diff: ${dtl/1e6:.1f}M")

usable_dta = dta_valuation(dta_gross=120_000_000, probability_of_utilization=0.6, haircut=0.15)
print(f"Usable DTA value: ${usable_dta/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- Tax rates directly affect NOPAT and free cash flow; small changes can materially move DCF value.
- Deferred taxes affect equity bridges and can be a hidden source of leverage (DTLs) or value (DTAs).
- ETR volatility can signal one-offs, tax planning, or sustainability issues in cash taxes.

Impact on multiples:
- P/E: net income is sensitive to ETR and deferred tax adjustments; normalise for one-off tax benefits/charges.
- EV/EBITDA: less directly affected, but cash taxes and net debt adjustments (including DTAs/DTLs) affect equity value.

Impact on DCF inputs:
- Use a sustainable cash tax rate rather than a one-year accounting ETR distorted by deferred movements.
- Model loss utilisation and tax shields explicitly where material (especially for highly leveraged or loss-making firms).

Comparability issues:
- Different jurisdictions and group structures can drive structurally different tax rates.
- Revaluations/business combinations can create DTLs that differ by accounting policy and tax regime.

Practical adjustments:
```python
def normalized_nopat(ebit: float, normalized_tax_rate: float) -> float:
    """
    NOPAT = EBIT × (1 - tax_rate)

    Raises:
        ValueError: tax_rate out of bounds.
    """
    e = _num(ebit, "ebit")
    tr = _num(normalized_tax_rate, "normalized_tax_rate")
    if not (0.0 <= tr <= 1.0):
        raise ValueError("normalized_tax_rate must be between 0 and 1.")
    return e * (1.0 - tr)
```

---

### Quality of Earnings Flags
⚠️ Large tax credits/benefits in a single year boosting net income with limited recurrence.  
⚠️ Sudden recognition of DTAs (loss carryforwards) without clear evidence of future taxable profits.  
⚠️ Significant uncertain tax positions, disputes, or aggressive transfer pricing signals.  
⚠️ Major ETR swings driven by one-off “rate change remeasurement” or reclassification items.  
✅ Stable cash tax paid relative to operating profit and transparent reconciliation from statutory to effective rate.

---

### Sector-Specific Considerations

| Sector | Key tax issue | Typical treatment |
|---|---|---|
| Asset-heavy (industrials, utilities) | accelerated depreciation and tax shields | consider DTL as quasi-permanent; model cash tax vs accounting tax |
| Tech / IP-heavy | transfer pricing, IP location, tax rate sustainability | assess long-run sustainable tax rate; sensitivity to policy changes |
| Real estate | deferred tax on revaluations and disposals | consider reversal timing and transaction structuring impacts |
| Financials | DTAs from credit losses and regulatory constraints | haircuts to DTAs for capital usability; focus on CET1 implications where relevant |

---

### Real-World Example
Scenario: You are valuing an operating business with EBIT $250m. Reported ETR is 18% due to a one-off tax credit; statutory is 25%. You want a normalised NOPAT and equity adjustment for net DTL.

```python
ebit = 250_000_000
reported_etr = 0.18
normalized_rate = 0.25
reported_nopat = normalized_nopat(ebit, reported_etr)
norm_nopat = normalized_nopat(ebit, normalized_rate)

print(f"Reported NOPAT (18%): ${reported_nopat/1e6:.1f}M")
print(f"Normalised NOPAT (25%): ${norm_nopat/1e6:.1f}M")

equity = 3_200_000_000
net_dtl = 300_000_000
adj_equity = adjust_equity_for_deferred_taxes(equity, net_dtl, include_dtl_as_debt_like=True)
print(f"Equity after net DTL adjustment: ${adj_equity/1e9:.2f}B")
```

Interpretation: Normalising tax can reduce sustainable earnings; net DTL reduces equity value if treated as debt-like.

See also: Chapter 25 (fair value) for deferred tax on remeasurements; Chapter 15 (business combinations) for acquisition-related DTLs; Chapter 13 (impairment) where DTAs/DTLs may arise on write-downs; Chapter 28 (segments) for different tax regimes by geography.
