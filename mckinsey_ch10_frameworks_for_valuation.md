# McKinsey Valuation - Key Concepts for Valuation

## Chapter 10: Frameworks for Valuation (DCF, Economic Profit, APV, Multiples Cross-Checks)

### Core Concept
A valuation “framework” is the structure that links operating performance and capital deployment to enterprise value and equity value. The core principle is consistency: cash flows, discount rates, terminal assumptions, and balance-sheet items must be defined on the same basis (firm vs equity; operating vs non-operating). Use multiple frameworks (DCF, economic profit, APV, multiples) as cross-checks, not substitutes for fundamentals.

See also: Chapter 11 on reorganising financial statements (operating vs financing); Chapter 14 on continuing value; Chapter 18 on multiples; Chapter 16 on EV to value per share.

---

### Core Concept: The Main Valuation Frameworks and When to Use Them

| Framework | What you value | Core inputs | Best for | Common failure mode |
|---|---|---|---|---|
| DCF (FCFF) | Enterprise value | FCFF, WACC, terminal value | Most operating companies | Mixing equity/firm items; inconsistent terminal assumptions |
| Economic Profit (EP) | Enterprise value | Invested capital, ROIC, WACC, growth/reinvestment | ROIC-centric analysis; value creation decomposition | Wrong invested capital definition; ROIC not normalized |
| APV | Enterprise value | Unlevered value + PV(tax shields) − PV(distress/issuance costs) | Changing leverage; LBO/restructuring; tax shield focus | Double-counting tax shields; assuming full usability |
| Equity DCF (FCFE) | Equity value | FCFE, cost of equity, terminal value | Financial firms; stable leverage | Using FCFE with WACC; ignoring net debt definition changes |
| Multiples | EV or Equity value | Peer multiple × metric | Sanity checks; market pricing context | Peer mismatch; bad metric normalization |

Practical rule: build one “primary” model (usually FCFF DCF) and use EP/APV/multiples as coherence checks.

---

### Core Concept: DCF (FCFF) as the Backbone

### Formula/Methodology
```text
FCFF = NOPAT − Reinvestment
NOPAT = EBIT × (1 − Tax Rate)

Enterprise Value (EV) = Σ [ FCFF_t / (1 + WACC)^t ] + Terminal Value / (1 + WACC)^N
Equity Value = EV − Net Debt − Other debt-like claims + Non-operating assets
```

Where:
- FCFF: Free cash flow to the firm (after operating taxes, before financing)
- Reinvestment: Net investment to support growth (capex − D&A + ΔNWC) or equivalent operating reinvestment definition
- Net Debt: Interest-bearing debt minus cash (adjust for excess cash, restricted cash, debt-like liabilities)

### Python Implementation
```python
from typing import List, Optional, Dict

def nopat(ebit: float, tax_rate: float) -> float:
    """
    Compute NOPAT from EBIT and tax rate.

    Args:
        ebit: Earnings before interest and taxes (currency)
        tax_rate: Effective operating tax rate (decimal in [0,1))

    Returns:
        float: NOPAT (currency)

    Raises:
        ValueError: If inputs invalid.
    """
    if not isinstance(ebit, (int, float)) or ebit != ebit:
        raise ValueError("ebit must be numeric and not NaN")
    if not isinstance(tax_rate, (int, float)) or tax_rate != tax_rate:
        raise ValueError("tax_rate must be numeric and not NaN")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0, 1)")
    return ebit * (1.0 - tax_rate)


def fcff(nopat_value: float,
         capex: float,
         depreciation: float,
         delta_nwc: float) -> float:
    """
    Compute FCFF using a common reinvestment definition:
    FCFF = NOPAT - (Capex - Depreciation + ΔNWC)

    Args:
        nopat_value: NOPAT (currency)
        capex: Gross capital expenditures (currency, >= 0)
        depreciation: Depreciation & amortisation (currency, >= 0)
        delta_nwc: Change in net working capital (currency; positive = cash outflow)

    Returns:
        float: FCFF (currency)

    Raises:
        ValueError: For invalid inputs.
    """
    for name, v in {
        "nopat_value": nopat_value,
        "capex": capex,
        "depreciation": depreciation,
        "delta_nwc": delta_nwc,
    }.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must be numeric and not NaN")
    if capex < 0 or depreciation < 0:
        raise ValueError("capex and depreciation must be >= 0")
    reinvestment = capex - depreciation + delta_nwc
    return nopat_value - reinvestment


def pv_cash_flows(cash_flows: List[float], discount_rate: float) -> float:
    """
    Present value of a list of annual cash flows discounted at discount_rate.

    Args:
        cash_flows: list of cash flows (currency)
        discount_rate: discount rate (decimal > 0)

    Returns:
        float: present value

    Raises:
        ValueError: For invalid inputs.
    """
    if not isinstance(cash_flows, (list, tuple)) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list")
    if not isinstance(discount_rate, (int, float)) or discount_rate != discount_rate:
        raise ValueError("discount_rate must be numeric and not NaN")
    if discount_rate <= 0:
        raise ValueError("discount_rate must be > 0")

    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not isinstance(cf, (int, float)) or cf != cf:
            raise ValueError("cash_flows must contain only numeric values")
        pv += cf / ((1.0 + discount_rate) ** t)
    return pv


# Example usage (5-year explicit forecast)
ebit_series = [300, 330, 360, 390, 420]  # $m
tax_rate = 0.25
capex_series = [120, 125, 130, 135, 140]
da_series = [80, 82, 84, 86, 88]
d_nwc_series = [10, 8, 7, 6, 5]
wacc = 0.09

fcffs = []
for ebit, capex, da, d_nwc in zip(ebit_series, capex_series, da_series, d_nwc_series):
    n = nopat(ebit, tax_rate)
    fcffs.append(fcff(n, capex, da, d_nwc))

print("FCFFs:", fcffs)
print("PV(FCFFs):", pv_cash_flows(fcffs, wacc))
```

### Valuation Impact
Why this matters:
- DCF is the accountability framework: every assumption must flow through to cash, capital, and risk.
- Multiples are outputs: EV/EBITDA and P/E are implied by your DCF assumptions; use them as cross-checks.
- Comparability: DCF forces you to normalise operating earnings (QoE) and define invested capital consistently.

Practical adjustments (enterprise-to-equity bridge, preliminary):
```python
def equity_value_from_ev(ev: float, net_debt: float, other_debt_like: float = 0.0, non_operating_assets: float = 0.0) -> float:
    """
    Convert enterprise value to equity value using a simple bridge.

    Equity = EV - Net Debt - Other debt-like claims + Non-operating assets
    """
    for name, v in {"ev": ev, "net_debt": net_debt, "other_debt_like": other_debt_like, "non_operating_assets": non_operating_assets}.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must be numeric and not NaN")
    return ev - net_debt - other_debt_like + non_operating_assets
```

### Quality of Earnings Flags
⚠️ FCFF boosted by one-off working capital releases or capex under-spend (not sustainable).  
⚠️ Tax rate uses GAAP effective tax including one-offs, not an operating/sustainable rate.  
⚠️ EBIT includes non-operating gains/losses (asset sales, fair value moves).  
✅ Clear split between operating and financing items and explicit reconciliation to cash flow statement.

---

### Core Concept: Economic Profit (EP) as a Value Creation Lens

### Formula/Methodology
```text
Economic Profit (EP)_t = Invested Capital_{t−1} × (ROIC_t − WACC)

Enterprise Value = Invested Capital_0 + PV(EP over explicit period) + PV(Continuing EP)
```

Where:
- Invested Capital_0: opening operating invested capital (excluding non-operating assets)
- EP translates operating performance into value creation directly (positive EP means ROIC > WACC).

### Python Implementation
```python
from typing import List

def economic_profit_series(invested_capital_opening: List[float], roic: List[float], wacc: float) -> List[float]:
    """
    Compute a series of economic profits using opening invested capital each year.

    Args:
        invested_capital_opening: list of opening IC per year (currency, >= 0)
        roic: list of ROIC per year (decimal)
        wacc: WACC (decimal > 0)

    Returns:
        list[float]: economic profit per year (currency)

    Raises:
        ValueError: If lengths mismatch or inputs invalid.
    """
    if len(invested_capital_opening) != len(roic):
        raise ValueError("invested_capital_opening and roic must have same length")
    if not isinstance(wacc, (int, float)) or wacc != wacc or wacc <= 0:
        raise ValueError("wacc must be numeric and > 0")

    eps = []
    for ic, r in zip(invested_capital_opening, roic):
        if ic < 0:
            raise ValueError("invested capital must be >= 0")
        if not isinstance(r, (int, float)) or r != r:
            raise ValueError("roic values must be numeric and not NaN")
        eps.append(ic * (r - wacc))
    return eps


# Example usage
ic_open = [2000, 2150, 2300, 2450, 2600]  # $m
roic_path = [0.12, 0.125, 0.13, 0.13, 0.128]
wacc = 0.09
eps = economic_profit_series(ic_open, roic_path, wacc)
print("EPs:", [round(x, 1) for x in eps])
```

### Valuation Impact
Why this matters:
- EP makes it harder to hide value destruction behind growth: if ROIC falls toward WACC as the company scales, EP growth can stall even when revenue rises.
- EP highlights the “engine” of value: improving ROIC, improving growth at high ROIC, or improving capital efficiency (lower IC needed per unit of revenue).

Practical adjustments:
- Normalise ROIC by removing non-recurring EBIT and using consistent invested capital definitions (see Chapter 11).

### Quality of Earnings Flags
⚠️ ROIC inflated by excluding operating leases or capitalised development costs inconsistently across peers.  
⚠️ Invested capital excludes material working capital items (e.g., deferred revenue) inconsistently.  
✅ ROIC derived from reorganised statements with transparent adjustments.

---

### Core Concept: APV for Changing Leverage (Tax Shields vs Distress Costs)

### Formula/Methodology
```text
APV = Unlevered Enterprise Value + PV(Tax Shields) − PV(Expected Distress / Issuance Costs)

Simple tax shield proxy (stable debt):
PV(Tax Shields) ≈ Tax Rate × Debt
```

Use APV when leverage changes materially through time (e.g., LBO, recapitalisation), making a single WACC path hard to defend.

### Python Implementation
```python
def apv(unlevered_value: float, pv_tax_shields: float, pv_distress_costs: float = 0.0) -> float:
    """
    Compute Adjusted Present Value (APV).

    Args:
        unlevered_value: PV of unlevered FCFF discounted at unlevered cost of capital (currency)
        pv_tax_shields: PV of financing benefits (currency)
        pv_distress_costs: PV of expected distress/issuance costs (currency)

    Returns:
        float: APV (currency)

    Raises:
        ValueError: If inputs invalid.
    """
    for name, v in {"unlevered_value": unlevered_value, "pv_tax_shields": pv_tax_shields, "pv_distress_costs": pv_distress_costs}.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must be numeric and not NaN")
    return unlevered_value + pv_tax_shields - pv_distress_costs
```

### Valuation Impact
Why this matters:
- Prevents the common error of using a single WACC when leverage is expected to change rapidly.
- Forces explicit modelling of tax shield usability and distress trade-offs (see Chapter 17 on leverage decisions).

### Quality of Earnings Flags
⚠️ Full tax shield assumed despite loss-making periods or limited taxable income.  
⚠️ Debt path inconsistent with coverage and refinancing constraints.

---

### Core Concept: Multiples as Outputs and Cross-Checks

### Formula/Methodology
```text
Implied EV/EBITDA = Enterprise Value / EBITDA
Implied P/E = Equity Value / Net Income

Implied multiple cross-check:
Implied EV/EBITDA from DCF should be broadly consistent with peers after normalisation.
```

### Python Implementation
```python
from typing import Optional

def safe_multiple(numerator: float, denominator: float) -> Optional[float]:
    """
    Compute a valuation multiple with defensive handling.

    Returns None if denominator is zero or negative (not meaningful for most multiples).
    """
    if not isinstance(numerator, (int, float)) or numerator != numerator:
        raise ValueError("numerator must be numeric and not NaN")
    if not isinstance(denominator, (int, float)) or denominator != denominator:
        raise ValueError("denominator must be numeric and not NaN")
    if denominator <= 0:
        return None
    return numerator / denominator


ev = 5_000_000_000
ebitda = 620_000_000
print("EV/EBITDA:", safe_multiple(ev, ebitda))
```

### Valuation Impact
Why this matters:
- Multiples are the market’s shorthand for DCF drivers: growth, margins, reinvestment, and risk.
- If your DCF implies a multiple far outside peers, it’s usually a signal that one of those drivers is inconsistent or mismeasured, not that the market is “wrong.”

### Quality of Earnings Flags
⚠️ Peer EBITDA definitions not comparable (IFRS 16 lease uplift, capitalised dev costs, one-off add-backs).  
⚠️ Multiples applied to unadjusted earnings with significant non-operating or non-recurring items.  
✅ Clear metric normalisation and consistent enterprise-to-equity bridges.

---

### Real-World Example (Framework Cross-Check)

Scenario: You built a 5-year FCFF DCF. Cross-check with implied EV/EBITDA and an EP decomposition.

```python
# Assume EV from your DCF
ev_dcf = 7_800_000_000

# Cross-check multiple
ebitda_yr1 = 750_000_000
implied = safe_multiple(ev_dcf, ebitda_yr1)
print(f"Implied EV/EBITDA (Yr1): {implied:.1f}x" if implied else "Not meaningful")

# EP cross-check (opening IC and ROIC path)
ic_open = [3_000_000_000, 3_150_000_000, 3_300_000_000, 3_450_000_000, 3_600_000_000]
roic_path = [0.11, 0.115, 0.12, 0.12, 0.118]
wacc = 0.09
eps = economic_profit_series(ic_open, roic_path, wacc)
print("Economic profits ($m):", [round(x/1e6, 1) for x in eps])
```

Interpretation:
- If implied EV/EBITDA is far above peers, investigate whether your model assumes unusually high growth, margins, or low reinvestment for the risk profile.
- If EP is negative (ROIC < WACC) while your DCF value looks high, it typically indicates a definition inconsistency (ROIC/IC mismeasured) or a terminal assumption error (see Chapter 14).
