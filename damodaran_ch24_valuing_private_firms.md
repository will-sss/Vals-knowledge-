# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 24: Valuing Private Firms (Illiquidity, Control, Normalisation, Key Person Risk, DLOM)

### Core Concept
Private firm valuation uses the same intrinsic and relative valuation logic as public firms, but you must explicitly address **illiquidity**, **information risk**, **control constraints**, and **owner-specific accounting**. Practically, private valuation is a sequence of adjustments: recast financials to an operating basis, align risk and capital structure assumptions, value using DCF or multiples, then apply appropriate discounts/premia (often with clear documentation rather than opaque “plug” discounts).

See also: Chapter 14–16 (intrinsic value models and per-share), Chapter 17–20 (relative valuation), Chapter 14 (control), Chapter 14–15 (cost of capital/APV), Chapter 30 (distress).

---

### Formula/Methodology

#### 1) Recasting private firm financials (owner-adjusted earnings)
Private accounts often include discretionary expenses and tax-motivated accounting.
```text
Adjusted Operating Income = Reported Operating Income
                          + Excess owner compensation
                          + Non-operating / discretionary expenses
                          − One-off gains
                          ± Reclassifications (rent vs owner perks, etc.)
```
Key principle:
- Adjust to a **market participant** view (arm’s-length compensation, commercial pricing).

#### 2) FCFF framework still applies
```text
FCFF = EBIT × (1 − Tax) + D&A − Capex − ΔNWC
```
Private-firm specifics:
- Taxes: use a realistic marginal tax rate (not necessarily reported effective rate).
- Working capital: adjust for owner-managed payment terms if non-sustainable.
- Capex: separate maintenance vs growth if possible.

#### 3) Cost of equity and discount rates (private firm risk)
Use the same components, but be explicit about inputs:
```text
Cost of Equity (k_e) = Risk-free Rate + Beta × Equity Risk Premium + Size Premium + Company-Specific Risk Premium

Where:
- Beta can be estimated from public comps (unlevered then relevered)
- Size premium reflects higher risk in small firms (if you use it, document it)
- Company-specific risk premium should be used sparingly and justified
```
Control: avoid double-counting risk via both high discount rates and large discounts later.

#### 4) Private company relative valuation (illiquidity and comparability)
Use public comps but adjust for differences:
```text
Private Value = Public Multiple × Private Fundamental × (1 − Discount)

Discount may reflect:
- lack of marketability (DLOM)
- smaller scale / higher risk (but avoid double counting)
- control differences (minority vs control)
```
If you use a discount, tie it to a reason and ensure you are not already capturing it in the multiple selection (e.g., using small-cap comps already embeds some illiquidity/size).

#### 5) Discount for Lack of Marketability (DLOM) — practical approaches
Common practical approaches (choose one and document):
```text
A) Empirical: restricted stock / pre-IPO studies (use a range, avoid false precision)
B) Option-based: treat illiquidity as a “put option” on inability to sell
C) Liquidity horizon: higher discount for longer expected holding period
```
Option-based intuition (high-level diagnostic):
- DLOM increases with volatility, longer lock-up, and lower dividend/payout.

#### 6) Control and minority interests (avoid mixing bases)
```text
Control value includes ability to change:
- management, strategy, capital structure, dividend policy, asset sales

Minority value excludes those controls.

Rule:
- If cash flows assume operational improvements only available with control, value is a control value.
- If the interest is minority, do not add control value unless justified by rights.
```
Be explicit about:
- valuation basis (minority vs controlling),
- rights (board seats, vetoes, drag/tag, liquidation preference).

---

### Python Implementation
```python
from typing import Optional, Dict, List
import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def recast_owner_earnings(
    reported_ebit: float,
    excess_owner_comp: float = 0.0,
    discretionary_expenses: float = 0.0,
    one_off_gains: float = 0.0,
    one_off_charges: float = 0.0
) -> float:
    """Adjusted EBIT for private firm normalisation."""
    for name, v in {
        "reported_ebit": reported_ebit,
        "excess_owner_comp": excess_owner_comp,
        "discretionary_expenses": discretionary_expenses,
        "one_off_gains": one_off_gains,
        "one_off_charges": one_off_charges,
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if any(v < 0 for v in [excess_owner_comp, discretionary_expenses, one_off_gains, one_off_charges]):
        raise ValueError("adjustment components must be >= 0")
    return reported_ebit + excess_owner_comp + discretionary_expenses + one_off_charges - one_off_gains


def fcff(ebit: float, tax_rate: float, da: float, capex: float, delta_nwc: float) -> float:
    """FCFF = EBIT*(1-tax) + D&A - Capex - ΔNWC."""
    if not all(_num(v) for v in [ebit, tax_rate, da, capex, delta_nwc]):
        raise ValueError("inputs must be numeric")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    return ebit * (1.0 - tax_rate) + da - capex - delta_nwc


def pv_cash_flows(cash_flows: List[float], r: float) -> float:
    """PV of cash flows discounted at r."""
    if not _num(r) or r <= -1:
        raise ValueError("r must be numeric and > -100%")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values")
        pv += cf / ((1.0 + r) ** t)
    return pv


def terminal_value_perpetuity(cf_next: float, r: float, g: float) -> float:
    """TV = CF_{n+1} / (r - g)."""
    if not all(_num(v) for v in [cf_next, r, g]):
        raise ValueError("inputs must be numeric")
    if r <= g:
        raise ValueError("Need r > g for finite TV")
    return cf_next / (r - g)


def apply_discount(value: float, discount: float) -> float:
    """Apply a discount (e.g., DLOM)."""
    if not _num(value) or not _num(discount):
        raise ValueError("inputs must be numeric")
    if value < 0:
        raise ValueError("value must be >= 0")
    if not (0.0 <= discount < 1.0):
        raise ValueError("discount must be in [0,1)")
    return value * (1.0 - discount)


def ke_private(
    rf: float,
    beta: float,
    erp: float,
    size_premium: float = 0.0,
    company_specific: float = 0.0
) -> float:
    """Cost of equity proxy for private firms."""
    for name, v in {"rf": rf, "beta": beta, "erp": erp, "size_premium": size_premium, "company_specific": company_specific}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if rf < -0.05:  # guardrail
        raise ValueError("rf appears too low; check units")
    if erp < 0:
        raise ValueError("erp must be >= 0")
    return rf + beta * erp + size_premium + company_specific


def option_based_dlom_proxy(vol: float, lockup_years: float, base: float = 0.05) -> float:
    """
    Simple diagnostic DLOM proxy (NOT a full option model).
    Increases with volatility and time; capped to avoid nonsense.
    """
    if not all(_num(v) for v in [vol, lockup_years, base]):
        raise ValueError("inputs must be numeric")
    if vol < 0 or lockup_years <= 0:
        raise ValueError("vol must be >=0 and lockup_years > 0")
    if not (0.0 <= base < 0.5):
        raise ValueError("base must be in [0,0.5)")
    dlom = base + 0.6 * vol * math.sqrt(lockup_years)
    return max(0.0, min(dlom, 0.45))


# Example usage (all $m)
reported_ebit = 18.0
adj_ebit = recast_owner_earnings(
    reported_ebit=reported_ebit,
    excess_owner_comp=4.0,
    discretionary_expenses=1.5,
    one_off_gains=0.8,
    one_off_charges=0.0
)

tax_rate = 0.25
da = 2.0
capex = 3.5
delta_nwc = 1.0

fcff_y1 = fcff(adj_ebit, tax_rate, da, capex, delta_nwc)

# Forecast simple 5-year FCFF growth and terminal
fcff_forecast = [fcff_y1 * (1.0 + g) ** (t - 1) for t, g in enumerate([0.20, 0.15, 0.10, 0.08, 0.06], start=1)]

rf = 0.04
beta = 1.1
erp = 0.05
size = 0.02
csrp = 0.01
ke = ke_private(rf, beta, erp, size, csrp)
wacc_proxy = ke  # for simplicity; add debt cost if modelling debt

pv_stage1 = pv_cash_flows(fcff_forecast, wacc_proxy)
g_term = 0.03
tv5 = terminal_value_perpetuity(fcff_forecast[-1] * (1.0 + g_term), wacc_proxy, g_term)
pv_tv = tv5 / ((1.0 + wacc_proxy) ** 5)

ev = pv_stage1 + pv_tv

# Apply a DLOM proxy (diagnostic)
dlom = option_based_dlom_proxy(vol=0.35, lockup_years=3.0, base=0.08)
ev_marketable = apply_discount(ev, dlom)

print(f"Adjusted EBIT: ${adj_ebit:,.1f}m")
print(f"EV (pre-DLOM): ${ev:,.1f}m")
print(f"DLOM proxy: {dlom:.1%}")
print(f"EV (post-DLOM): ${ev_marketable:,.1f}m")
```

---

### Valuation Impact
Why this matters:
- Private financials often reflect owner decisions (comp, perks, tax choices). Recasting earnings is usually the single biggest driver of a defendable value.
- Illiquidity discounts can be large and contentious; if you apply them, they must be consistent with the valuation basis (minority vs control) and not double-counted in discount rates or peer selection.
- Control assumptions matter: if your forecast assumes operational changes that require control, you are valuing a control interest.

Impact on multiples:
- Public multiples embed liquidity and disclosure; applying them to private firms typically requires an explicit marketability adjustment unless comps are already illiquid small-caps.
- P/E and EV/EBITDA require normalised earnings; otherwise owners can “choose” their multiple via expense classification.

Impact on DCF inputs:
- Discount rates can incorporate size and information risk, but be careful: a large DLOM plus a large company-specific risk premium can double-count.
- Terminal value should reflect long-run stable economics; for small private firms, conservative terminal growth and explicit reinvestment closure matter.

Practical adjustments:
```python
def check_double_counting(size_premium: float, dlom: float) -> str:
    """Heuristic warning for potential double-counting of illiquidity/size."""
    if not all(_num(v) for v in [size_premium, dlom]):
        raise ValueError("inputs must be numeric")
    if size_premium > 0.03 and dlom > 0.25:
        return "⚠️ potential double-counting: high size premium and high DLOM; reconcile basis"
    return "✅ no obvious double-counting (heuristic)"
```

---

### Quality of Earnings Flags (Private Firms)
⚠️ Owner compensation materially above/below market rates (inflates/deflates earnings).  
⚠️ Discretionary expenses (cars, travel, “consulting”) embedded in opex.  
⚠️ Tax-motivated accounting (accelerated expenses) distorts comparability.  
⚠️ Related-party transactions (rent, management fees) not at arm’s length.  
⚠️ Working capital managed aggressively near valuation date (supplier stretch, receivable pull-in).  
✅ Green flag: clean reconciliation from reported to adjusted earnings with documentation and stable operating cash conversion.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Professional services | Key person risk | Explicitly model attrition risk; consider higher discount rate or scenario weighting. |
| Manufacturing | Capex intensity | Separate maintenance vs growth capex; normalise through-cycle margins. |
| Retail | Lease commitments | Treat lease costs consistently; consider EV/EBITDAR if relevant. |
| Private credit/finance | Balance-sheet risk | Use P/B or earnings with provisioning controls; capital adequacy is critical. |
| Real estate holding | Asset-based valuation | Use NAV/DCF of property cash flows; separate operating company from property. |

---

### Practical Comparison Tables

#### 1) Private firm adjustment map
| Adjustment | What changes | Why it matters |
|---|---|---|
| Owner comp normalisation | EBIT/EBITDA | Align to market participant earnings |
| Discretionary spend removal | EBIT/FCFF | Removes non-operating owner perks |
| Related-party pricing | Revenue/expenses | Fix non-arm’s-length distortions |
| Working capital normalisation | FCFF | Prevent window dressing |
| DLOM | Equity value | Reflects saleability constraints |
| Control basis | Value basis | Ensures forecast rights match interest valued |

#### 2) DLOM methods (practical)
| Method | Pros | Cons |
|---|---|---|
| Empirical studies | Market-based ranges | Not firm-specific; wide dispersion |
| Option-based | Links to vol and horizon | Input-sensitive; model risk |
| Hold-period heuristic | Simple, transparent | Coarse; may miss key drivers |

---

### Real-World Example
Scenario: A founder-owned company reports $18m EBIT but pays the owner above-market compensation and includes discretionary expenses. You recast EBIT to a market basis, value with a DCF proxy, then apply a marketability discount consistent with a minority interest.

```python
reported_ebit = 18.0
adj_ebit = recast_owner_earnings(reported_ebit, excess_owner_comp=4.0, discretionary_expenses=1.5, one_off_gains=0.8)

print(f"Reported EBIT: ${reported_ebit:.1f}m")
print(f"Adjusted EBIT: ${adj_ebit:.1f}m")
```

Interpretation: The adjusted earnings base anchors both DCF and multiples. Marketability and control adjustments should be applied only after clarifying the interest being valued and ensuring risk is not double-counted via discount rate add-ons.
