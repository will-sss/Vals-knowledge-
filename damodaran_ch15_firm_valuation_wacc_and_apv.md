# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 15: Firm Valuation — Cost of Capital and Adjusted Present Value Approaches (FCFF/WACC, APV, Capital Structure Consistency)

### Core Concept
Firm valuation values the operating assets of the business by discounting **free cash flow to the firm (FCFF)** at the **WACC**, then bridging to equity by subtracting debt-like claims. The APV framework separates operating value (unlevered) from financing side effects (tax shields, distress costs). Practically, choose WACC when leverage is stable and APV when leverage changes materially.

See also: Chapter 8 (building discount rates), Chapter 12 (terminal value), Chapter 16 (equity value per share), Chapter 30 (distressed firms: financing and default risk).

---

### Formula/Methodology

#### 1) FCFF (cash flow to firm)
```text
FCFF = EBIT × (1 − Tax Rate) + Depreciation & Amortization − Capex − ΔNWC
Equivalently:
FCFF = NOPAT + D&A − Capex − ΔNWC
Where:
NOPAT = EBIT × (1 − Tax Rate)
ΔNWC = change in non-cash working capital
```
Implementation note:
- Use **operating** taxes (sustainable marginal or effective operating rate) consistent with operating income definition.
- Keep consistency on lease/SBC/capitalised costs when measuring EBIT and reinvestment.

#### 2) WACC (discount rate for FCFF)
```text
WACC = (E/V) × k_e + (D/V) × k_d × (1 − Tc)
Where:
E = market value of equity
D = market value of debt (debt-like claims)
V = E + D
k_e = cost of equity
k_d = pre-tax cost of debt
Tc = marginal tax rate
```
Practical rule:
- Use **target/long-run** weights when capital structure is expected to revert; avoid transient quarter-end leverage.

#### 3) Enterprise value and equity bridge
```text
Enterprise Value (EV) = PV(FCFF) + PV(Terminal Value)
Equity Value = EV − Net Debt − Other Debt-like Claims + Non-operating Assets

Where:
Net Debt = Debt − Cash (use excess cash concept where relevant)
Other claims = leases (if treated as debt), pensions, preferred, minorities, etc.
Non-operating assets = investments, surplus real estate, etc.
```
Consistency rules:
- If you capitalise leases into debt, EV must subtract lease debt in the bridge and operating costs must be adjusted (EBIT/EBITDA consistency).

#### 4) APV (Adjusted Present Value)
```text
APV = Unlevered Firm Value + PV(Tax Shields) − PV(Expected Distress Costs) − PV(Other Financing Side Effects)

Unlevered Firm Value = PV(FCFF discounted at k_u)
Where:
k_u = unlevered cost of capital (asset cost)
```
When APV is preferred:
- leverage changes significantly over time (LBO, recap, project finance)
- debt policy is driven by financing constraints (distressed firms)

#### 5) Tax shield present value (common practical variants)
If debt is proportional and stable (WACC world), tax shields are embedded in WACC. If using APV:
```text
Tax Shield_t = Tc × Interest_t
PV(Tax Shields) = Σ [Tax Shield_t / (1 + r_ts)^t]
Where:
r_ts = discount rate for tax shields (often approximated by k_d or k_u depending on assumptions)
```
Practical note:
- The “right” r_ts depends on the riskiness of the tax shields; be explicit and run sensitivity.

---

### Python Implementation
```python
from typing import List, Optional, Dict


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def nopat(ebit: float, tax_rate: float) -> float:
    """NOPAT = EBIT * (1 - tax_rate)."""
    if not _num(ebit):
        raise ValueError("ebit must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    return ebit * (1.0 - tax_rate)


def fcff(ebit: float, tax_rate: float, da: float, capex: float, delta_nwc: float) -> float:
    """FCFF = NOPAT + D&A - Capex - ΔNWC."""
    for name, v in {"ebit": ebit, "tax_rate": tax_rate, "da": da, "capex": capex, "delta_nwc": delta_nwc}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if capex < 0:
        raise ValueError("capex must be >= 0")
    return nopat(ebit, tax_rate) + da - capex - delta_nwc


def wacc(equity: float, debt: float, ke: float, kd: float, tax_rate: float) -> Optional[float]:
    """WACC = (E/V)ke + (D/V)kd(1-Tc). Returns None if V<=0."""
    for name, v in {"equity": equity, "debt": debt, "ke": ke, "kd": kd, "tax_rate": tax_rate}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if equity < 0 or debt < 0:
        raise ValueError("equity and debt must be >= 0")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    v = equity + debt
    if v <= 0:
        return None
    return (equity / v) * ke + (debt / v) * kd * (1.0 - tax_rate)


def pv_cash_flows(cash_flows: List[float], r: float) -> float:
    """PV of cash flows discounted at r."""
    if not _num(r) or r <= -1:
        raise ValueError("r must be numeric and > -100%.")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list.")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values.")
        pv += cf / ((1.0 + r) ** t)
    return pv


def terminal_value_perpetuity(cf_next: float, r: float, g: float) -> float:
    """TV = CF_{n+1} / (r - g)."""
    for name, v in {"cf_next": cf_next, "r": r, "g": g}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if r <= g:
        raise ValueError("Need r > g for a finite perpetuity terminal value.")
    return cf_next / (r - g)


def enterprise_value_from_fcff(
    fcff_forecast: List[float],
    discount_rate: float,
    terminal_growth: float
) -> float:
    """EV = PV(FCFF_1..n) + PV(TV_n)."""
    if not isinstance(fcff_forecast, list) or len(fcff_forecast) == 0:
        raise ValueError("fcff_forecast must be non-empty")
    n = len(fcff_forecast)
    pv_stage1 = pv_cash_flows(fcff_forecast, discount_rate)

    fcff_n = fcff_forecast[-1]
    fcff_n1 = fcff_n * (1.0 + terminal_growth)  # only if year n is near-stable
    tv_n = terminal_value_perpetuity(fcff_n1, discount_rate, terminal_growth)

    pv_tv = tv_n / ((1.0 + discount_rate) ** n)
    return pv_stage1 + pv_tv


def equity_bridge_from_ev(
    ev: float,
    debt_like: float,
    cash: float,
    other_claims: float = 0.0,
    non_operating_assets: float = 0.0
) -> float:
    """Equity = EV - (Debt-like - Cash) - Other claims + Non-operating assets."""
    for name, v in {"ev": ev, "debt_like": debt_like, "cash": cash, "other_claims": other_claims, "non_operating_assets": non_operating_assets}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if debt_like < 0 or cash < 0 or other_claims < 0:
        raise ValueError("debt_like, cash, other_claims must be >= 0")
    net_debt = debt_like - cash
    return ev - net_debt - other_claims + non_operating_assets


def apv(
    unlevered_value: float,
    pv_tax_shields: float,
    pv_distress_costs: float = 0.0,
    pv_other_financing: float = 0.0
) -> float:
    """APV = Vu + PV(TS) - PV(distress) - PV(other financing)."""
    for name, v in {"unlevered_value": unlevered_value, "pv_tax_shields": pv_tax_shields,
                    "pv_distress_costs": pv_distress_costs, "pv_other_financing": pv_other_financing}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if pv_tax_shields < 0 or pv_distress_costs < 0 or pv_other_financing < 0:
        raise ValueError("PV components must be >= 0")
    return unlevered_value + pv_tax_shields - pv_distress_costs - pv_other_financing


def pv_tax_shields_from_interest(
    interest_forecast: List[float],
    tax_rate: float,
    discount_rate_ts: float
) -> float:
    """PV(TS) where TS_t = tax_rate * interest_t."""
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    ts = []
    for i in interest_forecast:
        if not _num(i):
            raise ValueError("interest_forecast must be numeric")
        ts.append(tax_rate * i)
    return pv_cash_flows(ts, discount_rate_ts)


# Example usage (all $m)
# WACC method: value firm from FCFF, then bridge to equity.
fcff_forecast = [260, 290, 315, 340, 360]
ke = 0.10
kd = 0.06
E = 5_000
D = 2_500
tax = 0.24

WACC = wacc(E, D, ke, kd, tax)
if WACC is None:
    raise ValueError("Invalid capital structure for WACC")

EV = enterprise_value_from_fcff(fcff_forecast, WACC, terminal_growth=0.03)

equity_value = equity_bridge_from_ev(
    ev=EV,
    debt_like=D,
    cash=400,
    other_claims=150,           # e.g., pension deficit, preferred
    non_operating_assets=200    # e.g., investments
)

print(f"WACC: {WACC:.1%}")
print(f"Enterprise Value: ${EV:,.0f}m")
print(f"Equity Value: ${equity_value:,.0f}m")

# APV method: unlevered value + PV tax shields - PV distress costs.
Vu = enterprise_value_from_fcff(fcff_forecast, discount_rate=0.09, terminal_growth=0.03)  # unlevered discount
pv_ts = pv_tax_shields_from_interest([120, 130, 135, 140, 140], tax_rate=tax, discount_rate_ts=kd)
apv_value = apv(Vu, pv_ts, pv_distress_costs=80)

print(f"Unlevered Value: ${Vu:,.0f}m")
print(f"PV(Tax Shields): ${pv_ts:,.0f}m")
print(f"APV (after distress): ${apv_value:,.0f}m")
```

---

### Valuation Impact
Why this matters:
- Firm valuation is the workhorse for most operating companies; FCFF/WACC is standard because it separates operating performance from financing, then adds financing via the equity bridge.
- APV prevents “modeling by accident” when leverage is changing: WACC is not constant in that world, and using a single WACC can misstate value.
- Consistency across EV, debt-like claims, and operating earnings definitions prevents double-counting (e.g., leases treated both as operating expense and as debt).

Impact on multiples:
- EV multiples require EV to include debt-like claims consistently; if you adjust EBITDA for leases, you must adjust EV for lease debt.
- DCF-derived EV provides a cross-check on market EV and implied multiples; large gaps should reconcile to drivers (ROIC, growth, risk).

Impact on DCF inputs:
- Forecast reinvestment (capex, working capital) must support growth; otherwise FCFF is overstated.
- Terminal value closure is critical: stable growth and WACC must satisfy r > g and reinvestment logic (Chapter 12).

Practical adjustments:
```python
def ev_consistency_check(ev: float, equity: float, net_debt: float, other_claims: float = 0.0) -> None:
    """Check EV ≈ Equity + Net Debt + Other Claims."""
    for v in [ev, equity, net_debt, other_claims]:
        if not _num(v):
            raise ValueError("inputs must be numeric")
    lhs = ev
    rhs = equity + net_debt + other_claims
    if abs(lhs - rhs) > max(1.0, 0.01 * abs(lhs)):
        raise ValueError("EV bridge inconsistency: check debt-like items and cash treatment.")
```

---

### Quality of Earnings Flags (FCFF/WACC/APV)
⚠️ EBIT includes non-operating income/expenses (recast needed) while FCFF is treated as operating.  
⚠️ Capex understated (maintenance capex ignored) leading to inflated FCFF and terminal value.  
⚠️ Working capital modeled as a “plug” to hit cash flow targets (story-to-numbers mismatch).  
⚠️ Using a single WACC despite an explicit plan to materially change leverage (APV is safer).  
⚠️ Debt-like items omitted from EV bridge (leases, pensions, minorities), overstating equity value.  
✅ Green flag: clean separation of operating vs financing items, explicit debt-like definitions, and reconciliation checks.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Financials | Debt is operating | Firm valuation/WACC less meaningful; prefer equity models (Chapter 14). |
| Infrastructure/project finance | Leverage varies | APV often clearer; model tax shields and covenants explicitly. |
| Retail | Lease-heavy | Ensure lease policy consistent in EBIT/EBITDA and EV bridge. |
| Cyclicals | FCFF volatility | Use mid-cycle normalization and stress tests; consider probability-weighted outcomes. |

---

### Practical Comparison Tables

#### 1) WACC vs APV decision rule
| Situation | Use | Why |
|---|---|---|
| Stable leverage / target D/E | WACC | Simple and consistent |
| Leverage changing materially | APV | Separates operating value from financing effects |
| Distressed firms | APV + distress costs | WACC breaks down; distress and default risk are first-order |
| Project finance | APV | Debt schedule and shields are explicit |

#### 2) EV bridge checklist (avoid double counting)
| Item | If treated as debt-like | Earnings/cash flow treatment |
|---|---|---|
| Operating leases | Add PV leases to debt | Remove lease expense from opex; use depreciation + interest-like split |
| Pension deficit | Add to debt-like claims | Keep service cost in EBIT; treat financing cost separately if needed |
| Excess cash | Add as non-operating asset | Do not net against operating cash needs |

---

### Real-World Example
Scenario: A company plans a leveraged recap over 3 years. WACC is not constant, so you compute unlevered value from FCFF at k_u and add PV of tax shields, net of expected distress costs.

```python
fcff = [260, 290, 315, 340, 360]
ku = 0.09
kd = 0.06
tax = 0.24

Vu = enterprise_value_from_fcff(fcff, discount_rate=ku, terminal_growth=0.03)

# Debt increases as part of recap -> rising interest -> explicit tax shields
interest = [90, 120, 150, 160, 160]
pv_ts = pv_tax_shields_from_interest(interest, tax_rate=tax, discount_rate_ts=kd)

# Distress costs rise with leverage; treat as a PV estimate from scenario work
apv_value = apv(Vu, pv_ts, pv_distress_costs=120)

print(f"Vu: ${Vu:,.0f}m, PV(TS): ${pv_ts:,.0f}m, APV: ${apv_value:,.0f}m")
```

Interpretation: APV makes financing effects explicit and prevents hidden WACC tuning. If APV is materially higher than a naive constant-WACC DCF, it may be because the constant WACC overstates discounting during high-leverage years or misprices the tax shields and distress costs.
