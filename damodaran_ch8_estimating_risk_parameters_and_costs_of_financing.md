# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 8: Estimating Risk Parameters and Costs of Financing (Bottom-Up Beta, Cost of Equity, Cost of Debt, WACC/APV Inputs)

### Core Concept
Valuation discount rates should be built from **business risk** (operating exposure) and **financial risk** (leverage), not from noisy single-firm regressions. Practically, you estimate risk parameters using **bottom-up betas** from comparable firms, adjust for operating leverage and country exposure, then compute the **cost of equity**, **cost of debt** (often via a synthetic rating / default spread), and the **WACC** (or APV components). fileciteturn3file2

See also: Chapter 7 (risk-free and ERP), Chapter 15 (WACC/APV valuation), Chapter 16 (equity value per share), Chapter 33 (scenario/simulation for risk ranges).

---

### Formula/Methodology

#### 1) Beta construction: unlever and relever (core mechanics)
```text
Unlevered beta (asset beta approximation):
Beta_U = Beta_L / (1 + (1 − Tc) × D/E)

Relevered beta:
Beta_L = Beta_U × (1 + (1 − Tc) × D/E)

Where:
Beta_L = levered equity beta
Beta_U = unlevered (business) beta
D/E    = market-value debt-to-equity
Tc     = marginal tax rate
```
Practical rules:
- Use **market values** for D and E whenever possible (book D is a fallback when markets are illiquid).
- If the company has material debt-like items (leases, pensions), include them in D for D/E consistency. fileciteturn3file2

#### 2) Bottom-up beta (reduce noise)
```text
Step A: For each comparable i, compute Beta_U,i (unlever each comp beta).
Step B: Take a central tendency (median/mean) of Beta_U across comps.
Step C: Relever to the target’s D/E (target or industry-average).
```
Why: regression betas are unstable for single firms; bottom-up betas diversify estimation error and better reflect business risk. fileciteturn3file2

#### 3) Cost of equity (CAPM baseline with extensions)
```text
k_e = R_f + Beta_L × ERP_total
ERP_total = ERP_mature + CRP_exposure (if applicable)
```
Where:
- Rf and ERP must match cash-flow currency (Chapter 7).
- CRP_exposure can be exposure-weighted by revenue/operating assets if country risk is material.

#### 4) Cost of debt (after-tax) and synthetic rating concept
```text
Pre-tax cost of debt:
k_d = R_f + Default Spread (+ other debt premia if needed)

After-tax cost of debt:
k_d_after_tax = k_d × (1 − Tc)
```
A common practical approach is to estimate default spread from a **synthetic rating**, using interest coverage ratios (ICR) and mapping to spreads.

Interest coverage ratio:
```text
ICR = EBIT / Interest Expense
```
(Alternative for EBITDA-based coverage when EBIT is distorted; be consistent.)

#### 5) WACC (market-value weights)
```text
WACC = (E/V) × k_e + (D/V) × k_d × (1 − Tc)
Where:
V = E + D
```
Key: weights should reflect the **target long-run capital structure**, not a transient quarter-end. fileciteturn3file2

#### 6) APV framing (useful when leverage changes materially)
```text
APV = Unlevered Firm Value + PV(Tax Shields) − PV(Distress/Financing Costs)
```
Use APV when the firm’s D/E is expected to change significantly (LBO, recap, project finance). fileciteturn3file2

---

### Python Implementation
```python
from typing import List, Dict, Optional, Tuple


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def unlever_beta(beta_l: float, debt: float, equity: float, tax_rate: float) -> Optional[float]:
    """
    Beta_U = Beta_L / (1 + (1 - Tc) * D/E). Returns None if equity <= 0.
    """
    for name, v in {"beta_l": beta_l, "debt": debt, "equity": equity, "tax_rate": tax_rate}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if debt < 0:
        raise ValueError("debt must be >= 0")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if equity <= 0:
        return None
    return beta_l / (1.0 + (1.0 - tax_rate) * (debt / equity))


def relever_beta(beta_u: float, debt: float, equity: float, tax_rate: float) -> Optional[float]:
    """
    Beta_L = Beta_U * (1 + (1 - Tc) * D/E). Returns None if equity <= 0.
    """
    for name, v in {"beta_u": beta_u, "debt": debt, "equity": equity, "tax_rate": tax_rate}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if beta_u < 0:
        raise ValueError("beta_u must be >= 0")
    if debt < 0:
        raise ValueError("debt must be >= 0")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if equity <= 0:
        return None
    return beta_u * (1.0 + (1.0 - tax_rate) * (debt / equity))


def bottom_up_beta(comps: List[Dict], tax_rate: float, method: str = "median") -> float:
    """
    Compute a bottom-up unlevered beta from comparables.

    Args:
        comps: list of dicts with keys:
            - beta_l (levered beta)
            - debt (market value, or debt-like proxy)
            - equity (market value)
        tax_rate: marginal tax rate (0-1).
        method: 'median' or 'mean'.

    Returns:
        float: bottom-up unlevered beta (beta_u).

    Raises:
        ValueError: invalid inputs or no valid comps.
    """
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    betas_u = []
    for c in comps:
        bu = unlever_beta(c.get("beta_l"), c.get("debt"), c.get("equity"), tax_rate)
        if bu is not None and _num(bu) and bu >= 0:
            betas_u.append(bu)
    if len(betas_u) == 0:
        raise ValueError("No valid comparables to compute bottom-up beta.")
    betas_u.sort()
    if method == "median":
        mid = len(betas_u) // 2
        return betas_u[mid] if len(betas_u) % 2 == 1 else 0.5 * (betas_u[mid - 1] + betas_u[mid])
    return sum(betas_u) / len(betas_u)


def capm_cost_of_equity(rf: float, beta_l: float, erp: float) -> float:
    """k_e = rf + beta_l * ERP"""
    for name, v in {"rf": rf, "beta_l": beta_l, "erp": erp}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if rf <= -1:
        raise ValueError("rf must be > -100%")
    if erp < 0:
        raise ValueError("erp must be >= 0")
    return rf + beta_l * erp


def interest_coverage_ratio(ebit: float, interest_expense: float) -> Optional[float]:
    """ICR = EBIT / Interest. Returns None if interest <= 0."""
    if not _num(ebit) or not _num(interest_expense):
        raise ValueError("inputs must be numeric")
    if interest_expense <= 0:
        return None
    return ebit / interest_expense


def synthetic_default_spread_from_icr(icr: float, table: List[Tuple[float, float]]) -> float:
    """
    Map ICR to a default spread using a provided table of (min_icr, spread).

    Args:
        icr: interest coverage ratio.
        table: sorted descending by min_icr, e.g. [(8.0, 0.010), (6.0,0.012)...].

    Returns:
        float: default spread (decimal).

    Raises:
        ValueError: invalid inputs or empty table.
    """
    if not _num(icr):
        raise ValueError("icr must be numeric")
    if not isinstance(table, list) or len(table) == 0:
        raise ValueError("table must be a non-empty list")
    # iterate thresholds from high to low; pick first matching
    for min_icr, spread in table:
        if not _num(min_icr) or not _num(spread):
            raise ValueError("table entries must be numeric")
        if icr >= min_icr:
            if spread < 0:
                raise ValueError("spread must be >= 0")
            return spread
    # if below all thresholds, use last spread
    last_spread = table[-1][1]
    if last_spread < 0:
        raise ValueError("spread must be >= 0")
    return last_spread


def cost_of_debt(rf: float, default_spread: float, extra_premium: float = 0.0) -> float:
    """k_d = rf + default_spread + extra_premium"""
    for name, v in {"rf": rf, "default_spread": default_spread, "extra_premium": extra_premium}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if default_spread < 0 or extra_premium < 0:
        raise ValueError("spreads/premia must be >= 0")
    return rf + default_spread + extra_premium


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


# Example usage: bottom-up beta + synthetic cost of debt + WACC (all in decimals, $ currency)
rf = 0.042
mature_erp = 0.050
tax = 0.24

comparables = [
    {"beta_l": 1.10, "debt": 2_000, "equity": 8_000},
    {"beta_l": 0.95, "debt": 1_500, "equity": 6_500},
    {"beta_l": 1.25, "debt": 3_000, "equity": 7_000},
]

beta_u = bottom_up_beta(comparables, tax_rate=tax, method="median")

# Target leverage (market-value): D/E = 0.35
target_equity = 10_000
target_debt = 3_500
beta_l_target = relever_beta(beta_u, debt=target_debt, equity=target_equity, tax_rate=tax) or 0.0
ke = capm_cost_of_equity(rf, beta_l_target, mature_erp)

# Synthetic rating table (illustrative; replace with your firm-standard mapping)
# Each tuple: (min_ICR, default_spread)
spread_table = [
    (8.0, 0.010),
    (6.0, 0.012),
    (4.5, 0.015),
    (3.0, 0.020),
    (2.0, 0.030),
    (1.0, 0.050),
    (0.0, 0.080),
]

icr = interest_coverage_ratio(ebit=900, interest_expense=180)  # 5.0x
default_spread = synthetic_default_spread_from_icr(icr, spread_table)
kd = cost_of_debt(rf, default_spread)

w = wacc(equity=target_equity, debt=target_debt, ke=ke, kd=kd, tax_rate=tax) or 0.0

print(f"Bottom-up beta_u: {beta_u:.2f}")
print(f"Relevered beta_l: {beta_l_target:.2f}")
print(f"Cost of equity:   {ke:.1%}")
print(f"ICR:              {icr:.2f}x")
print(f"Default spread:   {default_spread:.1%}")
print(f"Cost of debt:     {kd:.1%}")
print(f"WACC:             {w:.1%}")
```

---

### Valuation Impact
Why this matters:
- **Bottom-up beta** reduces estimation noise and forces clarity on what drives risk (business mix) vs what changes with leverage (financial policy). fileciteturn3file2  
- Synthetic debt costs anchor WACC even when traded bonds/credit curves are unavailable (common in private-company valuations).  
- Errors in beta, ERP, default spread, or weights can dominate value when terminal value is large (high duration).

Impact on multiples:
- Higher beta/leverage → higher discount rates → lower justified P/E and EV/EBITDA, all else equal.
- Synthetic ratings help explain why “cheap” multiples can be justified by weak coverage and distress risk (not undervaluation).

Impact on DCF inputs:
- Use target (long-run) capital structure in WACC (not current temporary debt).
- Make debt-like items explicit (leases, pensions) so D/E and WACC match enterprise-to-equity bridges.

Practical adjustments:
```python
def debt_like_adjustment(interest_bearing_debt: float, pv_leases: float = 0.0, pension_deficit: float = 0.0) -> float:
    """Create a debt-like measure for leverage and EV bridge consistency."""
    for v in [interest_bearing_debt, pv_leases, pension_deficit]:
        if not _num(v):
            raise ValueError("all inputs must be numeric")
    if interest_bearing_debt < 0 or pv_leases < 0 or pension_deficit < 0:
        raise ValueError("inputs must be >= 0")
    return interest_bearing_debt + pv_leases + pension_deficit
```

---

### Quality of Earnings Flags (Risk Parameter Pitfalls)
⚠️ Using a single regression beta for a thinly traded or recently listed firm (unstable).  
⚠️ Relevering with book D/E while equity is market-valued (inconsistent).  
⚠️ Ignoring leases/pensions when peers include them in debt-like measures (comparability break).  
⚠️ Interest coverage boosted by one-offs or capitalised costs (synthetic rating too optimistic).  
⚠️ Using effective tax rate when the relevant rate for debt tax shields is marginal and sustainable.  
✅ Green flag: bottom-up beta from a defensible peer set, with leverage and country exposure stated explicitly and sensitivity analysis shown.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Utilities | Regulated stability | Lower beta; debt capacity higher; synthetic spreads often tight unless regulatory risk. |
| Software | SBC and intangible reinvestment | Ensure EBIT/ICR uses consistent SBC treatment; consider higher uncertainty despite low current leverage. |
| Cyclical industrials | Coverage collapses in downturns | Use trough EBIT for ICR mapping or run stress ICR scenarios and spreads. |
| Financials | Operating leverage is leverage | WACC framework differs; focus on cost of equity and regulatory capital metrics rather than synthetic debt spreads. |

---

### Practical Comparison Tables

#### 1) Risk parameter build (bottom-up) checklist
| Step | What you do | Common mistake |
|---|---|---|
| Define peers | Same business drivers, not just same SIC | “Industry” peers with different economics |
| Unlever betas | Remove financial leverage effect | Use book D/E with market betas |
| Average beta_u | Median/trimmed mean | Let an outlier dominate |
| Relever to target | Use long-run D/E | Use current distressed D/E |
| Add ERP/CRP | Currency + exposure consistent | Mix currencies or ignore exposure |
| Cost of debt | Synthetic rating or observed yields | Use too-optimistic coverage from one-off EBIT |

#### 2) Synthetic cost of debt: implementation notes
| Input | Better practice | Why |
|---|---|---|
| EBIT for ICR | Use normalised or through-the-cycle EBIT | Avoid optimistic spreads in peak earnings |
| Interest expense | Use current run-rate or expected | Avoid temporary low rates misleading kd |
| Spread table | Use your firm’s standard mapping | Consistency across valuations |
| Tax rate | Use sustainable marginal | Align with tax shield economics |

---

### Real-World Example
Scenario: A private industrial firm has no traded beta and no traded debt. You build a bottom-up beta from public peers, relever to target D/E, estimate a synthetic spread from interest coverage, and compute WACC for FCFF discounting.

```python
rf = 0.042
erp = 0.050
tax = 0.24

peers = [
    {"beta_l": 1.05, "debt": 2500, "equity": 7500},
    {"beta_l": 0.90, "debt": 1800, "equity": 8200},
    {"beta_l": 1.20, "debt": 3200, "equity": 6800},
]

beta_u = bottom_up_beta(peers, tax_rate=tax)
target_debt = 2_800
target_equity = 8_000

beta_l = relever_beta(beta_u, target_debt, target_equity, tax_rate=tax) or 0.0
ke = capm_cost_of_equity(rf, beta_l, erp)

icr = interest_coverage_ratio(ebit=520, interest_expense=130)  # 4.0x
spread_table = [(8.0, 0.010), (6.0, 0.012), (4.5, 0.015), (3.0, 0.020), (2.0, 0.030), (1.0, 0.050), (0.0, 0.080)]
spread = synthetic_default_spread_from_icr(icr, spread_table)
kd = cost_of_debt(rf, spread)

w = wacc(target_equity, target_debt, ke, kd, tax) or 0.0

print(f"beta_u={beta_u:.2f}, beta_l={beta_l:.2f}")
print(f"ke={ke:.1%}, kd={kd:.1%}, WACC={w:.1%}")
```

Interpretation: This builds a coherent, auditable discount-rate stack from observable peer risk and the firm’s financing capacity. For cyclical firms, repeat the exercise using stressed EBIT to see how spreads and WACC behave in downturns.
