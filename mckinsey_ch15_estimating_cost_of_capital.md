# McKinsey Valuation - Key Concepts for Valuation

## Chapter 15: Estimating the Cost of Capital (WACC, CAPM, Beta, Cost of Debt, Weights, Cross-Checks)

### Core Concept
The cost of capital is the discount rate that converts forecast cash flows into present value. In enterprise DCFs, WACC is used to discount free cash flow to the firm (FCFF) and must be consistent with the business risk, capital structure assumptions, and tax treatment. Practical accuracy comes from disciplined inputs (risk-free rate, equity risk premium, beta, credit spread) and robust cross-checks (implied spreads, reasonableness vs peers, sensitivity ranges).

See also: Chapter 14 on continuing value (WACC vs g stability); Chapter 16 on moving from enterprise value to value per share (net debt and debt-like items); Chapter 33 on capital structure, dividends, and buybacks.

---

### Core Concept: Match Discount Rate to Cash Flow Definition
Use a discount rate that matches the cash flow:
- FCFF → WACC (after corporate tax shield on debt in WACC formula)
- FCFE → Cost of equity (Re), with leverage explicitly reflected in cash flows

Common error: discounting FCFF with cost of equity (over-discounts or misprices leverage effects).

---

### Formula/Methodology

#### 1) Weighted Average Cost of Capital (WACC)
```text
WACC = (E/V) × Re + (D/V) × Rd × (1 − Tc)
```
Where:
- E: market value of equity
- D: market value of interest-bearing debt (use market if available; otherwise book as proxy with judgement)
- V: total firm value = E + D
- Re: cost of equity
- Rd: pre-tax cost of debt
- Tc: marginal corporate tax rate (or sustainable statutory rate if marginal not observable)

Practical notes:
- Use target capital structure when forecasting long-run steady state; use current structure if it is stable and expected to persist.
- If material debt-like items exist (leases, pensions, certain provisions), consider whether they should be included in D and in EV consistently.

#### 2) Cost of Equity (CAPM)
```text
Re = Rf + βL × ERP + Size/Specific Risk Premium (if justified)
```
Where:
- Rf: risk-free rate (currency-consistent with cash flows; maturity matched to horizon in practice)
- βL: levered beta (equity beta)
- ERP: equity risk premium (market)
- Premiums: use only with explicit justification; avoid double-counting risk already embedded in cash flows or beta

#### 3) Unlevering and Relevering Beta (Capital Structure Normalisation)
```text
βU = βL / (1 + (1 − Tc) × (D/E))

βL_target = βU × (1 + (1 − Tc) × (D/E)_target)
```
Where:
- βU: unlevered (asset) beta
- (D/E): debt-to-equity ratio (market-value basis preferred)

#### 4) Cost of Debt (Rd)
```text
Rd = Risk-free Rate + Credit Spread
```
Practical ways to estimate credit spread:
- Use observed yields on the company’s traded debt (best if liquid)
- Infer from credit rating and current spreads (proxy approach)
- Use interest expense / average debt as a starting point, then adjust to current market conditions

---

### Python Implementation
```python
from typing import Optional, Dict, List, Tuple

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def wacc(equity_value: float, debt_value: float, cost_of_equity: float, cost_of_debt: float, tax_rate: float) -> float:
    """
    Compute WACC.

    Args:
        equity_value: Market value of equity, E (currency, >= 0)
        debt_value: Market value of debt, D (currency, >= 0)
        cost_of_equity: Re (decimal, >= 0)
        cost_of_debt: Rd pre-tax (decimal, >= 0)
        tax_rate: Tc (decimal in [0,1))

    Returns:
        float: WACC (decimal)

    Raises:
        ValueError: invalid inputs or V = 0.
    """
    for name, v in {"equity_value": equity_value, "debt_value": debt_value}.items():
        if not _num(v) or v < 0:
            raise ValueError(f"{name} must be numeric and >= 0")
    for name, v in {"cost_of_equity": cost_of_equity, "cost_of_debt": cost_of_debt}.items():
        if not _num(v) or v < 0:
            raise ValueError(f"{name} must be numeric and >= 0")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0, 1)")
    v_total = equity_value + debt_value
    if v_total <= 0:
        raise ValueError("Total firm value V = E + D must be > 0")
    w_e = equity_value / v_total
    w_d = debt_value / v_total
    return w_e * cost_of_equity + w_d * cost_of_debt * (1.0 - tax_rate)


def cost_of_equity_capm(rf: float, beta_levered: float, erp: float, extra_premium: float = 0.0) -> float:
    """
    CAPM-based cost of equity.

    Re = Rf + beta * ERP + extra_premium

    Args:
        rf: risk-free rate (decimal, can be 0+; negative allowed if market is negative, but validate)
        beta_levered: levered beta (>= 0 typically; allow small negatives but flag via validation)
        erp: equity risk premium (decimal, >= 0)
        extra_premium: additional premium (decimal, >= 0). Use only if justified.

    Returns:
        float: Re (decimal)

    Raises:
        ValueError: invalid inputs.
    """
    for name, v in {"rf": rf, "beta_levered": beta_levered, "erp": erp, "extra_premium": extra_premium}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if erp < 0 or extra_premium < 0:
        raise ValueError("erp and extra_premium must be >= 0")
    # beta can be negative for hedges; for operating companies, negative beta is unusual
    if beta_levered < -1.0:
        raise ValueError("beta_levered is implausibly low; check input")
    return rf + beta_levered * erp + extra_premium


def unlever_beta(beta_levered: float, debt_to_equity: float, tax_rate: float) -> float:
    """
    Unlever beta using Hamada adjustment.

    beta_u = beta_l / (1 + (1 - Tc) * D/E)
    """
    for name, v in {"beta_levered": beta_levered, "debt_to_equity": debt_to_equity}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0, 1)")
    if debt_to_equity < 0:
        raise ValueError("debt_to_equity must be >= 0")
    denom = 1.0 + (1.0 - tax_rate) * debt_to_equity
    if denom <= 0:
        raise ValueError("Invalid denominator in unlevering; check inputs")
    return beta_levered / denom


def relever_beta(beta_unlevered: float, debt_to_equity_target: float, tax_rate: float) -> float:
    """
    Relever beta to a target capital structure.

    beta_l_target = beta_u * (1 + (1 - Tc) * D/E_target)
    """
    if not _num(beta_unlevered):
        raise ValueError("beta_unlevered must be numeric")
    if not _num(debt_to_equity_target) or debt_to_equity_target < 0:
        raise ValueError("debt_to_equity_target must be numeric and >= 0")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0, 1)")
    return beta_unlevered * (1.0 + (1.0 - tax_rate) * debt_to_equity_target)


def cost_of_debt(rf: float, credit_spread: float) -> float:
    """
    Rd = Rf + credit spread.

    Args:
        rf: risk-free rate (decimal)
        credit_spread: spread over rf (decimal, >= 0)

    Returns:
        float: Rd pre-tax (decimal)
    """
    if not _num(rf):
        raise ValueError("rf must be numeric")
    if not _num(credit_spread) or credit_spread < 0:
        raise ValueError("credit_spread must be numeric and >= 0")
    return rf + credit_spread


def implied_credit_spread(rd: float, rf: float) -> float:
    """
    Credit spread implied by Rd - Rf.
    """
    if not _num(rd) or not _num(rf):
        raise ValueError("rd and rf must be numeric")
    spread = rd - rf
    if spread < -0.01:
        raise ValueError("Implied spread is materially negative; check inputs")
    return max(0.0, spread)


# Example usage
inputs = {
    "rf": 0.04,
    "erp": 0.05,
    "beta_l": 1.15,
    "tax": 0.25,
    "credit_spread": 0.02,
    "equity_value": 5_500_000_000,
    "debt_value": 1_800_000_000
}

re = cost_of_equity_capm(inputs["rf"], inputs["beta_l"], inputs["erp"])
rd = cost_of_debt(inputs["rf"], inputs["credit_spread"])
w = wacc(inputs["equity_value"], inputs["debt_value"], re, rd, inputs["tax"])

print(f"Cost of equity (Re): {re:.1%}")
print(f"Cost of debt (Rd): {rd:.1%}")
print(f"WACC: {w:.1%}")

# Beta normalisation example
d_to_e = inputs["debt_value"] / inputs["equity_value"] if inputs["equity_value"] > 0 else 0.0
beta_u = unlever_beta(inputs["beta_l"], d_to_e, inputs["tax"])
beta_l_target = relever_beta(beta_u, debt_to_equity_target=0.50, tax_rate=inputs["tax"])
print(f"Unlevered beta: {beta_u:.2f}")
print(f"Relevered beta (target D/E=0.50): {beta_l_target:.2f}")
```

---

### Valuation Impact
Why this matters:
- WACC is a primary driver of DCF value; small changes can materially change enterprise value and implied multiples.
- Beta normalisation and target leverage ensure comparability across companies and consistency with a steady-state capital structure.
- Rd and tax assumptions embed the value of the interest tax shield; errors here can overstate or understate value systematically.

Impact on multiples:
- A lower WACC supports higher valuation multiples mechanically; sanity-check that implied multiples are consistent with peer risk, growth, and ROIC.
- If your DCF implies a multiple far above peers, check WACC inputs first (Rf, ERP, beta, debt spread) before changing operating forecasts.

Impact on DCF inputs:
- Use nominal rates with nominal cash flows and real rates with real cash flows (consistency check).
- Align tax rate used in WACC (Tc) with how taxes are modelled in FCFF; avoid double-counting tax effects.

Practical adjustments:
```python
def adjust_wacc_for_target_leverage(rf: float, erp: float, beta_l_observed: float,
                                   d_over_e_observed: float, d_over_e_target: float,
                                   tax_rate: float, credit_spread: float,
                                   equity_value_target: float, debt_value_target: float,
                                   extra_premium: float = 0.0) -> float:
    """
    Compute WACC using a target leverage by unlevering and relevering beta.

    Returns:
        float: target WACC
    """
    beta_u = unlever_beta(beta_l_observed, d_over_e_observed, tax_rate)
    beta_l_t = relever_beta(beta_u, d_over_e_target, tax_rate)
    re = cost_of_equity_capm(rf, beta_l_t, erp, extra_premium=extra_premium)
    rd = cost_of_debt(rf, credit_spread)
    return wacc(equity_value_target, debt_value_target, re, rd, tax_rate)
```

---

### Quality of Earnings / Risk Flags (Cost of Capital Red Flags)
⚠️ Using book-value weights when market values are materially different (especially for distressed or high-growth companies).  
⚠️ Beta taken “as is” without adjusting for leverage differences vs peers (D/E mismatch).  
⚠️ “Company-specific premium” added on top of conservative cash flows (double-counting risk).  
⚠️ Rd derived from historical interest expense despite major changes in market rates or capital structure.  
⚠️ Mixing currencies (e.g., USD cash flows with GBP risk-free rate).  
✅ Clear disclosure of each input source and sensitivity analysis (WACC range, g range).

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Regulated utilities | Allowed returns and regulated leverage | Use regulatory capital structure assumptions; cross-check vs allowed ROE/WACC |
| Banks/insurers | WACC not the right tool for core operations | Prefer equity-based valuation (cost of equity, ROE, cost of capital by business line) |
| Cyclicals | Credit spreads and betas move with cycle | Use mid-cycle spreads/betas or scenario ranges; avoid single-point peak spreads |
| High-growth / venture-like | Beta unstable; capital structure evolving | Use industry asset betas; rely on scenario analysis and conservative premiums only with clear logic |
| Real estate / infrastructure | Debt-like leverage and asset-specific risk | Consider project-level discount rates; ensure debt-like items are consistently in EV bridge |

---

### Practical Cross-Checks (Use in Valuation Review)

#### 1) WACC reasonableness vs peers and history
```text
Check:
- Is beta consistent with comparable companies after leverage adjustment?
- Does Rd match current credit conditions (spreads for similar rating/tenor)?
- Is Tc aligned with how taxes are modelled in FCFF?
```

#### 2) Gordon stability (WACC vs g)
```text
Requirement (perpetuity method):
WACC > g

If WACC is close to g, terminal value becomes hypersensitive.
Consider tightening g, extending explicit forecast, or validating steady-state economics.
```

#### 3) Implied spread check (from observed yield or interest expense)
```text
Implied spread = Rd − Rf

If implied spread is materially negative or far outside market levels, re-check debt inputs.
```

---

### Real-World Example (Compute WACC + Sensitivity Grid)

Scenario: Estimate WACC and show sensitivity to beta and credit spread.

```python
def wacc_sensitivity(rf: float, erp: float, beta_list: List[float], spread_list: List[float],
                     tax_rate: float, E: float, D: float) -> List[Tuple[float, float, float]]:
    """
    Return a list of (beta, spread, wacc) tuples.
    """
    out = []
    for b in beta_list:
        re = cost_of_equity_capm(rf, b, erp)
        for s in spread_list:
            rd = cost_of_debt(rf, s)
            out.append((b, s, wacc(E, D, re, rd, tax_rate)))
    return out

grid = wacc_sensitivity(
    rf=0.04, erp=0.05,
    beta_list=[0.9, 1.1, 1.3],
    spread_list=[0.015, 0.020, 0.030],
    tax_rate=0.25,
    E=5_500_000_000,
    D=1_800_000_000
)

for b, s, w in grid:
    print(f"beta={b:.1f}, spread={s:.1%} -> WACC={w:.1%}")
```

Interpretation:
- If valuation is extremely sensitive across reasonable beta/spread ranges, prioritise better input evidence (peer beta set, credit spread) and widen scenario disclosure.
