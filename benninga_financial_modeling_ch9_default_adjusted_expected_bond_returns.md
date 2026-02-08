# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 9: Calculating Default-Adjusted Expected Bond Returns (PD/LGD, expected yield, spread decomposition)

### Core Concept
A bond’s promised yield is not its expected return when default risk exists. Default-adjusted expected return decomposes yield into (i) compensation for time value (risk-free), (ii) expected credit loss (driven by probability of default and loss given default), and (iii) any remaining risk premium for bearing systematic credit risk and liquidity risk. In valuation, this framework supports consistent discounting for risky debt cash flows, credit spread selection, and stress testing capital structure and equity value.

### Formula/Methodology

#### 1) Expected payoff with default (one-period, simplified)
```text
Expected payoff at t=1:
E[Payoff] = (1 - PD) × (Principal + Coupon) + PD × Recovery

Where:
PD = probability of default over the period (decimal)
Recovery = recovery value if default occurs (currency)
```

If recovery is expressed as a fraction of par (Recovery rate, RR):
```text
Recovery = RR × Principal
LGD = 1 - RR
```

#### 2) Expected return (one-period, from price)
```text
Expected return:
E[R] = E[Payoff] / Price_0 - 1
```

#### 3) Expected credit loss (ECL) and credit loss rate
```text
Expected credit loss (currency):
ECL = PD × LGD × Exposure

Expected loss rate (as % of exposure):
EL = PD × LGD
```

Where:
Exposure is typically par (or current outstanding principal).

#### 4) Linking spread to expected loss vs risk premium (intuition)
For small rates and short horizons, a rough decomposition is:
```text
Credit spread ≈ Expected loss rate + Risk premium + Liquidity premium
```

A practical approximation for expected loss component per year:
```text
Expected loss component ≈ PD_annual × LGD
```

#### 5) Multi-period expected PV with default (survival approach)
Let S(t) be survival probability to time t (no default up to t). If default can occur between coupon dates, a discrete approximation:
```text
PV = Σ_{t=1..N} [CF_t × S(t)] / (1 + r)^t + Σ_{t=1..N} [Recovery_t × (S(t-1) - S(t))] / (1 + r)^t

Where:
S(0) = 1
S(t) = Π_{k=1..t} (1 - PD_k)
r = discount rate used for PV (often risk-free for expected cash flows; or include risk premium separately)
```

Interpretation:
- Coupon/principal paid only if the bond survives to payment date.
- Recovery paid in default states, weighted by default probability in the period.

---

### Practical Application (How to use default-adjusted returns in valuation)

#### Step 1: Decide your modeling stance (expected cash flows vs promised cash flows)
Two consistent approaches:

**A) Expected cash flow approach (recommended for transparency)**
- Model expected CFs using PD/LGD and survival probabilities.
- Discount expected CFs at a rate aligned to risk-free + appropriate risk premium (or risk-free if risk premiums are handled elsewhere).

**B) Promised cash flow approach (market-yield approach)**
- Discount promised CFs at a risky yield that embeds expected loss + risk premium + liquidity.
- Good for “market” valuations when bonds trade and yields are observable.

Avoid mixing:
- Do not apply PD/LGD haircuts to cash flows and also discount at full risky yield without careful rationale (double-counting credit risk).

#### Step 2: Source and calibrate PD and LGD
- PD can come from rating transition/default tables, CDS-implied hazard rates, or internal credit models.
- LGD from historical recoveries by seniority/sector or deal terms (secured vs unsecured).
- Use horizon-consistent PD: annual vs cumulative.

#### Step 3: Convert PD into survival probabilities and expected cash flows
- Use period-by-period PDs, compute S(t), then expected coupons/principal.
- Model recovery timing: immediate at default vs paid at end of period (choose one assumption and document).

#### Step 4: Use outputs for valuation decisions
- Debt market value (MTM): PV of expected CFs or promised CFs at market yields.
- Equity value sensitivity: changes in PD/LGD affect net debt and interest expense (capital structure feedback).
- WACC / discount rate narrative: explain if credit spreads are consistent with implied PD/LGD and sector conditions.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def one_period_expected_return(price0: float, principal: float, coupon: float, pd: float, recovery_rate: float) -> float:
    """Compute one-period default-adjusted expected return.

    Args:
        price0: current price (currency)
        principal: principal due at period end (currency)
        coupon: coupon due at period end (currency)
        pd: probability of default over the period (decimal)
        recovery_rate: recovery rate as fraction of principal (decimal)

    Returns:
        float: expected return E[R] (decimal)

    Raises:
        ValueError: invalid inputs.
    """
    p0 = _num(price0, "price0")
    prin = _num(principal, "principal")
    cpn = _num(coupon, "coupon")
    pd = _num(pd, "pd")
    rr = _num(recovery_rate, "recovery_rate")

    if p0 <= 0:
        raise ValueError("price0 must be > 0.")
    if prin < 0 or cpn < 0:
        raise ValueError("principal and coupon must be >= 0.")
    if not (0.0 <= pd <= 1.0):
        raise ValueError("pd must be between 0 and 1.")
    if not (0.0 <= rr <= 1.0):
        raise ValueError("recovery_rate must be between 0 and 1.")

    recovery = rr * prin
    expected_payoff = (1.0 - pd) * (prin + cpn) + pd * recovery
    return expected_payoff / p0 - 1.0

def expected_loss_rate(pd: float, lgd: float) -> float:
    """Expected loss rate: EL = PD * LGD."""
    pd = _num(pd, "pd")
    lgd = _num(lgd, "lgd")
    if not (0.0 <= pd <= 1.0):
        raise ValueError("pd must be between 0 and 1.")
    if not (0.0 <= lgd <= 1.0):
        raise ValueError("lgd must be between 0 and 1.")
    return pd * lgd

def survival_probabilities(pds: List[float]) -> List[float]:
    """Compute survival probabilities S(t) for t=1..N given period PDs.

    S(t) = Π_{k=1..t} (1 - PD_k)

    Args:
        pds: list of PD_k for periods 1..N

    Returns:
        list of S(t) for t=1..N

    Raises:
        ValueError: invalid inputs.
    """
    if not pds:
        raise ValueError("pds must not be empty.")
    s = 1.0
    out = []
    for i, pd in enumerate(pds, start=1):
        pd = _num(pd, f"pds[{i}]")
        if not (0.0 <= pd <= 1.0):
            raise ValueError("Each PD must be between 0 and 1.")
        s *= (1.0 - pd)
        out.append(s)
    return out

def pv_expected_cashflows_with_default(
    cashflows: List[float],
    discount_rate: float,
    pds: List[float],
    principal_exposure: float,
    recovery_rate: float,
    recovery_timing: str = "end_of_period"
) -> float:
    """Present value of expected cash flows with default using survival probabilities (discrete-time).

    PV = Σ CF_t * S(t)/(1+r)^t + Σ Recovery_t * (S(t-1)-S(t))/(1+r)^t

    Assumptions:
        - cashflows are scheduled payments (including principal at maturity if included)
        - recovery paid either at end_of_period (discount to t) or immediately (discount to t-1 approx)

    Args:
        cashflows: scheduled CF_t for t=1..N (currency)
        discount_rate: r per period (decimal)
        pds: PD per period (same length as cashflows)
        principal_exposure: exposure used for recovery (typically par principal)
        recovery_rate: RR (decimal)
        recovery_timing: 'end_of_period' or 'start_of_period'

    Returns:
        float: PV (currency)

    Raises:
        ValueError: invalid inputs.
    """
    if len(cashflows) != len(pds):
        raise ValueError("cashflows and pds must have the same length.")
    r = _num(discount_rate, "discount_rate")
    if r <= -0.99:
        raise ValueError("discount_rate must be > -0.99.")
    expo = _num(principal_exposure, "principal_exposure")
    rr = _num(recovery_rate, "recovery_rate")
    if expo < 0:
        raise ValueError("principal_exposure must be >= 0.")
    if not (0.0 <= rr <= 1.0):
        raise ValueError("recovery_rate must be between 0 and 1.")
    if recovery_timing not in {"end_of_period", "start_of_period"}:
        raise ValueError("recovery_timing must be 'end_of_period' or 'start_of_period'.")

    # survival probabilities S(t)
    S = survival_probabilities(pds)
    pv = 0.0

    S_prev = 1.0
    for t, (cf, s_t, pd_t) in enumerate(zip(cashflows, S, pds), start=1):
        cf = _num(cf, f"cashflows[{t}]")
        # Expected scheduled cashflow if survives to t
        pv += (cf * s_t) / ((1.0 + r) ** t)

        # Default probability in period t: S(t-1) - S(t)
        default_prob = S_prev - s_t
        recovery = rr * expo

        if recovery_timing == "end_of_period":
            pv += (recovery * default_prob) / ((1.0 + r) ** t)
        else:
            # approximate as received at start of period (t-1)
            pv += (recovery * default_prob) / ((1.0 + r) ** (t - 1))

        S_prev = s_t

    return pv

def implied_spread_components(risky_yield: float, risk_free_yield: float, pd_annual: float, lgd: float) -> Dict[str, float]:
    """Approximate spread decomposition: spread ≈ expected loss + residual (risk+liquidity).

    Args:
        risky_yield: observed risky yield (decimal)
        risk_free_yield: risk-free yield (decimal)
        pd_annual: annual PD (decimal)
        lgd: loss given default (decimal)

    Returns:
        dict: {'spread', 'expected_loss', 'residual'}

    Raises:
        ValueError: invalid inputs.
    """
    ry = _num(risky_yield, "risky_yield")
    rf = _num(risk_free_yield, "risk_free_yield")
    el = expected_loss_rate(pd_annual, lgd)
    spread = ry - rf
    return {"spread": spread, "expected_loss": el, "residual": spread - el}

# Example usage (illustrative)
price0 = 980.0
principal = 1_000.0
coupon = 50.0
pd = 0.03
recovery_rate = 0.40

er = one_period_expected_return(price0, principal, coupon, pd, recovery_rate)
print(f"One-period expected return: {er:.2%}")

# Multi-period PV example: 5-year annual coupon bond, with constant PD
n = 5
cashflows = [50.0] * (n - 1) + [1_050.0]
pds = [0.03] * n
pv = pv_expected_cashflows_with_default(cashflows, discount_rate=0.03, pds=pds, principal_exposure=1_000.0, recovery_rate=0.40)
print(f"PV of expected CFs (default-adjusted): ${pv:.2f}")

# Spread decomposition (rough)
comp = implied_spread_components(risky_yield=0.075, risk_free_yield=0.035, pd_annual=0.03, lgd=0.60)
print({k: f"{v:.2%}" for k, v in comp.items()})
```

---

### Valuation Impact
Why this matters:
- Credit spreads embed expected losses plus risk and liquidity premia; understanding the split helps justify discount rates and avoid inconsistent treatment across valuation models.
- For distressed or high-yield capital structures, small changes in PD/LGD assumptions can materially change net debt (market value), equity value, and implied multiples.

Impact on multiples:
- If EV uses market value of debt (or if debt trades at a discount/premium), EV/EBITDA can move significantly with PD/LGD or spreads.
- Comparing firms with different credit quality using the same multiple without considering spread and default risk can mis-rank “cheap vs expensive.”

Impact on DCF inputs:
- In APV, discounting tax shields or distress costs requires explicit credit risk assumptions; PD/LGD are core.
- For FCFF valuation, WACC includes cost of debt; if cost of debt is based on a spread that already reflects expected loss, do not separately haircut cash flows for default.

Comparability issues across companies:
- PD differs by rating, leverage, covenant package, and sector cyclicality; LGD differs by collateral/seniority.
- Two bonds with the same yield can have different expected returns if one has lower recovery prospects.

Practical adjustments:
```python
def normalize_credit_spread(observed_spread: float, target_pd: float, target_lgd: float) -> float:
    """Simple normalization: subtract expected loss component for one credit profile and add for another."""
    sp = _num(observed_spread, "observed_spread")
    el_target = expected_loss_rate(target_pd, target_lgd)
    # Placeholder: in practice you may also adjust for liquidity and risk premium differences
    return el_target + max(0.0, sp - el_target)
```

---

### Quality of Earnings Flags
⚠️ Rapidly rising interest expense or refinancing at higher yields not reflected in forecasts (understated debt cost).  
⚠️ Large “one-off” gains from debt repurchases at a discount boosting earnings/FCF (non-recurring; adjust for sustainable earnings).  
⚠️ Covenant breaches, payment-in-kind (PIK) toggles, or maturity walls indicating elevated PD not captured in assumed spreads.  
✅ Transparent disclosure of debt terms, maturity profile, and refinancing plans; conservative stress cases on PD/LGD/spreads.

---

### Sector-Specific Considerations

| Sector | Key default-return issue | Typical handling |
|---|---|---|
| Cyclical industrials | PD is state-dependent (downturn spikes) | scenario-weighted PDs; mid-cycle normalization |
| Real estate | collateral affects LGD | model LGD by asset LTV and valuation haircuts |
| Financials | default modeling differs; recovery complex | use regulatory capital and asset quality metrics |
| Utilities | lower PD but event risk | include tail-risk scenarios; regulatory stress |

---

### Real-World Example
Scenario: A company’s bonds trade at 90 with a 9% yield. You want an expected return and PV of expected payments, then assess how this impacts EV and equity.

```python
price0 = 900.0
principal = 1_000.0
coupon = 80.0
pd = 0.06
recovery_rate = 0.35

er = one_period_expected_return(price0, principal, coupon, pd, recovery_rate)
print(f"Expected return (1y): {er:.2%}")

cashflows = [80.0, 80.0, 80.0, 80.0, 1_080.0]
pds = [0.06] * 5
pv = pv_expected_cashflows_with_default(cashflows, discount_rate=0.04, pds=pds, principal_exposure=1_000.0, recovery_rate=0.35)
print(f"PV of expected CFs at 4% base discount: ${pv:.2f}")
```

Interpretation: If the expected PV of debt is materially below face, net debt (market value) is lower than book net debt, which can increase implied equity value in an EV→equity bridge. However, the same credit risk driving lower debt PV typically depresses operating value through higher WACC and lower multiples—ensure consistent risk treatment.

See also: Chapter 7 (duration) and Chapter 8 (term structure) for rate sensitivity; Chapter 3 (WACC) for cost of debt usage; IFRS 9 for credit loss measurement and classification; Damodaran chapters on distress for equity valuation in default-prone firms.
