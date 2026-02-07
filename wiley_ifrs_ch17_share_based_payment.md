# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 17: Share-Based Payment (IFRS 2, SBC expense, dilution, option valuation, EBITDA comparability)

### Core Concept
IFRS 2 requires entities to recognise an expense for share-based payment (SBC) over the vesting period, measured at fair value at grant date for equity-settled awards (and remeasured for cash-settled awards). For valuation, SBC is both a profitability/quality-of-earnings issue (non-cash expense that reduces reported earnings) and an ownership issue (dilution transfers value from existing shareholders to employees). The correct treatment depends on whether you value enterprise cash flows (DCF) or equity per share and how peers report “adjusted” metrics.

### Formula/Methodology

#### 1) Equity-settled awards (grant-date FV, no remeasurement for equity FV)
```text
Total Compensation Cost = Grant-date fair value per award × Number of awards expected to vest

Period Expense (straight-line, simplified) = Total Compensation Cost / Vesting Period (years)
Cumulative Expense at time t = Total Compensation Cost × (elapsed vesting / total vesting) × vesting probability adjustments

Where:
Grant-date FV is typically estimated using an option pricing model (e.g., Black-Scholes) for options
Adjust for forfeitures (expected non-vesting) based on policy and estimates
```

#### 2) Cash-settled awards (remeasured each period)
```text
Liability_end = Fair value of obligation at reporting date (for vested/earned portion)
Period Expense = Change in liability + cash payments (depending on presentation)

Key point: remeasure each reporting date until settlement -> P&L volatility
```

#### 3) Black-Scholes (common for employee options, simplified)
```text
Call Option Value = S × N(d1) - K × exp(-rT) × N(d2)

d1 = (ln(S/K) + (r + 0.5σ^2)T) / (σ × sqrt(T))
d2 = d1 - σ × sqrt(T)

Where:
S = current share price
K = exercise price
r = risk-free rate (continuous compounding, decimal)
σ = volatility (annualised, decimal)
T = time to maturity (years)
N(.) = standard normal CDF
```

#### 4) EBITDA and SBC
```text
EBITDA includes SBC expense if classified within operating expenses.
Some companies present “Adjusted EBITDA excluding SBC” (non-GAAP).
Analyst must decide whether to:
- treat SBC as a real cost (recommended for comparability), and/or
- account for dilution explicitly in per-share value.
```

---

### Practical Application for Valuation Analysts

#### 1) Decide: treat SBC as an operating cost vs treat dilution explicitly (or both)
Common approaches:

A) Operating-cost approach (enterprise view)
- Keep SBC expense in operating costs (reduces EBIT/EBITDA and FCF).
- Do not add back SBC when comparing to peers unless you also adjust peers consistently.
- This aligns with economic reality: SBC is compensation.

B) Dilution approach (equity per-share focus)
- For DCF to equity value per share, use **fully diluted share count** (options/RSUs).
- You may adjust earnings to remove non-cash SBC, but then must model dilution carefully (otherwise overstates value).

Practical rule:
- For institutional-quality valuations, **treat SBC as a cost AND use diluted shares** if material. This avoids double-counting, but requires consistency:
  - If you keep SBC expense in FCF, do not also deduct value of options separately, unless you are explicitly separating operating vs equity claims.

#### 2) Multiple-based valuation: ensure consistent treatment across numerator/denominator
- EV/EBITDA: if EBITDA includes SBC, then keep it; if you remove it, apply the same policy to all comps and consider adding back to FCF similarly.
- P/E: SBC reduces earnings; removing it creates a non-comparable “adjusted EPS” unless reconciled.

#### 3) Cash-settled awards are debt-like
Cash-settled SBC creates a liability that:
- behaves like short-term/long-term debt-like obligation
- should be included in net debt adjustments for EV bridge (if material)

#### 4) Common SBC mechanics that impact valuation
- Accelerated vesting on change of control: can spike expense.
- Performance conditions: affects expected vesting and expense recognition.
- Net settlement / tax withholding: reduces shares issued but creates cash outflow; check share count impact.

---

### Python Implementation
```python
from typing import Any, Dict, Optional
import numpy as np
import math

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes call option value (European), simplified.

    Args:
        S: current share price
        K: strike price
        r: risk-free rate (decimal, continuous compounding assumption in formula)
        sigma: volatility (annualised, decimal)
        T: time to maturity (years)

    Returns:
        float: option value per option

    Raises:
        ValueError: invalid inputs.
    """
    S = _num(S, "S")
    K = _num(K, "K")
    r = _num(r, "r")
    sigma = _num(sigma, "sigma")
    T = _num(T, "T")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if T <= 0:
        raise ValueError("T must be > 0.")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def sbc_total_comp_cost(grant_fv_per_award: float, awards_granted: float, expected_vesting_pct: float) -> float:
    """
    Total compensation cost for equity-settled awards.

    Args:
        grant_fv_per_award: grant-date FV per award
        awards_granted: number of awards granted
        expected_vesting_pct: expected vesting (0..1)

    Returns:
        float: total cost

    Raises:
        ValueError: invalid inputs.
    """
    fv = _num(grant_fv_per_award, "grant_fv_per_award")
    n = _num(awards_granted, "awards_granted")
    p = _num(expected_vesting_pct, "expected_vesting_pct")
    if fv < 0:
        raise ValueError("grant_fv_per_award must be >= 0.")
    if n < 0:
        raise ValueError("awards_granted must be >= 0.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("expected_vesting_pct must be between 0 and 1.")
    return fv * n * p

def sbc_period_expense_straight_line(total_comp_cost: float, vesting_years: float) -> float:
    """
    Straight-line annual SBC expense.

    Raises:
        ValueError: vesting_years <= 0.
    """
    total = _num(total_comp_cost, "total_comp_cost")
    v = _num(vesting_years, "vesting_years")
    if total < 0:
        raise ValueError("total_comp_cost must be >= 0.")
    if v <= 0:
        raise ValueError("vesting_years must be > 0.")
    return total / v

def fully_diluted_shares(
    basic_shares: float,
    in_the_money_options: float = 0.0,
    rsus: float = 0.0,
    treasury_stock_method_buyback_shares: float = 0.0
) -> float:
    """
    Simplified fully diluted shares.

    Notes:
      - Treasury stock method requires estimating shares repurchased with option proceeds.
      - Here, provide treasury_stock_method_buyback_shares as an input (>=0).

    Raises:
        ValueError: invalid inputs.
    """
    b = _num(basic_shares, "basic_shares")
    opt = _num(in_the_money_options, "in_the_money_options")
    r = _num(rsus, "rsus")
    buy = _num(treasury_stock_method_buyback_shares, "treasury_stock_method_buyback_shares")
    if b <= 0:
        raise ValueError("basic_shares must be > 0.")
    if opt < 0 or r < 0 or buy < 0:
        raise ValueError("option/rsu/buyback shares must be >= 0.")
    diluted = b + opt + r - buy
    if diluted <= 0:
        raise ValueError("diluted shares must be > 0.")
    return diluted

# Example usage: value employee options and compute SBC cost
S = 20.0
K = 22.0
r = 0.03
sigma = 0.35
T = 5.0

fv_option = black_scholes_call(S, K, r, sigma, T)
print(f"Option grant-date FV (per option): ${fv_option:.2f}")

total_cost = sbc_total_comp_cost(grant_fv_per_award=fv_option, awards_granted=10_000_000, expected_vesting_pct=0.85)
annual_expense = sbc_period_expense_straight_line(total_cost, vesting_years=4)
print(f"Total SBC cost: ${total_cost/1e6:.1f}M")
print(f"Annual SBC expense (straight-line): ${annual_expense/1e6:.1f}M")

diluted = fully_diluted_shares(basic_shares=1_200_000_000, in_the_money_options=35_000_000, rsus=18_000_000, treasury_stock_method_buyback_shares=10_000_000)
print(f"Fully diluted shares: {diluted/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- SBC is compensation: excluding it can overstate sustainable margins and inflate EV/EBITDA comparables.
- Dilution reduces equity value per share even if enterprise value is unchanged; option overhang is often material in high-growth companies.
- Cash-settled SBC creates liabilities that should be treated as debt-like in EV bridges.

Impact on multiples:
- EV/EBITDA: decide whether EBITDA includes SBC and apply consistently across comps; if comps exclude SBC, document and adjust.
- P/E: “adjusted EPS” excluding SBC can be misleading; reconcile back to IFRS profit and treat dilution explicitly.

Impact on DCF inputs:
- If SBC is treated as an operating cost, it reduces NOPAT and FCF.
- If SBC is excluded from FCF, you must reflect value transfer via higher share count or explicit option valuation.

Practical adjustments:
```python
def normalize_ebitda_for_sbc(reported_ebitda: float, sbc_expense: float, policy: str = "include") -> float:
    """
    If policy='exclude', add back SBC to reported EBITDA.
    Default policy='include' returns reported EBITDA (recommended for comparability).

    Raises:
        ValueError: invalid policy or sbc negative.
    """
    e = _num(reported_ebitda, "reported_ebitda")
    s = _num(sbc_expense, "sbc_expense")
    if s < 0:
        raise ValueError("sbc_expense must be >= 0.")
    if policy not in {"include", "exclude"}:
        raise ValueError("policy must be 'include' or 'exclude'.")
    return e if policy == "include" else e + s
```

---

### Quality of Earnings Flags
⚠️ “Adjusted EBITDA/EPS” excludes SBC without a robust dilution discussion.  
⚠️ Rapidly increasing SBC as % of revenue, masking deteriorating cash profitability.  
⚠️ Option repricing or frequent refresh grants to retain employees (may imply weak retention or weak economics).  
⚠️ Large cash-settled SBC liabilities not treated as debt-like in leverage metrics.  
✅ Clear reconciliation of SBC adjustments, detailed share count/dilution disclosures, and consistent policy over time.

---

### Sector-Specific Considerations

| Sector | Key SBC issue | Typical treatment |
|---|---|---|
| High-growth Tech / SaaS | SBC can be very large vs revenue; option overhang | Keep SBC in margins for comparables; use diluted shares; consider SBC run-rate normalisation |
| Financial services | Lower SBC, more cash bonuses | SBC often less material; still ensure diluted EPS and capital stack consistency |
| Industrials | SBC usually modest; performance shares | Focus on vesting conditions and change-of-control acceleration |
| Early-stage / biotech | High SBC but low revenue | Treat as real cost; dilution often the primary valuation impact |

---

### Real-World Example
Scenario: Company reports $600m EBITDA including $120m SBC. It markets “Adjusted EBITDA” excluding SBC. You are building a peer EV/EBITDA screen and per-share valuation.

```python
reported_ebitda = 600.0
sbc = 120.0

ebitda_including = normalize_ebitda_for_sbc(reported_ebitda, sbc, policy="include")
ebitda_excluding = normalize_ebitda_for_sbc(reported_ebitda, sbc, policy="exclude")

print(f"EBITDA (include SBC): ${ebitda_including:.0f}m")
print(f"EBITDA (exclude SBC): ${ebitda_excluding:.0f}m")

# Per-share impact via dilution (simplified)
basic_shares = 900_000_000
options_itm = 45_000_000
rsus = 20_000_000
buyback = 12_000_000
diluted = fully_diluted_shares(basic_shares, options_itm, rsus, buyback)
print(f"Diluted shares: {diluted/1e6:.1f}M")
```

Interpretation: For comparables, including SBC avoids overstating margins; for per-share value, dilution can be a first-order driver.

See also: Chapter 16 (shareholders’ equity) for classification and dilution; Chapter 27 (EPS) for per-share mechanics; Chapter 24 (financial instruments) for liability vs equity classification of awards and embedded features.
