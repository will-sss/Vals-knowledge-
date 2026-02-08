# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 8: Modeling the Term Structure (spot rates, forward rates, bootstrapping, discount factors)

### Core Concept
The term structure (yield curve) maps discount rates by maturity and is the backbone of valuing bonds, leases, pensions, and any multi-period cash flow stream. In valuation work, term structure modeling improves accuracy when discounting long-dated or irregular cash flows (debt schedules, lease liabilities, APV tax shields) and supports scenario analysis on rate shifts and curve shape (level, slope, curvature).

### Formula/Methodology

#### 1) Discount factors and present value
```text
Discount factor for maturity t: DF(t)

PV = Σ_{t} CF(t) × DF(t)
```

Common conventions (choose one and be consistent):
- Annual compounding:
```text
DF(t) = 1 / (1 + s_t)^t
```
- Continuous compounding:
```text
DF(t) = exp(-s_t × t)
```

Where:
- s_t = spot (zero) rate for maturity t (decimal, per year)
- t = maturity in years (can be fractional)

#### 2) Spot rates, par yields, and bond pricing (par bond)
A par bond has price = par (e.g., 100). For an annual coupon par bond with coupon rate c_n and maturity n:
```text
100 = Σ_{t=1..n} (100 × c_n) × DF(t) + 100 × DF(n)
```

Given DF(1..n-1), you can solve for DF(n) (bootstrapping).

#### 3) Bootstrapping discount factors from par yields (annual coupons)
Rearranging the par bond equation:
```text
DF(n) = [100 - Σ_{t=1..n-1} (100 × c_n) × DF(t)] / [100 × (1 + c_n)]
```

Where:
- c_n = par coupon rate for maturity n (decimal)
- DF(t) are discount factors for earlier maturities

If the curve is quoted in par yields y_n and the bond is priced at par:
```text
c_n = y_n   (for standard par-yield instruments)
```

#### 4) Forward rates from discount factors
One-period forward rate between t-1 and t (annual periods):
```text
1 + f_{t-1,t} = DF(t-1) / DF(t)
f_{t-1,t} = DF(t-1) / DF(t) - 1
```

General forward rate from T1 to T2 (annual compounding):
```text
(1 + f_{T1,T2})^(T2 - T1) = DF(T1) / DF(T2)
f_{T1,T2} = (DF(T1) / DF(T2))^(1/(T2 - T1)) - 1
```

#### 5) Curve shifts for sensitivity (common stress)
Parallel shift:
```text
s'_t = s_t + Δ
```

Key-rate shift (localized shock):
```text
s'_t = s_t + Δ at selected maturities; interpolate between nodes
```

---

### Practical Application (How to use term structure in valuation work)

#### Step 1: Choose curve type and units
- Use a risk-free base curve aligned to the currency (e.g., govies/OIS proxy), then add a credit spread for risky cash flows.
- For valuation models that use a single WACC, the curve is still useful for debt/lease valuation and sanity checks.

#### Step 2: Bootstrap discount factors for accurate PV
Use bootstrapping when you have par yields by maturity and need DF(t) for:
- mark-to-market debt
- lease liability PV
- APV tax shield discounting using debt curve
- pension obligation PV (liability duration)

#### Step 3: Derive forward rates for scenario and planning
Forward rates help interpret the curve:
- steep curve → higher implied future short rates
- inverted curve → lower implied future short rates / recession signal
In modeling, forward rates support interest expense projections for floating-rate debt (base rate path + spread).

#### Step 4: Discount different cash flows appropriately
Common splits:
- risk-free curve: contractual, near-riskless flows (some pension/sovereign contexts)
- risk-free + credit spread: debt cash flows (issuer-specific)
- WACC: operating FCFF (enterprise valuation DCF)

Avoid mixing:
- Do not discount debt at WACC.
- Do not discount FCFF at risk-free.

#### Step 5: Curve-driven sensitivities
- DV01 by bucket: compute PV under key-rate shocks.
- Equity value sensitivity: if net debt is marked to market, the EV→equity bridge is curve-sensitive.

---

### Python Implementation
```python
from typing import Any, Dict, List, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def bootstrap_discount_factors_from_par_yields(par_yields: List[float], face: float = 100.0) -> List[float]:
    """Bootstrap annual discount factors from annual par yields (coupon = par yield), assuming price = par.

    Args:
        par_yields: list of par yields for maturities 1..N (decimals, e.g., 0.035)
        face: par value used for pricing (default 100)

    Returns:
        List[float]: discount factors DF(1)..DF(N)

    Raises:
        ValueError: invalid inputs or impossible curve (e.g., negative DF).
    """
    if not par_yields:
        raise ValueError("par_yields must not be empty.")
    F = _num(face, "face")
    if F <= 0:
        raise ValueError("face must be > 0.")

    dfs: List[float] = []
    for n, y in enumerate(par_yields, start=1):
        y = _num(y, f"par_yields[{n}]")
        if y <= -0.99:
            raise ValueError("par yields must be > -0.99.")
        c = y  # coupon rate for par instrument under this simplification

        if n == 1:
            # F = F*c*DF1 + F*DF1 = F*(1+c)*DF1
            df1 = 1.0 / (1.0 + c)
            if df1 <= 0:
                raise ValueError("Bootstrapped DF(1) <= 0; check inputs.")
            dfs.append(df1)
            continue

        pv_coupons = 0.0
        for t in range(1, n):
            pv_coupons += (F * c) * dfs[t - 1]

        denom = F * (1.0 + c)
        num = F - pv_coupons
        df_n = num / denom

        if df_n <= 0:
            raise ValueError(f"Bootstrapped DF({n}) <= 0; check curve inputs.")
        if df_n > dfs[-1] + 1e-9:
            # DF should usually be non-increasing with maturity for non-negative rates,
            # but allow small numerical slack; flag extreme violations.
            raise ValueError(f"Bootstrapped DF({n}) > DF({n-1}); curve implies strongly negative rates or inconsistent inputs.")
        dfs.append(df_n)

    return dfs

def spot_rates_from_discount_factors(dfs: List[float]) -> List[float]:
    """Convert annual discount factors DF(t) to annual-compounded spot rates s_t.

    s_t = DF(t)^(-1/t) - 1
    """
    if not dfs:
        raise ValueError("dfs must not be empty.")
    spots = []
    for t, df in enumerate(dfs, start=1):
        df = _num(df, f"dfs[{t}]")
        if df <= 0:
            raise ValueError("discount factors must be > 0.")
        s = (df ** (-1.0 / t)) - 1.0
        spots.append(s)
    return spots

def forward_rates_from_discount_factors(dfs: List[float]) -> List[float]:
    """Compute one-period forward rates f_{t-1,t} for t=1..N.
    Convention:
        f_{0,1} uses DF(0)=1, so f_{0,1} = 1/DF(1)-1
        For t>=2, f_{t-1,t} = DF(t-1)/DF(t)-1
    """
    if not dfs:
        raise ValueError("dfs must not be empty.")
    fwds = []
    df_prev = 1.0
    for t, df in enumerate(dfs, start=1):
        df = _num(df, f"dfs[{t}]")
        if df <= 0:
            raise ValueError("discount factors must be > 0.")
        f = (df_prev / df) - 1.0
        fwds.append(f)
        df_prev = df
    return fwds

def pv_from_discount_factors(cashflows: List[float], dfs: List[float]) -> float:
    """Present value using discount factors.
    Args:
        cashflows: CF_t for t=1..N
        dfs: DF(t) for t=1..N (same length)
    """
    if len(cashflows) != len(dfs):
        raise ValueError("cashflows and dfs must have the same length.")
    pv = 0.0
    for t, (cf, df) in enumerate(zip(cashflows, dfs), start=1):
        c = _num(cf, f"cashflows[{t}]")
        d = _num(df, f"dfs[{t}]")
        if d <= 0:
            raise ValueError("discount factors must be > 0.")
        pv += c * d
    return pv

def apply_parallel_shift_to_spot_rates(spots: List[float], delta: float) -> List[float]:
    """Parallel shift spot rates by delta (decimal)."""
    delta = _num(delta, "delta")
    out = []
    for t, s in enumerate(spots, start=1):
        s = _num(s, f"spots[{t}]")
        out.append(s + delta)
    return out

def discount_factors_from_spot_rates(spots: List[float]) -> List[float]:
    """Compute DF(t) = 1/(1+s_t)^t from annual spot rates."""
    if not spots:
        raise ValueError("spots must not be empty.")
    dfs = []
    for t, s in enumerate(spots, start=1):
        s = _num(s, f"spots[{t}]")
        if s <= -0.99:
            raise ValueError("spot rates must be > -0.99.")
        df = 1.0 / ((1.0 + s) ** t)
        dfs.append(df)
    return dfs

# Example usage (illustrative par-yield curve)
par_yields = [0.03, 0.032, 0.034, 0.035, 0.036]  # maturities 1..5
dfs = bootstrap_discount_factors_from_par_yields(par_yields)
spots = spot_rates_from_discount_factors(dfs)
fwds = forward_rates_from_discount_factors(dfs)

print("Discount factors:", [round(x, 6) for x in dfs])
print("Spot rates:", [f"{x:.2%}" for x in spots])
print("1Y forward rates:", [f"{x:.2%}" for x in fwds])

# PV a simple cashflow stream using curve
cashflows = [0, 0, 0, 0, 110]  # bullet payment at year 5
pv = pv_from_discount_factors(cashflows, dfs)
print(f"PV of $110 in year 5: ${pv:.2f}")

# Parallel shift +100bp and recompute PV
spots_up = apply_parallel_shift_to_spot_rates(spots, 0.01)
dfs_up = discount_factors_from_spot_rates(spots_up)
pv_up = pv_from_discount_factors(cashflows, dfs_up)
print(f"PV after +100bp parallel shift: ${pv_up:.2f}")
```

---

### Valuation Impact
Why this matters:
- Discounting with a single rate can misprice long-dated or back-ended cash flows; the term structure improves accuracy and supports more defensible liability valuation (debt, leases, pensions).
- Forward rates derived from the curve support explicit interest expense forecasting for floating-rate debt and scenario analysis.

Impact on multiples:
- If net debt is measured at market value, EV can change with curve shifts; this mechanically changes EV/EBITDA even without operating changes.
- Comparable company analysis can be distorted if one firm has long-duration fixed debt and another has short-duration floating debt under fast-moving rates.

Impact on DCF inputs:
- WACC often uses a point-in-time risk-free rate; a curve provides a better framework for multi-year discounting in APV or project finance where discount rates vary by phase.
- Terminal value sensitivity: long-dated terminal value is highly rate-sensitive; curve scenarios help stress plausibility under higher/steeper rate environments.

Comparability issues across companies:
- Different hedging policies (swaps) change effective duration and exposure; the curve is needed to evaluate hedges and fair value impacts (IFRS 9).

Practical adjustments:
```python
def implied_spread(risky_yield: float, risk_free_spot: float) -> float:
    """Simple implied credit spread approximation (same maturity)."""
    ry = _num(risky_yield, "risky_yield")
    rf = _num(risk_free_spot, "risk_free_spot")
    return ry - rf
```

---

### Quality of Earnings Flags
⚠️ Large fair value gains/losses on debt or derivatives without clear disclosure of curve moves and hedge relationships (could be accounting mismatches).  
⚠️ Using inconsistent discount curves across periods (changing methodology) creates artificial volatility in liabilities and OCI.  
⚠️ Ignoring curve shape when valuing long-dated obligations (pensions, leases) can materially misstate liabilities.  
✅ Consistent curve selection and clear disclosure of inputs (rates, spreads, methodology), with sensitivity analyses.

---

### Sector-Specific Considerations

| Sector | Key curve issue | Typical handling |
|---|---|---|
| Banks | curve slope affects NIM | scenario on level/slope; forward-rate based interest income/expense modeling |
| Infrastructure / project finance | long tenor cash flows | use term structure + project risk premia; phase-specific discounting |
| Insurers | liability discounting and duration matching | build risk-free curve; evaluate ALM duration gap and reinvestment |
| Real estate | cap rates and debt pricing both rate-linked | stress both discount curve and asset yield assumptions |

---

### Real-World Example
Scenario: Value a 5-year bullet debt at market using a bootstrapped curve and quantify PV impact under a +100bp shift.

```python
par_yields = [0.03, 0.032, 0.034, 0.035, 0.036]
dfs = bootstrap_discount_factors_from_par_yields(par_yields)
cashflows = [0, 0, 0, 0, 110]
pv_base = pv_from_discount_factors(cashflows, dfs)

spots = spot_rates_from_discount_factors(dfs)
dfs_up = discount_factors_from_spot_rates(apply_parallel_shift_to_spot_rates(spots, 0.01))
pv_up = pv_from_discount_factors(cashflows, dfs_up)

print(f"Base PV: ${pv_base:.2f}; PV after +100bp: ${pv_up:.2f}; Change: ${pv_up - pv_base:.2f}")
```

Interpretation: The curve-based PV shift approximates the mark-to-market impact on debt (or liabilities) from rate moves; use this in EV→equity bridges and risk narratives.

See also: Chapter 7 (duration/DV01) for sensitivity measures; Chapter 3 (WACC) and Chapter 13 (betas) for discount rate components; IFRS 9 (financial instruments) for fair value and hedge accounting implications.
