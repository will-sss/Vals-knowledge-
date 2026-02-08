# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 7: Bond’s Duration (Macaulay, modified, DV01, convexity basics)

### Core Concept
Duration measures a bond’s sensitivity to interest rate changes and is foundational for valuing fixed income and debt-like liabilities (including lease liabilities and pension obligations). In valuation, duration and DV01 help quantify how changes in discount rates affect the present value of debt cash flows, credit spreads, and capital structure decisions.

### Formula/Methodology

#### 1) Bond price (PV of cash flows)
```text
Price = Σ_{t=1..N} CF_t / (1 + y)^t

Where:
CF_t = coupon and principal cash flow at time t
y = yield to maturity per period (decimal)
N = number of periods
```

#### 2) Macaulay duration
```text
Macaulay Duration (D_M) = [Σ_{t=1..N} t × PV(CF_t)] / Price
PV(CF_t) = CF_t / (1 + y)^t
```

Units:
- If t is measured in years, D_M is in years.
- If t is measured in periods, D_M is in periods (convert to years by dividing by periods per year).

#### 3) Modified duration (price sensitivity)
```text
Modified Duration (D_mod) = D_M / (1 + y)

Approx price change:
ΔP / P ≈ -D_mod × Δy
```

#### 4) DV01 (Dollar Value of 1 basis point)
```text
DV01 ≈ Price × D_mod × 0.0001
```

Interpretation:
- DV01 is the approximate $ change in price for a +1 bp change in yield.

#### 5) Convexity (second-order correction, practical)
```text
Convexity (C) = [Σ_{t=1..N} t(t+1) × PV(CF_t)] / [Price × (1 + y)^2]

Second-order approximation:
ΔP / P ≈ -D_mod × Δy + 0.5 × C × (Δy)^2
```

---

### Practical Application (How to apply duration in valuation and modeling)

#### Step 1: Use duration to stress-test debt valuation
- For a fixed-rate debt tranche, compute D_mod and DV01 to quantify sensitivity of market value to rate movements.
- For floating-rate debt, duration is typically low (rate resets), but spread duration may be meaningful if credit spreads change.

#### Step 2: Link debt valuation to enterprise/equity value
- In EV → equity bridge, net debt changes with debt market value if you mark-to-market.
- Higher duration debt increases volatility of equity value (all else equal) when rates move.

#### Step 3: Use duration to validate discount rate changes
- If WACC or risk-free rate changes, debt market values may move materially; duration provides a sanity-check for the magnitude.

#### Step 4: Portfolio / treasury risk management
- Compare duration of assets and liabilities (ALM). Mismatch can create equity value volatility.
- DV01 aggregation across instruments gives a quick “rate risk” profile.

---

### Python Implementation
```python
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def bond_price(cashflows: List[float], y: float) -> float:
    """Present value (price) of a bond given cashflows and yield per period.

    Args:
        cashflows: list of CF_t for t=1..N (currency)
        y: yield per period (decimal)

    Returns:
        float: price (currency)

    Raises:
        ValueError: if inputs invalid.
    """
    y = _num(y, "y")
    if y <= -0.99:
        raise ValueError("y must be > -0.99.")
    if not cashflows:
        raise ValueError("cashflows must not be empty.")
    price = 0.0
    for t, cf in enumerate(cashflows, start=1):
        c = _num(cf, f"cashflows[{t}]")
        price += c / ((1.0 + y) ** t)
    return price

def macaulay_duration(cashflows: List[float], y: float) -> float:
    """Macaulay duration in periods for discrete cash flows.

    Returns:
        float: Macaulay duration in periods.

    Raises:
        ValueError: if price is zero or inputs invalid.
    """
    y = _num(y, "y")
    p = bond_price(cashflows, y)
    if p == 0:
        raise ValueError("Bond price is zero; cannot compute duration.")
    num = 0.0
    for t, cf in enumerate(cashflows, start=1):
        c = _num(cf, f"cashflows[{t}]")
        pv = c / ((1.0 + y) ** t)
        num += t * pv
    return num / p

def modified_duration(cashflows: List[float], y: float) -> float:
    """Modified duration in periods.

    D_mod = D_M / (1 + y)
    """
    y = _num(y, "y")
    if y <= -0.99:
        raise ValueError("y must be > -0.99.")
    d_m = macaulay_duration(cashflows, y)
    return d_m / (1.0 + y)

def dv01(cashflows: List[float], y: float) -> float:
    """Approximate DV01 using duration approximation.

    DV01 ≈ Price × D_mod × 0.0001
    """
    y = _num(y, "y")
    p = bond_price(cashflows, y)
    d_mod = modified_duration(cashflows, y)
    return p * d_mod * 0.0001

def convexity(cashflows: List[float], y: float) -> float:
    """Discrete convexity measure (in periods^2).

    C = Σ t(t+1) PV(CF_t) / [Price × (1+y)^2]
    """
    y = _num(y, "y")
    if y <= -0.99:
        raise ValueError("y must be > -0.99.")
    p = bond_price(cashflows, y)
    if p == 0:
        raise ValueError("Bond price is zero; cannot compute convexity.")
    num = 0.0
    for t, cf in enumerate(cashflows, start=1):
        c = _num(cf, f"cashflows[{t}]")
        pv = c / ((1.0 + y) ** t)
        num += t * (t + 1) * pv
    return num / (p * ((1.0 + y) ** 2))

def price_change_approx(cashflows: List[float], y: float, dy: float, use_convexity: bool = True) -> Tuple[float, float]:
    """Approximate new price after a yield change.

    Args:
        cashflows: CF_t list
        y: starting yield per period
        dy: yield change (decimal, e.g. +0.01 for +100bp)
        use_convexity: include convexity term if True

    Returns:
        (p0, p1_approx): original price and approximated new price

    Raises:
        ValueError: invalid inputs.
    """
    dy = _num(dy, "dy")
    p0 = bond_price(cashflows, y)
    dmod = modified_duration(cashflows, y)
    dp_over_p = -dmod * dy
    if use_convexity:
        c = convexity(cashflows, y)
        dp_over_p += 0.5 * c * (dy ** 2)
    p1 = p0 * (1.0 + dp_over_p)
    return p0, p1

# Example usage (annual coupon bond)
face = 1_000.0
coupon_rate = 0.05
y = 0.04
n_years = 7

# annual cashflows: coupons, then coupon+principal at maturity
cashflows = [face * coupon_rate] * (n_years - 1) + [face * coupon_rate + face]

p = bond_price(cashflows, y)
d_m = macaulay_duration(cashflows, y)
d_mod = modified_duration(cashflows, y)
dv = dv01(cashflows, y)
c = convexity(cashflows, y)

print(f"Price: ${p:.2f}")
print(f"Macaulay duration: {d_m:.2f} periods (years if annual)")
print(f"Modified duration: {d_mod:.2f}")
print(f"DV01: ${dv:.4f}")
print(f"Convexity: {c:.2f}")

# Shock +100bp
p0, p1 = price_change_approx(cashflows, y, dy=0.01, use_convexity=True)
print(f"Approx price after +100bp: ${p1:.2f} (from ${p0:.2f})")
```

---

### Valuation Impact
Why this matters:
- Duration quantifies interest-rate risk embedded in debt, leases, pensions, and other liabilities that influence enterprise value and equity value.
- In capital structure analysis, the same face value of debt can have very different market value sensitivity depending on coupon and maturity (duration).

Impact on multiples:
- If debt is marked-to-market (or if you’re comparing firms where debt trading levels differ), EV can shift with rates/spreads, changing EV/EBITDA mechanically.
- High-duration debt can make equity appear “more volatile” and affect perceived leverage risk, influencing multiple selection and discount rates.

Impact on DCF inputs:
- Changing risk-free rates affects WACC; duration helps separate “discount-rate effect” on EV from “debt valuation effect” when moving from EV to equity value.
- For APV-style valuation, the PV of tax shields depends on debt policy and discounting; duration provides intuition for sensitivity.

Comparability issues across companies:
- Two firms with identical net debt can have different rate sensitivity if one has long-dated fixed-rate bonds and the other uses short-dated or floating debt.

Practical adjustments:
```python
def mark_to_market_debt(face_value: float, price_per_100: float) -> float:
    """Convert quoted bond price (per 100 of par) into market value.

    Args:
        face_value: par amount (currency)
        price_per_100: quoted clean price per 100 par (e.g., 97.5)

    Returns:
        float: market value
    """
    fv = _num(face_value, "face_value")
    px = _num(price_per_100, "price_per_100")
    if fv < 0:
        raise ValueError("face_value must be >= 0.")
    return fv * (px / 100.0)
```

---

### Quality of Earnings Flags
⚠️ Large interest expense changes without corresponding changes in average debt balances (could be refinancing, hedges, or capitalised interest).  
⚠️ Significant embedded derivatives or callable features not reflected in a simple duration estimate (duration can be misleading).  
⚠️ Debt described as “fixed” but with step-ups, make-whole calls, or covenants that change economics.  
✅ Clear debt maturity schedule, fixed vs floating split, and hedging disclosure; stable relationship between debt balances and interest cost.

---

### Sector-Specific Considerations

| Sector | Key duration issue | Typical handling |
|---|---|---|
| Banks / insurers | ALM mismatch drives equity sensitivity | measure duration gap; focus on NIM sensitivity and hedges |
| Utilities | long-dated fixed-rate debt common | quantify DV01; consider regulatory pass-through |
| Real estate | debt + cap rates sensitive to rates | model both debt MV and property yield shifts |
| High yield / distressed | spread duration dominates | stress credit spreads; incorporate default risk |

---

### Real-World Example
Scenario: A firm has $1.0B of 7-year fixed-rate debt; rates rise 100bp. Use DV01 to estimate the market value hit and reflect in EV→equity bridge.

```python
# Scale DV01 from the example bond to a $1.0B face amount (approx)
face_amount = 1_000_000_000
example_face = 1_000.0
scale = face_amount / example_face
debt_dv01 = dv * scale
mv_change_100bp = debt_dv01 * 100  # 100bp = 100 * 1bp

print(f"Approx DV01 on $1.0B face: ${debt_dv01:,.0f}")
print(f"Approx MV change for +100bp: ${mv_change_100bp:,.0f} (negative sign implied)")
```

Interpretation: If you treat debt at market value, a +100bp move can change equity value meaningfully for long-duration debt, even if operating forecasts are unchanged.

See also: Chapter 3 (WACC) and Chapter 13 (betas) for discount rate components; Chapter 9 (default-adjusted bond returns) for credit risk; IFRS 9 (financial instruments) for classification and measurement considerations.
