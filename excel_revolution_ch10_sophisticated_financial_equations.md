# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 10: Sophisticated Financial Equations (TVM, IRR/XIRR, NPV, Duration, Option Basics)

### Core Concept
Chapter 10 consolidates financial math that shows up in valuation, credit, and investment models—time value of money, discounting, IRR/NPV, bond duration/convexity, and basic option-like payoffs. In valuation work, these equations drive DCF mechanics, debt valuation, WACC inputs, earn-outs/contingent consideration, and scenario-based analyses. The main application point is implementing calculations defensively (rate bounds, date handling, convergence checks) and validating against Excel outputs.

See also: McKinsey chapters on continuing value and cost of capital; Damodaran chapters on terminal value and risk parameters.

---

### Formula/Methodology

#### 1) Discounting and compounding
```text
Present Value (PV) = FV / (1 + r)^t
Where:
FV = future value (currency)
r = discount rate per period (decimal)
t = number of periods
```

```text
Net Present Value (NPV) = Σ [ CF_t / (1 + r)^t ]  for t = 0..T
Where:
CF_0 is typically negative (investment)
r is the discount rate
```

#### 2) IRR (internal rate of return)
```text
IRR is the rate r such that:
0 = Σ [ CF_t / (1 + r)^t ]  for t = 0..T
Where:
At least one positive and one negative cash flow are required.
```

#### 3) Terminal value (perpetuity growth) and implied multiple cross-check
```text
Terminal Value (Perpetuity) at T = FCF_(T+1) / (WACC - g)
Where:
WACC > g
```

```text
Implied Exit Multiple = Terminal Value / Metric_T
Common Metric_T: EBITDA_T, EBIT_T, Revenue_T
```

#### 4) Bond pricing, duration, convexity (useful for debt valuation and WACC debt cost)
```text
Bond Price = Σ [ Coupon_t / (1 + y)^t ] + Face / (1 + y)^T
Where:
y = yield per period (decimal)
```

```text
Macaulay Duration = Σ [ t × PV(CF_t) ] / Price
Modified Duration = Macaulay Duration / (1 + y)
```

---

### Python Implementation
```python
from typing import Sequence, Union, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

Number = Union[int, float]

def _finite(x: Number, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def pv(fv: Number, r: Number, t: Number) -> float:
    """
    Present value of a single future amount.

    Args:
        fv: future value ($)
        r: rate per period (decimal)
        t: number of periods (>=0)

    Returns:
        float: present value ($)

    Raises:
        ValueError: invalid inputs.
    """
    FV = _finite(fv, "fv")
    rate = _finite(r, "r")
    tt = _finite(t, "t")
    if tt < 0:
        raise ValueError("t must be >= 0.")
    if rate <= -1.0:
        raise ValueError("r must be > -1.0.")
    return FV / ((1.0 + rate) ** tt)

def npv(rate: Number, cashflows: Sequence[Number]) -> float:
    """
    Net present value of a series of cash flows at a constant rate.

    Args:
        rate: discount rate per period (decimal)
        cashflows: sequence CF_0..CF_T

    Returns:
        float: NPV

    Raises:
        ValueError: if empty or invalid rate.
    """
    r = _finite(rate, "rate")
    if r <= -1.0:
        raise ValueError("rate must be > -1.0.")
    if cashflows is None or len(cashflows) == 0:
        raise ValueError("cashflows must be non-empty.")
    total = 0.0
    for t, cf in enumerate(cashflows):
        c = _finite(cf, f"cf_{t}")
        total += c / ((1.0 + r) ** t)
    return total

def irr(cashflows: Sequence[Number], guess: float = 0.10, max_iter: int = 200, tol: float = 1e-8) -> float:
    """
    Internal rate of return using Newton-Raphson with fallback scanning.

    Args:
        cashflows: sequence CF_0..CF_T (must include at least one positive and one negative)
        guess: initial guess
        max_iter: max iterations
        tol: convergence tolerance on NPV

    Returns:
        float: IRR (decimal)

    Raises:
        ValueError: if cashflows invalid or no sign change.
    """
    if cashflows is None or len(cashflows) < 2:
        raise ValueError("cashflows must have length >= 2.")
    cfs = np.asarray(list(cashflows), dtype=float)
    if np.any(~np.isfinite(cfs)):
        raise ValueError("cashflows contain non-finite values.")
    if not (np.any(cfs > 0) and np.any(cfs < 0)):
        raise ValueError("cashflows must include at least one positive and one negative value.")

    r = float(guess)
    if r <= -0.999:
        r = -0.5

    def f(rate: float) -> float:
        return npv(rate, cfs)

    def fprime(rate: float) -> float:
        # derivative of NPV w.r.t r
        if rate <= -0.999999:
            return np.nan
        deriv = 0.0
        for t, cf in enumerate(cfs):
            if t == 0:
                continue
            deriv += -t * cf / ((1.0 + rate) ** (t + 1))
        return deriv

    for _ in range(max_iter):
        val = f(r)
        if abs(val) < tol:
            return r
        d = fprime(r)
        if not np.isfinite(d) or abs(d) < 1e-12:
            break
        r_new = r - val / d
        if r_new <= -0.999999:
            r_new = (r - 0.999) / 2
        r = r_new

    # Fallback: scan for a sign change in NPV over a reasonable range
    grid = np.linspace(-0.9, 2.0, 300)  # -90% to 200%
    vals = np.array([f(x) for x in grid], dtype=float)
    sign = np.sign(vals)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if idx.size == 0:
        raise ValueError("IRR did not converge and no sign change found in scan range.")
    a, b = float(grid[idx[0]]), float(grid[idx[0] + 1])
    # bisection
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        fa = f(a)
        if np.sign(fa) * np.sign(fm) <= 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)

def perpetuity_tv(fcf1: Number, wacc: Number, g: Number) -> float:
    """
    Perpetuity growth terminal value.

    Raises:
        ValueError: if wacc <= g.
    """
    f = _finite(fcf1, "fcf1")
    r = _finite(wacc, "wacc")
    gg = _finite(g, "g")
    if r <= gg:
        raise ValueError("wacc must be > g for perpetuity terminal value.")
    return f / (r - gg)

def implied_exit_multiple(tv: Number, metric_t: Number) -> float:
    """
    Implied terminal multiple: TV / metric_T.

    Raises:
        ValueError: if metric_t is zero.
    """
    T = _finite(tv, "tv")
    m = _finite(metric_t, "metric_t")
    if abs(m) < 1e-12:
        raise ValueError("metric_t must be non-zero for implied multiple.")
    return T / m

def bond_price(face: Number, coupon_rate: Number, ytm: Number, periods: int, freq: int = 2) -> float:
    """
    Price a plain-vanilla fixed coupon bond.

    Args:
        face: face value ($)
        coupon_rate: annual coupon rate (decimal)
        ytm: annual yield to maturity (decimal)
        periods: years to maturity (integer)
        freq: coupon payments per year

    Returns:
        float: bond price ($)

    Raises:
        ValueError: invalid inputs.
    """
    F = _finite(face, "face")
    cr = _finite(coupon_rate, "coupon_rate")
    y = _finite(ytm, "ytm")
    if periods <= 0 or freq <= 0:
        raise ValueError("periods and freq must be positive.")
    if y <= -1.0:
        raise ValueError("ytm must be > -1.0.")
    n = periods * freq
    c = F * cr / freq
    r = y / freq
    price = 0.0
    for t in range(1, n + 1):
        price += c / ((1.0 + r) ** t)
    price += F / ((1.0 + r) ** n)
    return price

def macaulay_duration(face: Number, coupon_rate: Number, ytm: Number, periods: int, freq: int = 2) -> float:
    """
    Macaulay duration (in years).

    Raises:
        ValueError: if price is zero/non-finite.
    """
    F = _finite(face, "face")
    cr = _finite(coupon_rate, "coupon_rate")
    y = _finite(ytm, "ytm")
    if periods <= 0 or freq <= 0:
        raise ValueError("periods and freq must be positive.")
    n = periods * freq
    c = F * cr / freq
    r = y / freq

    price = bond_price(F, cr, y, periods, freq)
    if not np.isfinite(price) or abs(price) < 1e-12:
        raise ValueError("Bond price invalid for duration.")
    weighted = 0.0
    for t in range(1, n + 1):
        cf = c if t < n else (c + F)
        pv_cf = cf / ((1.0 + r) ** t)
        weighted += (t / freq) * pv_cf
    return weighted / price

# Example usage
cfs = [-500_000_000, 120_000_000, 150_000_000, 180_000_000, 210_000_000, 240_000_000]
print(f"NPV @ 10%: ${npv(0.10, cfs)/1e6:.1f}M")
print(f"IRR: {irr(cfs):.2%}")

tv = perpetuity_tv(260_000_000, 0.10, 0.03)
print(f"Terminal value: ${tv/1e9:.2f}B")
print(f"Implied EV/EBITDA multiple: {implied_exit_multiple(tv, 300_000_000):.1f}x")

bp = bond_price(1_000, 0.06, 0.05, periods=5, freq=2)
dur = macaulay_duration(1_000, 0.06, 0.05, periods=5, freq=2)
print(f"Bond price: ${bp:.2f} | Duration: {dur:.2f} years")
```

---

### Valuation Impact
Why this matters:
- DCF mechanics rely on correct discounting, terminal value, and rate handling; small errors compound into large value swings.
- IRR/NPV are used for project finance, acquisition returns, and sanity checks versus DCF-derived values.
- Debt valuation and duration inform cost of debt, leverage analysis, and WACC assumptions.

Impact on multiples:
- Terminal value often implies an exit multiple; checking the implied multiple against peers is a key plausibility test.
- Correct handling of denominators and period alignment prevents misleading valuation ranges.

Impact on DCF inputs:
- WACC bounds and terminal growth constraints (WACC > g) are critical.
- Duration supports interest-rate sensitivity and debt mark-to-market adjustments when relevant.

Practical adjustments:
```python
def pv_factor(r: float, t: int) -> float:
    """PV factor with validation for discounting tables."""
    rr = float(r)
    if not np.isfinite(rr) or rr <= -1.0:
        raise ValueError("r must be finite and > -1.0.")
    if t < 0:
        raise ValueError("t must be >= 0.")
    return 1.0 / ((1.0 + rr) ** t)
```

---

### Quality of Earnings Flags
⚠️ Red flag: terminal value dominates (e.g., >70–80% of EV) without strong support for long-run assumptions.  
⚠️ Red flag: IRR used as a headline metric when cash flows include large non-operating items or one-offs.  
⚠️ Red flag: inconsistent period assumptions (annual WACC with quarterly cash flows) without converting rates.  
✅ Green flag: implied exit multiple cross-check and discounting reconciliation vs Excel functions (NPV/XNPV, IRR/XIRR).

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | High sensitivity to WACC/g | Publish WACC×g sensitivity; sanity-check implied multiples vs comps |
| Banking | Discount rates and cash flows differ | Prefer equity-based models; be cautious applying generic IRR/TV mechanics |
| Real Estate | Cap rates resemble discounting | Translate cap rate logic into PV and sensitivity; reconcile NOI/cap rate to value |
| Infrastructure/Project Finance | IRR central metric | Use date-based cash flows (XIRR equivalent); stress test with delays and cost overruns |

---

### Real-World Example
Scenario: Build a DCF that outputs PV, terminal value, and implied exit multiple; run an IRR check on the same cash flow profile.

```python
fcf = [120_000_000, 135_000_000, 150_000_000, 165_000_000, 180_000_000]
wacc = 0.10
g = 0.03
fcf1 = 186_000_000
tv = perpetuity_tv(fcf1, wacc, g)
pv_total = npv(wacc, [0.0] + fcf) + tv / ((1 + wacc) ** len(fcf))
implied_mult = implied_exit_multiple(tv, 300_000_000)
print(f"PV total: ${pv_total/1e9:.2f}B | TV: ${tv/1e9:.2f}B | Implied multiple: {implied_mult:.1f}x")
```

Interpretation: If implied exit multiple is far above peer ranges, revisit long-run growth, margins, reinvestment needs, or WACC.

See also: McKinsey continuing value and cost of capital chapters; Damodaran terminal value chapter.
