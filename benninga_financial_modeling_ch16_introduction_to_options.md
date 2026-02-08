# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 16: Introduction to Options (payoffs, no-arbitrage bounds, put-call parity)

### Core Concept
Options are contingent claims whose value depends on an underlying asset’s price and time. In valuation, option logic is used directly in (i) equity valuation for firms with significant financial leverage (equity as a call on firm value), (ii) real options (expansion/abandonment), and (iii) valuing embedded features (convertibles, warrants, earn-outs). The practical valuation takeaway is that optionality is convex: it can make downside limited while preserving upside, which standard DCF can miss.

---

### Formula/Methodology

#### 1) Payoff at maturity
Call option payoff:
```text
Call Payoff (T) = max(S_T - K, 0)
```

Put option payoff:
```text
Put Payoff (T) = max(K - S_T, 0)
```

Where:
- S_T = underlying price at maturity
- K = strike price
- T = maturity date

#### 2) Profit vs payoff
Profit = payoff - premium (price paid for the option):
```text
Call Profit (T) = max(S_T - K, 0) - C0
Put Profit (T)  = max(K - S_T, 0) - P0
```

Where:
- C0 = call premium today
- P0 = put premium today

#### 3) Intrinsic value and time value
```text
Call Intrinsic = max(S0 - K, 0)
Put Intrinsic  = max(K - S0, 0)

Time Value = Option Price - Intrinsic Value
```

#### 4) No-arbitrage bounds (European options, non-dividend-paying)
Lower and upper bounds:
```text
0 ≤ C0 ≤ S0
0 ≤ P0 ≤ K / (1 + r)^T   (discrete compounding)
```

Call lower bound:
```text
C0 ≥ max(S0 - K/(1+r)^T, 0)
```

Put lower bound:
```text
P0 ≥ max(K/(1+r)^T - S0, 0)
```

Where:
- r = risk-free rate (per period, matched to T)

#### 5) Put–call parity (European, non-dividend-paying)
```text
C0 - P0 = S0 - K/(1+r)^T
```

If the underlying pays a known PV of dividends (PV(Div)):
```text
C0 - P0 = (S0 - PV(Div)) - K/(1+r)^T
```

---

### Practical Application (How to apply option logic in valuation)

#### A) Identify “option-like” features in your valuation problem
Typical valuation use cases:
- Equity of a levered firm: resembles a call on firm value with strike = debt face value (stylized).
- Convertibles and warrants: embedded calls on equity.
- Earn-outs: contingent payments resembling options on performance metrics.
- Project flexibility: expand/abandon/switch timing resembles real options.

#### B) Use no-arbitrage bounds as sanity checks
Even if you are not pricing options formally, bounds help detect model errors:
- If implied call price > S0, something is inconsistent.
- If put–call parity is violated materially, check dividend assumptions, early exercise (American vs European), or rate compounding.

#### C) Build payoff diagrams to validate economic exposure
Before running Black–Scholes or binomial:
- Graph payoff across S_T range
- Confirm the instrument behaves as expected under downside and upside scenarios

#### D) Map options to valuation inputs
- Optionality changes downside risk and expected value distribution.
- Instruments with convex payoffs can justify valuation approaches beyond linear DCF (e.g., binomial/Monte Carlo).

---

### Python Implementation
```python
from typing import Any, Dict, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def pv_discrete(amount: float, r: float, T: float) -> float:
    """Present value with discrete compounding.

    Args:
        amount: future value
        r: risk-free rate per period (decimal)
        T: number of periods (can be fractional if consistent)

    Returns:
        float: present value

    Raises:
        ValueError: if inputs invalid
    """
    amt = _num(amount, "amount")
    rr = _num(r, "r")
    tt = _num(T, "T")
    if tt < 0:
        raise ValueError("T must be >= 0.")
    denom = (1.0 + rr) ** tt
    if denom <= 0:
        raise ValueError("Invalid discount factor.")
    return amt / denom

def call_payoff(S_T: float, K: float) -> float:
    """European call payoff at maturity."""
    s = _num(S_T, "S_T")
    k = _num(K, "K")
    return max(s - k, 0.0)

def put_payoff(S_T: float, K: float) -> float:
    """European put payoff at maturity."""
    s = _num(S_T, "S_T")
    k = _num(K, "K")
    return max(k - s, 0.0)

def option_profit(payoff: float, premium: float) -> float:
    """Profit = payoff - premium."""
    p = _num(payoff, "payoff")
    prem = _num(premium, "premium")
    return p - prem

def intrinsic_value_call(S0: float, K: float) -> float:
    s = _num(S0, "S0")
    k = _num(K, "K")
    return max(s - k, 0.0)

def intrinsic_value_put(S0: float, K: float) -> float:
    s = _num(S0, "S0")
    k = _num(K, "K")
    return max(k - s, 0.0)

def time_value(option_price: float, intrinsic: float) -> float:
    op = _num(option_price, "option_price")
    iv = _num(intrinsic, "intrinsic")
    tv = op - iv
    # Time value can be ~0 but should not be negative beyond small numerical noise
    if tv < -1e-9:
        raise ValueError("Option price is below intrinsic value (arbitrage violation).")
    return max(tv, 0.0)

def no_arbitrage_bounds_call(S0: float) -> Dict[str, float]:
    """Bounds: 0 <= C0 <= S0."""
    s = _num(S0, "S0")
    if s < 0:
        raise ValueError("S0 must be >= 0.")
    return {"lower": 0.0, "upper": s}

def no_arbitrage_bounds_put(K: float, r: float, T: float) -> Dict[str, float]:
    """Bounds: 0 <= P0 <= PV(K)."""
    k = _num(K, "K")
    if k < 0:
        raise ValueError("K must be >= 0.")
    upper = pv_discrete(k, r, T)
    return {"lower": 0.0, "upper": upper}

def lower_bound_call(S0: float, K: float, r: float, T: float) -> float:
    """C0 >= max(S0 - PV(K), 0)."""
    s = _num(S0, "S0")
    k = _num(K, "K")
    pvk = pv_discrete(k, r, T)
    return max(s - pvk, 0.0)

def lower_bound_put(S0: float, K: float, r: float, T: float) -> float:
    """P0 >= max(PV(K) - S0, 0)."""
    s = _num(S0, "S0")
    k = _num(K, "K")
    pvk = pv_discrete(k, r, T)
    return max(pvk - s, 0.0)

def put_call_parity_call_price(S0: float, K: float, r: float, T: float, P0: float, pv_div: float = 0.0) -> float:
    """Solve for call price using put-call parity: C0 = P0 + (S0 - PV(Div)) - PV(K)."""
    s = _num(S0, "S0")
    p = _num(P0, "P0")
    d = _num(pv_div, "pv_div")
    if d < 0:
        raise ValueError("pv_div must be >= 0.")
    pvk = pv_discrete(_num(K, "K"), r, T)
    return p + (s - d) - pvk

def put_call_parity_put_price(S0: float, K: float, r: float, T: float, C0: float, pv_div: float = 0.0) -> float:
    """Solve for put price using put-call parity: P0 = C0 - (S0 - PV(Div)) + PV(K)."""
    s = _num(S0, "S0")
    c = _num(C0, "C0")
    d = _num(pv_div, "pv_div")
    if d < 0:
        raise ValueError("pv_div must be >= 0.")
    pvk = pv_discrete(_num(K, "K"), r, T)
    return c - (s - d) + pvk

def parity_deviation(S0: float, K: float, r: float, T: float, C0: float, P0: float, pv_div: float = 0.0) -> float:
    """Compute parity deviation: (C0 - P0) - ((S0 - PV(Div)) - PV(K))."""
    s = _num(S0, "S0")
    c = _num(C0, "C0")
    p = _num(P0, "P0")
    d = _num(pv_div, "pv_div")
    pvk = pv_discrete(_num(K, "K"), r, T)
    lhs = c - p
    rhs = (s - d) - pvk
    return lhs - rhs

# Example usage
S0 = 100.0
K = 95.0
r = 0.04
T = 1.0
C0 = 10.5
P0 = 3.2

print("Call intrinsic:", intrinsic_value_call(S0, K))
print("Put intrinsic :", intrinsic_value_put(S0, K))
print("Call lower bound:", lower_bound_call(S0, K, r, T))
print("Put  lower bound:", lower_bound_put(S0, K, r, T))
print("Put-call parity deviation:", parity_deviation(S0, K, r, T, C0, P0))
print("Implied call from parity:", put_call_parity_call_price(S0, K, r, T, P0))
```

---

### Valuation Impact
Why this matters:
- Option convexity captures upside value from uncertainty and managerial flexibility; DCF (single path) typically undervalues these features.
- Put–call parity and bounds provide fast sanity checks for embedded options in securities and contracts.

Impact on multiples:
- Firms with meaningful optionality (early-stage tech, biotech, natural resources) may trade on revenue or user metrics partly because earnings-based multiples miss convexity.
- Convertibles/warrants can distort per-share metrics; you often need a fully diluted equity bridge.

Impact on DCF inputs:
- Optionality can justify scenario-weighted cash flows and/or real-options overlays (binomial/Monte Carlo).
- Equity-as-option framing is useful for distressed firms where debt overhang caps equity downside at zero but retains upside.

Comparability issues across companies:
- Different capital structures create different degrees of “equity optionality” (leverage).
- Embedded derivatives and contingent features can make reported EPS and diluted shares non-comparable without normalization.

Practical adjustments:
```python
def fully_diluted_share_count(basic_shares: float, in_the_money_options: float, treasury_stock_method_shares: float = 0.0) -> float:
    """Simple fully diluted count = basic + incremental option shares (approx)."""
    bs = _num(basic_shares, "basic_shares")
    opt = _num(in_the_money_options, "in_the_money_options")
    tsm = _num(treasury_stock_method_shares, "treasury_stock_method_shares")
    if bs < 0 or opt < 0 or tsm < 0:
        raise ValueError("Share inputs must be >= 0.")
    return bs + max(opt - tsm, 0.0)
```

---

### Quality of Earnings Flags
⚠️ EPS and valuation per share computed without fully diluted shares (options, warrants, convertibles).  
⚠️ Ignoring contingent payments (earn-outs) that are economically option-like and can be material to value allocation.  
⚠️ Using DCF without scenario or real-option treatment for businesses where outcomes are binary (drug approval, exploration success).  
✅ Clear disclosure of dilution assumptions and reconciliation from basic to fully diluted value per share.

---

### Sector-Specific Considerations

| Sector | Key optionality source | Typical treatment |
|---|---|---|
| Biotech | drug approval milestones | scenario-weighted DCF / real options; dilution critical |
| Natural resources | exploration success and commodity prices | real options; volatility drives value |
| Financials | structured products, embedded options | use parity/bounds; consistent discounting and hedge assumptions |
| Tech (growth) | platform scaling and strategic pivots | scenario valuation; treat employee options carefully in per-share bridge |

---

### Real-World Example
Scenario: A company has a large employee option pool. Your DCF produces enterprise value, and you need value per share.

1) Compute equity value = enterprise value - net debt (and other claims).  
2) Estimate fully diluted shares (basic + in-the-money options net of treasury stock method).  
3) Value per share = equity value / fully diluted shares.

```python
enterprise_value = 2_500_000_000  # $2.5B
net_debt = 400_000_000            # $0.4B
equity_value = enterprise_value - net_debt

basic_shares = 180_000_000
options = 30_000_000
tsm_shares = 8_000_000

fd_shares = fully_diluted_share_count(basic_shares, options, tsm_shares)
value_per_share = equity_value / fd_shares

print(f"Equity value: ${equity_value/1e9:.2f}B")
print(f"Fully diluted shares: {fd_shares/1e6:.1f}M")
print(f"Value per share: ${value_per_share:.2f}")
```

Interpretation: Per-share value is highly sensitive to dilution assumptions; for option-heavy capital structures, a “clean” enterprise DCF is not enough without a correct equity bridge.

See also: Chapter 16 (enterprise to equity bridge concepts), Chapter 17 (binomial option pricing), Chapter 20 (real options) for valuation of flexibility.
