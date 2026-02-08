# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 26: Replicating Options and Option Strategies (replication, delta hedging, payoff construction)

### Core Concept
Option replication constructs an option payoff using a dynamic position in the underlying asset and a risk-free asset (cash/borrowing). For valuation, replication underpins arbitrage-free pricing, helps audit model outputs (binomial/Black–Scholes), and supports practical hedging logic for embedded derivatives and real options.

---

### Formula/Methodology

#### 1) Basic replication in one-step binomial
Underlying price today: S0  
Next step: Su (up) or Sd (down)  
Option payoff at next step: Cu and Cd

Replicating portfolio:
- Δ shares of underlying
- B in risk-free asset (cash, could be negative if borrowing)
Risk-free growth factor per step: (1 + r)

Solve:
```text
Δ·Su + B·(1 + r) = Cu
Δ·Sd + B·(1 + r) = Cd
```

Closed-form solution:
```text
Δ = (Cu - Cd) / (Su - Sd)
B = (Su·Cd - Sd·Cu) / ((Su - Sd)·(1 + r))
```

Option value today:
```text
C0 = Δ·S0 + B
```

Where:
- S0 = underlying spot price at t=0
- Su, Sd = underlying price in up/down states
- Cu, Cd = option payoffs in up/down states
- r = risk-free rate per step (decimal)
- Δ = hedge ratio (shares)
- B = cash position (positive lending, negative borrowing)

#### 2) Risk-neutral probability (same result)
Risk-neutral probability p:
```text
p = ((1 + r)·S0 - Sd) / (Su - Sd)
```

Option value:
```text
C0 = (p·Cu + (1 - p)·Cd) / (1 + r)
```

#### 3) Continuous-time delta hedging intuition
Delta (local hedge ratio):
```text
Δ = ∂C/∂S
```

Discretely hedged portfolio (rebalanced each step) approximates replication. Hedge error increases with:
- larger time steps
- higher volatility
- nonlinearity (gamma) and jumps
- transaction costs

#### 4) Common option strategies (payoff construction)
Let call payoff at expiry: max(S_T - K, 0)  
Put payoff: max(K - S_T, 0)

| Strategy | Construction | Payoff intuition |
|---|---|---|
| Covered call | +Stock +Short Call | capped upside, downside like stock |
| Protective put | +Stock +Put | floor on losses, pays for insurance |
| Bull call spread | +Call(K1) -Call(K2) | limited upside, reduced premium |
| Straddle | +Call(K) +Put(K) | long volatility (move either way) |
| Collar | +Stock +Put(Kp) -Call(Kc) | downside floor + capped upside |

---

### Practical Application (How to apply in valuation work)

#### A) Validate option model outputs by replication (binomial)
1) Build the one-step or multi-step binomial tree.
2) At each node, compute Δ and B that replicate next-step option values.
3) Check that C = Δ·S + B holds at every node (numerical sanity check).

#### B) Embedded derivatives and real options
- Convertible instruments, earn-outs, performance-linked consideration, or contingent liabilities often behave like options.
- Replication logic helps identify key drivers (delta, gamma) and where linear approximations fail.

#### C) Hedging and risk management narrative for valuation committees
- Delta shows exposure of option value to underlying movement (sensitivity).
- Gamma shows convexity; high gamma implies hedging requires frequent rebalancing (higher friction, risk).

---

### Python Implementation
```python
from typing import Any, Dict, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def one_step_replication(
    s0: float,
    su: float,
    sd: float,
    cu: float,
    cd: float,
    r: float,
) -> Dict[str, float]:
    """Compute one-step replicating portfolio for an option.

    Args:
        s0: spot price at t0
        su: underlying up-state price at t1
        sd: underlying down-state price at t1
        cu: option payoff in up-state at t1
        cd: option payoff in down-state at t1
        r: risk-free rate per step (decimal), e.g., 0.01 for 1%

    Returns:
        dict: delta, bond (cash), option_value, rn_prob

    Raises:
        ValueError: invalid or arbitrage-inconsistent inputs
    """
    S0 = _num(s0, "s0")
    Su = _num(su, "su")
    Sd = _num(sd, "sd")
    Cu = _num(cu, "cu")
    Cd = _num(cd, "cd")
    rr = _num(r, "r")

    if Su <= Sd:
        raise ValueError("Require su > sd for a valid binomial step.")
    if (1.0 + rr) <= 0:
        raise ValueError("Require (1 + r) > 0.")

    denom = Su - Sd
    delta = (Cu - Cd) / denom
    bond = (Su * Cd - Sd * Cu) / (denom * (1.0 + rr))
    c0 = delta * S0 + bond

    # risk-neutral probability
    p = ((1.0 + rr) * S0 - Sd) / denom
    # Basic no-arbitrage bounds for p in [0,1] imply Sd <= (1+r)S0 <= Su
    if p < -1e-8 or p > 1.0 + 1e-8:
        raise ValueError("Inputs imply arbitrage (risk-neutral probability not in [0,1]).")

    c0_rn = (p * Cu + (1.0 - p) * Cd) / (1.0 + rr)

    # numerical consistency check
    if abs(c0 - c0_rn) > 1e-6 * max(1.0, abs(c0_rn)):
        raise ValueError("Replication and risk-neutral pricing are inconsistent; check inputs.")

    return {"delta": float(delta), "bond": float(bond), "option_value": float(c0_rn), "rn_prob": float(p)}

def payoff_call(st: np.ndarray, k: float) -> np.ndarray:
    K = float(_num(k, "k"))
    S = np.array(st, dtype=float)
    if not np.all(np.isfinite(S)):
        raise ValueError("st must be finite.")
    return np.maximum(S - K, 0.0)

def payoff_put(st: np.ndarray, k: float) -> np.ndarray:
    K = float(_num(k, "k"))
    S = np.array(st, dtype=float)
    if not np.all(np.isfinite(S)):
        raise ValueError("st must be finite.")
    return np.maximum(K - S, 0.0)

def strategy_payoff(
    st: np.ndarray,
    legs: Dict[str, Tuple[str, float, float]],
) -> np.ndarray:
    """Build payoff from option legs.

    Args:
        st: array of terminal underlying prices
        legs: dict of leg_name -> (instrument, strike, quantity)
              instrument in {'call','put','stock','cash'}
              For 'stock' and 'cash', strike is ignored.

    Returns:
        total payoff across legs

    Raises:
        ValueError: invalid legs
    """
    S = np.array(st, dtype=float)
    if S.ndim != 1:
        raise ValueError("st must be 1D array of terminal prices.")
    if not np.all(np.isfinite(S)):
        raise ValueError("st must be finite.")

    total = np.zeros_like(S, dtype=float)
    for name, (inst, strike, qty) in legs.items():
        q = float(_num(qty, f"{name}.qty"))
        inst_l = str(inst).lower().strip()
        if inst_l == "call":
            total += q * payoff_call(S, strike)
        elif inst_l == "put":
            total += q * payoff_put(S, strike)
        elif inst_l == "stock":
            total += q * S
        elif inst_l == "cash":
            total += q * float(_num(strike, f"{name}.cash_amount"))
        else:
            raise ValueError(f"Unknown instrument '{inst}'. Use call/put/stock/cash.")
    return total

# Example usage: one-step replication of a call
S0 = 100.0
Su = 120.0
Sd = 90.0
K = 100.0
r = 0.02

Cu = max(Su - K, 0.0)
Cd = max(Sd - K, 0.0)

rep = one_step_replication(S0, Su, Sd, Cu, Cd, r)
print(rep)  # delta, bond, option value

# Example usage: payoff of a protective put
st_grid = np.linspace(50, 150, 101)
legs = {
    "stock": ("stock", 0.0, 1.0),
    "put": ("put", 100.0, 1.0),
}
pp = strategy_payoff(st_grid, legs)
print(f"Protective put payoff at ST=80: {pp[np.argmin(np.abs(st_grid-80))]:.2f}")
```

---

### Valuation Impact
Why this matters:
- Replication is the core “sanity check” behind arbitrage-free valuation: if an option can be replicated, its price is pinned down by the replicating portfolio.
- In valuations involving earn-outs, convertibles, warrants, or contingent consideration, replication intuition clarifies what drives value and which risks are effectively being priced.

Impact on multiples:
- Option-like instruments can distort EV/Equity bridges and per-share value if not recognized (dilution, embedded leverage).
- For firms with significant options/convertibles, P/E comparability breaks; use fully diluted shares and adjust equity value.

Impact on DCF inputs:
- If a project has expansion/abandonment flexibility (real options), DCF may understate value; replication-based option valuation adds flexibility value rather than forcing it into growth.

Comparability issues across companies:
- Equity with heavy option overhang (employee options/warrants) is less comparable to “plain vanilla” equity; normalize by fully diluted market cap and option-adjusted leverage.

Practical adjustments:
```python
def fully_diluted_share_count(basic_shares: float, in_the_money_options: float, assumed_net_shares: float = 0.0) -> float:
    """Simple fully diluted share count scaffold.

    Use a treasury-stock method in practice; this is a minimal placeholder.
    """
    bs = float(_num(basic_shares, "basic_shares"))
    itmo = float(_num(in_the_money_options, "in_the_money_options"))
    net = float(_num(assumed_net_shares, "assumed_net_shares"))
    if bs <= 0:
        raise ValueError("basic_shares must be > 0.")
    if itmo < 0 or net < 0:
        raise ValueError("option counts must be >= 0.")
    return float(bs + max(itmo - net, 0.0))
```

---

### Quality of Earnings Flags
⚠️ Embedded derivatives ignored in EV-to-equity bridge (convertibles, warrants, earn-outs) leading to overstated equity value per share.  
⚠️ Option values computed with inconsistent inputs (term, volatility, risk-free rate, dividend yield) across periods or peers.  
⚠️ Hedging narrative used without acknowledging hedging error, transaction costs, and jump risk.  
✅ Clear disclosure of option-like instruments and consistent valuation inputs; dilution treated transparently.

---

### Sector-Specific Considerations

| Sector | Where replication shows up | Typical treatment |
|---|---|---|
| Tech / growth | warrants, employee options | fully diluted equity, option valuation |
| Energy / mining | real options (expand/abandon) | add flexibility value via options |
| Financials | structured notes, embedded derivatives | replication + model governance |
| PE / VC | liquidation preferences as options | scenario + option-like payoff modeling |

---

### Real-World Example
Scenario: Replicate a one-step call option and cross-check against risk-neutral pricing.

```python
S0, Su, Sd = 100.0, 120.0, 90.0
K, r = 100.0, 0.02
Cu, Cd = max(Su-K, 0.0), max(Sd-K, 0.0)

rep = one_step_replication(S0, Su, Sd, Cu, Cd, r)
print(f"Delta: {rep['delta']:.4f} shares")
print(f"Bond:  {rep['bond']:.2f} cash")
print(f"Value: {rep['option_value']:.2f}  (risk-neutral)")
```

Interpretation: Delta indicates how many shares replicate the option locally; the bond position finances the hedge. If risk-neutral probability is outside [0,1], the assumed Su/Sd and r imply arbitrage (inputs inconsistent).

See also: Chapter 17 (binomial option pricing), Chapter 18 (Black–Scholes), Chapter 19 (Greeks).
