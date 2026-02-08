# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 17: The Binomial Option Pricing Model (replication, risk-neutral pricing, American exercise)

### Core Concept
The binomial model prices options by modeling the underlying price as moving up or down each step, then valuing the option by backward induction using no-arbitrage and replication. In valuation, binomial trees are practical for real options (expand/abandon), employee options with vesting/exercise features, and American-style options where early exercise matters. The core advantage over closed-form formulas is flexibility: you can model discrete cash flows, changing volatility, and early exercise in a transparent framework.

---

### Formula/Methodology

#### 1) One-step binomial stock process
```text
S_up   = S0 × u
S_down = S0 × d
```

Where:
- S0 = underlying price today
- u = up factor (> 1)
- d = down factor (< 1), typically d = 1/u (but not required)

#### 2) Risk-neutral probability (discrete compounding)
```text
p = ((1 + r) - d) / (u - d)
```

Where:
- r = risk-free rate per step (decimal)
- p must satisfy 0 ≤ p ≤ 1 (otherwise arbitrage or inconsistent inputs)

#### 3) Risk-neutral valuation (European, one step)
```text
Option0 = ( p × Option_up + (1 - p) × Option_down ) / (1 + r)
```

#### 4) Replicating portfolio (Δ shares + risk-free borrowing/lending)
```text
Δ = (Option_up - Option_down) / (S_up - S_down)
B = (Option_up - Δ × S_up) / (1 + r)   (value invested in risk-free asset today)
Option0 = Δ × S0 + B
```

Interpretation:
- Δ is the hedge ratio (shares per option)
- B is the risk-free position needed to replicate the option payoff

#### 5) Multi-step (N steps) backward induction
At each node:
```text
Continuation = ( p × V_up + (1 - p) × V_down ) / (1 + r)
```

For American options:
```text
V_node = max( Intrinsic_value, Continuation_value )
```

#### 6) Common u/d parameterization (CRR)
If using volatility σ and step length Δt (years):
```text
u = exp(σ × sqrt(Δt))
d = exp(-σ × sqrt(Δt))
R = exp(r × Δt)   (continuous compounding per step)
p = (R - d) / (u - d)
```

---

### Practical Application (How to apply binomial trees in valuation)

#### A) When you should use binomial instead of Black–Scholes
Use binomial when:
- Early exercise matters (American puts; dividend-paying stocks; employee options).
- Payoffs depend on path or discrete features (barriers, vesting, step-up strikes).
- You need to embed managerial flexibility into project valuation (real options).

Use closed-form models when:
- European options with standard assumptions and no discrete features.

#### B) Choose a tree specification that matches your data
Inputs you must align:
- r: risk-free rate consistent with Δt
- σ: volatility per annum if using CRR; ensure Δt in years
- Dividend treatment:
  - If continuous dividend yield q is used, adjust drift in risk-neutral probability (see below).
  - For discrete dividends, adjust stock price at the dividend node.

Dividend yield variant (continuous yield q, CRR):
```text
R = exp((r - q) × Δt)
p = (R - d) / (u - d)
```

#### C) Sanity checks before running the tree
- u > d
- 0 ≤ p ≤ 1
- Increasing N should stabilize the option value (convergence check)
- American option value ≥ European option value (same parameters)

#### D) Map outputs into valuation deliverables
- Warrants/convertibles: price option-like features and incorporate into equity bridge.
- Real options: treat project value as underlying; treat investment cost as strike; model exercise (invest/expand) as American-style decision.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Tuple
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def crr_ud(sigma: float, dt: float) -> Tuple[float, float]:
    """Cox-Ross-Rubinstein up/down factors."""
    sig = _num(sigma, "sigma")
    dtt = _num(dt, "dt")
    if sig < 0:
        raise ValueError("sigma must be >= 0.")
    if dtt <= 0:
        raise ValueError("dt must be > 0.")
    u = float(np.exp(sig * np.sqrt(dtt)))
    d = float(np.exp(-sig * np.sqrt(dtt)))
    if not (u > d):
        raise ValueError("Invalid u/d: need u > d.")
    return u, d

def risk_neutral_p(u: float, d: float, r: float, dt: float, q: float = 0.0) -> float:
    """Risk-neutral probability using continuous compounding per step.

    Args:
        u, d: up/down factors
        r: annual risk-free rate (decimal)
        dt: step length in years
        q: continuous dividend yield (decimal)

    Returns:
        p in [0,1]

    Raises:
        ValueError: invalid inputs or arbitrage condition.
    """
    uu = _num(u, "u")
    dd = _num(d, "d")
    rr = _num(r, "r")
    dtt = _num(dt, "dt")
    qq = _num(q, "q")
    if dtt <= 0:
        raise ValueError("dt must be > 0.")
    if uu <= dd:
        raise ValueError("Need u > d.")
    if qq < 0:
        raise ValueError("q must be >= 0.")
    R = float(np.exp((rr - qq) * dtt))
    p = (R - dd) / (uu - dd)
    if p < -1e-10 or p > 1 + 1e-10:
        raise ValueError(f"Risk-neutral probability out of bounds: p={p:.6f}. Check r, q, sigma, dt.")
    return float(min(max(p, 0.0), 1.0))

def build_stock_tree(S0: float, u: float, d: float, N: int) -> np.ndarray:
    """Build a recombining binomial stock price tree.

    Returns:
        tree of shape (N+1, N+1) where tree[j, i] is price at time i with j down moves.
        (Only entries with 0<=j<=i are used.)
    """
    s0 = _num(S0, "S0")
    uu = _num(u, "u")
    dd = _num(d, "d")
    if s0 < 0:
        raise ValueError("S0 must be >= 0.")
    if uu <= 0 or dd <= 0:
        raise ValueError("u and d must be > 0.")
    if uu <= dd:
        raise ValueError("Need u > d.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    tree = np.zeros((N + 1, N + 1), dtype=float)
    tree[0, 0] = s0
    for i in range(1, N + 1):
        tree[0, i] = tree[0, i - 1] * uu
        for j in range(1, i + 1):
            tree[j, i] = tree[j - 1, i - 1] * dd
    return tree

def payoff_call(S: float, K: float) -> float:
    return max(_num(S, "S") - _num(K, "K"), 0.0)

def payoff_put(S: float, K: float) -> float:
    return max(_num(K, "K") - _num(S, "S"), 0.0)

def binomial_option_price(
    S0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    N: int,
    option_type: str = "call",
    american: bool = False,
    q: float = 0.0,
) -> Dict[str, Any]:
    """Price an option using a CRR recombining binomial tree.

    Args:
        S0: underlying price today
        K: strike
        r: annual risk-free rate (decimal)
        T: time to maturity (years)
        sigma: volatility (annual, decimal)
        N: number of steps
        option_type: 'call' or 'put'
        american: whether American (early exercise)
        q: continuous dividend yield (annual, decimal)

    Returns:
        dict with price, p, u, d, and (option_tree) for debugging

    Raises:
        ValueError: invalid inputs or arbitrage.
    """
    s0 = _num(S0, "S0")
    k = _num(K, "K")
    rr = _num(r, "r")
    tt = _num(T, "T")
    sig = _num(sigma, "sigma")
    qq = _num(q, "q")

    if s0 < 0 or k < 0:
        raise ValueError("S0 and K must be >= 0.")
    if tt <= 0:
        raise ValueError("T must be > 0.")
    if sig < 0:
        raise ValueError("sigma must be >= 0.")
    if qq < 0:
        raise ValueError("q must be >= 0.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    dt = tt / N
    u, d = crr_ud(sig, dt)
    p = risk_neutral_p(u, d, rr, dt, q=qq)
    disc = float(np.exp(-rr * dt))

    S_tree = build_stock_tree(s0, u, d, N)
    V = np.zeros_like(S_tree)

    payoff_fn = payoff_call if option_type == "call" else payoff_put

    # Terminal payoffs
    for j in range(N + 1):
        V[j, N] = payoff_fn(S_tree[j, N], k)

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * V[j, i + 1] + (1 - p) * V[j + 1, i + 1])
            if american:
                intrinsic = payoff_fn(S_tree[j, i], k)
                V[j, i] = max(intrinsic, cont)
            else:
                V[j, i] = cont

    return {"price": float(V[0, 0]), "p": p, "u": u, "d": d, "dt": dt, "option_tree": V}

# Example usage
S0 = 100.0
K = 100.0
r = 0.04
T = 1.0
sigma = 0.25
N = 200

eu_call = binomial_option_price(S0, K, r, T, sigma, N, option_type="call", american=False, q=0.0)
am_put  = binomial_option_price(S0, K, r, T, sigma, N, option_type="put", american=True, q=0.0)

print(f"European call (binomial): {eu_call['price']:.4f}")
print(f"American put  (binomial): {am_put['price']:.4f}")
print(f"p={eu_call['p']:.4f}, u={eu_call['u']:.4f}, d={eu_call['d']:.4f}")
```

---

### Valuation Impact
Why this matters:
- Binomial trees are a valuation workhorse for instruments and projects where flexibility/early exercise is real: employee options, American-style derivatives, and real options.
- The replication logic provides an internally consistent, no-arbitrage value, which is critical for auditability in valuation memos.

Impact on multiples:
- Optionality can justify why earnings multiples understate value (convex payoffs, especially in early-stage or distressed cases).
- If you ignore option value embedded in contracts (warrants/convertibles), EV/Equity bridge and per-share multiples will be distorted.

Impact on DCF inputs:
- Binomial can be layered on top of DCF by treating project value as the underlying and the investment outlay as strike (exercise).
- Early exercise logic can approximate optimal timing decisions (e.g., invest only if project value crosses a threshold).

Comparability issues across companies:
- Different disclosure of option terms (vesting, early exercise, dilution) can produce non-comparable per-share valuations.
- Volatility inputs (σ) are often the largest driver; ensure consistent estimation (historical vs implied vs peer).

Practical adjustments:
```python
def real_option_expand(
    project_value: float,
    expansion_cost: float,
    r: float,
    T: float,
    sigma: float,
    N: int,
    q: float = 0.0,
) -> float:
    """Treat expansion right as a call option on project value."""
    res = binomial_option_price(
        S0=project_value, K=expansion_cost, r=r, T=T, sigma=sigma, N=N,
        option_type="call", american=True, q=q
    )
    return float(res["price"])
```

---

### Quality of Earnings Flags
⚠️ Material warrant/option liabilities not modeled, but equity value per share is presented as precise.  
⚠️ Volatility assumption chosen to “make the answer work” (needs a defendable basis).  
⚠️ Early exercise ignored for employee options (American-like behavior, forfeiture/vesting).  
✅ Sensitivity table over σ, r, and N, and documented mapping from contract terms to tree assumptions.

---

### Sector-Specific Considerations

| Sector | Common binomial use | Key issue | Typical handling |
|---|---|---|---|
| Biotech | milestone / continuation options | binary outcomes | scenario trees + binomial overlay |
| Resources | development timing options | commodity-linked volatility | stress-test σ and exercise thresholds |
| Financials | structured notes | early exercise / call features | calibrate to market-implied σ where possible |
| High leverage | equity as option | default boundary | integrate with distress modelling (firm value tree) |

---

### Real-World Example
Scenario: Value a management expansion option on a project.

Inputs:
- Project value today: $150M
- Expansion cost (strike): $120M
- Time to decision: 2 years
- Risk-free rate: 4%
- Volatility of project value: 35%
- Use 200 steps

```python
opt_val = real_option_expand(
    project_value=150_000_000,
    expansion_cost=120_000_000,
    r=0.04,
    T=2.0,
    sigma=0.35,
    N=200
)
print(f"Expansion option value: ${opt_val/1e6:.1f}M")
```

Interpretation: If the expansion option is worth a material fraction of base project NPV, a single-path DCF that ignores timing flexibility will understate value.

See also: Chapter 16 (option basics and parity/bounds), Chapter 18 (Black–Scholes), Chapter 20 (real options).
