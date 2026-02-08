# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 20: Real Options (managerial flexibility, expand/abandon/defer, project volatility)

### Core Concept
Real options treat managerial flexibility in capital projects as option-like payoffs: the right (not obligation) to invest, expand, contract, defer, or abandon based on future information. In valuation, real options are most useful when uncertainty is high and decisions are staged, because a single-path DCF typically undervalues flexibility. Practical real-options work is about translating a business decision into an option structure (underlying, strike, time, volatility), then selecting a model that matches the exercise rules (European vs American) and cash-flow structure (discrete vs continuous).

---

### Formula/Methodology

#### 1) Core mapping from project to option
```text
Underlying (S) = value of project cash flows if exercised today (PV of expected CFs)
Strike (K)     = required investment outlay to exercise (capex, acquisition price, expansion cost)
Maturity (T)   = decision window / time until opportunity expires
Volatility (σ) = uncertainty in project value (often proxied from peer assets/commodities)
Risk-free (r)  = risk-free rate consistent with currency and maturity
Yield (q)      = value leakage while waiting (e.g., competitive erosion, foregone cash flows)
```

Interpretation:
- Real options are usually valued under risk-neutral logic (no-arbitrage analog), but the inputs require judgment and documentation.

#### 2) Option types you repeatedly use in corporate valuation
Option to delay (timing option): like an American call on the project
```text
Value(delay) ≈ Call(S, K, T, r, σ, q)   with American exercise preferred
```

Option to expand: call on incremental value
```text
Value(expand) ≈ Call(S_incremental, K_expand, T, r, σ, q)
```

Option to abandon: put on project value (sell project for salvage)
```text
Value(abandon) ≈ Put(S, SalvageValue, T, r, σ, q)   with American exercise preferred
```

Option to switch (inputs/outputs): often needs a lattice or simulation (multiple state variables)

#### 3) Real-options adjusted project value
```text
Strategic Project Value = Base DCF NPV + Option Value(s)
```

Where:
- Base DCF NPV is the value under a fixed policy (no flexibility).
- Option Value(s) capture managerial responses to uncertainty.

---

### Practical Application (How to apply real options in valuation)

#### A) Decision rules: when real options are worth doing
Real options are most valuable when ALL are true:
- Uncertainty is high (σ is high)
- Management has real flexibility (can delay/abandon/scale)
- Decisions are staged and information arrives over time
- Downside is limited relative to upside (convex payoff)

Real options add little when:
- Project is “now or never” and fully committed
- Cash flows are stable and well forecastable
- Exercise is effectively forced by contracts/regulation

#### B) Choose the right valuation method
- Black–Scholes (European): OK for simple, single decision at a fixed date; limited flexibility
- Binomial tree (American): preferred for delay/abandon where early exercise matters
- Decision tree (discrete scenarios): useful for staged milestones; complements a binomial overlay
- Monte Carlo: useful with multiple uncertainties (commodity + volume + FX), but needs careful exercise logic

#### C) Estimating project volatility (σ) in a defendable way
Common approaches (document assumptions):
- Market proxy: volatility of comparable traded assets (peer equities, commodity prices)
- Cash-flow proxy: volatility of key value driver (price, volume, margin) propagated into project value
- Implied from transaction dispersion: rarely clean, but may support a range

Practical guidance:
- Use annualized σ in decimals (0.30 = 30%).
- Align σ to the underlying definition: project value volatility, not accounting earnings volatility.

#### D) Handling “dividend yield” analogue (q) for projects
Waiting can destroy value (opportunity cost). Model this as a yield q:
- Competitive erosion / lost market share while waiting
- Foregone cash flows if project could operate today
- Technology obsolescence

Rule of thumb:
- If delaying causes you to lose a fraction of value per year, set q to that fraction (with justification).

#### E) Integrate into enterprise valuation without double-counting
- Base DCF should represent a fixed policy (e.g., invest now and operate).
- Option valuation should represent the incremental value from flexibility vs that fixed policy.
- Avoid double-counting by ensuring scenarios in DCF do not already assume optimal exercise.

---

### Python Implementation
```python
from typing import Any, Dict, Optional
import numpy as np

# Reuse binomial option pricer (American/European) from Chapter 17 style, included here for self-containment.

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def crr_ud(sigma: float, dt: float) -> Dict[str, float]:
    sig = _num(sigma, "sigma")
    dtt = _num(dt, "dt")
    if sig < 0:
        raise ValueError("sigma must be >= 0.")
    if dtt <= 0:
        raise ValueError("dt must be > 0.")
    u = float(np.exp(sig * np.sqrt(dtt)))
    d = float(np.exp(-sig * np.sqrt(dtt)))
    if u <= d:
        raise ValueError("Need u > d.")
    return {"u": u, "d": d}

def risk_neutral_p(u: float, d: float, r: float, dt: float, q: float = 0.0) -> float:
    uu = _num(u, "u"); dd = _num(d, "d"); rr = _num(r, "r"); dtt = _num(dt, "dt"); qq = _num(q, "q")
    if dtt <= 0:
        raise ValueError("dt must be > 0.")
    if uu <= dd:
        raise ValueError("Need u > d.")
    if qq < 0:
        raise ValueError("q must be >= 0.")
    R = float(np.exp((rr - qq) * dtt))
    p = (R - dd) / (uu - dd)
    if p < -1e-10 or p > 1 + 1e-10:
        raise ValueError(f"Risk-neutral probability out of bounds: p={p:.6f}. Check inputs.")
    return float(min(max(p, 0.0), 1.0))

def build_stock_tree(S0: float, u: float, d: float, N: int) -> np.ndarray:
    s0 = _num(S0, "S0")
    uu = _num(u, "u"); dd = _num(d, "d")
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
    american: bool = True,
    q: float = 0.0,
) -> float:
    """CRR recombining binomial pricer (supports American exercise).

    Args:
        S0: underlying today
        K: strike
        r: annual risk-free rate (cont. comp.)
        T: years to expiry
        sigma: annual volatility
        N: steps
        option_type: 'call' or 'put'
        american: True for American
        q: dividend/yield proxy

    Returns:
        float option value
    """
    s0 = _num(S0, "S0"); k = _num(K, "K"); rr = _num(r, "r"); tt = _num(T, "T"); sig = _num(sigma, "sigma"); qq = _num(q, "q")
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
    ud = crr_ud(sig, dt)
    u, d = ud["u"], ud["d"]
    p = risk_neutral_p(u, d, rr, dt, q=qq)
    disc = float(np.exp(-rr * dt))

    S_tree = build_stock_tree(s0, u, d, N)
    V = np.zeros_like(S_tree)
    payoff = payoff_call if option_type == "call" else payoff_put

    for j in range(N + 1):
        V[j, N] = payoff(S_tree[j, N], k)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * V[j, i + 1] + (1 - p) * V[j + 1, i + 1])
            if american:
                intrinsic = payoff(S_tree[j, i], k)
                V[j, i] = max(intrinsic, cont)
            else:
                V[j, i] = cont

    return float(V[0, 0])

def real_option_delay(project_value: float, investment_cost: float, r: float, T: float, sigma: float, N: int, q: float = 0.0) -> float:
    """Option to delay investment: treat as an American call on project value."""
    return binomial_option_price(S0=project_value, K=investment_cost, r=r, T=T, sigma=sigma, N=N, option_type="call", american=True, q=q)

def real_option_abandon(project_value: float, salvage_value: float, r: float, T: float, sigma: float, N: int, q: float = 0.0) -> float:
    """Option to abandon: treat as an American put with strike = salvage value."""
    return binomial_option_price(S0=project_value, K=salvage_value, r=r, T=T, sigma=sigma, N=N, option_type="put", american=True, q=q)

def real_options_adjusted_npv(base_npv: float, *option_values: float) -> float:
    """Combine base NPV with additive option values (avoid double-counting)."""
    base = _num(base_npv, "base_npv")
    total = base
    for i, ov in enumerate(option_values, start=1):
        total += _num(ov, f"option_value_{i}")
    return float(total)

# Example usage
project_value_now = 180_000_000    # PV of project CFs if invested today
investment_cost = 150_000_000      # capex required to invest
salvage_value = 120_000_000        # recovery if abandon
r = 0.04
T = 2.0
sigma = 0.35
N = 200
q = 0.02  # value leakage while waiting

delay_val = real_option_delay(project_value_now, investment_cost, r, T, sigma, N, q=q)
abandon_val = real_option_abandon(project_value_now, salvage_value, r, T, sigma, N, q=0.0)

print(f"Delay option value:   ${delay_val/1e6:.1f}M")
print(f"Abandon option value: ${abandon_val/1e6:.1f}M")

base_npv = project_value_now - investment_cost
adj = real_options_adjusted_npv(base_npv, delay_val)  # delay adds value vs invest-now policy
print(f"Base NPV:             ${base_npv/1e6:.1f}M")
print(f"Real-options adj NPV: ${adj/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- Real options can justify valuation premia where traditional DCF undervalues flexibility (staging, deferral, abandonment).
- They provide a structured way to incorporate uncertainty and management actions into valuation, improving decision usefulness for capex, M&A, and R&D portfolios.

Impact on multiples:
- Multiples often embed option value implicitly (especially in growth/biotech/resource companies). Real options help explain why “current earnings” multiples can look extreme relative to value.
- Firms with valuable growth pipelines may screen “overvalued” on EV/EBITDA but not after recognizing embedded expansion options.

Impact on DCF inputs:
- Real options usually do not change WACC itself; they change the distribution and timing of cash flows by allowing optimal decisions.
- q (value leakage) is a practical proxy for competitive erosion or foregone cash flows and can materially reduce option value.

Comparability issues across companies:
- Different pipeline optionality and managerial flexibility reduce comparability of multiples across firms even within the same sector.
- Volatility proxies differ widely; document peer selection and leverage adjustments where applicable.

Practical adjustments:
```python
def option_value_per_share(option_value: float, shares_outstanding: float) -> float:
    """Translate project option value into per-share impact for equity bridge."""
    ov = _num(option_value, "option_value")
    sh = _num(shares_outstanding, "shares_outstanding")
    if sh <= 0:
        raise ValueError("shares_outstanding must be > 0.")
    return float(ov / sh)
```

---

### Quality of Earnings Flags
⚠️ Management narrative claims “significant optionality” but no quantified options analysis and no evidence of staged decision rights.  
⚠️ Base DCF already embeds optimistic ramp/expansion assumptions, then real options are added on top (double-counting).  
⚠️ Volatility proxy chosen opportunistically without peer/driver justification; no sensitivity table.  
✅ Clear mapping of decision to option structure, with sensitivity to σ, T, and q and explicit statement of the base policy used for NPV.

---

### Sector-Specific Considerations

| Sector | Typical real option | Key uncertainty | Common modeling choice |
|---|---|---|---|
| Biotech | continue/abandon trials; launch option | regulatory success | decision tree + option overlay |
| Resources | defer/develop/abandon mine | commodity price | binomial or Monte Carlo on price |
| Infrastructure | expand capacity | demand growth | binomial on project value/demand |
| SaaS/Tech | scale GTM spend | adoption/retention | scenario tree + staged investment |

---

### Real-World Example
Scenario: Evaluate whether to invest now or wait up to 2 years for more clarity.

Assumptions:
- Invest now policy base NPV = S - K
- Option to delay adds value because you invest only if value improves enough
- q captures erosion/foregone cash flows from waiting

```python
S = 180_000_000
K = 150_000_000
r = 0.04
T = 2.0
sigma = 0.35
N = 200
q = 0.02

delay = real_option_delay(S, K, r, T, sigma, N, q=q)
base_npv = S - K
adj_npv = base_npv + delay

print(f"Base invest-now NPV: ${base_npv/1e6:.1f}M")
print(f"Delay option value:  ${delay/1e6:.1f}M")
print(f"Adj NPV (policy+option): ${adj_npv/1e6:.1f}M")
```

Interpretation: If delay option value is material, the investment decision should be staged (or contingent) rather than treated as fixed, and valuation should recognize the convexity of outcomes.

See also: Chapter 17 (binomial trees for American exercise), Chapter 18 (Black–Scholes as baseline), Chapter 22/24 (Monte Carlo for multiple uncertainties).
