# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 27: Using Monte Carlo Methods for Option Pricing (risk-neutral simulation, payoff discounting, variance reduction)

### Core Concept
Monte Carlo option pricing estimates an option’s value by simulating many possible paths for the underlying under the risk-neutral measure, computing payoffs, and discounting expected payoffs. In valuation, Monte Carlo is the workhorse for pricing complex, path-dependent, or multi-factor contingent claims (earn-outs, performance hurdles, commodity-linked contracts, and real options) where closed-form models are unavailable or unreliable.

---

### Formula/Methodology

#### 1) Risk-neutral pricing identity
Option value equals discounted expected payoff under risk-neutral probabilities:
```text
C0 = E_Q[ Payoff(ST, path) ] / (1 + r)^T
```

Where:
- C0 = option value today
- E_Q[·] = expectation under risk-neutral measure Q
- r = risk-free rate per period (decimal)
- T = number of periods (or use continuous discounting)
- Payoff(...) = payoff function (may depend on entire path, not just ST)

Continuous discounting form:
```text
C0 = exp(-r·t) · E_Q[ Payoff ]
```

#### 2) Geometric Brownian Motion (GBM) under risk-neutral measure
A common equity price process:
```text
dS/S = r·dt + σ·dW
```

Discretized (exact) step for Δt:
```text
S_{t+Δt} = S_t · exp( (r - 0.5·σ^2)·Δt + σ·sqrt(Δt)·Z )
```

Where:
- σ = volatility (annualized if dt in years)
- Z ~ Normal(0,1) i.i.d.

#### 3) Monte Carlo estimator (plain)
Simulate N paths, compute discounted payoffs:
```text
C0_hat = exp(-r·t) · (1/N) · Σ_{i=1..N} Payoff_i
```

Standard error (SE) estimate:
```text
SE(C0_hat) = exp(-r·t) · stdev(Payoff) / sqrt(N)
```

#### 4) Variance reduction (practical)
**Antithetic variates:** use Z and -Z in paired paths; reduces variance for monotone payoffs.  
**Control variates:** adjust estimator using a correlated variable with known expectation (e.g., underlying ST or Black–Scholes price for a related vanilla option).  
**Moment matching:** force simulated Z to have mean 0 and variance 1.

#### 5) American options (note)
Standard Monte Carlo does not handle early exercise cleanly. Common workarounds:
- Longstaff–Schwartz least-squares Monte Carlo (LSM)
- Binomial/trinomial trees (if feasible)
- PDE/finite-difference methods

(Implement LSM if the option has early exercise features; otherwise treat as European.)

---

### Practical Application (How to apply in valuation work)

#### A) Choose the right simulation “measure”
- For pricing contingent claims: simulate under risk-neutral (drift = risk-free rate, adjusted for dividends/forwards).
- For forecasting business outcomes (DCF scenarios): simulate under real-world (drift = expected return) but discount with risk-adjusted rates.
Mixing these leads to systematic mispricing.

#### B) Inputs governance checklist (model audit)
| Input | Practical source | Common mistake |
|---|---|---|
| Risk-free rate r | matching tenor government curve | using mismatched horizon |
| Volatility σ | implied vol / historical vol | mixing frequencies/tenors |
| Dividend yield q (if any) | forecast dividend/forward curve | ignoring carry |
| Correlations | historical or implied | unstable estimates ignored |
| Path rules | contract terms | payoff coded incorrectly |

#### C) Path-dependent payoffs (earn-outs / hurdles)
- Code payoff using contract logic (caps, floors, averaging, measurement windows).
- Store and validate intermediate metrics (e.g., average price, max drawdown) to ensure payoff matches term sheet.

#### D) Reporting outputs (decision-ready)
Always report:
- Point estimate, standard error, and a convergence diagnostic (estimate vs N).
- Sensitivities: Δ (delta), Vega (σ sensitivity), and key parameter stress.
- Back-of-envelope cross-check: compare to a simplified closed-form approximation where possible.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Tuple
import numpy as np
from math import exp, sqrt, log
from statistics import NormalDist

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
    n_paths: int,
    dividend_yield: float = 0.0,
    seed: Optional[int] = None,
    antithetic: bool = True,
    moment_match: bool = True,
) -> np.ndarray:
    """Simulate GBM price paths under risk-neutral measure.

    Args:
        s0: initial price
        r: risk-free rate (annualized if t in years)
        sigma: volatility (annualized)
        t: time to maturity in years
        steps: number of time steps
        n_paths: number of simulated paths
        dividend_yield: continuous dividend yield q (optional)
        seed: RNG seed
        antithetic: if True, use antithetic variates (pairs)
        moment_match: if True, moment-match Z to mean 0, var 1

    Returns:
        paths: ndarray shape (n_paths_eff, steps+1)

    Raises:
        ValueError: on invalid inputs
    """
    S0 = _num(s0, "s0")
    rr = _num(r, "r")
    sig = _num(sigma, "sigma")
    tt = _num(t, "t")
    qq = _num(dividend_yield, "dividend_yield")

    if S0 <= 0:
        raise ValueError("s0 must be > 0.")
    if sig < 0:
        raise ValueError("sigma must be >= 0.")
    if tt <= 0:
        raise ValueError("t must be > 0.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2.")

    dt = tt / steps
    drift = (rr - qq - 0.5 * sig * sig) * dt
    vol = sig * sqrt(dt)

    rng = np.random.default_rng(seed)

    # Determine effective n for antithetic pairing
    if antithetic:
        half = (n_paths + 1) // 2
        Z = rng.standard_normal(size=(half, steps))
        Z = np.vstack([Z, -Z])[:n_paths, :]
    else:
        Z = rng.standard_normal(size=(n_paths, steps))

    if moment_match:
        m = float(Z.mean())
        s = float(Z.std(ddof=0))
        if s == 0:
            raise ValueError("Degenerate Z draws; try different seed or settings.")
        Z = (Z - m) / s

    # Build paths
    paths = np.empty((Z.shape[0], steps + 1), dtype=float)
    paths[:, 0] = S0
    if sig == 0:
        # deterministic forward under carry (r-q)
        for j in range(steps):
            paths[:, j + 1] = paths[:, j] * exp((rr - qq) * dt)
        return paths

    for j in range(steps):
        paths[:, j + 1] = paths[:, j] * np.exp(drift + vol * Z[:, j])
    return paths

def european_call_payoff(st: np.ndarray, k: float) -> np.ndarray:
    K = float(_num(k, "k"))
    S = np.array(st, dtype=float)
    if not np.all(np.isfinite(S)):
        raise ValueError("st must be finite.")
    return np.maximum(S - K, 0.0)

def european_put_payoff(st: np.ndarray, k: float) -> np.ndarray:
    K = float(_num(k, "k"))
    S = np.array(st, dtype=float)
    if not np.all(np.isfinite(S)):
        raise ValueError("st must be finite.")
    return np.maximum(K - S, 0.0)

def mc_price_european(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
    n_paths: int,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Monte Carlo price for a European call/put under GBM with SE.

    Returns discounted expected payoff and an estimated standard error.

    Raises:
        ValueError: invalid inputs
    """
    opt = str(option_type).lower().strip()
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    paths = simulate_gbm_paths(
        s0=s0, r=r, sigma=sigma, t=t, steps=steps, n_paths=n_paths,
        dividend_yield=dividend_yield, seed=seed, antithetic=True, moment_match=True
    )
    st = paths[:, -1]
    if opt == "call":
        payoff = european_call_payoff(st, k)
    else:
        payoff = european_put_payoff(st, k)

    rr = _num(r, "r")
    tt = _num(t, "t")
    disc = exp(-rr * tt)

    price = disc * float(np.mean(payoff))
    se = disc * float(np.std(payoff, ddof=1) / sqrt(payoff.size))
    return {"price": price, "se": se, "n_paths": float(payoff.size)}

def bs_call_price(s0: float, k: float, r: float, sigma: float, t: float, dividend_yield: float = 0.0) -> float:
    """Black–Scholes European call with continuous dividend yield q.

    Uses standard normal CDF from statistics.NormalDist (no scipy).
    """
    S0 = _num(s0, "s0")
    K = _num(k, "k")
    rr = _num(r, "r")
    sig = _num(sigma, "sigma")
    tt = _num(t, "t")
    qq = _num(dividend_yield, "dividend_yield")

    if S0 <= 0 or K <= 0:
        raise ValueError("s0 and k must be > 0.")
    if sig < 0:
        raise ValueError("sigma must be >= 0.")
    if tt <= 0:
        raise ValueError("t must be > 0.")
    if sig == 0:
        # deterministic forward
        fwd = S0 * exp((rr - qq) * tt)
        return float(exp(-rr * tt) * max(fwd - K, 0.0))

    d1 = (log(S0 / K) + (rr - qq + 0.5 * sig * sig) * tt) / (sig * sqrt(tt))
    d2 = d1 - sig * sqrt(tt)
    N = NormalDist()
    return float(S0 * exp(-qq * tt) * N.cdf(d1) - K * exp(-rr * tt) * N.cdf(d2))

def mc_with_control_variate_call(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
    n_paths: int,
    dividend_yield: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Monte Carlo call with a simple control variate: discounted ST.

    Control variate uses known E_Q[ST] = S0*exp((r-q)*t).
    This can reduce variance for many payoffs (especially near-the-money).
    """
    paths = simulate_gbm_paths(
        s0=s0, r=r, sigma=sigma, t=t, steps=steps, n_paths=n_paths,
        dividend_yield=dividend_yield, seed=seed, antithetic=True, moment_match=True
    )
    st = paths[:, -1]
    payoff = european_call_payoff(st, k)

    rr = _num(r, "r")
    tt = _num(t, "t")
    qq = _num(dividend_yield, "dividend_yield")
    disc = exp(-rr * tt)

    # Control: discounted ST
    y = disc * payoff
    x = disc * st
    x_mean_true = disc * (_num(s0, "s0") * exp((rr - qq) * tt))

    cov = float(np.cov(y, x, ddof=1)[0, 1])
    varx = float(np.var(x, ddof=1))
    if varx <= 0:
        raise ValueError("Control variate variance is non-positive; check inputs.")

    beta = cov / varx
    y_adj = y - beta * (x - x_mean_true)

    price = float(np.mean(y_adj))
    se = float(np.std(y_adj, ddof=1) / sqrt(y_adj.size))
    return {"price": price, "se": se, "beta": float(beta), "n_paths": float(y_adj.size)}

# Example usage: price a European call with MC and compare to Black–Scholes
params = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.25, "t": 1.0, "q": 0.01}
mc = mc_price_european(
    s0=params["s0"], k=params["k"], r=params["r"], sigma=params["sigma"], t=params["t"],
    steps=252, n_paths=50_000, option_type="call", dividend_yield=params["q"], seed=1
)
bs = bs_call_price(params["s0"], params["k"], params["r"], params["sigma"], params["t"], dividend_yield=params["q"])
mc_cv = mc_with_control_variate_call(
    s0=params["s0"], k=params["k"], r=params["r"], sigma=params["sigma"], t=params["t"],
    steps=252, n_paths=50_000, dividend_yield=params["q"], seed=1
)

print(f"MC call: {mc['price']:.4f} (SE {mc['se']:.4f})")
print(f"MC+CV:  {mc_cv['price']:.4f} (SE {mc_cv['se']:.4f})")
print(f"BS call: {bs:.4f}")
```

---

### Valuation Impact
Why this matters:
- Monte Carlo enables valuation of contingent claims with complex, path-dependent payoff logic (earn-outs, commodity-linked revenues, regulatory caps/floors), which directly affects enterprise value and equity bridges in transactions.
- It supports explicit tail risk analysis: price sensitivity to volatility, correlation, and barriers often dominates value in distressed or high-growth cases.

Impact on multiples:
- Earn-outs and contingent consideration can change implied EV/EBITDA and EV/Revenue. Without pricing them consistently, multiples across deals are not comparable.
- Volatility-rich businesses (biotech, commodities) can have option-like equity; simple multiples can mislead.

Impact on DCF inputs:
- Monte Carlo can replace “single-case” DCF with probability-weighted cash flows, but ensure consistency:
  - If simulating real-world cash flows, discount with risk-adjusted rates.
  - If pricing a claim, simulate risk-neutral and discount at risk-free.

Comparability issues across companies:
- Differences in hedging, contract structures, and volatility exposure make otherwise similar EBITDA profiles incomparable. Monte Carlo helps normalize by valuing those embedded options.

Practical adjustments:
```python
def probability_weighted_value(values: np.ndarray, probs: np.ndarray) -> float:
    """Generic probability-weighted expected value."""
    v = np.array(values, dtype=float).ravel()
    p = np.array(probs, dtype=float).ravel()
    if v.size != p.size or v.size < 1:
        raise ValueError("values and probs must have same non-zero length.")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(p))):
        raise ValueError("values and probs must be finite.")
    if np.any(p < 0):
        raise ValueError("probs must be >= 0.")
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("sum(probs) must be > 0.")
    p = p / s
    return float(np.sum(v * p))
```

---

### Quality of Earnings Flags
⚠️ Contingent consideration/earn-outs treated at face value (undiscounted) rather than fair value; understates liabilities or overstates “headline EV”.  
⚠️ Volatility/correlation inputs cherry-picked; no sensitivity analysis or convergence check.  
⚠️ Risk-neutral vs real-world measure mixed (e.g., drift set to expected return but discounting at risk-free).  
✅ Clear documentation of measure, inputs, convergence, and independent cross-check (e.g., Black–Scholes for a simplified payoff).

---

### Sector-Specific Considerations

| Sector | Typical Monte Carlo use | Key issue | Typical treatment |
|---|---|---|---|
| Commodities | price-linked contracts | fat tails, mean reversion | multi-factor models + stress |
| Biotech | milestone earn-outs | event timing uncertainty | scenario trees + Monte Carlo |
| Infrastructure | regulated caps/floors | path dependency | barrier/trigger logic |
| Financials | structured products | correlation/jumps | variance reduction + governance |

---

### Real-World Example
Scenario: Price a European call via Monte Carlo (with control variate) and compare to Black–Scholes as a cross-check.

```python
params = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.25, "t": 1.0, "q": 0.01}

mc = mc_price_european(
    s0=params["s0"], k=params["k"], r=params["r"], sigma=params["sigma"], t=params["t"],
    steps=252, n_paths=100_000, option_type="call", dividend_yield=params["q"], seed=7
)
mc_cv = mc_with_control_variate_call(
    s0=params["s0"], k=params["k"], r=params["r"], sigma=params["sigma"], t=params["t"],
    steps=252, n_paths=100_000, dividend_yield=params["q"], seed=7
)
bs = bs_call_price(params["s0"], params["k"], params["r"], params["sigma"], params["t"], dividend_yield=params["q"])

print(f"MC call: {mc['price']:.4f} (SE {mc['se']:.4f})")
print(f"MC+CV:  {mc_cv['price']:.4f} (SE {mc_cv['se']:.4f})")
print(f"BS call: {bs:.4f}")
```

Interpretation: If MC+CV matches Black–Scholes within a few standard errors, simulation logic is likely correct; you can then extend the same engine to path-dependent payoffs where closed-form pricing is unavailable.

See also: Chapter 21 (random numbers), Chapter 22 (Monte Carlo methods), Chapter 24 (Monte Carlo for investments), Chapter 17–19 (options and Greeks).
