# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 24: Monte Carlo Simulations for Investments (return paths, terminal wealth, risk metrics)

### Core Concept
Monte Carlo for investments simulates portfolio return paths to quantify the distribution of terminal wealth, drawdowns, and risk measures (VaR/CVaR). For valuation work, the same engine supports scenario-weighted discounting, pension/asset-liability valuation, and stress-testing equity value where financing, liquidity, or funding constraints depend on market outcomes.

---

### Formula/Methodology

#### 1) Single-period return and terminal wealth
If portfolio value evolves with simple returns R_t:
```text
V_T = V_0 · Π_{t=1..T} (1 + R_t)
```

Where:
- V0 = starting value
- Rt = portfolio return in period t (decimal)
- VT = terminal value after T periods

With log returns r_t = ln(1 + R_t):
```text
ln(V_T / V_0) = Σ_{t=1..T} r_t
```

#### 2) Portfolio return from asset returns
For weights w and asset returns vector R:
```text
R_p = wᵀ · R
```
Where:
- w sums to 1 (fully invested) unless modeling cash/leverage
- R is vector of asset returns for a period

#### 3) Multivariate normal model for returns
Assume asset returns:
```text
R ~ Normal(μ, Σ)
```
Generate correlated draws using Cholesky factor L where Σ = L·Lᵀ:
```text
R = μ + L · Z
```
Where:
- Z ~ Normal(0, I)

#### 4) Value at Risk (VaR) and Conditional VaR (CVaR)
For loss L = -(V_T - V_0) (or negative P&L), at confidence level q (e.g., 95%):
```text
VaR_q = quantile(L, q)
```

CVaR (expected shortfall) beyond VaR:
```text
CVaR_q = E[L | L ≥ VaR_q]
```

#### 5) Path-dependent risk: maximum drawdown
For path values V_t:
```text
Drawdown_t = 1 - V_t / max_{s≤t}(V_s)
MaxDrawdown = max_t(Drawdown_t)
```

---

### Practical Application (How to apply in investment and valuation contexts)

#### A) When Monte Carlo adds value
| Problem | Why simulation helps | Typical output |
|---|---|---|
| Portfolio/wealth projections | nonlinear compounding & tails | terminal wealth distribution |
| Funding/solvency constraints | probability of breaching covenants | P(breach), worst-case outcomes |
| Discount rate stress | market-linked financing cost | scenario-based discount factors |
| Real options timing | exercise depends on price paths | probability of exercise, option value |
| Pension/ALM valuation | correlation of assets & liabilities | funding ratio distribution |

#### B) Model building checklist (practical)
1) Choose horizon and step:
- monthly or annual; keep consistent with parameter estimation

2) Calibrate μ and Σ:
- use historical estimates, forward-looking overlays, or implied (document choice)
- ensure Σ is positive definite

3) Enforce realistic constraints:
- weights sum to 1 (unless leverage modeled)
- no negative wealth (floor at 0) for limited liability, if appropriate

4) Validate:
- backtest summary stats (mean, vol, correlation)
- sensitivity to μ (often dominates terminal wealth)
- convergence of VaR and tail metrics (requires higher N)

#### C) Practical pitfalls
- Using arithmetic mean returns for long horizons can overstate terminal wealth; consider log returns for compounding consistency.
- Assuming normality underestimates fat tails; consider t-distribution or scenario mixtures if needed (at minimum, stress-test).
- Ignoring rebalancing and cash flows misstates risk; add rules explicitly.

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

def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be an int >= 0 or None.")
    return np.random.default_rng(seed)

def cholesky_factor(cov: np.ndarray) -> np.ndarray:
    c = np.array(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square (k x k).")
    if not np.all(np.isfinite(c)):
        raise ValueError("cov must be finite.")
    if not np.allclose(c, c.T, atol=1e-10, rtol=1e-8):
        raise ValueError("cov must be symmetric.")
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError as e:
        raise ValueError("cov must be positive definite (Cholesky failed).") from e

def simulate_asset_returns(
    n_sims: int,
    n_steps: int,
    mu: np.ndarray,
    cov: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Simulate multivariate normal asset returns.

    Returns:
        returns: array shape (n_sims, n_steps, k)
    """
    if not isinstance(n_sims, int) or n_sims <= 0:
        raise ValueError("n_sims must be a positive int.")
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive int.")
    m = np.array(mu, dtype=float).ravel()
    if m.size == 0:
        raise ValueError("mu must not be empty.")
    if not np.all(np.isfinite(m)):
        raise ValueError("mu must be finite.")
    L = cholesky_factor(cov)
    k = m.size
    if L.shape != (k, k):
        raise ValueError("cov dimensions must match mu length.")

    rng = make_rng(seed)
    z = rng.standard_normal((n_sims, n_steps, k))
    # correlate: (..,k) @ L.T -> (..,k)
    corr = z @ L.T
    r = m + corr
    return r

def portfolio_returns(asset_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute portfolio return per step: R_p = w^T R."""
    R = np.array(asset_returns, dtype=float)
    if R.ndim != 3:
        raise ValueError("asset_returns must be (n_sims, n_steps, k).")
    w = np.array(weights, dtype=float).ravel()
    if w.size != R.shape[2]:
        raise ValueError("weights length must match number of assets (k).")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    s = float(np.sum(w))
    if abs(s - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0 (fully invested).")
    return np.einsum("stk,k->st", R, w)

def simulate_wealth_paths(
    R_p: np.ndarray,
    v0: float = 1.0,
    rebalance: bool = False,
) -> np.ndarray:
    """Simulate wealth paths from portfolio returns.

    Args:
        R_p: (n_sims, n_steps) returns
        v0: starting value
        rebalance: placeholder; for constant-weight portfolios, R_p already reflects rebalancing at step frequency

    Returns:
        wealth: (n_sims, n_steps+1) with wealth[:,0]=v0
    """
    V0 = _num(v0, "v0")
    if V0 <= 0:
        raise ValueError("v0 must be > 0.")
    rp = np.array(R_p, dtype=float)
    if rp.ndim != 2:
        raise ValueError("R_p must be (n_sims, n_steps).")
    if not np.all(np.isfinite(rp)):
        raise ValueError("R_p must be finite.")
    n_sims, n_steps = rp.shape
    wealth = np.empty((n_sims, n_steps + 1), dtype=float)
    wealth[:, 0] = V0
    for t in range(1, n_steps + 1):
        wealth[:, t] = wealth[:, t - 1] * (1.0 + rp[:, t - 1])
    return wealth

def max_drawdown(path: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each simulation path."""
    V = np.array(path, dtype=float)
    if V.ndim != 2:
        raise ValueError("path must be (n_sims, n_steps+1).")
    if not np.all(np.isfinite(V)):
        raise ValueError("path must be finite.")
    running_max = np.maximum.accumulate(V, axis=1)
    dd = 1.0 - (V / running_max)
    return np.max(dd, axis=1)

def var_cvar(losses: np.ndarray, q: float = 0.95) -> Dict[str, float]:
    """Compute VaR and CVaR for losses at confidence q."""
    L = np.array(losses, dtype=float).ravel()
    if L.size < 2:
        raise ValueError("Need at least 2 loss observations.")
    if not np.all(np.isfinite(L)):
        raise ValueError("losses must be finite.")
    qq = _num(q, "q")
    if not (0 < qq < 1):
        raise ValueError("q must be between 0 and 1.")
    var = float(np.quantile(L, qq))
    tail = L[L >= var]
    cvar = float(np.mean(tail)) if tail.size else var
    return {"VaR": var, "CVaR": cvar}

def summarize_terminal_wealth(wealth: np.ndarray) -> Dict[str, float]:
    V = np.array(wealth, dtype=float)
    if V.ndim != 2:
        raise ValueError("wealth must be (n_sims, n_steps+1).")
    terminal = V[:, -1]
    if terminal.size < 2:
        raise ValueError("Need at least 2 simulations.")
    return {
        "mean": float(np.mean(terminal)),
        "median": float(np.median(terminal)),
        "p10": float(np.percentile(terminal, 10)),
        "p90": float(np.percentile(terminal, 90)),
        "prob_below_v0": float(np.mean(terminal < V[:, 0])),
    }

# Example usage
seed = 7
n_sims = 20000
n_steps = 12  # monthly for 1 year

# Two-asset example: equity and bonds (illustrative parameters per step)
mu = np.array([0.006, 0.002])  # mean monthly returns
cov = np.array([[0.04**2, 0.04*0.01*0.2],
                [0.04*0.01*0.2, 0.01**2]])  # monthly covariance matrix
w = np.array([0.60, 0.40])

asset_r = simulate_asset_returns(n_sims, n_steps, mu, cov, seed=seed)
rp = portfolio_returns(asset_r, w)
wealth = simulate_wealth_paths(rp, v0=1.0)

terminal_summary = summarize_terminal_wealth(wealth)
mdd = max_drawdown(wealth)

# Loss = 1 - terminal wealth (for initial wealth 1.0)
loss = 1.0 - wealth[:, -1]
risk = var_cvar(loss, q=0.95)

print({k: round(v, 4) for k, v in terminal_summary.items()})
print(f"Mean Max Drawdown: {float(np.mean(mdd)):.2%}")
print(f"VaR95 (loss): {risk['VaR']:.4f}  |  CVaR95 (loss): {risk['CVaR']:.4f}")
```

---

### Valuation Impact
Why this matters:
- Investment-path simulation supports valuation where equity value depends on capital-market conditions (ability to refinance, cost of capital, liquidity of holdings, covenant headroom).
- For pension-backed or investment-heavy businesses (insurers, asset managers, PE funds, holding companies), scenario distributions can be required to evaluate solvency, dividend capacity, and intrinsic value under risk.

Impact on multiples:
- Higher downside risk (VaR/CVaR, drawdowns) can justify lower P/E or EV/EBITDA multiples due to higher equity risk and constrained distributions.
- For financials, market often prices tail risk more than mean outcomes; simulation metrics help explain valuation gaps.

Impact on DCF inputs:
- If funding/discount rates are state-dependent, you can link WACC or discount factors to simulated market states (rates/spreads), rather than holding them fixed.
- Simulation provides probability-weighted scenarios for terminal value assumptions (e.g., downturn regime vs normal regime).

Comparability issues across companies:
- Differences in asset mix, leverage, and liquidity produce different tail risk even with similar mean returns; adjust peer comparisons for risk exposure.

Practical adjustments:
```python
def probability_of_covenant_breach(wealth: np.ndarray, threshold: float) -> float:
    """Example: proxy covenant breach as terminal wealth below threshold."""
    V = np.array(wealth, dtype=float)
    if V.ndim != 2:
        raise ValueError("wealth must be 2D.")
    th = float(_num(threshold, "threshold"))
    return float(np.mean(V[:, -1] < th))
```

---

### Quality of Earnings Flags
⚠️ Return distributions assume normality with thin tails (understates downside risk).  
⚠️ μ calibrated from short history or boom period without regime checks.  
⚠️ Correlation ignored or unstable (risk clustering missed).  
✅ Parameter documentation includes time window, frequency, and stress overlays; tail metrics (VaR/CVaR, drawdown) reported.

---

### Sector-Specific Considerations

| Sector | Investment simulation relevance | Key metric to monitor | Typical adjustment |
|---|---|---|---|
| Insurers | solvency and asset-liability | funding ratio distribution | link discount curve to rates |
| Asset managers | fee base depends on AUM paths | P(AUM drawdown) | scenario-based fee margin |
| Holding companies | NAV driven by portfolio | NAV P10/P90 | apply liquidity haircuts |
| Banks | market risk & capital | VaR/CVaR and stress loss | integrate with capital buffers |

---

### Real-World Example
Scenario: Report terminal wealth distribution and VaR/CVaR for a 60/40 portfolio.

```python
seed = 99
asset_r = simulate_asset_returns(
    n_sims=30000,
    n_steps=24,
    mu=np.array([0.005, 0.002]),
    cov=np.array([[0.045**2, 0.045*0.012*0.15],
                  [0.045*0.012*0.15, 0.012**2]]),
    seed=seed,
)
rp = portfolio_returns(asset_r, np.array([0.6, 0.4]))
wealth = simulate_wealth_paths(rp, v0=1.0)

loss = 1.0 - wealth[:, -1]
risk = var_cvar(loss, q=0.95)

print(f"Terminal wealth p10/p90: {np.percentile(wealth[:,-1],10):.3f} / {np.percentile(wealth[:,-1],90):.3f}")
print(f"VaR95 loss: {risk['VaR']:.3f} | CVaR95 loss: {risk['CVaR']:.3f}")
```

Interpretation: If CVaR is materially worse than VaR, tails dominate risk; incorporate this when assessing dividend capacity, leverage tolerance, or discount-rate conservatism.

See also: Chapter 21 (random numbers), Chapter 22 (Monte Carlo foundations), Chapter 25 (VaR).
