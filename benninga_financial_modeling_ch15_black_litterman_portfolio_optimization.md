# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 15: The Black-Litterman Approach to Portfolio Optimization (implied returns, views, confidence)

### Core Concept
Black–Litterman (BL) produces more stable, realistic expected returns for portfolio optimization by combining (i) equilibrium “implied” returns from market weights with (ii) investor views, weighted by confidence. In valuation and capital markets work, BL is useful for building defensible cost of capital and scenario narratives (e.g., forward-looking risk premia and factor tilts) when historical mean returns are too noisy. The key value is governance: BL forces you to document assumptions about expected returns and confidence rather than letting unstable sample means drive decisions.

### Formula/Methodology

#### 1) Equilibrium (implied) excess returns from market weights
```text
π = δ × Σ × w_mkt
```

Where:
- π = vector of implied excess returns (Nx1)
- δ = risk aversion (scalar)
- Σ = covariance matrix of asset excess returns (NxN)
- w_mkt = market-cap weights (Nx1)

Practical note:
- π is “what the market must be pricing” given Σ, w_mkt, and δ.

#### 2) Views (linear constraints)
```text
P × μ = Q
```

Where:
- μ = (unknown) expected excess return vector (Nx1)
- P = view matrix (KxN), each row maps assets to a view
- Q = view returns (Kx1), the expected excess return in each view

Examples:
- “Asset A will outperform Asset B by 2%”: row has +1 for A, -1 for B, Q = 0.02
- “Asset C expected return is 5%”: row has 1 for C, Q = 0.05

#### 3) View uncertainty (confidence)
```text
Ω = diag(ω1, ω2, ..., ωK)
```

Where:
- Ω = KxK covariance of view errors (smaller ω = higher confidence)

#### 4) Posterior (combined) expected returns
One common BL posterior mean form:
```text
μ_BL = [ (τΣ)^(-1) + P' Ω^(-1) P ]^(-1) × [ (τΣ)^(-1) π + P' Ω^(-1) Q ]
```

Where:
- τ = scalar that scales uncertainty in the prior (typical small value, e.g., 0.025)

#### 5) Posterior covariance (optional)
```text
Σ_BL = Σ + [ (τΣ)^(-1) + P' Ω^(-1) P ]^(-1)
```

---

### Practical Application (How to apply BL in real finance workflows)

#### Step 1: Build a robust covariance matrix Σ
- Use aligned return series (same dates, same currency).
- Annualize consistently.
- Stabilize Σ if needed (shrinkage). Unstable Σ causes unstable π and unstable optimization.

#### Step 2: Choose market weights w_mkt
- Use market-cap weights for public assets.
- For private/institutional mixes, use policy weights as the “market proxy”.

#### Step 3: Calibrate risk aversion δ
Two common practical choices:
1) Match observed market Sharpe ratio:
```text
δ ≈ (E[Rm] - Rf) / Var(Rm)
```
2) Choose δ to make implied returns “reasonable” and consistent with governance (documented).

#### Step 4: Define views P and Q (keep them few and explicit)
Good views:
- Sector tilt (e.g., cyclicals outperform defensives by X).
- Rate regime views (e.g., duration assets underperform by X if yields rise).
Bad views:
- Many micro-views with low confidence (overfitting).

#### Step 5: Set confidence (Ω) in a transparent way
Approaches:
- Absolute: set ωk = (σ_view)^2 where σ_view reflects uncertainty in that view.
- Relative: scale ωk so “high confidence” views dominate prior, low confidence barely move μ.

Rule: Start conservative; stress test outcomes.

#### Step 6: Use μ_BL in optimization or scenario valuation
- Portfolio optimization: replace historical mean returns with μ_BL.
- Valuation: translate macro/sector views into forward-looking risk premia assumptions that influence discount rates and relative valuation narratives.

---

### Python Implementation
```python
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def _to_colvec(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"{name} must be a column vector (N x 1).")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values.")
    return x

def _to_mat(x: np.ndarray, shape: Optional[Tuple[int,int]], name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if shape is not None and x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values.")
    return x

def implied_equilibrium_returns(Sigma: np.ndarray, w_mkt: np.ndarray, delta: float) -> np.ndarray:
    """Compute implied equilibrium excess returns: pi = delta * Sigma * w_mkt."""
    delta = _num(delta, "delta")
    S = _to_mat(Sigma, None, "Sigma")
    n = S.shape[0]
    if S.shape[1] != n:
        raise ValueError("Sigma must be square (N x N).")
    w = _to_colvec(w_mkt, "w_mkt")
    if w.shape[0] != n:
        raise ValueError("w_mkt length must match Sigma dimension.")
    return delta * (S @ w)

def black_litterman_posterior_mean(
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    delta: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.025,
) -> Dict[str, np.ndarray]:
    """Compute Black-Litterman posterior mean returns.

    Args:
        Sigma: NxN covariance matrix (annualized excess returns)
        w_mkt: Nx1 market weights
        delta: risk aversion
        P: KxN view matrix
        Q: Kx1 view returns
        Omega: KxK diagonal (or PSD) view error covariance
        tau: scalar prior uncertainty scale

    Returns:
        dict with pi (prior), mu_bl (posterior), M (posterior adjustment matrix)

    Raises:
        ValueError: invalid shapes or non-invertible matrices.
    """
    tau = _num(tau, "tau")
    if tau <= 0:
        raise ValueError("tau must be > 0.")

    S = _to_mat(Sigma, None, "Sigma")
    n = S.shape[0]
    if S.shape[1] != n:
        raise ValueError("Sigma must be square.")
    w = _to_colvec(w_mkt, "w_mkt")
    if w.shape[0] != n:
        raise ValueError("w_mkt length must match Sigma dimension.")

    Pm = _to_mat(P, None, "P")
    k = Pm.shape[0]
    if Pm.shape[1] != n:
        raise ValueError("P must have shape (K x N).")

    q = _to_colvec(Q, "Q")
    if q.shape[0] != k:
        raise ValueError("Q length must equal number of views K.")

    Om = _to_mat(Omega, None, "Omega")
    if Om.shape != (k, k):
        raise ValueError("Omega must have shape (K x K).")

    # Prior implied returns
    pi = implied_equilibrium_returns(S, w, delta)

    # Compute posterior mean:
    # mu = inv(inv(tau*Sigma) + P' inv(Omega) P) * (inv(tau*Sigma)*pi + P' inv(Omega) Q)
    try:
        inv_tauS = np.linalg.inv(tau * S)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot invert tau*Sigma (check Sigma PSD/conditioning): {e}")

    try:
        inv_Om = np.linalg.inv(Om)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot invert Omega (check view covariance): {e}")

    A = inv_tauS + (Pm.T @ inv_Om @ Pm)
    b = (inv_tauS @ pi) + (Pm.T @ inv_Om @ q)

    try:
        mu_bl = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot solve for posterior mean (check A conditioning): {e}")

    return {"pi": pi, "mu_bl": mu_bl, "A": A}

def optimize_mean_variance(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Simple unconstrained mean-variance optimal weights: w* = (1/delta) * inv(Sigma) * mu.

    Note: In production, you usually need constraints (long-only, tracking error, etc.).
    """
    d = _num(risk_aversion, "risk_aversion")
    if d <= 0:
        raise ValueError("risk_aversion must be > 0.")
    S = _to_mat(Sigma, None, "Sigma")
    n = S.shape[0]
    if S.shape[1] != n:
        raise ValueError("Sigma must be square.")
    m = _to_colvec(mu, "mu")
    if m.shape[0] != n:
        raise ValueError("mu length must match Sigma dimension.")
    try:
        invS_mu = np.linalg.solve(S, m)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot solve Sigma\mu (check Sigma PSD/conditioning): {e}")
    w = (1.0 / d) * invS_mu
    return w

# Example usage (toy 3-asset case)
Sigma = np.array([
    [0.040, 0.010, 0.008],
    [0.010, 0.090, 0.015],
    [0.008, 0.015, 0.060],
], dtype=float)

w_mkt = np.array([0.50, 0.30, 0.20], dtype=float)  # market weights
delta = 2.5  # risk aversion
tau = 0.025

# Two views:
# View 1: Asset 1 will outperform Asset 2 by 2%
# View 2: Asset 3 expected excess return is 4%
P = np.array([
    [1.0, -1.0, 0.0],
    [0.0,  0.0, 1.0],
], dtype=float)

Q = np.array([0.02, 0.04], dtype=float)

# Confidence: smaller omega -> higher confidence
Omega = np.diag([0.0004, 0.0016])  # (2%)^2 and (4%)^2

res = black_litterman_posterior_mean(Sigma, w_mkt, delta, P, Q, Omega, tau=tau)
pi = res["pi"].flatten()
mu_bl = res["mu_bl"].flatten()

print("Implied prior returns (pi):", np.round(pi, 4))
print("Posterior returns (mu_bl):", np.round(mu_bl, 4))

w_star = optimize_mean_variance(mu_bl, Sigma, risk_aversion=delta).flatten()
print("Unconstrained MV weights (not normalized):", np.round(w_star, 3))
```

---

### Valuation Impact
Why this matters:
- Historical average returns are extremely noisy; BL provides a structured, auditable way to build forward-looking expected returns and risk premia.
- Forward-looking risk premia can inform cost of capital narratives (e.g., why ERP or sector premia are higher/lower under a macro view).

Impact on multiples:
- If BL-implied expected returns increase (higher required returns), the market-clearing discount rate rises → lower justified P/E and EV/EBITDA.
- If sector views imply lower risk premia (higher confidence), multiples may be expected to expand (all else equal).

Impact on DCF inputs:
- CAPM components (ERP and beta inputs) and “equity risk premium by region/sector” narratives can be framed as BL views.
- For portfolio-held assets or conglomerates, BL helps create consistent, scenario-based discount rates across segments.

Comparability issues across companies:
- Different assumptions about δ, τ, and Ω materially change outputs; document them to keep comparability.
- Covariance regimes change through cycles; BL outputs must be stress-tested under alternative Σ.

Practical adjustments:
```python
def implied_risk_aversion_from_market(market_excess_return: float, market_variance: float) -> float:
    """delta ≈ (E[Rm]-Rf) / Var(Rm)."""
    mr = _num(market_excess_return, "market_excess_return")
    mv = _num(market_variance, "market_variance")
    if mv <= 0:
        raise ValueError("market_variance must be > 0.")
    return mr / mv
```

---

### Quality of Earnings Flags
⚠️ Treating historical average returns as “expected returns” without shrinkage/governance (optimizers become unstable).  
⚠️ Using a covariance matrix built from misaligned data or without PSD checks (can cause impossible results).  
⚠️ Overloading the model with many low-quality views (overfitting; false precision).  
✅ Few, explicit views with documented confidence, plus sensitivity to Ω and τ.

---

### Sector-Specific Considerations

| Sector | Typical view type | Key issue | Typical handling |
|---|---|---|---|
| Financials | credit-cycle / spread views | regime shifts dominate | use scenario-specific Ω (lower confidence) |
| Energy | commodity-price regime views | covariance changes sharply | stress-test Σ under high-volatility regimes |
| Tech | growth premium / duration-like behavior | correlations rise in risk-off | include risk-off scenario with higher Ω |
| Real estate | rate sensitivity views | leverage amplifies | link views to rate scenarios and debt costs |

---

### Real-World Example
Scenario: You manage valuation inputs for a multi-asset portfolio and need defendable forward-looking risk premia for scenario DCF work.

Approach:
1) Build Σ from 5-year weekly returns (and apply shrinkage if needed).  
2) Use market weights as baseline.  
3) Encode 1–3 macro views (rates up, credit spreads widen, defensives outperform).  
4) Set Ω to reflect confidence (low confidence → small movement from π).  
5) Use μ_BL to update expected returns and narrate how required returns shift.

```python
# Example: translate a view into a simple “required return shift” narrative
# If mu_bl for an equity basket rises by +1%, discount rates in DCF scenarios may rise similarly (subject to beta and ERP mapping).
```

Interpretation: BL doesn’t “predict” returns; it provides a disciplined way to blend equilibrium priors with explicit, confidence-weighted views—useful for defensible scenario valuation ranges.

See also: Chapter 12 (covariance matrix), Chapter 11 (efficient frontier), Chapter 13 (beta/SML) for how Σ and β connect to discount rates and portfolio risk.
