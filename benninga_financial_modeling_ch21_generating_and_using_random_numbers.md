# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 21: Generating and Using Random Numbers (simulation inputs, reproducibility, risk controls)

### Core Concept
Random numbers are the foundation of Monte Carlo valuation (real options, VaR, risk-adjusted project valuation) because they convert probability assumptions into simulated outcomes. In practice, the key is not “randomness” but control: choosing the correct distribution, calibrating parameters, enforcing correlation, and ensuring reproducibility (seed governance). Poor random-number design leads to biased valuation results, unstable outputs, and audit challenges.

---

### Formula/Methodology

#### 1) Uniform random numbers (base source)
```text
U ~ Uniform(0, 1)
```
Where:
- U is a random draw between 0 and 1.

Typical uses:
- Transform U into other distributions (Normal, Lognormal, Bernoulli).
- Inverse-CDF sampling and stratified sampling.

#### 2) Normal random numbers (for returns, risk factors)
Standard normal:
```text
Z ~ Normal(0, 1)
```

Convert to Normal with mean μ and standard deviation σ:
```text
X = μ + σ · Z
```

Where:
- μ = expected value of X
- σ = standard deviation of X (same units as X)

#### 3) Lognormal random numbers (positive-only, multiplicative processes)
If:
```text
Y = exp(μ + σ · Z)
```
then Y is lognormally distributed and always positive.

Link to continuous compounding returns:
```text
S_T = S_0 · exp( (μ - 0.5σ^2)T + σ√T · Z )
```

Where:
- S0 = starting level (price, project value index)
- μ = drift (depends on real-world vs risk-neutral context)
- σ = volatility
- T = time horizon (years)

#### 4) Correlated normals (multiple risk factors)
Given:
- Z ~ N(0, I) (independent standard normals)
- Σ = covariance matrix
- L = Cholesky factor such that Σ = L · Lᵀ

Generate correlated normals:
```text
X = L · Z
```

Then X has covariance Σ.

---

### Practical Application (How to apply in valuation models)

#### A) Use cases in valuation
- Monte Carlo DCF: simulate revenue growth, margins, FX, commodity prices.
- Real options: simulate project value drivers and exercise rules.
- Credit valuation: default timing and recovery simulations.
- Risk reporting: VaR, CVaR, stress outcomes.

#### B) Governance and reproducibility (non-negotiable for auditability)
- Set a fixed seed for deterministic re-runs.
- Store seed, distribution choice, parameter assumptions, and calibration sources.
- Record simulation count (N) and convergence checks.

#### C) Distribution choice: decision rules
| Driver | Common distribution | Why | Typical pitfalls |
|---|---|---|---|
| Revenue growth | Normal / truncated normal | symmetric shocks | negative growth unrealistic without truncation |
| Prices (commodities/FX) | Lognormal / GBM | positive, multiplicative | wrong drift (real-world vs risk-neutral) |
| Default indicator | Bernoulli(p) | event occurrence | p inconsistent with horizon or rating |
| Time-to-default | Exponential/Weibull | hazard-based | mixing PD and hazard incorrectly |

#### D) Convergence checks (practical)
Monte Carlo outputs should stabilize as N increases. Track:
- Mean estimate
- Standard error (SE)
- Confidence interval (CI)

For an estimated mean ̅X:
```text
SE( X̄ ) = s / √N
CI_95% ≈ X̄ ± 1.96 · SE
```

Where:
- s = sample standard deviation of outcomes
- N = number of simulations

#### E) Use variance reduction when N is constrained (Excel/Python-in-Excel)
When runtime is limited:
- Antithetic variates: pair Z and -Z
- Stratification: use evenly-spaced U (quasi-random-lite)
- Common random numbers: same seed across scenarios to isolate changes

---

### Python Implementation
```python
from typing import Any, Dict, Tuple, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a NumPy random generator with optional seed for reproducibility."""
    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError("seed must be an int or None.")
        if seed < 0:
            raise ValueError("seed must be >= 0.")
    return np.random.default_rng(seed)

def draw_uniform(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Draw U~Uniform(0,1)."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int.")
    rng = make_rng(seed)
    return rng.random(n)

def draw_normal(n: int, mu: float = 0.0, sigma: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Draw X = mu + sigma*Z where Z~N(0,1)."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int.")
    m = _num(mu, "mu")
    s = _num(sigma, "sigma")
    if s <= 0:
        raise ValueError("sigma must be > 0.")
    rng = make_rng(seed)
    return m + s * rng.standard_normal(n)

def draw_lognormal(n: int, mu: float, sigma: float, seed: Optional[int] = None) -> np.ndarray:
    """Draw Y = exp(mu + sigma*Z) (lognormal)."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int.")
    m = _num(mu, "mu")
    s = _num(sigma, "sigma")
    if s <= 0:
        raise ValueError("sigma must be > 0.")
    rng = make_rng(seed)
    z = rng.standard_normal(n)
    return np.exp(m + s * z)

def cholesky_correlated_normals(n: int, cov: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Generate correlated N(0, cov) draws using Cholesky.

    Args:
        n: number of simulations
        cov: covariance matrix (k x k), symmetric positive definite

    Returns:
        array of shape (n, k)
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int.")
    if not isinstance(cov, np.ndarray):
        cov = np.array(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square (k x k) matrix.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("cov must be finite.")
    # Symmetry check (tolerant)
    if not np.allclose(cov, cov.T, atol=1e-10, rtol=1e-8):
        raise ValueError("cov must be symmetric.")
    # Cholesky will fail if not PD; catch and raise cleanly
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as e:
        raise ValueError("cov must be positive definite for Cholesky.") from e

    rng = make_rng(seed)
    k = cov.shape[0]
    z = rng.standard_normal((n, k))
    x = z @ L.T  # (n,k)
    return x

def mc_mean_ci(outcomes: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Compute mean, std, SE, and approx normal CI for Monte Carlo outcomes."""
    if outcomes is None:
        raise ValueError("outcomes is missing.")
    x = np.array(outcomes, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("Need at least 2 outcomes to compute CI.")
    if not np.all(np.isfinite(x)):
        raise ValueError("outcomes must be finite.")
    a = _num(alpha, "alpha")
    if not (0 < a < 1):
        raise ValueError("alpha must be between 0 and 1.")
    n = x.size
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(n)

    # 95% normal approx; for other alpha you'd use a z-quantile (kept simple for Excel/Python-in-Excel)
    z = 1.96 if abs(a - 0.05) < 1e-9 else 1.96
    lo = mean - z * se
    hi = mean + z * se
    return {"n": float(n), "mean": mean, "std": std, "se": float(se), "ci_lo": float(lo), "ci_hi": float(hi)}

def antithetic_normals(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate antithetic standard normal draws (variance reduction)."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int.")
    rng = make_rng(seed)
    half = (n + 1) // 2
    z = rng.standard_normal(half)
    z_full = np.concatenate([z, -z])[:n]
    return z_full

# Example usage
seed = 42
n = 100_000

# 1) Normal driver
rev_growth = draw_normal(n, mu=0.06, sigma=0.10, seed=seed)

# 2) Lognormal price index
price_index = draw_lognormal(n, mu=0.0, sigma=0.25, seed=seed + 1)

# 3) Correlated shocks (e.g., revenue growth and margin change)
cov = np.array([[1.0, 0.4],
                [0.4, 1.0]], dtype=float)
corr_shocks = cholesky_correlated_normals(n, cov, seed=seed + 2)

# 4) Convergence stats
stats = mc_mean_ci(rev_growth)
print({k: round(v, 6) for k, v in stats.items() if k != "n"})
```

---

### Valuation Impact
Why this matters:
- Random-number design determines whether simulation-based valuations are credible. The same DCF assumptions can produce materially different values depending on distribution choice, correlation handling, and exercise logic (for real options).
- Reproducibility is essential for audit and version control. A valuation that cannot be reproduced from documented inputs is effectively unusable in governance-heavy contexts.

Impact on multiples:
- Simulated distributions can help explain why a company trades at a premium/discount multiple (market pricing the upside convexity or downside tail risk).
- For cyclicals/commodities, lognormal price risk can produce asymmetric valuation outcomes not captured by point estimates.

Impact on DCF inputs:
- Simulation turns single-point forecasts into a distribution of FCF and value. It can inform:
  - downside-adjusted terminal value assumptions
  - probability-weighted scenarios
  - risk management overlays (e.g., haircut to growth when tail risk is large)

Comparability issues across companies:
- Two companies using different volatility and correlation calibrations will produce different “option-adjusted” values; ensure consistent methodology within a peer set.

Practical adjustments:
```python
def seed_registry(base_seed: int, scenario_id: int) -> int:
    """Deterministic seed assignment to make scenario comparisons consistent."""
    b = int(_num(base_seed, "base_seed"))
    s = int(_num(scenario_id, "scenario_id"))
    if b < 0 or s < 0:
        raise ValueError("Seeds/IDs must be >= 0.")
    return b + 10_000 * s
```

---

### Quality of Earnings Flags
⚠️ Simulation uses unconstrained normals for drivers that should be bounded (e.g., negative volumes, negative prices).  
⚠️ Correlations assumed with no evidence; covariance matrix not PD (forces ad-hoc fixes).  
⚠️ Different seeds used across scenarios, making comparisons noisy and misleading.  
✅ Documented seed policy, distribution rationale, correlation calibration, and convergence evidence (CI widths).

---

### Sector-Specific Considerations

| Sector | Key uncertain driver | Recommended distribution | Typical correlation to model |
|---|---|---|---|
| Resources | commodity price | lognormal/GBM | price vs volume (often negative), price vs costs (positive) |
| Retail/Consumer | demand growth | truncated normal | demand vs margin (often negative) |
| SaaS | retention / net revenue retention | beta/logit transform | growth vs margin (can be positive in scale-up) |
| Banking | defaults / recoveries | Bernoulli + beta | defaults vs GDP/rates |

---

### Real-World Example
Scenario: Build a Monte Carlo on a single driver (revenue growth) and compute a 95% CI for the mean.

```python
n = 50_000
seed = 123
g = draw_normal(n, mu=0.05, sigma=0.12, seed=seed)

# Suppose valuation is roughly linear in growth for a small range:
# Value ≈ BaseValue × (1 + sensitivity × (growth - base_growth))
base_value = 500_000_000
base_growth = 0.05
sensitivity = 4.0  # illustrative: 4x multiplier sensitivity to growth deviations

values = base_value * (1.0 + sensitivity * (g - base_growth))
summary = mc_mean_ci(values)

print(f"MC mean value: ${summary['mean']/1e6:.1f}M")
print(f"95% CI:        ${summary['ci_lo']/1e6:.1f}M to ${summary['ci_hi']/1e6:.1f}M")
```

Interpretation: If CI width is large, your valuation is highly sensitive to the driver and you should either increase N, apply variance reduction, or tighten driver calibration.

See also: Chapter 22 (Monte Carlo methods), Chapter 24 (Monte Carlo for investments), Chapter 25 (VaR).
