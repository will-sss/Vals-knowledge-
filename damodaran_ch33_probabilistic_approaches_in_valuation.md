# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 33: Probabilistic Approaches in Valuation (Scenarios, Decision Trees, Monte Carlo, Simulation-Ready DCF)

### Core Concept
Probabilistic valuation replaces single-point assumptions with **distributions and discrete outcomes**, making uncertainty explicit and auditable. Practically, you use probabilistic tools when (a) outcomes are **binary or path-dependent** (regulatory approval, product launch, litigation, distress) or (b) key inputs are **uncertain and interacting** (growth, margins, reinvestment, discount rate). The output is a **distribution of values** (not just one value) and clear identification of the drivers of downside and upside.

See also: Chapter 12 (terminal value), Chapter 22 (money-losing firms), Chapter 23 (start-ups), Chapter 30 (distress), Chapter 25 (M&A earn-outs).

---

### Formula/Methodology

#### 1) Scenario analysis (discrete outcomes)
Scenario analysis uses a small set of internally consistent futures:
```text
Expected Value = Σ [ p_s × Value_s ]

Where:
p_s = probability of scenario s (sum to 1)
Value_s = valuation under scenario s
```
Best when:
- the number of plausible states is small (3–6),
- scenarios are qualitatively distinct (success / base / failure),
- you can tie each scenario to specific driver sets (growth, margin, reinvestment, risk).

#### 2) Decision trees (path-dependent outcomes)
Decision trees model outcomes that unfold in stages, where early events affect later options:
```text
Node Value = (p × PV(Up branch)) + ((1 − p) × PV(Down branch))

Decision nodes:
choose the branch with the higher value (max operator)
```
Best when:
- the firm has real options (delay, expand, abandon),
- the payoff depends on sequential uncertainty (R&D phases, permits, trials).

#### 3) Monte Carlo simulation (continuous uncertainty)
Simulation samples uncertain inputs from distributions and recomputes valuation many times:
```text
For i in 1..N:
  sample inputs θ_i from distributions
  compute value_i = f(θ_i)
Output: distribution of {value_i}
```
Best when:
- multiple inputs are uncertain simultaneously,
- you need a value range and tail risk,
- there is interaction (e.g., growth affects margins and reinvestment).

#### 4) DCF simulation-ready design (avoid fragile models)
Make the valuation function:
- deterministic given inputs (no hidden state),
- input-validated (guardrails),
- modular (FCFF builder, terminal, PV),
- returns both value and intermediate drivers for diagnostics.

#### 5) Choosing distributions (practical defaults)
Distributions should reflect bounds and skew:
- Growth rates: bounded; often use triangular or truncated normal.
- Margins: bounded between (−1, 1); use triangular or truncated normal.
- Discount rates: bounded > −100%; use normal with guardrails or lognormal proxy.
- Multiples: positive; use lognormal or triangular.

If uncertain about distribution form, use **triangular** (min, mode, max) for transparency.

#### 6) Correlation (do not assume independence by default)
Common real-world correlations:
- Higher growth ↔ higher reinvestment needs (positive)
- Higher growth ↔ lower margins short-term (negative, for scale-up costs)
- Higher risk ↔ lower multiples (negative)
Simulation should allow correlated sampling when material.

#### 7) Reporting outputs (what decision-makers need)
Provide:
- mean, median, p10/p90 (or p5/p95),
- probability value exceeds current price (if relevant),
- sensitivity ranking (which input drives variance),
- scenario narrative mapping (why downside happens).

---

### Python Implementation
```python
from typing import Dict, List, Tuple, Optional, Callable
import math
import numpy as np


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def pv_cash_flows(cash_flows: List[float], r: float) -> float:
    """PV of cash flows discounted at r."""
    if not _num(r) or r <= -1:
        raise ValueError("r must be numeric and > -100%")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values")
        pv += cf / ((1.0 + r) ** t)
    return pv


def terminal_value_perpetuity(cf_next: float, r: float, g: float) -> float:
    """TV = CF_{n+1} / (r - g)."""
    if not all(_num(v) for v in [cf_next, r, g]):
        raise ValueError("inputs must be numeric")
    if r <= g:
        raise ValueError("Need r > g for finite TV")
    return cf_next / (r - g)


def scenario_expected_value(scenarios: List[Tuple[float, float]]) -> float:
    """Expected value from (probability, value) pairs."""
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        raise ValueError("scenarios must be a non-empty list")
    total_p = 0.0
    ev = 0.0
    for p, v in scenarios:
        if not _num(p) or not _num(v):
            raise ValueError("probabilities and values must be numeric")
        if not (0.0 <= p <= 1.0):
            raise ValueError("probabilities must be in [0,1]")
        total_p += p
        ev += p * v
    if abs(total_p - 1.0) > 1e-3:
        raise ValueError("probabilities should sum to 1 (within tolerance)")
    return ev


def triangular(n: int, low: float, mode: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Triangular distribution sampling with validation."""
    if n <= 0:
        raise ValueError("n must be > 0")
    for name, v in {"low": low, "mode": mode, "high": high}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if not (low <= mode <= high):
        raise ValueError("Require low <= mode <= high for triangular")
    return rng.triangular(left=low, mode=mode, right=high, size=n)


def clipped_normal(n: int, mean: float, sd: float, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Normal sampling clipped to [low, high] with validation."""
    if n <= 0:
        raise ValueError("n must be > 0")
    for name, v in {"mean": mean, "sd": sd, "low": low, "high": high}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if sd <= 0:
        raise ValueError("sd must be > 0")
    if low >= high:
        raise ValueError("low must be < high")
    x = rng.normal(loc=mean, scale=sd, size=n)
    return np.clip(x, low, high)


def dcf_value_from_inputs(
    revenue0: float,
    margin: float,
    tax_rate: float,
    sales_to_capital: float,
    g: float,
    years: int,
    r: float,
    g_terminal: float
) -> float:
    """Simple unlevered DCF from a compact driver set (simulation-ready)."""
    for name, v in {
        "revenue0": revenue0, "margin": margin, "tax_rate": tax_rate,
        "sales_to_capital": sales_to_capital, "g": g, "years": years, "r": r, "g_terminal": g_terminal
    }.items():
        if name != "years" and not _num(v):
            raise ValueError(f"{name} must be numeric")
    if years <= 0:
        raise ValueError("years must be > 0")
    if revenue0 < 0:
        raise ValueError("revenue0 must be >= 0")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if sales_to_capital <= 0:
        raise ValueError("sales_to_capital must be > 0")
    if r <= -1:
        raise ValueError("r must be > -100%")
    if r <= g_terminal:
        raise ValueError("Need r > g_terminal for finite terminal value")

    # Build revenues
    revenues = []
    rev = revenue0
    for _ in range(years):
        rev = rev * (1.0 + g)
        revenues.append(rev)

    # Reinvestment closure
    reinvest = []
    prev = revenue0
    for rev in revenues:
        reinvest.append((rev - prev) / sales_to_capital)
        prev = rev

    # FCFF approximation: EBIT*(1-tax) - reinvest (ignoring D&A/Capex split for compactness)
    fcff = []
    for rev, reinv in zip(revenues, reinvest):
        ebit = rev * margin
        fcff.append(ebit * (1.0 - tax_rate) - reinv)

    pv_stage = pv_cash_flows(fcff, r)

    # Terminal based on year+1 FCFF using terminal growth on revenue and same margin & S/C
    rev_T = revenues[-1]
    rev_T1 = rev_T * (1.0 + g_terminal)
    reinv_T1 = (rev_T1 - rev_T) / sales_to_capital
    ebit_T1 = rev_T1 * margin
    fcff_T1 = ebit_T1 * (1.0 - tax_rate) - reinv_T1

    tv = terminal_value_perpetuity(fcff_T1, r, g_terminal)
    pv_tv = tv / ((1.0 + r) ** years)

    return pv_stage + pv_tv


def monte_carlo_valuation(
    n: int,
    seed: int,
    revenue0: float,
    years: int,
    tax_rate: float,
    g_params: Tuple[float, float, float],          # low, mode, high
    margin_params: Tuple[float, float, float],     # low, mode, high
    r_params: Tuple[float, float, float],          # low, mode, high
    sales_to_capital_params: Tuple[float, float, float],
    g_terminal: float,
    corr: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Monte Carlo DCF with triangular inputs; optional correlation via Gaussian copula."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if years <= 0:
        raise ValueError("years must be > 0")
    if not _num(revenue0) or revenue0 < 0:
        raise ValueError("revenue0 must be numeric and >= 0")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if not _num(g_terminal):
        raise ValueError("g_terminal must be numeric")

    rng = np.random.default_rng(seed)

    # Sample inputs
    g = triangular(n, *g_params, rng=rng)
    m = triangular(n, *margin_params, rng=rng)
    r = triangular(n, *r_params, rng=rng)
    stc = triangular(n, *sales_to_capital_params, rng=rng)

    # Optional correlation (Gaussian copula on ranks)
    if corr is not None:
        corr = np.asarray(corr, dtype=float)
        if corr.shape != (4, 4):
            raise ValueError("corr must be 4x4 for [g, margin, r, stc]")
        # Generate correlated normals then map to uniforms via CDF, then to triangular via inverse CDF
        z = rng.multivariate_normal(mean=np.zeros(4), cov=corr, size=n)
        u = 0.5 * (1.0 + np.erf(z / math.sqrt(2.0)))  # normal CDF approximation
        # Convert uniforms to triangular using numpy's percent-point function logic
        def inv_tri(u_, low, mode, high):
            c = (mode - low) / (high - low) if high > low else 0.5
            out = np.empty_like(u_)
            left = u_ < c
            out[left] = low + np.sqrt(u_[left] * (high - low) * (mode - low))
            out[~left] = high - np.sqrt((1.0 - u_[~left]) * (high - low) * (high - mode))
            return out

        g = inv_tri(u[:, 0], *g_params)
        m = inv_tri(u[:, 1], *margin_params)
        r = inv_tri(u[:, 2], *r_params)
        stc = inv_tri(u[:, 3], *sales_to_capital_params)

    values = np.empty(n, dtype=float)
    for i in range(n):
        # Guardrails per draw to avoid invalid terminal conditions
        ri = float(r[i])
        gi = float(g_terminal)
        if ri <= gi + 1e-6:
            values[i] = np.nan
            continue
        try:
            values[i] = dcf_value_from_inputs(
                revenue0=revenue0,
                margin=float(m[i]),
                tax_rate=tax_rate,
                sales_to_capital=float(stc[i]),
                g=float(g[i]),
                years=years,
                r=ri,
                g_terminal=gi
            )
        except ValueError:
            values[i] = np.nan

    clean = values[~np.isnan(values)]
    if clean.size == 0:
        raise ValueError("all simulation draws failed; loosen guardrails or check inputs")

    return {
        "n": float(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p90": float(np.percentile(clean, 90)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


# Example usage (all $m)
summary = monte_carlo_valuation(
    n=20_000,
    seed=7,
    revenue0=1_000.0,
    years=6,
    tax_rate=0.25,
    g_params=(0.02, 0.08, 0.16),            # growth
    margin_params=(-0.05, 0.12, 0.22),      # operating margin
    r_params=(0.09, 0.12, 0.16),            # discount rate
    sales_to_capital_params=(1.3, 2.0, 3.0),
    g_terminal=0.03
)

print("Monte Carlo DCF summary (EV):")
for k, v in summary.items():
    if k == "n":
        print(f"  n_valid: {int(v):,}")
    else:
        print(f"  {k}: ${v:,.0f}m")
```

---

### Valuation Impact
Why this matters:
- Probabilistic valuation converts “hand-wavy uncertainty” into measurable risk ranges (p10/p90), supporting better decisions on pricing, capital allocation, and deal terms.
- It prevents false precision: point estimates hide tail risk and can lead to systematically overpaying in M&A, underestimating distress, or mispricing growth options.
- Output distributions enable governance: you can document the assumptions that generate downside and set risk limits (e.g., “do not proceed if p10 < 0”).

Impact on multiples:
- Multiples embed market expectations about growth/risk; simulation helps translate a multiple into implied distributions of drivers (or vice versa).
- For volatile sectors, comparing point multiples is fragile; ranges and scenario-consistent multiples are more defensible.

Impact on DCF inputs:
- Terminal value dominates; probabilistic inputs for r and g_terminal must be constrained to avoid nonsensical draws (r ≤ g).
- Correlation is often the difference between realistic and misleading results; assuming independence can understate downside.

Practical adjustments:
```python
def prob_value_exceeds(values: List[float], threshold: float) -> float:
    """Probability that value exceeds a threshold."""
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("values must be a non-empty list")
    if not _num(threshold):
        raise ValueError("threshold must be numeric")
    clean = [v for v in values if _num(v)]
    if len(clean) == 0:
        raise ValueError("no valid values")
    return sum(1 for v in clean if v > threshold) / len(clean)
```

---

### Quality of Earnings / Modelling Flags (Probabilistic Work)
⚠️ Using distributions with no bounds (normal growth rates can imply impossible outcomes).  
⚠️ Ignoring correlation between growth, margins, and reinvestment (can understate downside).  
⚠️ Treating terminal value parameters as highly variable without economic closure (r close to g creates explosive TV).  
⚠️ “Optimism bias” in scenario probabilities (success probability not benchmarked).  
⚠️ Reporting only mean value (hides skew); always show median and tails (p10/p90).  
✅ Green flag: transparent distributions (min/mode/max), guardrails, and a clear narrative for each scenario tail.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Biotech | Binary approvals | Decision trees + scenario PV; explicit milestone probabilities. |
| Start-ups/SaaS | Growth/margin uncertainty | Monte Carlo on growth, retention, margin ramps; correlation with CAC. |
| Cyclicals | Mean reversion | Scenario sets around cycle states; through-cycle terminal assumptions. |
| Distressed firms | Default outcomes | Scenario EV + waterfall; probability-weighted equity as option-like residual. |
| Real estate | Lease-up / exit cap risk | Scenario exit caps, vacancy paths; stress test refinancing. |

---

### Practical Comparison Tables

#### 1) When to use which probabilistic tool
| Tool | Best for | Output | Common mistake |
|---|---|---|---|
| Scenario analysis | few discrete futures | expected value + narrative | inconsistent scenario inputs |
| Decision tree | staged/binary events | node values + option choices | ignoring decision flexibility |
| Monte Carlo | many interacting uncertainties | value distribution | unrealistic distributions, no correlation |
| Sensitivity table | local effects | rank drivers | ignores interactions/tails |

#### 2) Minimum reporting set (knowledge base standard)
| Metric | Why |
|---|---|---|
| Mean and median | skew diagnostics |
| p10/p90 (or p5/p95) | tail risk |
| % draws invalid | model robustness |
| Top 3 variance drivers | focus management attention |
| Scenario narratives | explain tails |

---

### Real-World Example
Scenario: You value a firm with uncertain growth, margin, discount rate, and reinvestment efficiency. You define triangular distributions for each input, run 20,000 DCF simulations, and report mean, median, and p10/p90.

```python
# The example code above prints a Monte Carlo DCF summary:
# - n_valid
# - mean/median
# - p10/p90
# - min/max
```
Interpretation: Use median as a robust central estimate and p10 as a downside planning case. If the p10 is below a financing covenant threshold (or below purchase price), risk mitigation (structure, earn-out, staged investment) is required rather than “hoping the base case happens.”
