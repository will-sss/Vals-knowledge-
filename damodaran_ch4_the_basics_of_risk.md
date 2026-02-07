# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 4: The Basics of Risk (Expected Returns, Variance, Correlation, Risk-Free vs Risk Premium)

### Core Concept
Risk in valuation is about **uncertainty in future cash flows** and how markets price that uncertainty through required returns (discount rates) and risk adjustments. Practically, you need to (1) distinguish **diversifiable (idiosyncratic)** from **non-diversifiable (market)** risk, (2) estimate expected returns and dispersion (volatility), and (3) translate risk into a **discount rate (cost of equity/WACC)** or into **scenario probabilities**. fileciteturn3file2

See also: Chapter 7 (Riskless Rates and Risk Premiums), Chapter 8 (Estimating Risk Parameters and Costs of Financing), Chapter 33 (Probabilistic Approaches: scenarios/simulations).

---

### Formula/Methodology

#### 1) Expected return and variance (discrete outcomes)
```text
Expected Return:   E(R) = Σ p_i × R_i
Variance:          Var(R) = Σ p_i × (R_i − E(R))^2
Std Dev (vol):     σ = sqrt(Var(R))
Where:
p_i = probability of outcome i (Σ p_i = 1)
R_i = return in outcome i
```
Application:
- Use for scenario trees (discrete states) or when you can define explicit outcome distributions.

#### 2) Portfolio return and variance (two assets; intuition for diversification)
```text
Portfolio return:  E(R_p) = w_A × E(R_A) + w_B × E(R_B)

Portfolio variance:
Var(R_p) = w_A^2 × σ_A^2 + w_B^2 × σ_B^2 + 2 × w_A × w_B × ρ_AB × σ_A × σ_B

Where:
w_A, w_B = portfolio weights (sum to 1)
σ_A, σ_B = standard deviations
ρ_AB     = correlation between asset returns (−1 to +1)
```
Key implication:
- Lower correlation reduces portfolio risk even if individual volatilities are high.
- Diversification eliminates idiosyncratic risk; market risk remains.

#### 3) CAPM (core pricing model used in many valuations)
```text
Cost of Equity (CAPM):  k_e = R_f + β × ERP
Where:
R_f = risk-free rate (currency/tenor-consistent)
β   = measure of exposure to market risk
ERP = equity risk premium (market’s price of risk)
```
Practical note:
- CAPM is a tool, not a truth: it’s commonly used because inputs are estimable and comparable across firms.
- If CAPM inputs are weak, use cross-checks (implied ERP, multi-factor sanity checks, scenario WACC ranges).

#### 4) Beta from regression (concept)
```text
β = Cov(R_i, R_m) / Var(R_m)
```
Where:
R_i = asset/stock return
R_m = market return

Practical note:
- Regression beta is noisy; improve by using longer windows, weekly returns, or bottom-up betas (Chapter 8).

---

### Python Implementation
```python
from typing import List, Tuple, Optional
import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def expected_return(probabilities: List[float], returns: List[float]) -> float:
    """
    Compute expected return E(R) for discrete outcomes.

    Args:
        probabilities: list of probabilities p_i (must sum to 1 within tolerance).
        returns: list of returns R_i as decimals (e.g., 0.12 for 12%).

    Returns:
        float: expected return as decimal.

    Raises:
        ValueError: on invalid inputs.
    """
    if not isinstance(probabilities, list) or not isinstance(returns, list):
        raise ValueError("probabilities and returns must be lists")
    if len(probabilities) == 0 or len(probabilities) != len(returns):
        raise ValueError("probabilities and returns must be same non-zero length")
    if any((not _num(p) or p < 0) for p in probabilities):
        raise ValueError("probabilities must be numeric and >= 0")
    if any((not _num(r)) for r in returns):
        raise ValueError("returns must be numeric")
    s = sum(probabilities)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1 (got {s})")
    return sum(p * r for p, r in zip(probabilities, returns))


def variance(probabilities: List[float], returns: List[float]) -> float:
    """
    Compute variance Var(R) for discrete outcomes.

    Returns:
        float: variance (decimal^2).
    """
    mu = expected_return(probabilities, returns)
    return sum(p * (r - mu) ** 2 for p, r in zip(probabilities, returns))


def std_dev(probabilities: List[float], returns: List[float]) -> float:
    """Standard deviation (volatility) for discrete outcomes."""
    return math.sqrt(variance(probabilities, returns))


def portfolio_variance_two_assets(w_a: float, sigma_a: float, w_b: float, sigma_b: float, corr_ab: float) -> float:
    """
    Two-asset portfolio variance.

    Args:
        w_a, w_b: weights (should sum to 1).
        sigma_a, sigma_b: vols as decimals (>=0).
        corr_ab: correlation in [-1, 1].

    Returns:
        float: variance.

    Raises:
        ValueError: invalid inputs.
    """
    for name, v in {"w_a": w_a, "w_b": w_b, "sigma_a": sigma_a, "sigma_b": sigma_b, "corr_ab": corr_ab}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if sigma_a < 0 or sigma_b < 0:
        raise ValueError("sigmas must be >= 0")
    if corr_ab < -1 or corr_ab > 1:
        raise ValueError("corr_ab must be between -1 and 1")
    if abs((w_a + w_b) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")
    return (w_a**2) * (sigma_a**2) + (w_b**2) * (sigma_b**2) + 2 * w_a * w_b * corr_ab * sigma_a * sigma_b


def capm_cost_of_equity(rf: float, beta: float, erp: float) -> float:
    """
    Cost of equity using CAPM: ke = rf + beta * ERP.

    Args:
        rf: risk-free rate (decimal).
        beta: equity beta (can be negative in rare cases).
        erp: equity risk premium (decimal, should be >= 0).

    Returns:
        float: cost of equity.

    Raises:
        ValueError: invalid inputs.
    """
    for name, v in {"rf": rf, "beta": beta, "erp": erp}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if erp < 0:
        raise ValueError("erp must be >= 0")
    if rf <= -1:
        raise ValueError("rf must be > -100%")
    return rf + beta * erp


# Example usage: discrete outcomes + diversification intuition + CAPM
probs = [0.25, 0.50, 0.25]
rets = [-0.10, 0.08, 0.22]  # -10%, +8%, +22%

mu = expected_return(probs, rets)
vol = std_dev(probs, rets)

print(f"E(R): {mu:.1%}")
print(f"Vol:  {vol:.1%}")

var_p = portfolio_variance_two_assets(w_a=0.6, sigma_a=0.20, w_b=0.4, sigma_b=0.15, corr_ab=0.2)
print(f"Portfolio vol (2 assets): {math.sqrt(var_p):.1%}")

ke = capm_cost_of_equity(rf=0.04, beta=1.1, erp=0.05)
print(f"CAPM cost of equity: {ke:.1%}")
```

---

### Valuation Impact
Why this matters:
- Discount rates are the **price of risk** in DCF: higher uncertainty in operating cash flows or higher exposure to macro risks should increase required returns (lower value), all else equal. fileciteturn3file2  
- Risk is not just “volatility”: for valuation, focus on **non-diversifiable risk** (systematic exposure) and the company’s ability to survive downside cases (liquidity, leverage, cyclicality).  
- In relative valuation, higher risk should show up as **lower multiples** (e.g., lower P/E for higher beta/leverage), so risk adjustment is central to comparability.

Impact on multiples:
- Two firms with the same growth and margins can deserve different EV/EBITDA or P/E due to different risk (beta, cyclicality, leverage).  
- If the market is risk-on/off, multiples can shift broadly; sanity-check with a DCF range or implied ERP.

Impact on DCF inputs:
- CAPM cost of equity is a core input into WACC; errors in beta or ERP have large effects on value, especially when terminal value is a big share of total value.
- For firms with concentrated risks (commodity exposure, regulatory cliffs), scenario analysis can be more informative than a single-point discount rate.

Practical adjustments:
```python
def risk_adjusted_discount_rate(base_rate: float, incremental_risk_premium: float) -> float:
    """Simple additive risk premium approach for project-specific risks."""
    if base_rate <= -1:
        raise ValueError("base_rate must be > -100%")
    if incremental_risk_premium < 0:
        raise ValueError("incremental_risk_premium must be >= 0")
    return base_rate + incremental_risk_premium
```

---

### Quality of Earnings Flags (Risk Signals)
⚠️ Earnings are stable only because of aggressive smoothing (reserve releases, capitalization) while cash flows are volatile (risk is understated).  
⚠️ High reported margins but extreme revenue concentration (key customer/supplier) → idiosyncratic risk that may become systematic in stress.  
⚠️ Leverage makes equity “option-like” (high sensitivity to downside), so small operating disappointments cause large equity value swings.  
⚠️ Volatility is low because the business is illiquid or thinly traded (beta/vol estimates are unreliable).  
✅ Transparent disclosures of risk drivers (pricing, volumes, FX, commodity hedging) and stable cash conversion improve confidence in risk inputs.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Cyclical industrials | High macro sensitivity | Use higher beta/ERP sensitivity; stress margins and working capital in downturns. |
| Commodities | Price risk dominates | Prefer scenario/simulation; consider hedging policy and reserve life. |
| Financials | Leverage is operating | Use equity-focused risk measures; regulatory capital buffers matter more than EBITDA volatility. |
| Early-stage tech | Discrete success/failure outcomes | Use scenario trees or simulations; don’t rely solely on regression betas. |
| Utilities | Regulatory and interest-rate exposure | Lower beta typically; model regulatory reset risk explicitly. |

---

### Practical Comparison Tables

#### 1) Risk Type Map (What to adjust in valuation)
| Risk type | Diversifiable? | Where it matters | Practical handling |
|---|---|---|---|
| Company-specific (single product, key customer) | Mostly | Scenario cash flows | Scenario probabilities, margin/volume stress |
| Macro exposure (cycle, rates, FX) | No | Discount rate + scenarios | Beta/WACC + macro scenarios |
| Leverage / liquidity | No (for equity) | Equity value and distress risk | Capital structure stress tests, distress cost adjustments |
| Regulatory cliff | Often no | Cash flows discontinuities | Scenario trees, contingent claims if option-like |

#### 2) Beta/discount rate sanity checks
| Check | Expectation | Action if fails |
|---|---|---|
| Beta vs business cyclicality | More cyclical → higher beta | Revisit comp set and unlevering |
| Beta vs leverage | Higher D/E → higher equity beta | Ensure market-value D/E and tax rate consistent |
| ERP vs market regime | ERP should not be arbitrary | Cross-check implied ERP or use a published estimate consistently |
| Terminal value share | High TV share → rate sensitivity | Run WACC and terminal g sensitivity grid |

---

### Real-World Example
Scenario: A company’s valuation hinges on whether it is “defensive” or “cyclical.” You quantify risk using CAPM and check diversification benefits conceptually via correlation.

```python
rf = 0.04
erp = 0.05

# Defensive vs cyclical beta comparison
for name, beta in [("Defensive", 0.7), ("Cyclical", 1.3)]:
    ke = capm_cost_of_equity(rf, beta, erp)
    print(f"{name} beta={beta:.1f} -> k_e={ke:.1%}")

# Diversification intuition: same vols, different correlations
import math
for corr in [-0.2, 0.2, 0.8]:
    var_p = portfolio_variance_two_assets(0.5, 0.25, 0.5, 0.25, corr)
    print(f"corr={corr:+.1f} -> portfolio vol={math.sqrt(var_p):.1%}")
```

Interpretation: Required returns (discount rates) and downside scenarios must reflect systematic exposure. Treating a cyclical business as defensive (too low beta/WACC) overvalues it, often most visibly through terminal value.
