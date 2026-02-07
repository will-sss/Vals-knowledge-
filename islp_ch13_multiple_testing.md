# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 13: Multiple Testing (p-values, FWER, FDR, Bonferroni, Holm, Benjamini–Hochberg)

### Core Concept
Multiple testing arises when you run many hypothesis tests at once (e.g., screening dozens of KPIs, factors, or accounting adjustments for “significance”). In valuation and financial analysis, failing to correct for multiple comparisons leads to **false discoveries**—drivers that appear significant by chance—causing overconfident narratives, unstable forecasts, and mis-specified comparable selection.

See also: Chapter 6 (model selection/regularization), Chapter 5 (validation), Chapter 12 (unsupervised screening), Damodaran relative valuation (driver selection discipline).

---

### Formula/Methodology

#### 1) What goes wrong: false positives scale with the number of tests
If you test m independent null hypotheses at significance level α:
```text
Expected false positives ≈ m × α
```
Example: m=100 tests at α=0.05 ⇒ ~5 false positives expected even if all nulls are true.

#### 2) Family-wise error rate (FWER)
FWER controls the probability of **at least one** false rejection:
```text
FWER = Pr( V ≥ 1 )
```
Where:
- V = number of false positives

##### Bonferroni correction (simple, conservative)
Reject H_i if:
```text
p_i ≤ α / m
```
Or equivalently, adjusted p-value:
```text
p_i_adj = min(1, m × p_i)
```

##### Holm step-down (uniformly more powerful than Bonferroni)
Sort p-values ascending: p_(1) ≤ ... ≤ p_(m).
Find first k where:
```text
p_(k) > α / (m − k + 1)
```
Reject all hypotheses with ranks < k.

Holm adjusted p-values:
```text
p_(k)_adj = max_{j≤k} ( (m − j + 1) × p_(j) )
```
Then cap at 1.

#### 3) False discovery rate (FDR)
FDR controls the expected fraction of false rejections among all rejections:
```text
FDR = E[ V / max(R, 1) ]
```
Where:
- R = number of rejections

##### Benjamini–Hochberg (BH) procedure (common in factor/KPI screening)
Sort p-values ascending: p_(1) ≤ ... ≤ p_(m).
Find the largest k such that:
```text
p_(k) ≤ (k/m) × q
```
Reject all hypotheses with ranks 1..k.
Where:
- q is the target FDR (e.g., 0.10)

BH adjusted p-values (q-values in BH sense):
```text
p_(k)_adj = min_{j≥k} ( (m/j) × p_(j) )
```
Then enforce monotonicity and cap at 1.

Practical implication:
- FDR is typically better than FWER for **exploratory screens** (many candidate drivers), while FWER is more appropriate for **high-stakes single conclusions**.

---

### Python Implementation
```python
from typing import List, Dict, Tuple
import numpy as np


def _validate_pvalues(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("pvals must not be empty.")
    if np.any(~np.isfinite(p)):
        raise ValueError("pvals contains non-finite values (NaN/Inf).")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("All p-values must be in [0, 1].")
    return p


def bonferroni(pvals: List[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Bonferroni correction for controlling FWER.

    Args:
        pvals: list/array of p-values
        alpha: desired family-wise level

    Returns:
        dict:
          p_adj: adjusted p-values (Bonferroni)
          reject: boolean array for rejection at alpha

    Raises:
        ValueError: invalid inputs.
    """
    p = _validate_pvalues(pvals)
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    m = p.size
    p_adj = np.minimum(1.0, m * p)
    reject = p <= (alpha / m)
    return {"p_adj": p_adj, "reject": reject}


def holm(pvals: List[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Holm step-down correction (FWER), more powerful than Bonferroni.

    Returns:
        dict:
          p_adj: Holm adjusted p-values
          reject: boolean array at alpha
    """
    p = _validate_pvalues(pvals)
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]

    # Raw adjusted values: (m - k + 1) * p_(k)
    raw = (m - np.arange(m)) * p_sorted
    # Step-down max to enforce monotonicity
    p_adj_sorted = np.maximum.accumulate(raw)
    p_adj_sorted = np.minimum(1.0, p_adj_sorted)

    # Rejection rule: find first k where p_(k) > alpha/(m-k)
    thresh = alpha / (m - np.arange(m))
    # Holm rejects all before first failure
    fail = np.where(p_sorted > thresh)[0]
    k0 = int(fail[0]) if fail.size > 0 else m
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[:k0] = True

    # Unsort
    p_adj = np.empty_like(p_adj_sorted)
    reject = np.empty_like(reject_sorted)
    p_adj[order] = p_adj_sorted
    reject[order] = reject_sorted
    return {"p_adj": p_adj, "reject": reject}


def benjamini_hochberg(pvals: List[float], q: float = 0.10) -> Dict[str, np.ndarray]:
    """
    Benjamini–Hochberg procedure to control FDR at level q.

    Args:
        pvals: list/array of p-values
        q: target FDR in (0,1)

    Returns:
        dict:
          p_adj: BH adjusted p-values
          reject: boolean array at FDR level q
          k_star: number of rejections
    """
    p = _validate_pvalues(pvals)
    if not np.isfinite(q) or not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1).")

    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)

    # Find largest k where p_(k) <= (k/m)*q
    crit = (ranks / m) * q
    ok = np.where(p_sorted <= crit)[0]
    k_star = int(ok[-1] + 1) if ok.size > 0 else 0  # number rejected

    reject_sorted = np.zeros(m, dtype=bool)
    if k_star > 0:
        reject_sorted[:k_star] = True

    # BH adjusted p-values: p_adj(k) = min_{j>=k} (m/j)*p(j)
    raw_adj = (m / ranks) * p_sorted
    p_adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    p_adj_sorted = np.minimum(1.0, p_adj_sorted)

    # Unsort
    p_adj = np.empty_like(p_adj_sorted)
    reject = np.empty_like(reject_sorted)
    p_adj[order] = p_adj_sorted
    reject[order] = reject_sorted
    return {"p_adj": p_adj, "reject": reject, "k_star": np.array([k_star], dtype=int)}


def summarize_rejections(names: List[str], pvals: List[float], method_out: Dict[str, np.ndarray]) -> List[Tuple[str, float, float, bool]]:
    """
    Create a summary list for reporting.

    Returns:
        list of tuples: (name, p, p_adj, reject)
    """
    p = _validate_pvalues(pvals)
    if len(names) != p.size:
        raise ValueError("names length must match number of p-values.")
    p_adj = method_out["p_adj"]
    reject = method_out["reject"]
    out = [(names[i], float(p[i]), float(p_adj[i]), bool(reject[i])) for i in range(p.size)]
    # sort by adjusted p
    out.sort(key=lambda x: x[2])
    return out


# Example usage (valuation context): screening 12 candidate KPI drivers of EV/EBITDA multiple
driver_names = [
    "rev_growth", "gross_margin", "ebitda_margin", "roic",
    "nd_to_ebitda", "capex_sales", "wc_intensity", "customer_churn",
    "nrr", "r_and_d_sales", "fx_exposure", "pricing_power_index"
]

# Suppose these are p-values from separate regressions/correlations (illustrative)
pvals = [0.012, 0.041, 0.006, 0.021, 0.180, 0.090, 0.055, 0.008, 0.004, 0.030, 0.250, 0.015]

bh = benjamini_hochberg(pvals, q=0.10)
holm_out = holm(pvals, alpha=0.05)
bonf = bonferroni(pvals, alpha=0.05)

print("BH rejections:", int(np.sum(bh["reject"])))
print("Holm rejections:", int(np.sum(holm_out["reject"])))
print("Bonferroni rejections:", int(np.sum(bonf["reject"])))

# Produce a compact report (sorted by adjusted p)
report = summarize_rejections(driver_names, pvals, bh)
for name, p, p_adj, rej in report:
    print(name, round(p, 4), round(p_adj, 4), rej)
```

---

### Valuation Impact
Why this matters:
- **Avoid spurious drivers:** Multiple testing corrections prevent “discovering” drivers of valuation multiples or cash-flow outcomes that are purely noise.
- **More defensible narratives:** Corrected significance supports investment committee defensibility and reduces “p-hacking” risk.
- **Model risk reduction:** Stabilises factor selection for forecasting modules (growth, margins, default risk).

Impact on multiples:
- In “multiple driver” analysis (e.g., EV/EBITDA explained by growth, ROIC, leverage), multiple testing helps keep only drivers that survive FDR/FWER control.
- Prevents overconfident peer multiple adjustments driven by chance correlations.

Impact on DCF inputs:
- When screening which KPIs to include in margin/growth forecasting, use FDR to reduce false inclusions.
- Use corrected results as a filter, then validate out-of-sample (do not treat statistical significance as causal).

Practical adjustments:
```python
import numpy as np

def select_drivers(pvals: list, names: list, q: float = 0.10) -> list:
    """
    Select candidate drivers using BH FDR control.

    Returns:
        list of selected driver names.
    """
    if len(pvals) != len(names):
        raise ValueError("names and pvals must have same length.")
    out = benjamini_hochberg(pvals, q=q)
    return [names[i] for i, keep in enumerate(out["reject"]) if bool(keep)]
```

---

### Quality of Earnings Flags
⚠️ **Red flag: “significant” findings from many tests without correction** — classic multiple-testing failure, often disguised as a compelling narrative.  
⚠️ **Red flag: iterative re-testing until significance** (p-hacking) — especially when KPIs are redefined, winsorized differently, or samples are tweaked.  
⚠️ **Red flag: correlated tests interpreted as independent** — corrections assume independence or certain dependence structures; treat results cautiously.  
✅ **Green flag: pre-registered driver set** (defined before testing) and corrected p-values plus out-of-sample validation.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS | Many correlated KPIs (NRR, churn, CAC, ARPA) | Use FDR (BH) for exploratory screens; then regularize (elastic net) and validate by time split |
| Banking | Many ratios and macro factors | Use FWER for high-stakes conclusions; ensure consistent definitions across banks |
| Industrials | Cyclicality creates regime-dependent significance | Test within regimes; avoid pooling across cycles without controls |
| Real Estate | Many correlated valuation metrics (cap rate, DSCR, LTV) | Use FDR + economic constraints; beware mechanical correlations |

---

### Real-World Example
Scenario: You test 12 drivers for explaining EV/EBITDA across a peer set. Several appear significant at 5% without correction. After BH FDR control at q=10%, only a subset remains—these are the drivers you carry forward into a more robust forecasting or regularized model.

```python
# The example code above:
# - applies BH (FDR), Holm (FWER), and Bonferroni (FWER)
# - compares how many drivers survive each method
# - prints a sorted report by adjusted p-values
```

Interpretation:
- Use BH for exploratory screening of many candidates.
- Use Holm/Bonferroni when a single false positive is costly (e.g., regulatory or audit-facing claims).
- Always follow with out-of-sample validation and economic sanity checks.

See also: Chapter 6 (regularization for high-dimensional drivers), Chapter 5 (validation), Chapter 12 (unsupervised screens)
