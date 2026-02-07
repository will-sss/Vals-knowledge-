# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 17: Fundamental Principles of Relative Valuation (Comparable Selection, Consistency, Standardisation, Cross-Checks)

### Core Concept
Relative valuation prices a company by comparing it to peers using multiples (EV/EBITDA, P/E, EV/Sales, P/B), but it only works when you standardise definitions and control for differences in growth, risk, profitability, and reinvestment. The practical goal is to build a repeatable workflow: define the right multiple for the business, select truly comparable firms, normalise metrics, and run consistency checks against intrinsic value drivers.

See also: Chapter 9 (earnings normalisation), Chapter 3 (recasting statements), Chapter 18–20 (multiple-specific mechanics), Chapter 12 (terminal value implies an “exit multiple”).

---

### Formula/Methodology

#### 1) Multiples: what they are (and why consistency matters)
```text
Multiple = Price / Fundamental

Examples:
P/E = Equity Value / Net Income
EV/EBITDA = Enterprise Value / EBITDA
EV/Sales = Enterprise Value / Revenue
P/B = Equity Value / Book Equity
```
Consistency rules:
- Equity multiples (P/E, P/B) must use **equity** numerators and **after-interest** fundamentals.
- Enterprise multiples (EV/EBITDA, EV/EBIT, EV/Sales) must use **enterprise** numerators and **pre-interest** fundamentals.
- Adjusted fundamentals require adjusted numerators (leases, pensions, minorities, preferred).

#### 2) Comparable company selection (decision rules)
Relative valuation is only as good as the peer set. Practical filters:
```text
Business model similarity: product/service, customer, monetisation
Operating drivers: margin structure, unit economics, capital intensity
Risk profile: cyclicality, leverage, country risk, customer concentration
Growth stage: mature vs high growth; avoid mixing life-cycle stages
Accounting comparability: IFRS vs US GAAP, lease/SBC policies, capitalisation
```
Minimum viable peer set:
- 6–15 firms is often a workable range; fewer increases noise, more often increases heterogeneity.

#### 3) Standardising multiples (to reduce noise)
A multiple is a function of growth, risk, and cash-flow conversion. Use simple “standardisation” by controlling for one or more drivers:
```text
PEG = (P/E) / Growth

Adjusted EV/EBITDA can be compared after controlling for margin and growth (via regression or buckets).
```
Practical application:
- Use PEG only when growth is meaningful and positive and earnings quality is acceptable.
- For EV/Sales, control for gross margin / contribution margin differences.

#### 4) Using fundamentals to explain multiples (directional mapping)
```text
Higher growth -> higher multiples (all else equal)
Higher risk (k_e or WACC) -> lower multiples
Higher profitability (margins, ROIC) -> higher multiples
Higher reinvestment needs (capital intensity) -> lower multiples (for same growth)
```
Cross-check logic:
- If a firm has a higher multiple than peers, you should be able to point to higher growth, higher margins/ROIC, or lower risk. If not, the multiple is likely “story premium” and fragile.

#### 5) Relative valuation workflow (repeatable)
```text
Step 1: Choose appropriate multiple(s) for the business
Step 2: Build peer set (business + risk + growth stage)
Step 3: Standardise definitions (EV, EBITDA, NI, book equity)
Step 4: Compute peer distribution (median, IQR, trimmed mean)
Step 5: Apply to target fundamental (normalised)
Step 6: Bridge to equity/per-share (if EV multiple)
Step 7: Sanity checks: implied growth/margins; reconcile vs DCF
```
Preferred peer statistics:
- median, interquartile range (IQR), trimmed mean (exclude top/bottom 10–20%)

---

### Python Implementation
```python
from typing import List, Dict, Optional, Tuple
import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def safe_div(n: float, d: float) -> Optional[float]:
    """Safe division. Returns None if denominator is 0 or invalid."""
    if not _num(n) or not _num(d):
        raise ValueError("Inputs must be numeric")
    if d == 0:
        return None
    return n / d


def compute_ev(market_cap: float, debt_like: float, cash: float, other_claims: float = 0.0) -> float:
    """EV = Market Cap + Debt-like + Other Claims - Cash."""
    for name, v in {"market_cap": market_cap, "debt_like": debt_like, "cash": cash, "other_claims": other_claims}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if debt_like < 0 or cash < 0 or other_claims < 0 or market_cap < 0:
        raise ValueError("Inputs must be >= 0")
    return market_cap + debt_like + other_claims - cash


def multiple_ev_to_metric(ev: float, metric: float) -> Optional[float]:
    """EV/metric (returns None if metric <= 0)."""
    if not _num(ev) or not _num(metric):
        raise ValueError("Inputs must be numeric")
    if metric <= 0:
        return None
    return ev / metric


def multiple_equity_to_metric(equity_value: float, metric: float) -> Optional[float]:
    """Equity/metric (returns None if metric <= 0)."""
    if not _num(equity_value) or not _num(metric):
        raise ValueError("Inputs must be numeric")
    if metric <= 0:
        return None
    return equity_value / metric


def robust_stats(values: List[float], trim: float = 0.1) -> Dict[str, float]:
    """Median, IQR, trimmed mean. Expects numeric list; ignores None."""
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("values must be a non-empty list")
    clean = [v for v in values if _num(v)]
    if len(clean) == 0:
        raise ValueError("No numeric values provided")
    clean.sort()
    n = len(clean)

    def pct(p: float) -> float:
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return clean[lo]
        w = idx - lo
        return clean[lo] * (1 - w) + clean[hi] * w

    median = pct(0.5)
    q1 = pct(0.25)
    q3 = pct(0.75)
    iqr = q3 - q1

    k = int(round(n * trim))
    trimmed = clean[k:n - k] if n - 2 * k > 0 else clean
    tmean = sum(trimmed) / len(trimmed)

    return {"median": median, "q1": q1, "q3": q3, "iqr": iqr, "trimmed_mean": tmean, "n": float(n)}


def peer_screen(
    peers: List[Dict],
    multiple_name: str,
    min_n: int = 6
) -> Dict[str, float]:
    """Compute robust stats for a chosen multiple across peers."""
    if multiple_name not in {"ev_ebitda", "ev_ebit", "ev_sales", "pe", "pb"}:
        raise ValueError("Unsupported multiple_name")
    vals = []
    for p in peers:
        v = p.get(multiple_name, None)
        if v is not None and _num(v):
            vals.append(v)
    stats = robust_stats(vals, trim=0.1)
    if int(stats["n"]) < min_n:
        raise ValueError(f"Peer count too low after cleaning: {int(stats['n'])} (<{min_n})")
    return stats


def apply_multiple_to_target(stats: Dict[str, float], target_metric: float, method: str = "median") -> float:
    """Apply peer multiple to target metric to get value (EV or equity depending on multiple)."""
    if method not in {"median", "trimmed_mean"}:
        raise ValueError("method must be 'median' or 'trimmed_mean'")
    if not _num(target_metric) or target_metric <= 0:
        raise ValueError("target_metric must be numeric and > 0")
    m = stats[method]
    return m * target_metric


# Example usage: build peer multiples and value a target (all $m)
peers = [
    {"name": "PeerA", "ev_ebitda": 9.5, "ev_sales": 2.1, "pe": 18.0},
    {"name": "PeerB", "ev_ebitda": 11.2, "ev_sales": 2.8, "pe": 22.0},
    {"name": "PeerC", "ev_ebitda": 8.7, "ev_sales": 1.9, "pe": 16.5},
    {"name": "PeerD", "ev_ebitda": 10.1, "ev_sales": 2.4, "pe": 19.0},
    {"name": "PeerE", "ev_ebitda": 12.0, "ev_sales": 3.0, "pe": 24.5},
    {"name": "PeerF", "ev_ebitda": 9.9, "ev_sales": 2.2, "pe": 17.8},
    {"name": "PeerG", "ev_ebitda": 35.0, "ev_sales": 6.5, "pe": 60.0},  # outlier
]

stats_ev_ebitda = peer_screen(peers, "ev_ebitda", min_n=6)
target_ebitda = 420.0
target_ev = apply_multiple_to_target(stats_ev_ebitda, target_ebitda, method="median")

# Bridge EV to equity (simple)
debt_like = 1_100.0
cash = 200.0
other_claims = 150.0
equity = target_ev - (debt_like - cash) - other_claims

shares = 300.0
value_per_share = equity / shares

print(f"Peer EV/EBITDA median: {stats_ev_ebitda['median']:.1f}x (IQR {stats_ev_ebitda['q1']:.1f}x–{stats_ev_ebitda['q3']:.1f}x)")
print(f"Target EV (median multiple): ${target_ev:,.0f}m")
print(f"Equity value: ${equity:,.0f}m")
print(f"Value per share: ${value_per_share:,.2f}")
```

---

### Valuation Impact
Why this matters:
- Relative valuation is the fastest way to triangulate value, but also the easiest to get wrong due to inconsistent definitions (EBITDA adjustments, lease treatment, debt-like claims).
- Peer selection is a valuation decision: wrong peers produce wrong medians. A smaller, cleaner peer set is usually better than a broad, noisy one.
- Multiples can be reverse-engineered into implied fundamentals; if implied growth/margins are inconsistent with the story, adjust the peer set or the multiple used.

Impact on multiples:
- EV/EBITDA is sensitive to lease policy, capitalisation, and SBC adjustments; standardise or adjust.
- P/E is sensitive to leverage, tax regimes, and one-offs; avoid mixing structurally different capital structures without controls.
- EV/Sales is useful when earnings are negative but requires margin controls.

Impact on DCF inputs:
- Multiples provide a market-based check on your terminal value and discount rate assumptions.
- DCF can explain why the target deserves a premium/discount multiple via drivers (growth, ROIC, risk).

Practical adjustments:
```python
def flag_peer_outliers(multiples: List[float], z: float = 2.5) -> List[int]:
    """Return indices of outliers using a robust MAD-based z-score."""
    clean = [m for m in multiples if _num(m)]
    if len(clean) < 5:
        return []
    med = sorted(clean)[len(clean)//2]
    abs_dev = [abs(m - med) for m in clean]
    mad = sorted(abs_dev)[len(abs_dev)//2]
    if mad == 0:
        return []
    out = []
    for i, m in enumerate(multiples):
        if not _num(m):
            continue
        rz = 0.6745 * (m - med) / mad
        if abs(rz) > z:
            out.append(i)
    return out
```

---

### Quality of Earnings Flags (Relative Valuation Pitfalls)
⚠️ “Adjusted EBITDA” differs across peers (lease, SBC, capitalised costs), but EV is not adjusted accordingly.  
⚠️ Peer set includes firms at different life-cycle stages (early hypergrowth vs mature), inflating dispersion.  
⚠️ Using P/E when earnings are volatile or near zero (multiples explode).  
⚠️ Using EV/Sales without controlling for gross margin/contribution margin (unfair comparisons).  
⚠️ Mixing consolidation treatments (minority interests, associates) without consistent EV and metrics.  
✅ Green flag: clear peer selection logic, consistent metric definitions, robust statistics, and reconciliation to fundamentals.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Software | SBC and R&D capitalisation differences | Standardise policies; consider EV/Sales with margin controls. |
| Retail | Lease intensity | Either keep reported EV/EBITDA consistently or lease-adjust both EV and EBITDA. |
| Banks | Balance-sheet driven | Use P/B and ROE, not EV/EBITDA; control for capital ratios. |
| Commodities | Cycle effects | Use mid-cycle earnings and normalised multiples; avoid peak/trough anchoring. |

---

### Practical Comparison Tables

#### 1) Multiple selection decision rules
| Company situation | Primary multiple | Why | Secondary check |
|---|---|---|---|
| Stable positive earnings | EV/EBITDA or P/E | Widely comparable | EV/EBIT, DCF implied multiple |
| High leverage differences | EV multiples | Capital structure-neutral | P/E with leverage controls |
| Loss-making but scaling | EV/Sales | Earnings not meaningful | EV/Gross Profit; unit economics |
| Financial services | P/B, P/E | Debt is operating | ROE/COE, capital ratios |

#### 2) Standardisation checklist
| Item | Standardise by | Why it matters |
|---|---|---|
| EV | include debt-like claims consistently | Avoid overstating/understating EV |
| EBITDA | consistent add-backs (or none) | Prevent multiple drift |
| Net income | remove one-offs | P/E stability |
| Book equity | adjust for intangibles/OCI policy (if needed) | P/B comparability |
| Currency/geography | align risk and inflation | Avoid mixing regimes |

---

### Real-World Example
Scenario: Target is a lease-heavy retailer. You want EV/EBITDA, but peers have mixed lease policies. You decide either to keep everything reported (no lease adjustment) or adjust both EV and EBITDA consistently.

```python
# Option A: stay reported for everyone (simplest)
# - Use reported EV and reported EBITDA for target and peers.

# Option B: lease-adjust
# - Add PV of leases to debt-like claims in EV for all
# - Add back lease expense and subtract depreciation/interest-like components to get a consistent EBITDA basis
# If you can't do it consistently across peers, do not do it for only the target.
```

Interpretation: Relative valuation is a consistency exercise. A defensible multiple conclusion requires: clean peer logic, standardised definitions, robust statistics, and a fundamentals-based explanation for premiums/discounts.
