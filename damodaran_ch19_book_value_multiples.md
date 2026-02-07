# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 19: Book Value Multiples (P/B, ROE Link, Intangibles, Financials, Consistency Rules)

### Core Concept
Book value multiples (especially P/B) are most useful when **book equity is meaningful** and accounting captures economic capital reasonably well (often true for financials, sometimes for asset-heavy firms). Practically, P/B only becomes interpretable when you connect it to **ROE vs cost of equity** and standardise book equity for intangibles, write-downs, and accounting differences.

See also: Chapter 17 (relative valuation principles), Chapter 18 (earnings multiples), Chapter 14 (residual income), Chapter 3 (recasting statements).

---

### Formula/Methodology

#### 1) P/B definition and equivalents
```text
P/B = Price per Share / Book Value per Share
P/B = Equity Market Value / Book Equity
```
Book equity definition (common equity focus):
- Common shareholders’ equity excluding preferred (or treat preferred as a claim and exclude it from book equity when valuing common).

#### 2) ROE and P/B link (fundamental driver mapping)
Residual income logic ties P/B to expected excess returns:
```text
Residual Income_t = Net Income_t − (k_e × Book Equity_{t−1})

If expected ROE > k_e persistently -> P/B > 1
If expected ROE ≈ k_e -> P/B ≈ 1
If expected ROE < k_e -> P/B < 1
```
A practical stable-growth intuition (directional):
```text
Higher P/B is justified by:
- Higher sustainable ROE
- Higher growth (if ROE stays above k_e)
- Lower cost of equity (risk)
```
Avoid using P/B without ROE context.

#### 3) When P/B works best
```text
Works best when:
- Assets and liabilities are marked or close to economic values
- Earnings are linked to the book equity base
- Accounting is comparable across firms

Commonly best for:
- Banks/insurers (equity is the key capital base)
- Some asset-heavy sectors with stable accounting
```
Works poorly when:
- Large internally generated intangibles are expensed (book equity understated)
- Accounting write-offs and M&A distort book equity
- Large off-balance-sheet exposures exist

#### 4) Book equity adjustments (comparability)
Common adjustments (case-by-case):
```text
Adjusted Book Equity = Reported Book Equity
                     + Write-downs reversed (if you treat them as non-economic)
                     − Goodwill and acquired intangibles (if comparing tangible book)
                     ± AOCI/OCI adjustments (if peers differ and it matters)
                     − Underfunded pensions (if not already captured)
                     − Preferred equity (if valuing common and preferred is a claim)
```
Use the adjustment that matches the multiple:
- If you use P/TBV, denominator must be tangible book (exclude goodwill/intangibles).

#### 5) Banking-specific cross-check (P/B vs ROE vs COE)
For banks, a common interpretability check is:
```text
If ROE sustainably exceeds cost of equity -> P/B should be > 1
If ROE is below cost of equity -> P/B should be < 1
```
Then identify whether the market is pricing:
- future ROE improvement/decline,
- risk changes (COE),
- asset quality issues (provisions, NPLs).

---

### Python Implementation
```python
from typing import Optional, List, Dict


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def pb_ratio(market_cap: float, book_equity: float) -> Optional[float]:
    """P/B = market cap / book equity. Returns None if book_equity <= 0."""
    if not _num(market_cap) or not _num(book_equity):
        raise ValueError("inputs must be numeric")
    if market_cap < 0:
        raise ValueError("market_cap must be >= 0")
    if book_equity <= 0:
        return None
    return market_cap / book_equity


def roe(net_income: float, book_equity_begin: float) -> Optional[float]:
    """ROE = net income / beginning book equity. Returns None if book_equity_begin <= 0."""
    if not _num(net_income) or not _num(book_equity_begin):
        raise ValueError("inputs must be numeric")
    if book_equity_begin <= 0:
        return None
    return net_income / book_equity_begin


def adjusted_book_equity(
    reported_book_equity: float,
    goodwill: float = 0.0,
    acquired_intangibles: float = 0.0,
    preferred_equity: float = 0.0,
    pension_deficit: float = 0.0,
    aoci_adjustment: float = 0.0,
) -> float:
    """
    Adjust book equity for comparability. By default returns tangible common equity if you subtract goodwill, intangibles, preferred, pensions.
    """
    for name, v in {
        "reported_book_equity": reported_book_equity, "goodwill": goodwill, "acquired_intangibles": acquired_intangibles,
        "preferred_equity": preferred_equity, "pension_deficit": pension_deficit, "aoci_adjustment": aoci_adjustment
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if reported_book_equity < 0:
        raise ValueError("reported_book_equity must be >= 0")
    if any(v < 0 for v in [goodwill, acquired_intangibles, preferred_equity, pension_deficit]):
        raise ValueError("goodwill, acquired_intangibles, preferred_equity, pension_deficit must be >= 0")
    adj = reported_book_equity + aoci_adjustment - goodwill - acquired_intangibles - preferred_equity - pension_deficit
    return adj


def residual_income(net_income: float, ke: float, book_equity_begin: float) -> float:
    """RI = NI - ke * BE_{t-1}."""
    for name, v in {"net_income": net_income, "ke": ke, "book_equity_begin": book_equity_begin}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if book_equity_begin < 0:
        raise ValueError("book_equity_begin must be >= 0")
    return net_income - ke * book_equity_begin


def pb_signal(roe_value: float, ke: float) -> str:
    """Simple interpretive label for ROE vs cost of equity."""
    if roe_value is None or not _num(ke):
        return "n/a"
    if roe_value > ke + 0.01:
        return "ROE meaningfully above COE -> supports P/B > 1"
    if roe_value < ke - 0.01:
        return "ROE meaningfully below COE -> supports P/B < 1"
    return "ROE ~ COE -> supports P/B ~ 1"


# Example usage (all $m)
market_cap = 12_000.0
book_equity_reported = 9_500.0
goodwill = 1_800.0
intangibles = 700.0
preferred = 300.0
pension_deficit = 250.0

book_equity_tangible_common = adjusted_book_equity(
    reported_book_equity=book_equity_reported,
    goodwill=goodwill,
    acquired_intangibles=intangibles,
    preferred_equity=preferred,
    pension_deficit=pension_deficit,
    aoci_adjustment=0.0
)

pb_reported = pb_ratio(market_cap, book_equity_reported)
pb_tangible = pb_ratio(market_cap, book_equity_tangible_common)

net_income = 1_050.0
roe_rep = roe(net_income, book_equity_reported)
ke = 0.11

print(f"Reported P/B: {pb_reported:.2f}x" if pb_reported else "Reported P/B: n/a")
print(f"Tangible common P/B: {pb_tangible:.2f}x" if pb_tangible else "Tangible P/B: n/a")
print(f"ROE (reported BE): {roe_rep:.1%}" if roe_rep is not None else "ROE: n/a")
print(pb_signal(roe_rep, ke))

ri = residual_income(net_income, ke, book_equity_reported)
print(f"Residual income (year): ${ri:,.0f}m")
```

---

### Valuation Impact
Why this matters:
- P/B compresses a lot of information: expected ROE, risk (COE), and growth in equity base. Without ROE vs COE, P/B is just a number.
- Book equity can be distorted by M&A (goodwill), write-downs, and accounting choices; tangible book variants often improve comparability.
- For banks and insurers, the balance sheet is the business; P/B anchored to sustainable ROE is often more informative than EV multiples.

Impact on multiples:
- Higher P/B is justified only if you can defend higher sustainable ROE (or lower COE). If a bank trades at a premium P/B without superior ROE, investigate hidden risk or overly optimistic market expectations.
- P/TBV is commonly used to avoid goodwill distortions; but beware: goodwill can reflect real franchise value (you are choosing to exclude it).

Impact on DCF inputs:
- Residual income links book value and earnings to intrinsic value; it can cross-check FCFE valuations when cash flows are hard to pin down.
- If DCF implies high value but P/B is low, reconcile whether the DCF assumes ROE improvements not yet delivered or ignores balance-sheet risk.

Practical adjustments:
```python
def choose_pb_denominator(use_tangible: bool, reported_be: float, goodwill: float, intangibles: float) -> float:
    """Select book equity base (reported or tangible) consistently."""
    if not all(_num(v) for v in [reported_be, goodwill, intangibles]):
        raise ValueError("inputs must be numeric")
    if reported_be <= 0:
        raise ValueError("reported_be must be > 0")
    if use_tangible:
        be = reported_be - goodwill - intangibles
        if be <= 0:
            raise ValueError("tangible book equity <= 0; tangible P/B not meaningful")
        return be
    return reported_be
```

---

### Quality of Earnings Flags (Book Value Multiples)
⚠️ Book equity inflated/deflated by large one-time write-downs; ROE becomes temporarily distorted.  
⚠️ Repeated acquisitions create large goodwill; reported book equity may not represent economic capital.  
⚠️ OCI/AOCI volatility (rates/credit spreads) meaningfully affects book equity but peers treat it differently.  
⚠️ For banks, provisions are understated (optimistic loss recognition), boosting ROE and P/B unjustifiably.  
⚠️ Off-balance sheet risk exposures (guarantees, derivatives) not reflected in book equity comparability.  
✅ Green flag: P/B interpreted with sustainable ROE (through-cycle), capital adequacy, and asset quality metrics.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banks | Capital base drives earnings | Use P/B (or P/TBV) with ROE vs COE and capital ratios. |
| Insurers | Book equity sensitivity to rates | Use P/B with embedded value considerations; adjust for AOCI effects if material. |
| Asset-heavy industrials | Asset revaluations and depreciation policy | Consider tangible book; normalise ROA/ROE. |
| Tech/brands | Internally generated intangibles expensed | P/B often misleadingly high; prefer EV/Sales or P/E with quality controls. |

---

### Practical Comparison Tables

#### 1) P/B interpretation grid
| ROE vs COE | Expected P/B | Interpretation |
|---|---|---|
| ROE >> COE | > 1 | Sustainable value creation; franchise/moat or efficient capital use |
| ROE ≈ COE | ≈ 1 | Fair return on equity capital |
| ROE < COE | < 1 | Value destruction or risk under-recognised in earnings |

#### 2) Reported vs tangible book
| Denominator | Best for | Main risk |
|---|---|---|
| Reported book equity | Firms with limited goodwill/intangibles; consistent accounting | M&A and write-down distortions |
| Tangible book equity | Banks and acquisitive firms; comparability | Excludes franchise value embedded in goodwill/intangibles |

---

### Real-World Example
Scenario: A bank trades at 1.4x P/B, but current ROE is only slightly above estimated cost of equity. You compute tangible P/B and assess whether premium is supported by sustainable ROE or market expectations.

```python
market_cap = 12_000.0
be = 9_500.0
goodwill = 1_800.0
intangibles = 700.0

tbv = choose_pb_denominator(True, be, goodwill, intangibles)
pb_tbv = pb_ratio(market_cap, tbv)

net_income = 1_050.0
roe_tbv = roe(net_income, tbv)
ke = 0.11

print(f"P/TBV: {pb_tbv:.2f}x" if pb_tbv else "P/TBV: n/a")
print(f"ROE on TBV: {roe_tbv:.1%}" if roe_tbv is not None else "ROE: n/a")
print(pb_signal(roe_tbv, ke))
```

Interpretation: If ROE on tangible equity is not clearly above COE, a premium P/TBV likely reflects expected ROE improvement, lower perceived risk, or potential mispricing. Investigate asset quality, provisioning adequacy, and structural earnings power.
