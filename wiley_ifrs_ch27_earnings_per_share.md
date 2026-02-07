# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 27: Earnings Per Share (IAS 33, basic vs diluted EPS, instruments, valuation per-share)

### Core Concept
IAS 33 standardises how entities compute and present earnings per share (EPS), including basic EPS and diluted EPS. For valuation, EPS underpins P/E multiples, equity story comparisons, and per-share intrinsic value cross-checks. Misinterpreting dilution (options, convertibles, contingently issuable shares) can materially misstate per-share value and create false comparability across companies.

### Formula/Methodology

#### 1) Basic EPS
```text
Basic EPS = (Profit attributable to ordinary equity holders of the parent - preference dividends) / Weighted average ordinary shares outstanding

Where:
- Profit attributable: profit after tax attributable to ordinary shareholders
- Preference dividends: those classified as distributions to preference shareholders (if applicable)
- Weighted average shares: time-weighted shares outstanding during the period, adjusted for share issues/buybacks
```

#### 2) Weighted average shares (WASO) mechanics
```text
WASO = Σ (Shares outstanding during subperiod × (Days in subperiod / Total days in period))

Key adjustments:
- Share splits/bonus issues: restate prior periods and apply retroactively
- Rights issues: apply a theoretical ex-rights factor to restate prior periods
```

#### 3) Diluted EPS (overview)
```text
Diluted EPS includes the effect of dilutive potential ordinary shares.

Diluted EPS = Adjusted profit / Adjusted weighted average shares

Where:
Adjusted profit = profit attributable to equity holders + after-tax effects of assumed conversions (e.g., convertible interest net of tax)
Adjusted shares = WASO + incremental shares from assumed conversion/exercise
```

#### 4) Options and warrants: treasury stock method (TSM)
```text
Incremental shares = Shares issuable on exercise - Shares that could be bought back with exercise proceeds

Shares bought back = (Exercise price × Number of options) / Average market price

Dilutive only if average market price > exercise price (i.e., “in the money”).
```

#### 5) Convertible debt/equity: if-converted method
```text
Adjusted profit = Profit attributable + Interest expense × (1 - tax rate) (for convertible debt)
Adjusted shares = WASO + Shares issued on conversion

Dilutive only if diluted EPS < basic EPS (or, equivalently, inclusion reduces EPS).
```

#### 6) Contingently issuable shares and performance conditions
```text
Include contingently issuable shares in basic EPS only when conditions are satisfied.
In diluted EPS, include if conditions would be met based on current period performance at the reporting date.
```

#### 7) Anti-dilution rule
```text
Exclude potential ordinary shares if their inclusion increases EPS or reduces loss per share.
```

---

### Practical Application for Valuation Analysts

#### 1) Use diluted shares for per-share valuation checks
When comparing:
- Intrinsic value per share (DCF equity value / shares)
- Market price per share
- Target price per share

Use a “fully diluted” share count consistent with the instrument mix and market practice:
- Include in-the-money options/warrants (TSM)
- Include convertibles if dilutive or if valuation assumes conversion
- Include contingently issuable shares if probability/conditions justify inclusion

#### 2) Align per-share outputs to instrument treatment in your EV bridge
If your enterprise-to-equity bridge includes:
- Convertible debt treated as debt: shares should exclude conversion unless you model conversion separately.
- Convertible treated as equity (assume conversion): add conversion shares and remove the debt (or subtract its fair value appropriately).

Consistency rule:
- If you remove a claim from net debt (treat as equity-like), include its shares in the denominator.

#### 3) EPS vs “earnings power” in multiple-based valuation
EPS can be noisy due to:
- one-off items (impairment, fair value gains/losses, restructuring)
- tax anomalies
- discontinued operations

Actions:
- Use “normalised” earnings for P/E, but keep IAS 33 share count mechanics for comparability.
- Track both basic and diluted, and explain differences (dilution risk).

#### 4) Rights issues and comparability
Rights issues require a restatement factor that can change historical EPS without changing economics.
Actions:
- When doing trend analysis, ensure prior EPS is restated; otherwise growth rates can be wrong.

---

### Python Implementation
```python
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def weighted_average_shares(share_periods: List[Tuple[float, float]]) -> float:
    """
    Compute weighted average shares outstanding (WASO).

    Args:
        share_periods: list of (shares_outstanding, fraction_of_period)
            fraction_of_period should sum to ~1.0 (e.g., days/total_days)

    Returns:
        float: WASO

    Raises:
        ValueError: invalid fractions or shares.
    """
    if not share_periods:
        raise ValueError("share_periods must not be empty.")
    total_frac = 0.0
    waso = 0.0
    for i, (sh, frac) in enumerate(share_periods):
        s = _num(sh, f"shares[{i}]")
        f = _num(frac, f"fraction[{i}]")
        if s < 0:
            raise ValueError("shares must be >= 0.")
        if f < 0:
            raise ValueError("fraction_of_period must be >= 0.")
        waso += s * f
        total_frac += f
    # Allow small rounding error
    if not (0.98 <= total_frac <= 1.02):
        raise ValueError(f"fraction_of_period must sum to ~1.0; got {total_frac:.4f}")
    return waso

def basic_eps(profit_attrib: float, pref_dividends: float, waso: float) -> float:
    """
    Compute basic EPS.

    Args:
        profit_attrib: profit attributable to ordinary equity holders (currency)
        pref_dividends: preference dividends (currency)
        waso: weighted average shares outstanding

    Returns:
        float: basic EPS (currency per share)

    Raises:
        ValueError: waso <= 0.
    """
    p = _num(profit_attrib, "profit_attrib")
    d = _num(pref_dividends, "pref_dividends")
    w = _num(waso, "waso")
    if w <= 0:
        raise ValueError("waso must be > 0.")
    return (p - d) / w

def treasury_stock_incremental_shares(
    n_options: float,
    exercise_price: float,
    avg_market_price: float
) -> float:
    """
    Treasury stock method incremental shares for options/warrants.

    Args:
        n_options: number of options/warrants
        exercise_price: exercise price per share
        avg_market_price: average market price per share

    Returns:
        float: incremental shares (>= 0)

    Raises:
        ValueError: invalid inputs.
    """
    n = _num(n_options, "n_options")
    k = _num(exercise_price, "exercise_price")
    p = _num(avg_market_price, "avg_market_price")
    if n < 0:
        raise ValueError("n_options must be >= 0.")
    if k < 0:
        raise ValueError("exercise_price must be >= 0.")
    if p <= 0:
        raise ValueError("avg_market_price must be > 0.")
    # Not dilutive if out of the money
    if p <= k or n == 0:
        return 0.0
    proceeds = n * k
    buyback = proceeds / p
    inc = n - buyback
    return max(0.0, inc)

def if_converted_adjustments(
    interest_expense: float,
    tax_rate: float,
    conversion_shares: float
) -> Tuple[float, float]:
    """
    If-converted method adjustments for convertible debt.

    Args:
        interest_expense: annual interest expense on convertible (currency)
        tax_rate: tax rate (0-1)
        conversion_shares: shares issued on conversion

    Returns:
        (profit_adjustment, share_adjustment)

    Raises:
        ValueError: invalid inputs.
    """
    i = _num(interest_expense, "interest_expense")
    tr = _num(tax_rate, "tax_rate")
    sh = _num(conversion_shares, "conversion_shares")
    if not (0.0 <= tr <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    if sh < 0:
        raise ValueError("conversion_shares must be >= 0.")
    profit_adj = i * (1.0 - tr)
    return profit_adj, sh

def diluted_eps(
    profit_attrib: float,
    pref_dividends: float,
    waso: float,
    option_incremental_shares: float = 0.0,
    conv_profit_adj: float = 0.0,
    conv_share_adj: float = 0.0
) -> float:
    """
    Compute diluted EPS with options (TSM) and convertibles (if-converted).

    Args:
        profit_attrib: profit attributable to ordinary equity holders
        pref_dividends: preference dividends
        waso: weighted average shares
        option_incremental_shares: incremental shares from options/warrants (TSM)
        conv_profit_adj: after-tax interest add-back (if-converted)
        conv_share_adj: shares from conversion

    Returns:
        float: diluted EPS

    Raises:
        ValueError: denominator <= 0.
    """
    p = _num(profit_attrib, "profit_attrib")
    d = _num(pref_dividends, "pref_dividends")
    w = _num(waso, "waso")
    opt = _num(option_incremental_shares, "option_incremental_shares")
    pa = _num(conv_profit_adj, "conv_profit_adj")
    sa = _num(conv_share_adj, "conv_share_adj")

    denom = w + opt + sa
    if denom <= 0:
        raise ValueError("Diluted share count must be > 0.")
    num = (p - d) + pa
    return num / denom

def is_dilutive(basic: float, diluted: float, profit_attrib: float) -> bool:
    """
    IAS 33 anti-dilution check (simplified).

    Args:
        basic: basic EPS
        diluted: diluted EPS
        profit_attrib: profit attributable (sign determines profit vs loss context)

    Returns:
        bool: True if diluted EPS is <= basic EPS for profit periods; for loss periods, inclusion is anti-dilutive.
    """
    b = _num(basic, "basic_eps")
    dl = _num(diluted, "diluted_eps")
    p = _num(profit_attrib, "profit_attrib")
    if p >= 0:
        return dl <= b
    # Loss period: any potential shares that reduce loss per share are anti-dilutive
    return False

# Example usage
periods = [
    (100_000_000, 0.50),  # 100m shares for first half
    (110_000_000, 0.50)   # 110m shares for second half (issuance)
]
waso = weighted_average_shares(periods)
b_eps = basic_eps(profit_attrib=220_000_000, pref_dividends=0, waso=waso)

opt_inc = treasury_stock_incremental_shares(n_options=15_000_000, exercise_price=5.0, avg_market_price=8.0)
conv_profit_adj, conv_share_adj = if_converted_adjustments(interest_expense=18_000_000, tax_rate=0.25, conversion_shares=20_000_000)

d_eps = diluted_eps(
    profit_attrib=220_000_000,
    pref_dividends=0,
    waso=waso,
    option_incremental_shares=opt_inc,
    conv_profit_adj=conv_profit_adj,
    conv_share_adj=conv_share_adj
)

print(f"WASO: {waso/1e6:.1f}m shares")
print(f"Basic EPS: ${b_eps:.3f}")
print(f"Diluted EPS: ${d_eps:.3f}")
print("Dilutive?", is_dilutive(b_eps, d_eps, 220_000_000))
```

---

### Valuation Impact
Why this matters:
- Per-share valuation is sensitive to dilution; ignoring options/convertibles can overstate value per share.
- P/E and PEG comparisons depend on consistent EPS definitions (basic vs diluted) and consistent share counts.
- Deal models and investment memos need a defensible fully diluted share count that matches the equity bridge.

Impact on multiples:
- P/E: always confirm whether the market convention uses diluted EPS (often yes).
- EV/Equity value per share: share count differences can create false “cheap/expensive” conclusions.

Impact on DCF inputs:
- DCF produces equity value; dividing by the wrong share count distorts target price and investment conclusions.
- Dilution can also affect cost of equity assumptions (risk profile) if convertibles behave like equity.

Comparability issues:
- Different equity compensation intensity (tech vs industrials) leads to different dilution profiles.
- Companies may report adjusted EPS excluding some effects; keep the IAS 33 share count discipline.

Practical adjustments:
```python
def fully_diluted_shares(waso: float, option_inc: float, conv_shares: float, contingent_shares: float = 0.0) -> float:
    """
    Build a practical fully diluted share count.

    Raises:
        ValueError: if any component is negative.
    """
    w = _num(waso, "waso")
    o = _num(option_inc, "option_inc")
    c = _num(conv_shares, "conv_shares")
    g = _num(contingent_shares, "contingent_shares")
    if min(w, o, c, g) < 0:
        raise ValueError("share components must be >= 0.")
    return w + o + c + g
```

---

### Quality of Earnings Flags
⚠️ Large gap between basic and diluted EPS without a clear reconciliation (high dilution risk).  
⚠️ Significant option overhang with low exercise prices; future dilution likely even if currently anti-dilutive due to low market price.  
⚠️ Convertibles structured to avoid immediate dilution but economically equity-like (hidden dilution).  
⚠️ Contingently issuable shares excluded despite performance conditions being close to met.  
✅ Clear disclosure of potential ordinary shares and consistent diluted EPS calculation across periods.

---

### Sector-Specific Considerations

| Sector | Key EPS issue | Typical valuation handling |
|---|---|---|
| Technology | high SBC and option overhang | use fully diluted shares; assess SBC normalisation in earnings |
| Financials | convertibles and AT1-like instruments | check classification and whether instruments convert/write down; align share count and equity claims |
| Industrials | rights issues, buybacks | ensure restatements and time weighting; verify buyback timing |
| Biotech / early-stage | losses and anti-dilution | potential shares often anti-dilutive in IAS 33 but still economically relevant for target price scenarios |

---

### Real-World Example
Scenario: Equity value from DCF is $6.0bn. Company has WASO 120m, options incremental 8m, convertibles 15m shares. Compute value per share.

```python
equity_value = 6_000_000_000
fd_shares = fully_diluted_shares(120_000_000, 8_000_000, 15_000_000)
value_per_share = equity_value / fd_shares
print(f"Fully diluted shares: {fd_shares/1e6:.1f}m")
print(f"DCF value per share: ${value_per_share:.2f}")
```

Interpretation: Per-share value should be stated on a fully diluted basis if the market is likely to price in dilution; otherwise, include a clear bridge between basic and diluted outcomes.

See also: Chapter 16 (shareholders’ equity) for equity instruments; Chapter 17 (share-based payment) for SBC drivers of dilution; Chapter 24 (financial instruments) for convertibles classification; Chapter 25 (fair value) for option valuation inputs.
