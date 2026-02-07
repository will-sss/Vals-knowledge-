# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 16: Shareholders’ Equity (IAS 32/IFRS 9 linkages, equity vs liability, EPS dilution, valuation bridges)

### Core Concept
This chapter covers the classification and presentation of shareholders’ equity and equity-related instruments, including the critical boundary between equity and financial liabilities. For valuation, equity classification affects enterprise value bridges (net debt vs equity), earnings and EPS (including dilution), and comparability across companies that use different funding structures (convertibles, redeemable shares, preference shares, puttable instruments).

### Formula/Methodology

#### 1) Enterprise value bridge (equity to EV)
```text
Enterprise Value (EV) = Market Cap + Net Debt + NCI + Preferred Equity (if debt-like) + Other debt-like items - Non-operating assets
Equity Value = EV - Net Debt - NCI - Preferred Equity (if treated as debt-like) - Other adjustments + Non-operating assets
```

#### 2) Equity vs liability classification (IAS 32 practical decision rule)
```text
Financial liability if there is a contractual obligation to deliver cash or another financial asset.
Equity instrument if there is NO contractual obligation to deliver cash/another financial asset
and settlement is in the entity’s own equity instruments under “fixed-for-fixed” conditions (for many derivatives).

Key implications:
- Redeemable or mandatorily repayable instruments are typically liabilities.
- Preference shares can be equity or liability depending on redemption and dividend obligations.
```

#### 3) Preference shares classification (common patterns)
```text
If dividends are discretionary and there is no mandatory redemption -> more likely equity.
If dividends are mandatory (or cumulative with obligation) and/or redemption is mandatory -> more likely liability.
```

#### 4) Diluted EPS (conceptual mechanics)
```text
Basic EPS = Profit attributable to ordinary equity holders / Weighted average ordinary shares outstanding

Diluted EPS adjusts:
- Numerator: add back after-tax interest/dividends on dilutive potential ordinary shares (if they would be avoided on conversion)
- Denominator: add shares from conversion/exercise of dilutive instruments (options, warrants, convertibles)

Anti-dilution rule: exclude instruments that increase EPS (i.e., are anti-dilutive)
```

#### 5) Treasury shares / share buybacks (presentation)
```text
Treasury shares reduce equity (deducted from equity, not an asset)
Buyback does not affect profit, but affects EPS via lower share count going forward
```

---

### Practical Application for Valuation Analysts

#### 1) Build a “capital stack map” before valuing
For each instrument, identify:
- Legal form (ordinary shares, prefs, convertibles, warrants, put options)
- Accounting classification (equity vs liability)
- Economic classification for valuation (debt-like vs equity-like)
- Cash obligations (mandatory coupons, redemption)
- Conversion terms and dilution triggers

Analyst rule:
- EV should include **all debt-like claims**, whether accounted as liabilities or “equity” in financial statements, if they behave economically like debt.

#### 2) Normalise net debt and EV for hybrid instruments
Common hybrid instruments:
- Redeemable preference shares
- Perpetual notes with discretionary coupons
- Convertibles
- Puttable NCI
Valuation actions:
- Treat mandatory redemption/coupon instruments as debt-like (add to net debt or EV adjustments).
- For convertibles: use either (i) “if-converted” diluted equity approach, or (ii) value as debt + embedded option (advanced), depending on materiality.

#### 3) EPS and per-share value: always consider dilution
If you compute equity value per share:
- Use **diluted share count** if options/convertibles are in-the-money or likely to convert.
- For early-stage or high-growth companies, option overhang can materially reduce per-share value.

#### 4) Buybacks and capital returns: valuation interpretation
Buybacks:
- Increase per-share metrics if purchased below intrinsic value.
- Destroy value if purchased above intrinsic value or funded with expensive debt.
Dividends:
- Generally value-neutral in perfect markets, but signal maturity and capital allocation priorities.
For DCF:
- Value is driven by free cash flow; distribution policy affects per-share mechanics, not enterprise value (except via financing choices).

#### 5) Presentation traps
- “Equity” line can include instruments with debt-like features (prefs, puttable instruments).
- OCI reserves can be large (e.g., FVOCI), affecting book value but not immediate cash earnings.
- Share-based payment reserve affects equity but is non-cash; still implies dilution.

---

### Python Implementation
```python
from typing import Any, Dict, Optional
import numpy as np

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def enterprise_value_from_cap_stack(
    market_cap: float,
    net_debt: float,
    nci: float = 0.0,
    preferred_equity_debt_like: float = 0.0,
    other_debt_like: float = 0.0,
    non_operating_assets: float = 0.0
) -> float:
    """
    EV = Market cap + Net debt + NCI + debt-like preferred + other debt-like - non-operating assets

    Raises:
        ValueError: invalid inputs.
    """
    mc = _num(market_cap, "market_cap")
    nd = _num(net_debt, "net_debt")
    n = _num(nci, "nci")
    pe = _num(preferred_equity_debt_like, "preferred_equity_debt_like")
    od = _num(other_debt_like, "other_debt_like")
    noa = _num(non_operating_assets, "non_operating_assets")
    return mc + nd + n + pe + od - noa

def equity_value_from_ev(
    ev: float,
    net_debt: float,
    nci: float = 0.0,
    preferred_equity_debt_like: float = 0.0,
    other_debt_like: float = 0.0,
    non_operating_assets: float = 0.0
) -> float:
    """
    Equity value = EV - Net debt - NCI - debt-like preferred - other debt-like + non-operating assets

    Raises:
        ValueError: invalid inputs.
    """
    E = _num(ev, "ev")
    nd = _num(net_debt, "net_debt")
    n = _num(nci, "nci")
    pe = _num(preferred_equity_debt_like, "preferred_equity_debt_like")
    od = _num(other_debt_like, "other_debt_like")
    noa = _num(non_operating_assets, "non_operating_assets")
    return E - nd - n - pe - od + noa

def basic_eps(profit_attrib_to_ordinary: float, weighted_avg_shares: float) -> float:
    """
    Basic EPS.

    Raises:
        ValueError: shares <= 0.
    """
    p = _num(profit_attrib_to_ordinary, "profit_attrib_to_ordinary")
    s = _num(weighted_avg_shares, "weighted_avg_shares")
    if s <= 0:
        raise ValueError("weighted_avg_shares must be > 0.")
    return p / s

def diluted_eps(
    profit_attrib_to_ordinary: float,
    weighted_avg_shares: float,
    add_back_after_tax_interest: float = 0.0,
    incremental_shares: float = 0.0
) -> float:
    """
    Simplified diluted EPS:
      Diluted EPS = (profit + add_back_interest) / (shares + incremental_shares)

    Notes:
      - Apply anti-dilution screen externally (exclude if diluted EPS > basic EPS).

    Raises:
        ValueError: invalid shares.
    """
    p = _num(profit_attrib_to_ordinary, "profit_attrib_to_ordinary")
    s = _num(weighted_avg_shares, "weighted_avg_shares")
    ab = _num(add_back_after_tax_interest, "add_back_after_tax_interest")
    inc = _num(incremental_shares, "incremental_shares")
    if s <= 0:
        raise ValueError("weighted_avg_shares must be > 0.")
    if inc < 0:
        raise ValueError("incremental_shares must be >= 0.")
    denom = s + inc
    if denom <= 0:
        raise ValueError("diluted share count must be > 0.")
    return (p + ab) / denom

def if_converted_incremental_shares(conversion_shares: float, current_shares: float) -> float:
    """
    Helper for if-converted method: incremental shares from conversion.

    Raises:
        ValueError: invalid inputs.
    """
    conv = _num(conversion_shares, "conversion_shares")
    cur = _num(current_shares, "current_shares")
    if conv < 0 or cur <= 0:
        raise ValueError("conversion_shares must be >= 0 and current_shares must be > 0.")
    return conv

# Example usage
ev = enterprise_value_from_cap_stack(
    market_cap=4_800,
    net_debt=1_200,
    nci=150,
    preferred_equity_debt_like=300,
    other_debt_like=50,
    non_operating_assets=100
)
print(f"EV: ${ev:.0f}m")

eqv = equity_value_from_ev(
    ev=ev,
    net_debt=1_200,
    nci=150,
    preferred_equity_debt_like=300,
    other_debt_like=50,
    non_operating_assets=100
)
print(f"Equity value: ${eqv:.0f}m")

beps = basic_eps(profit_attrib_to_ordinary=420, weighted_avg_shares=1_050)
deps = diluted_eps(
    profit_attrib_to_ordinary=420,
    weighted_avg_shares=1_050,
    add_back_after_tax_interest=18,
    incremental_shares=90
)
# Anti-dilution screen
if deps > beps:
    deps = beps
print(f"Basic EPS: ${beps:.3f}")
print(f"Diluted EPS: ${deps:.3f}")
```

---

### Valuation Impact
Why this matters:
- Equity vs liability classification affects EV and leverage metrics. Debt-like instruments booked in equity can cause underestimation of net debt and overstatement of equity value.
- Dilution from options/convertibles materially impacts equity value per share; ignoring it leads to overvaluation on a per-share basis.
- Buybacks change per-share value mechanics and can meaningfully affect implied multiples if not aligned with intrinsic value.

Impact on multiples:
- EV/EBITDA and EV/EBIT: ensure numerator includes debt-like preference shares and other hybrid claims.
- P/E and per-share metrics: use diluted share counts where dilution is economically relevant.

Impact on DCF inputs:
- Financing choices affect WACC (capital structure) and free cash flow to equity vs firm.
- Equity-linked compensation increases dilution; treat as a claim on value (use diluted shares or option valuation where material).

Practical adjustments:
```python
def treat_redeemable_preference_shares_as_debt(net_debt_reported: float, redeemable_prefs: float) -> float:
    """
    Add redeemable preference shares to net debt for EV bridge comparability.

    Raises:
        ValueError: prefs negative.
    """
    nd = _num(net_debt_reported, "net_debt_reported")
    rp = _num(redeemable_prefs, "redeemable_prefs")
    if rp < 0:
        raise ValueError("redeemable_prefs must be >= 0.")
    return nd + rp
```

---

### Quality of Earnings Flags
⚠️ “Equity” instruments with mandatory coupons/redemption presented as equity (understates leverage).  
⚠️ Large option overhang with limited diluted EPS disclosure or aggressive “anti-dilution” assumptions.  
⚠️ Frequent buybacks funded by short-term borrowing in a weak cash flow profile.  
⚠️ Significant OCI reserves masking underlying volatility (e.g., FVOCI swings).  
✅ Clear note disclosures: instrument terms, redemption features, conversion prices, and fully diluted share reconciliation.

---

### Sector-Specific Considerations

| Sector | Key equity topic | Typical valuation handling |
|---|---|---|
| Banks/Insurers | Additional tier instruments, hybrids | Treat AT1-like instruments carefully (often debt-like with equity-like loss absorption); align with regulatory capital view |
| High-growth Tech | Options and convertibles | Use diluted shares; consider option valuation if material; watch SBC dilution |
| Real estate / REIT | Preference shares and NCI | Include prefs in EV if debt-like; ensure NCI included with consolidated EBITDA |
| Infrastructure | Structured equity and puttable interests | Identify put options/guarantees; treat debt-like claims in EV bridge |

---

### Real-World Example
Scenario: Company has $4.0bn market cap and $1.0bn net debt. It also has $0.4bn redeemable preference shares (accounted within equity). You are computing EV/EBITDA.

```python
ev = enterprise_value_from_cap_stack(
    market_cap=4_000,
    net_debt=1_000,
    preferred_equity_debt_like=400,
    nci=0,
    other_debt_like=0,
    non_operating_assets=0
)
print(f"EV including prefs: ${ev:.0f}m")

adj_net_debt = treat_redeemable_preference_shares_as_debt(1_000, 400)
print(f"Adjusted net debt: ${adj_net_debt:.0f}m")
```

Interpretation: Treating redeemable prefs as debt-like prevents underestimating EV and overstating equity value.

See also: Chapter 27 (EPS) for detailed per-share mechanics; Chapter 17 (share-based payment) for dilution and reserves; Chapter 24 (financial instruments) for classification and measurement linkages; Chapter 33 (capital structure) in valuation texts for broader capital policy context.
