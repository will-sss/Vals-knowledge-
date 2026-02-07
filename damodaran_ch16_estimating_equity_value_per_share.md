# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 16: Estimating Equity Value per Share (Equity Bridge, Dilution, Options, Non-Common Claims)

### Core Concept
After valuing the firm or equity, the final step is converting total equity value into **value per common share**, while correctly accounting for **debt-like claims**, **non-common equity claims**, and **dilutive securities** (options, convertibles, warrants). Practically, most per-share errors come from inconsistent bridges (EV ↔ equity), double-counting cash/debt-like items, or naive dilution handling.

See also: Chapter 15 (firm valuation and EV bridge), Chapter 14 (equity intrinsic models), Chapter 3 (recasting financial statements), Chapter 12 (terminal value dominates per-share value).

---

### Formula/Methodology

#### 1) Enterprise value to equity value (core bridge)
```text
Equity Value = Enterprise Value − Net Debt − Other Debt-like Claims + Non-operating Assets

Where:
Net Debt = Debt-like Claims − Cash (use excess cash concept where relevant)
Other debt-like claims = leases (if capitalised), pensions deficit, preferred, minorities, earn-outs, etc.
Non-operating assets = investments, surplus real estate, unconsolidated stakes
```
Practical rules:
- Define “debt-like” consistently with how you measured EV and operating cash flows.
- Separate **operating cash** (needed for working capital) from **excess cash** (non-operating asset).

#### 2) Equity value per share (basic)
```text
Value per Share = (Equity Value − Value of non-common claims) / (Diluted common shares)

Non-common claims can include:
- Preferred equity
- Employee options (if treated as a claim rather than share-count dilution)
- Convertibles (choose either “as-if converted” share count or claim value, not both)
```
Key discipline:
- Choose one method for dilutive securities: **Treasury Stock Method (TSM)** for share count *or* **option valuation as a claim** (e.g., Black–Scholes). Do not double count.

#### 3) Treasury Stock Method (options/warrants) — practical dilution
```text
Incremental Shares = Options Outstanding × (1 − Exercise Price / Share Price)

Diluted Shares = Basic Shares + Incremental Shares

Where:
Share Price should be the estimated intrinsic value per share (solve iteratively if needed)
Options that are out-of-the-money (Exercise >= Price) contribute ~0 incremental shares.
```
Practical note:
- TSM is a simplification; it assumes exercised options generate cash that repurchases shares at the prevailing price.

#### 4) Convertibles — consistent treatment
Two standard approaches:
```text
A) If treated as debt-like claim:
Equity Value = EV − (Debt including convertible) + Cash ...
Shares = basic shares (no as-converted)

B) If treated as equity (as-if converted):
Remove convertible debt from debt-like claims
Add converted shares to share count
```
Avoid mixing A and B.

#### 5) Employee options as a claim (option value approach)
For material option overhang, value options and subtract from equity value:
```text
Common Equity Value (to common shares) = Total Equity Value − Option Value − Preferred Value − Other non-common claims
Per-share = Common Equity Value / Basic Shares
```
Option value can be estimated with Black–Scholes for liquid/vanilla options; for private firms or complex vesting, use simplified approximations and disclose assumptions.

---

### Python Implementation
```python
from typing import Optional, Dict, List, Tuple


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def equity_bridge_from_ev(
    ev: float,
    debt_like: float,
    cash: float,
    other_claims: float = 0.0,
    non_operating_assets: float = 0.0,
) -> float:
    """Equity = EV - (Debt-like - Cash) - Other claims + Non-operating assets."""
    for name, v in {"ev": ev, "debt_like": debt_like, "cash": cash, "other_claims": other_claims, "non_operating_assets": non_operating_assets}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if debt_like < 0 or cash < 0 or other_claims < 0:
        raise ValueError("debt_like, cash, other_claims must be >= 0")
    return ev - (debt_like - cash) - other_claims + non_operating_assets


def value_per_share_basic(equity_value: float, basic_shares: float, non_common_claims_value: float = 0.0) -> float:
    """Per-share value using basic shares and deducting non-common claims."""
    for name, v in {"equity_value": equity_value, "basic_shares": basic_shares, "non_common_claims_value": non_common_claims_value}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if basic_shares <= 0:
        raise ValueError("basic_shares must be > 0")
    if non_common_claims_value < 0:
        raise ValueError("non_common_claims_value must be >= 0")
    return (equity_value - non_common_claims_value) / basic_shares


def tsm_incremental_shares(options: float, exercise_price: float, share_price: float) -> float:
    """Incremental shares under Treasury Stock Method."""
    for name, v in {"options": options, "exercise_price": exercise_price, "share_price": share_price}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if options < 0:
        raise ValueError("options must be >= 0")
    if exercise_price < 0:
        raise ValueError("exercise_price must be >= 0")
    if share_price <= 0:
        raise ValueError("share_price must be > 0")
    if exercise_price >= share_price:
        return 0.0
    return options * (1.0 - exercise_price / share_price)


def diluted_shares_tsm(basic_shares: float, option_tranches: List[Dict], share_price: float) -> float:
    """Compute diluted shares with multiple option tranches using TSM."""
    if not _num(basic_shares) or basic_shares <= 0:
        raise ValueError("basic_shares must be numeric and > 0")
    if not _num(share_price) or share_price <= 0:
        raise ValueError("share_price must be numeric and > 0")
    if not isinstance(option_tranches, list):
        raise ValueError("option_tranches must be a list")
    inc = 0.0
    for tr in option_tranches:
        opts = tr.get("options", 0.0)
        x = tr.get("exercise_price", 0.0)
        inc += tsm_incremental_shares(opts, x, share_price)
    return basic_shares + inc


def solve_per_share_with_tsm(
    equity_value_common: float,
    basic_shares: float,
    option_tranches: List[Dict],
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, float]:
    """
    Solve per-share value when dilution depends on share price (TSM uses share_price).
    Iterates: price -> diluted shares -> price until convergence.

    Returns:
        (price, diluted_shares)

    Raises:
        ValueError: if inputs invalid or fails to converge.
    """
    for name, v in {"equity_value_common": equity_value_common, "basic_shares": basic_shares}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if basic_shares <= 0:
        raise ValueError("basic_shares must be > 0")
    if equity_value_common <= 0:
        raise ValueError("equity_value_common must be > 0 for a meaningful per-share solve")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")

    price = equity_value_common / basic_shares  # start at undiluted
    for _ in range(max_iter):
        ds = diluted_shares_tsm(basic_shares, option_tranches, price)
        new_price = equity_value_common / ds
        if abs(new_price - price) < tol * max(1.0, abs(price)):
            return new_price, ds
        price = new_price
    raise ValueError("Failed to converge; check inputs or option tranches.")


# Example usage (all $m except shares)
EV = 8_500.0
debt_like = 2_900.0
cash = 450.0
other_claims = 300.0          # pension deficit, preferred, minorities, earn-outs, etc.
non_operating_assets = 250.0  # investments

equity_value = equity_bridge_from_ev(EV, debt_like, cash, other_claims, non_operating_assets)

# If preferred equity is a separate claim you want to deduct from common:
preferred_value = 180.0
equity_to_common = equity_value - preferred_value

basic_shares = 320.0  # million shares
option_tranches = [
    {"options": 12.0, "exercise_price": 15.0},
    {"options": 8.0, "exercise_price": 25.0},
]

price, diluted = solve_per_share_with_tsm(equity_to_common, basic_shares, option_tranches)

print(f"Enterprise Value: ${EV:,.0f}m")
print(f"Equity value: ${equity_value:,.0f}m")
print(f"Equity to common (after preferred): ${equity_to_common:,.0f}m")
print(f"Value per share (diluted, TSM): ${price:,.2f}")
print(f"Diluted shares: {diluted:,.1f}m")
```

---

### Valuation Impact
Why this matters:
- Valuation work fails at the finish line if the equity bridge is inconsistent or dilution is mishandled; these issues can shift per-share value materially, especially for high-growth firms with large option overhang.
- Correctly separating operating and non-operating items improves comparability and prevents overstated equity value (e.g., counting all cash as excess).
- Different claim priorities matter: preferred, convertibles, and earn-outs can have option-like payoffs; treating them as simple debt can misstate common equity value.

Impact on multiples:
- Per-share value links directly to P/E and implied equity multiples; if dilution is ignored, implied P/E can be understated and peer comparisons distorted.
- EV multiples require consistent EV construction; if EV includes lease debt, equity bridge and per-share must too.

Impact on DCF inputs:
- Capital structure assumptions (cash, debt-like items) affect the EV→equity bridge, not the operating FCFF itself.
- Option overhang affects per-share value but not necessarily enterprise value; treat separately and consistently.

Practical adjustments:
```python
def classify_cash(total_cash: float, operating_cash_needed: float) -> float:
    """Compute excess cash (non-operating asset) to add in equity bridge."""
    if not _num(total_cash) or not _num(operating_cash_needed):
        raise ValueError("inputs must be numeric")
    if total_cash < 0 or operating_cash_needed < 0:
        raise ValueError("inputs must be >= 0")
    return max(0.0, total_cash - operating_cash_needed)
```

---

### Quality of Earnings Flags (Per-Share and Claims)
⚠️ Cash is netted against debt without determining what portion is excess vs operating cash (overstates equity).  
⚠️ Lease liabilities/pension deficits ignored in net debt while EBIT/EBITDA is “lease-adjusted” (inconsistency).  
⚠️ Convertibles treated as debt in EV bridge but also included in diluted shares (double counting).  
⚠️ Options diluted using a market price inconsistent with intrinsic valuation (TSM misapplied).  
⚠️ Large earn-outs/contingent considerations ignored (option-like claims can be material).  
✅ Green flag: explicit claim stack, consistent EV bridge, and a documented approach to dilution (TSM or option value).

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Tech/high growth | Large option overhang | Use TSM iterative solve or option valuation; disclose assumptions. |
| Financials | Preferred/convertibles common | Treat non-common claims explicitly; be clear on conversion mechanics. |
| Real estate | Non-operating assets and JV stakes | Value and add non-operating assets separately; avoid burying them in EV. |
| Distressed | Priority claims dominate | Model claim stack carefully; equity may be out-of-the-money. |

---

### Practical Comparison Tables

#### 1) Dilution handling choices
| Method | How it works | When to use | Pitfall |
|---|---|---|---|
| TSM | Adds incremental shares based on intrinsic price | Vanilla options/warrants | Needs iterative solve if price unknown |
| Option value as claim | Value options (e.g., Black–Scholes) and subtract | Material option overhang, complex structures | Requires volatility/term assumptions |
| As-if converted | Add shares, remove convertible debt | In-the-money convertibles with likely conversion | Double counting if also treated as debt |

#### 2) EV → equity bridge checklist
| Item | Include where | Note |
|---|---|---|
| Debt-like claims | Subtract in bridge | Include leases/pensions if treated as debt-like |
| Excess cash | Add back | Determine operating cash vs excess |
| Non-operating investments | Add back | Value separately; avoid mixing with operating EV |
| Minority interests | Subtract (if EV includes 100% subs) | Ensure consolidation logic is consistent |
| Preferred | Subtract from equity to get common | Preferred can have option-like features |

---

### Real-World Example
Scenario: You have a DCF-derived enterprise value and need value per share for a venture-backed company with preferred shares and options. You compute equity value, subtract preferred, and solve diluted shares via TSM.

```python
EV = 8_500.0
debt_like = 2_900.0
cash_total = 450.0
operating_cash_needed = 150.0
excess_cash = classify_cash(cash_total, operating_cash_needed)

other_claims = 300.0
non_operating_assets = 250.0 + excess_cash  # add excess cash explicitly

equity_value = equity_bridge_from_ev(EV, debt_like, cash=0.0, other_claims=other_claims, non_operating_assets=non_operating_assets)

preferred = 180.0
equity_to_common = equity_value - preferred

basic_shares = 320.0
options = [{"options": 12.0, "exercise_price": 15.0}, {"options": 8.0, "exercise_price": 25.0}]

price, diluted = solve_per_share_with_tsm(equity_to_common, basic_shares, options)

print(f"Equity to common: ${equity_to_common:,.0f}m")
print(f"Diluted shares: {diluted:,.1f}m")
print(f"Per-share value: ${price:,.2f}")
```

Interpretation: The claim stack (preferred + other claims) and dilution mechanics can change per-share value materially even when enterprise value is stable. Make the bridge auditable and avoid double-counting dilutive securities.
