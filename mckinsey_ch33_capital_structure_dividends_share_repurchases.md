# McKinsey Valuation - Key Concepts for Valuation

## Chapter 33: Capital Structure, Dividends, and Share Repurchases (Value Neutrality, Tax Shields, Payout Policy, Buyback Economics)

### Core Concept
Capital structure and payout policy affect valuation primarily through taxes, financial distress risk, agency costs, and capital allocation discipline—not because “debt is cheaper” in a mechanical way. The practical objective is to evaluate whether changing leverage or returning cash (dividends/buybacks) increases intrinsic value after considering tax shields, higher required returns, loss of flexibility, and distress costs. Separately, buybacks create value only when shares are repurchased below intrinsic value per share and without introducing unacceptable risk.

See also: Chapter 15 on cost of capital (WACC and leverage); Chapter 16 on enterprise value to equity value (claims and excess cash); Chapter 20 on taxes (tax shield mechanics); Chapter 17 on analysing results (sensitivities to leverage and payout).

---

### Core Concept: Distinguish “Value” from “Price” and “Distribution” from “Creation”
- **Dividends and buybacks distribute value**; they do not create value unless they improve capital allocation (avoid wasteful reinvestment) or reduce agency costs.
- **Capital structure can create value** if the tax benefits of debt exceed the expected present value of financial distress and other leverage-related costs.
- **Repurchases create value per remaining share** only if repurchase price < intrinsic value and financing does not introduce excessive risk.

---

### Formula/Methodology

#### 1) Enterprise value is primarily operating-driven in FCFF/WACC
```text
Enterprise Value ≈ PV(FCFF discounted at WACC)
```

#### 2) Interest tax shield (APV framing)
```text
Value from Debt (tax shield) ≈ PV(Interest × Tax Rate)
APV = Unlevered Value + PV(Tax Shields) − PV(Distress Costs)
```

#### 3) WACC with taxes (target leverage approach)
```text
WACC = (E/V) × Re + (D/V) × Rd × (1 − Tc)
```
Where:
- E = market value of equity
- D = market value of debt
- V = E + D
- Re = cost of equity
- Rd = cost of debt
- Tc = marginal tax rate for interest deductibility

#### 4) Levered beta (for Re under leverage)
```text
Beta_L = Beta_U × (1 + (1 − Tc) × D/E)
```
Where:
- Beta_U = unlevered beta
- D/E uses market values

Cost of equity (CAPM):
```text
Re = Rf + Beta_L × ERP
```

#### 5) Dividend vs buyback economics (per-share value identity)
Dividends reduce equity value by the cash paid (ex-div), all else equal:
```text
Equity Value_after_div = Equity Value_before_div − Dividend Cash
```
Buybacks reduce equity value by cash spent and reduce shares by cash/price; value per share increases only if buyback price < intrinsic value per share.

---

### Python Implementation
```python
from typing import Optional, Dict

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def levered_beta(beta_unlevered: float, debt: float, equity: float, tax_rate: float) -> Optional[float]:
    """
    Beta_L = Beta_U * (1 + (1 - Tc) * D/E). Returns None if equity <= 0.
    """
    if not _num(beta_unlevered) or beta_unlevered < 0:
        raise ValueError("beta_unlevered must be numeric and >= 0")
    if not _num(debt) or debt < 0:
        raise ValueError("debt must be numeric and >= 0")
    if not _num(equity):
        raise ValueError("equity must be numeric")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if equity <= 0:
        return None
    return beta_unlevered * (1.0 + (1.0 - tax_rate) * (debt / equity))


def cost_of_equity(rf: float, beta_l: float, erp: float) -> float:
    """Re = Rf + beta * ERP."""
    for name, v in {"rf": rf, "beta_l": beta_l, "erp": erp}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return rf + beta_l * erp


def wacc(equity: float, debt: float, re: float, rd: float, tax_rate: float) -> Optional[float]:
    """WACC = (E/V)*Re + (D/V)*Rd*(1-Tc). Returns None if V <= 0."""
    for name, v in {"equity": equity, "debt": debt, "re": re, "rd": rd}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if equity < 0 or debt < 0:
        raise ValueError("equity and debt must be >= 0")
    if not _num(tax_rate) or not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    v = equity + debt
    if v <= 0:
        return None
    return (equity / v) * re + (debt / v) * rd * (1.0 - tax_rate)


def pv_level_tax_shield(debt: float, rd: float, tax_rate: float, discount_rate: float) -> float:
    """
    Approx PV of a perpetual interest tax shield for constant debt:
    PV = (Debt * Rd * Tc) / discount_rate
    Discount_rate depends on the risk of the shield (often Rd or unlevered rate as a simplification).
    """
    for name, v in {"debt": debt, "rd": rd, "tax_rate": tax_rate, "discount_rate": discount_rate}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if debt < 0:
        raise ValueError("debt must be >= 0")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if discount_rate <= 0:
        raise ValueError("discount_rate must be > 0")
    return (debt * rd * tax_rate) / discount_rate


def dividend_effect(equity_value_before: float, dividend_cash: float) -> float:
    """Equity after dividend = before - cash dividend."""
    if not _num(equity_value_before) or equity_value_before < 0:
        raise ValueError("equity_value_before must be numeric and >= 0")
    if not _num(dividend_cash) or dividend_cash < 0:
        raise ValueError("dividend_cash must be numeric and >= 0")
    if dividend_cash > equity_value_before:
        raise ValueError("dividend_cash cannot exceed equity value in this simple identity")
    return equity_value_before - dividend_cash


def buyback_value_per_share(equity_value_before: float,
                            cash_spent: float,
                            shares_outstanding: float,
                            buyback_price: float) -> Dict[str, float]:
    """
    Compute post-buyback equity value and value per share, assuming cash spent reduces equity value.

    This shows the mechanical identity; value per share increases only if buyback_price < intrinsic value per share.
    """
    for name, v in {
        "equity_value_before": equity_value_before,
        "cash_spent": cash_spent,
        "shares_outstanding": shares_outstanding,
        "buyback_price": buyback_price
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if equity_value_before < 0 or cash_spent < 0 or shares_outstanding <= 0 or buyback_price <= 0:
        raise ValueError("invalid inputs: equity >=0, cash_spent>=0, shares>0, price>0")
    if cash_spent > equity_value_before:
        raise ValueError("cash_spent cannot exceed equity_value_before in this simple identity")

    shares_repur = cash_spent / buyback_price
    if shares_repur >= shares_outstanding:
        raise ValueError("buyback spends too much relative to shares; would retire all shares")
    equity_after = equity_value_before - cash_spent
    shares_after = shares_outstanding - shares_repur
    vps_before = equity_value_before / shares_outstanding
    vps_after = equity_after / shares_after
    return {
        "shares_repur": shares_repur,
        "equity_after": equity_after,
        "shares_after": shares_after,
        "value_per_share_before": vps_before,
        "value_per_share_after": vps_after
    }


def buyback_intrinsic_test(intrinsic_value_per_share: float, buyback_price: float) -> bool:
    """True if buyback is value-accretive per share (price below intrinsic)."""
    if not _num(intrinsic_value_per_share) or intrinsic_value_per_share <= 0:
        raise ValueError("intrinsic_value_per_share must be > 0")
    if not _num(buyback_price) or buyback_price <= 0:
        raise ValueError("buyback_price must be > 0")
    return buyback_price < intrinsic_value_per_share


# Example usage
beta_u = 0.95
rf = 0.04
erp = 0.05
tax = 0.24
rd = 0.06

equity = 8_000_000_000
debt = 2_000_000_000

beta_l = levered_beta(beta_u, debt, equity, tax) or 0.0
re = cost_of_equity(rf, beta_l, erp)
w = wacc(equity, debt, re, rd, tax) or 0.0

print(f"Levered beta: {beta_l:.2f}")
print(f"Cost of equity: {re:.1%}")
print(f"WACC: {w:.1%}")

# Buyback (mechanical identity)
intrinsic_vps = 52.0
buyback_price = 40.0
print("Buyback below intrinsic?", buyback_intrinsic_test(intrinsic_vps, buyback_price))

buy = buyback_value_per_share(
    equity_value_before=equity,
    cash_spent=800_000_000,
    shares_outstanding=1_000_000_000,
    buyback_price=buyback_price
)
print(f"Value/share before: ${buy['value_per_share_before']:.2f}")
print(f"Value/share after:  ${buy['value_per_share_after']:.2f}")
```

---

### Valuation Impact
Why this matters:
- Optimal leverage is a trade-off: tax shields can increase value, but distress risk and reduced flexibility can destroy value, especially for cyclical and volatile businesses.
- Payout policy affects value indirectly via capital allocation discipline; intrinsic value improves when cash is returned instead of invested at sub-WACC returns.
- Buybacks are economically equivalent to dividends if done at fair value; they create value only when repurchasing below intrinsic value and without forcing costly financing.

Impact on multiples:
- Higher leverage can increase ROE and EPS mechanically, but risk rises and can compress valuation multiples.
- EV-based multiples (EV/EBITDA) should be compared using consistent debt-like treatment (leases, pensions) and sustainable earnings.

Impact on DCF inputs:
- WACC must reflect target leverage consistent with business risk and long-term policy.
- If modelling APV, keep unlevered cash flows and add tax shields explicitly; avoid mixing with WACC tax adjustments.

---

### Quality of Earnings / Capital Structure Red Flags
⚠️ Leverage increased to “hit” EPS targets without a robust downside liquidity plan.  
⚠️ Buybacks executed at cycle peaks while underinvesting in maintenance capex or R&D.  
⚠️ Dividend maintained despite weak cash conversion (dividend funded by debt).  
⚠️ Debt maturities concentrated with refinancing risk not reflected in downside cases.  
⚠️ Compensation tied to EPS/TSR encourages optical buybacks rather than value-based decisions.  
✅ Capital return policy explicitly tied to excess cash after funding value-creating reinvestment and maintaining resilient liquidity.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Cyclical industrials | Cash flow volatility | Conservative leverage; stress-cycle interest coverage and liquidity |
| Utilities | Stable regulated cash flows | Higher leverage feasible; incorporate regulatory constraints |
| Financials | Regulatory capital | Payouts constrained by CET1 and stress tests; valuation on equity/capital metrics |
| Technology | Intangible investment needs | Avoid levering up at expense of product/R&D; opportunistic buybacks |
| Real estate | Asset-backed leverage | Focus on LTV, interest coverage, refinancing ladder |

---

### Practical Comparison Tables

#### 1) Capital Structure Decision Map
| Business profile | Leverage capacity | Key constraints | Typical policy |
|---|---:|---|---|
| Stable, defensive | Higher | Regulation, covenants | Debt-funded dividends/buybacks can be acceptable |
| Growth, intangible-heavy | Lower–medium | Innovation and flexibility | Prefer low leverage; opportunistic buybacks |
| Highly cyclical | Low | Downturn liquidity | Maintain buffer; avoid aggressive payouts |
| Asset-backed | Medium–high | Refinancing and collateral | Ladder maturities; manage LTV |

#### 2) Buyback Checklist (Value-Based)
| Question | Why it matters |
|---|---|---|
| Is buyback price below intrinsic value? | Determines per-share value creation |
| Does buyback increase distress risk? | Tax shield vs distress trade-off |
| Are there higher-NPV uses of cash? | Opportunity cost (capex, M&A, debt reduction) |
| Is cash truly excess after liquidity buffer? | Avoid forced refinancing or underinvestment |
| Is compensation aligned with value? | Reduces optical decision risk |

---

### Real-World Example (Debt-Funded Buyback: Value vs Risk)

Scenario: A firm considers issuing debt to fund a buyback. The buyback is attractive if shares trade below intrinsic value, but leverage can raise required returns and increase distress risk. Use the intrinsic test and leverage sensitivity.

```python
beta_u = 0.95
rf = 0.04
erp = 0.05
tax = 0.24
rd = 0.06

equity = 8_000_000_000
intrinsic_vps = 52.0
buyback_price = 40.0

print("Buyback below intrinsic?", buyback_intrinsic_test(intrinsic_vps, buyback_price))

for debt in [2_000_000_000, 4_000_000_000, 6_000_000_000]:
    beta_l = levered_beta(beta_u, debt, equity, tax) or 0.0
    re = cost_of_equity(rf, beta_l, erp)
    w = wacc(equity, debt, re, rd, tax) or 0.0
    print(f"Debt ${debt/1e9:.0f}B -> beta {beta_l:.2f}, Re {re:.1%}, WACC {w:.1%}")
```

Interpretation:
- Passing the intrinsic-price test is necessary but not sufficient; funding and resilience under stress determine risk-adjusted value.
- A disciplined policy links leverage targets to downside liquidity and ensures capital returns do not crowd out high-ROIC reinvestment.
