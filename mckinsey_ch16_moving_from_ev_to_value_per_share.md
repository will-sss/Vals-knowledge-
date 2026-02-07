# McKinsey Valuation - Key Concepts for Valuation

## Chapter 16: Moving from Enterprise Value to Value per Share (EV-to-Equity Bridge, Net Debt, Debt-Like Items, Dilution)

### Core Concept
Enterprise value (EV) is the value of operating assets available to all capital providers. To get value per share, you must bridge EV to equity value by subtracting net debt and other non-equity claims, adding non-operating assets, then divide by the fully diluted share count. Most valuation mistakes at this step come from inconsistent definitions (cash, leases, pensions, provisions) and ignoring dilution mechanics (options, RSUs, convertibles).

See also: Chapter 11 on reorganising statements (operating vs financing classification); Chapter 15 on cost of capital (capital structure); Chapter 21 on non-operating items, provisions, and reserves.

---

### Core Concept: The EV-to-Equity Bridge is an Accounting Consistency Exercise
The bridge must mirror what is included in EV:
- If EV is derived from discounting FCFF of operations, then subtract all financing claims (debt and debt-like items) and add back non-operating assets.
- If EV includes lease liabilities (debt-like), ensure EBITDA/FCFF and terminal value assumptions are lease-consistent.

Practical rule: if an item is not part of operating cash flows being valued, it usually belongs in the bridge (either subtract as a claim or add as a non-operating asset).

---

### Formula/Methodology

#### 1) Equity Value from Enterprise Value
```text
Equity Value = Enterprise Value
            − Net Debt
            − Other Debt-like Claims
            − Preferred Equity (if any)
            − Non-controlling Interests (if EV reflects 100% of consolidated ops)
            + Non-operating Assets (excess cash, investments not needed for ops)
            ± Other Adjustments (e.g., underfunded pensions if treated as debt-like)
```

Where:
- Net Debt = Interest-bearing debt − Excess cash (not total cash by default)
- Other debt-like claims can include: lease liabilities, pension deficits, asset retirement obligations (if treated as financing), certain provisions, customer advance timing items (case-by-case), deferred tax liabilities (usually not, unless special situation), working capital facilities included in debt, etc.

#### 2) Net Debt
```text
Net Debt = Interest-bearing Debt − Excess Cash

Excess Cash = Cash & equivalents − Operating Cash Requirement (management/analyst judgement)
```

#### 3) Fully Diluted Shares (basic overview)
```text
Value per share = Equity Value / Fully Diluted Shares
```

Fully diluted shares typically include:
- Basic shares outstanding
- In-the-money options (treasury stock method)
- RSUs/PSUs expected to vest (net of shares withheld for taxes)
- Convertible instruments (if-converted method, subject to assumptions and whether conversion is in the money)

---

### Python Implementation
```python
from typing import Optional, Dict

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def net_debt(debt: float, cash: float, excess_cash: Optional[float] = None) -> float:
    """
    Net Debt = Debt - Excess Cash (excess cash defaults to total cash if not specified).

    Args:
        debt: interest-bearing debt (currency, >= 0)
        cash: total cash & equivalents (currency, >= 0)
        excess_cash: portion of cash deemed non-operating and nettable (currency, >= 0)

    Returns:
        float: net debt (currency). Can be negative.
    """
    if not _num(debt) or debt < 0:
        raise ValueError("debt must be numeric and >= 0")
    if not _num(cash) or cash < 0:
        raise ValueError("cash must be numeric and >= 0")
    cash_to_net = cash if excess_cash is None else excess_cash
    if not _num(cash_to_net) or cash_to_net < 0:
        raise ValueError("excess_cash must be numeric and >= 0")
    if cash_to_net > cash:
        raise ValueError("excess_cash cannot exceed total cash")
    return debt - cash_to_net


def equity_value_from_ev(
    enterprise_value: float,
    net_debt_value: float,
    other_debt_like: float = 0.0,
    preferred_equity: float = 0.0,
    non_controlling_interests: float = 0.0,
    non_operating_assets: float = 0.0,
    other_adjustments: float = 0.0
) -> float:
    """
    Bridge from enterprise value to equity value.

    Equity = EV - net debt - other debt-like - preferred - NCI + non-op assets + other adjustments

    Args:
        enterprise_value: EV (currency)
        net_debt_value: net debt (currency)
        other_debt_like: leases, pensions, etc. treated as debt-like (currency, >= 0)
        preferred_equity: preferred stock/other preferred claims (currency, >= 0)
        non_controlling_interests: minority interests if EV is for consolidated ops (currency, >= 0)
        non_operating_assets: investments/excess assets not in ops (currency, >= 0)
        other_adjustments: signed adjustments (currency), e.g., add surplus assets, subtract deficits

    Returns:
        float: equity value (currency)

    Raises:
        ValueError: if numeric inputs invalid.
    """
    for name, v in {
        "enterprise_value": enterprise_value,
        "net_debt_value": net_debt_value,
        "other_debt_like": other_debt_like,
        "preferred_equity": preferred_equity,
        "non_controlling_interests": non_controlling_interests,
        "non_operating_assets": non_operating_assets,
        "other_adjustments": other_adjustments
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric and not NaN")

    for name, v in {
        "other_debt_like": other_debt_like,
        "preferred_equity": preferred_equity,
        "non_controlling_interests": non_controlling_interests,
        "non_operating_assets": non_operating_assets
    }.items():
        if v < 0:
            raise ValueError(f"{name} must be >= 0")

    return (
        enterprise_value
        - net_debt_value
        - other_debt_like
        - preferred_equity
        - non_controlling_interests
        + non_operating_assets
        + other_adjustments
    )


def treasury_stock_method_dilution(
    options_in_the_money: int,
    strike_price: float,
    current_share_price: float
) -> int:
    """
    Treasury stock method for in-the-money options.

    New shares = options - shares repurchased with option proceeds.
    Proceeds = options * strike
    Repurchased = proceeds / current price

    Returns integer shares (rounded down).

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(options_in_the_money, int) or options_in_the_money < 0:
        raise ValueError("options_in_the_money must be an int >= 0")
    for name, v in {"strike_price": strike_price, "current_share_price": current_share_price}.items():
        if not _num(v) or v <= 0:
            raise ValueError(f"{name} must be numeric and > 0")
    proceeds = options_in_the_money * strike_price
    repurchased = int(proceeds / current_share_price)
    incremental = options_in_the_money - repurchased
    return max(0, incremental)


def fully_diluted_shares(
    basic_shares: int,
    incremental_options: int = 0,
    rsus_expected_to_vest: int = 0,
    convertibles_if_converted: int = 0
) -> int:
    """
    Construct a simple fully diluted share count.

    Args:
        basic_shares: shares outstanding (int >= 0)
        incremental_options: incremental shares from TSM (int >= 0)
        rsus_expected_to_vest: RSUs/PSUs expected to vest (int >= 0)
        convertibles_if_converted: incremental shares on conversion (int >= 0)

    Returns:
        int: fully diluted shares
    """
    for name, v in {
        "basic_shares": basic_shares,
        "incremental_options": incremental_options,
        "rsus_expected_to_vest": rsus_expected_to_vest,
        "convertibles_if_converted": convertibles_if_converted
    }.items():
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{name} must be an int >= 0")
    return basic_shares + incremental_options + rsus_expected_to_vest + convertibles_if_converted


def value_per_share(equity_value: float, diluted_shares: int) -> float:
    """
    Value per share = equity value / diluted shares.
    """
    if not _num(equity_value):
        raise ValueError("equity_value must be numeric")
    if not isinstance(diluted_shares, int) or diluted_shares <= 0:
        raise ValueError("diluted_shares must be an int > 0")
    return equity_value / diluted_shares


# Example usage
ev = 8_400_000_000
debt = 2_050_000_000
cash = 600_000_000
excess_cash = 350_000_000
lease_liability = 480_000_000
pension_deficit = 220_000_000  # treat as debt-like for the bridge in this example
nci = 90_000_000
non_operating_assets = 160_000_000

nd = net_debt(debt, cash, excess_cash=excess_cash)
eq = equity_value_from_ev(
    enterprise_value=ev,
    net_debt_value=nd,
    other_debt_like=lease_liability + pension_deficit,
    preferred_equity=0.0,
    non_controlling_interests=nci,
    non_operating_assets=non_operating_assets
)

basic_shares = 520_000_000
inc_options = treasury_stock_method_dilution(options_in_the_money=30_000_000, strike_price=12.0, current_share_price=20.0)
diluted = fully_diluted_shares(basic_shares, incremental_options=inc_options, rsus_expected_to_vest=8_000_000)

vps = value_per_share(eq, diluted)

print(f"Net debt: ${nd/1e9:.2f}B")
print(f"Equity value: ${eq/1e9:.2f}B")
print(f"Fully diluted shares: {diluted/1e6:.1f}M")
print(f"Value per share: ${vps:.2f}")
```

---

### Valuation Impact
Why this matters:
- The EV-to-equity bridge determines what portion of enterprise value belongs to common shareholders; errors here can exceed the “precision” gained from detailed forecasting.
- Debt-like items and non-operating assets drive comparability between EV-derived valuations and market capitalisation.
- Fully diluted shares often matter more than expected in option-heavy or convertible-heavy capital structures.

Impact on multiples:
- EV/EBITDA uses an enterprise definition; if you add leases to EV, ensure EBITDA is lease-consistent (IFRS 16 effects) or adjust EBITDA accordingly.
- Equity multiples (P/E) depend on equity value and diluted share count; ignoring dilution can overstate per-share value.

Impact on DCF:
- Use EV from FCFF discounting, then bridge to equity. If you value equity directly (FCFE), avoid double-counting leverage by also subtracting net debt.

Practical adjustments:
```python
def bridge_with_excess_cash_policy(ev: float, debt: float, cash: float, operating_cash_buffer: float,
                                  other_debt_like: float = 0.0, non_operating_assets: float = 0.0) -> float:
    """
    Apply an explicit operating cash buffer to determine excess cash and equity value.
    """
    if not _num(operating_cash_buffer) or operating_cash_buffer < 0:
        raise ValueError("operating_cash_buffer must be numeric and >= 0")
    excess = max(0.0, cash - operating_cash_buffer)
    nd = net_debt(debt, cash, excess_cash=excess)
    return equity_value_from_ev(ev, nd, other_debt_like=other_debt_like, non_operating_assets=non_operating_assets)
```

---

### Quality of Earnings / Bridge Red Flags
⚠️ Netting all cash against debt without assessing operating cash needs (inflates equity).  
⚠️ Lease liabilities excluded from EV bridge while EBITDA is IFRS 16-inflated (inconsistent EV/EBITDA).  
⚠️ Pension deficits and large provisions ignored despite being debt-like in economic substance.  
⚠️ Non-controlling interests omitted when EV reflects consolidated operations (equity overstated).  
⚠️ Share count uses basic shares only (ignores options/RSUs/convertibles) despite material dilution.  
✅ Clear reconciliation from EV to equity value, with explicit classification of each debt-like item and rationale.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Retail / airlines | Leases dominate financing profile | Include lease liabilities as debt-like; ensure EBITDA and EV are lease-consistent |
| Industrials | Pensions and environmental provisions | Often treat deficits as debt-like; document discount rate assumptions in pension valuation |
| Technology | Option and RSU dilution | Use fully diluted shares; consider net-share settlement and expected vesting |
| Financials | EV concept less meaningful | Prefer equity valuation frameworks (P/TB, residual income); bridge focuses on equity claims |
| Infrastructure | Project debt and restricted cash | Distinguish restricted cash and debt at project vs holdco level; avoid netting restricted cash improperly |

---

### Real-World Example (EV → Equity → Per Share)

Scenario: You have an enterprise DCF output (EV). Bridge to equity value using net debt, leases and pensions as debt-like items, then compute per-share value on a fully diluted basis.

```python
ev = 8_400_000_000
debt = 2_050_000_000
cash = 600_000_000
operating_cash_buffer = 250_000_000

lease_liability = 480_000_000
pension_deficit = 220_000_000
nci = 90_000_000
non_operating_assets = 160_000_000

equity = bridge_with_excess_cash_policy(
    ev=ev, debt=debt, cash=cash, operating_cash_buffer=operating_cash_buffer,
    other_debt_like=lease_liability + pension_deficit,
    non_operating_assets=non_operating_assets
) - nci  # subtract NCI separately if not included in helper

basic_shares = 520_000_000
inc_options = treasury_stock_method_dilution(30_000_000, strike_price=12.0, current_share_price=20.0)
diluted = fully_diluted_shares(basic_shares, incremental_options=inc_options, rsus_expected_to_vest=8_000_000)

vps = value_per_share(equity, diluted)

print(f"Equity value: ${equity/1e9:.2f}B")
print(f"Fully diluted shares: {diluted/1e6:.1f}M")
print(f"Value per share: ${vps:.2f}")
```

Interpretation:
- The largest “swing factors” are typically excess cash policy, leases/pensions classification, and dilution. Sensitise these before over-optimising operating forecasts.
