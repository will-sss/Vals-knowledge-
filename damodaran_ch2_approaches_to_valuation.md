# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 2: Approaches to Valuation (Intrinsic vs Relative vs Option-Based)

### Core Concept
Three valuation “families” cover most real-world work: **intrinsic valuation (DCF)**, **pricing/relative valuation (multiples)**, and **contingent claim valuation (option pricing/real options)**. Pick the approach that matches the asset’s economics, data availability, and what the decision-maker needs (absolute value vs “cheap/expensive vs peers” vs option-like payoff). fileciteturn3file2

### Formula/Methodology
```text
Intrinsic value (general present value rule):
Value = Σ_{t=1..n}  E(CF_t) / (1 + r)^t
Where:
E(CF_t) = expected cash flow in period t
r       = discount rate reflecting risk of the cash flows
n       = asset life / explicit forecast horizon
```

```text
Equity valuation (discount cash flows to equity at cost of equity):
Equity Value = Σ_{t=1..n}  E(FCFE_t) / (1 + k_e)^t  +  Terminal Equity Value / (1 + k_e)^n
Where:
FCFE_t = free cash flow to equity in period t (after reinvestment + debt cash flows)
k_e    = cost of equity
```

```text
Firm (enterprise) valuation (discount cash flows to the firm at WACC):
Enterprise Value = Σ_{t=1..n}  E(FCFF_t) / (1 + WACC)^t  +  Terminal Firm Value / (1 + WACC)^n
Where:
FCFF_t = free cash flow to firm (before debt cash flows; after taxes + reinvestment)
WACC   = weighted average cost of capital
```

```text
Relative valuation (pricing) - generic multiple:
Implied Value = Multiple_peer × Metric_target
Where:
Multiple_peer = median/mean peer multiple (or regression-predicted multiple)
Metric_target = consistent metric for target (e.g., EBITDA, EBIT, EPS, Book Value, Revenue)
```

```text
Contingent claim valuation (option-like payoffs, Black-Scholes call option):
Call Value = S·N(d1) - K·e^{-rT}·N(d2)
d1 = [ln(S/K) + (r + 0.5·σ^2)·T] / (σ·sqrt(T))
d2 = d1 - σ·sqrt(T)
Where:
S  = current value of underlying asset
K  = strike/exercise price
r  = risk-free rate (continuous)
T  = time to expiry (years)
σ  = volatility of underlying asset value (annualized)
N(.) = standard normal CDF
```

See also: Chapter 10 (From Earnings to Cash Flows), Chapter 12 (Terminal Value), Chapter 17 (Relative Valuation).

### Python Implementation
```python
from math import erf, exp, log, sqrt
from typing import Iterable, Optional


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no SciPy dependency)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def pv_cash_flows(cash_flows: Iterable[float], discount_rate: float) -> float:
    """
    Present value of a sequence of cash flows.

    Args:
        cash_flows: iterable of cash flows by period (t=1..n), in currency units.
        discount_rate: per-period discount rate as decimal (e.g., 0.10 for 10%).

    Returns:
        float: present value in currency units.

    Raises:
        ValueError: if discount_rate <= -1 or cash_flows is empty.
    """
    cf_list = list(cash_flows)
    if len(cf_list) == 0:
        raise ValueError("cash_flows must contain at least one period.")
    if discount_rate <= -1:
        raise ValueError("discount_rate must be greater than -100%.")
    pv = 0.0
    for t, cf in enumerate(cf_list, start=1):
        pv += cf / ((1.0 + discount_rate) ** t)
    return pv


def dcf_equity_value(
    fcfe: Iterable[float],
    cost_of_equity: float,
    terminal_value: Optional[float] = None
) -> float:
    """
    Equity DCF = PV(FCFE) + PV(Terminal Equity Value).

    Args:
        fcfe: free cash flow to equity per year (t=1..n), in currency units.
        cost_of_equity: k_e as decimal.
        terminal_value: terminal equity value at end of year n (optional), in currency units.

    Returns:
        float: equity value.

    Raises:
        ValueError: if cost_of_equity <= -1.
    """
    fcfe_list = list(fcfe)
    if cost_of_equity <= -1:
        raise ValueError("cost_of_equity must be greater than -100%.")
    value = pv_cash_flows(fcfe_list, cost_of_equity)
    if terminal_value is not None:
        n = len(fcfe_list)
        value += terminal_value / ((1.0 + cost_of_equity) ** n)
    return value


def dcf_enterprise_value(
    fcff: Iterable[float],
    wacc: float,
    terminal_value: Optional[float] = None
) -> float:
    """
    Firm/Enterprise DCF = PV(FCFF) + PV(Terminal Firm Value).

    Args:
        fcff: free cash flow to firm per year (t=1..n), in currency units.
        wacc: WACC as decimal.
        terminal_value: terminal firm value at end of year n (optional), in currency units.

    Returns:
        float: enterprise value.

    Raises:
        ValueError: if wacc <= -1.
    """
    fcff_list = list(fcff)
    if wacc <= -1:
        raise ValueError("wacc must be greater than -100%.")
    value = pv_cash_flows(fcff_list, wacc)
    if terminal_value is not None:
        n = len(fcff_list)
        value += terminal_value / ((1.0 + wacc) ** n)
    return value


def implied_value_from_multiple(metric_value: float, peer_multiple: float) -> float:
    """
    Implied value using a peer multiple.

    Args:
        metric_value: target metric in currency units (or per-share for P/E).
        peer_multiple: peer multiple (e.g., EV/EBITDA) as a unitless number.

    Returns:
        float: implied value.

    Raises:
        ValueError: if peer_multiple <= 0 or metric_value is None.
    """
    if metric_value is None:
        raise ValueError("metric_value cannot be None.")
    if peer_multiple <= 0:
        raise ValueError("peer_multiple must be > 0.")
    return peer_multiple * metric_value


def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes value of a European call option.

    Args:
        S: underlying value (currency).
        K: strike price (currency).
        r: risk-free rate (continuous comp, decimal).
        sigma: volatility of underlying (annualized, decimal).
        T: time to expiry in years.

    Returns:
        float: call option value.

    Raises:
        ValueError: for non-positive S, K, sigma, T.
    """
    if S <= 0:
        raise ValueError("S must be > 0.")
    if K <= 0:
        raise ValueError("K must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if T <= 0:
        raise ValueError("T must be > 0.")

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


# Example usage
if __name__ == "__main__":
    # 1) Simple intrinsic PV
    cash_flows = [50, 60, 70]  # $m
    pv = pv_cash_flows(cash_flows, discount_rate=0.10)
    print(f"PV of cash flows: ${pv:.1f}m")

    # 2) Relative valuation (EV/EBITDA)
    ev = implied_value_from_multiple(metric_value=200, peer_multiple=9.0)  # EBITDA=$200m, 9x
    print(f"Implied EV: ${ev:.0f}m")

    # 3) Option value (real option proxy)
    call = black_scholes_call(S=1200, K=1000, r=0.04, sigma=0.35, T=2.0)
    print(f"Call value: ${call:.1f}m")
```

### Valuation Impact
Why this matters:
- **Model choice changes value**: DCF answers “what is it worth on fundamentals?”, relative valuation answers “how is it priced vs similar assets?”, and option pricing answers “what is the value of flexibility / contingent payoffs?”. fileciteturn3file2  
- **Multiples vs DCF consistency**: If peers are collectively over/undervalued, a multiples-based valuation will inherit that market mood; DCF can diverge by design. fileciteturn3file14  
- **Capital structure alignment**: Use **equity DCF** (FCFE, k_e) when valuing per-share equity directly; use **enterprise DCF** (FCFF, WACC) when valuing operations then bridging to equity.

Practical adjustments (comparability + normalization):
```python
def normalize_metric(reported: float, add_backs: float = 0.0, remove: float = 0.0) -> float:
    """Generic normalization for multiples (e.g., adjust EBITDA for one-offs)."""
    if reported is None:
        raise ValueError("reported cannot be None.")
    return reported + add_backs - remove


def bridge_ev_to_equity(ev: float, net_debt: float, other_claims: float = 0.0) -> float:
    """
    Enterprise to Equity bridge (simplified).

    Equity Value = EV - Net Debt - Other Claims (e.g., pension deficit, minorities, pref)
    """
    if ev is None or net_debt is None:
        raise ValueError("ev and net_debt cannot be None.")
    return ev - net_debt - other_claims
```

### Quality of Earnings Flags
⚠️ **Relative valuation is vulnerable to “bad comps”**: inconsistent definitions (EBITDA before/after leases, SBC add-backs, capitalized R&D) can create false cheap/expensive signals.  
⚠️ **Market mood contamination**: peers bid up/down together → multiples reflect sentiment rather than fundamentals. fileciteturn3file14  
⚠️ **Terminal value dependence in DCF**: if most value is terminal, small errors in long-run assumptions dominate.  
✅ **Green flag**: intrinsic value, multiples, and “implied multiple from DCF” tell a consistent story (no large unexplained gaps).

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology / high-growth | Option-like upside (platform scale, product optionality) + volatile margins | Use DCF with scenario ranges; consider real-options framing for staged investments/patents where exercise is discretionary. |
| Natural resources | Develop only if commodity prices justify development | Model as real option when there is genuine “wait/defer” flexibility; otherwise use probability-weighted DCF. fileciteturn3file1 |
| Banking/insurers | Difficult FCFF definitions; leverage is core operating feature | Prefer equity valuation frameworks (dividends/FCFE) and relative valuation (P/B, P/E) with careful ROE/risk adjustments. |
| Distressed / high leverage | Equity resembles a call option on firm value | Option framing can be useful (equity-as-call) when default risk dominates. fileciteturn3file5 |

### Real-World Example
Scenario: A consumer services company is being valued for an acquisition. You want a **triangulation**: (1) enterprise DCF for fundamentals, (2) EV/EBITDA multiple sanity check, (3) call option proxy for a patent-like project embedded in the business.

```python
# Realistic numbers (all $m)
forecast = {
    "fcff": [120, 135, 150, 165, 180],  # years 1-5
    "wacc": 0.095,
    "terminal_value": 3200,             # at end of year 5
    "ebitda_ntm": 260,
    "peer_ev_ebitda": 10.5,
    "net_debt": 900,
}

ev_dcf = dcf_enterprise_value(
    forecast["fcff"], forecast["wacc"], terminal_value=forecast["terminal_value"]
)

ev_multiple = implied_value_from_multiple(
    metric_value=forecast["ebitda_ntm"], peer_multiple=forecast["peer_ev_ebitda"]
)

eq_from_dcf = bridge_ev_to_equity(ev=ev_dcf, net_debt=forecast["net_debt"])
eq_from_mult = bridge_ev_to_equity(ev=ev_multiple, net_debt=forecast["net_debt"])

print(f"Enterprise Value (DCF): ${ev_dcf:,.0f}m")
print(f"Enterprise Value (Multiple): ${ev_multiple:,.0f}m")
print(f"Equity Value (DCF bridge): ${eq_from_dcf:,.0f}m")
print(f"Equity Value (Multiple bridge): ${eq_from_mult:,.0f}m")
```

Interpretation: If DCF EV is materially below the multiple EV, check whether (a) peers are in a sentiment bubble, (b) your DCF margins/reinvestment/growth are too conservative, or (c) EBITDA comparability is broken by accounting/presentation differences (e.g., lease/SBC treatment). If the target has a meaningful “right but not obligation” to invest in growth projects (patents, staged rollout, reserves), consider layering a real-options valuation on top of the base DCF. fileciteturn3file1
