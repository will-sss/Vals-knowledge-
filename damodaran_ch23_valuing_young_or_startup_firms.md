# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 23: Valuing Young or Start-Up Firms (VC Method, Story-to-Numbers, Scaling Economics, Failure Risk)

### Core Concept
Young/start-up firms are hard to value because they have limited operating history, changing business models, and uncertain survival. Practically, valuation becomes a structured translation of **narrative → drivers → cash flows**, with explicit assumptions about **scaling margins**, **reinvestment**, **dilution**, and **probability of failure**. The goal is not precision; it is **discipline, consistency, and transparency**.

See also: Chapter 13 (narrative and numbers), Chapter 22 (money-losing firms), Chapter 33 (scenario analysis and simulations), Chapter 16 (per-share and dilution).

---

### Formula/Methodology

#### 1) “Story to numbers” (driver tree)
Start-ups should be forecast from operating drivers:
```text
Revenue = Customers × ARPU
Customers = Leads × Conversion × Retention (cohort logic)
Gross Profit = Revenue × Gross Margin
Operating Income = Revenue × Operating Margin (margin ramp)
Reinvestment = ΔRevenue / Sales-to-Capital  (closure)
FCFF = EBIT × (1 − Tax) + D&A − Reinvestment
```
Practical step: write the story in 6–10 bullet claims (market size, pricing, unit economics, moat, execution risk), then map each claim to a numeric driver (growth, margin, reinvestment, risk).

#### 2) Margin ramp and scaling economics
Young firms typically start with negative margins and converge toward a steady-state margin:
```text
Operating Margin_t = ramp(current -> target over N years)
```
Target margin must be defensible by:
- comparable mature firms,
- unit economics (gross margin minus CAC/servicing costs),
- competition and switching costs.

#### 3) Reinvestment closure (avoid “free growth”)
```text
Reinvestment_t ≈ ΔRevenue_t / (Sales-to-Capital)

Sales-to-Capital = Revenue / Invested Capital  (or proxy)
```
High growth with low reinvestment is a common valuation error.

#### 4) Failure risk (probability-weighted value)
```text
Expected Value = p_survive × Value_if_survive + (1 − p_survive) × Value_if_fail
```
For equity, Value_if_fail is often near zero (or liquidation value if assets matter).

You can also apply time-varying survival to cash flows:
```text
Expected PV = Σ [ (p_survive_to_t × CF_t) / (1 + r)^t ]
```
Use this if failure risk changes materially over time (e.g., after product-market fit).

#### 5) Venture Capital (VC) method (practical shortcut)
VC method works backward from an exit scenario:
```text
Exit Equity Value = Exit Multiple × Exit Metric   (e.g., P/E × Net Income, EV/Sales × Revenue)

Post-money Value today = Exit Equity Value / (1 + Target IRR) ^ N

Pre-money Value today = Post-money Value − New Investment

Investor Ownership % = New Investment / Post-money Value
```
Key discipline:
- Make the exit multiple consistent with expected risk and maturity at exit.
- Include dilution from future funding rounds and employee options in the path to exit.

#### 6) Expected dilution and option overhang
Start-ups frequently have large option pools and future funding dilution:
```text
Common equity value per share must reflect:
- existing options (option pool)
- expected future option grants
- future financing rounds (down/flat/up)
```
Treat dilution explicitly (share count path) or as claims (option valuation).

---

### Python Implementation
```python
from typing import List, Dict, Optional, Tuple
import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def linear_ramp(start: float, end: float, years: int) -> List[float]:
    """Linear ramp from start to end across `years` periods."""
    if not all(_num(v) for v in [start, end]):
        raise ValueError("start/end must be numeric")
    if years <= 0:
        raise ValueError("years must be > 0")
    return [start + (t / years) * (end - start) for t in range(1, years + 1)]


def forecast_revenue_from_customers(
    start_customers: float,
    customer_growth: List[float],
    arpu: List[float],
    churn: List[float]
) -> List[float]:
    """Forecast revenue with simple customer growth and churn dynamics."""
    if not _num(start_customers) or start_customers < 0:
        raise ValueError("start_customers must be numeric and >= 0")
    if not (len(customer_growth) == len(arpu) == len(churn)):
        raise ValueError("customer_growth, arpu, churn must have same length")
    customers = start_customers
    revenues = []
    for g, a, c in zip(customer_growth, arpu, churn):
        if not all(_num(v) for v in [g, a, c]):
            raise ValueError("inputs must be numeric")
        if a < 0:
            raise ValueError("arpu must be >= 0")
        if not (0.0 <= c < 1.0):
            raise ValueError("churn must be in [0,1)")
        # customers grow then churn
        customers = customers * (1.0 + g) * (1.0 - c)
        revenues.append(customers * a)
    return revenues


def reinvestment_from_sales_to_capital(revenues: List[float], sales_to_capital: float) -> List[float]:
    """Reinvestment ≈ ΔRevenue / (Sales-to-Capital)."""
    if not isinstance(revenues, list) or len(revenues) == 0:
        raise ValueError("revenues must be a non-empty list")
    if not _num(sales_to_capital) or sales_to_capital <= 0:
        raise ValueError("sales_to_capital must be numeric and > 0")
    reinv = []
    prev = 0.0
    for r in revenues:
        if not _num(r) or r < 0:
            raise ValueError("revenues must be numeric and >= 0")
        reinv.append((r - prev) / sales_to_capital)
        prev = r
    return reinv


def fcff_from_revenue_path(
    revenues: List[float],
    op_margins: List[float],
    tax_rate: float,
    da_pct_sales: float,
    sales_to_capital: float
) -> List[float]:
    """Build FCFF from revenue, margin ramp, D&A %, and reinvestment closure."""
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if not (0.0 <= da_pct_sales <= 1.0):
        raise ValueError("da_pct_sales must be in [0,1]")
    if len(revenues) != len(op_margins):
        raise ValueError("revenues and op_margins must have same length")
    reinv = reinvestment_from_sales_to_capital(revenues, sales_to_capital)
    out = []
    for r, m, rv in zip(revenues, op_margins, reinv):
        if not all(_num(v) for v in [r, m, rv]):
            raise ValueError("inputs must be numeric")
        ebit = r * m
        da = r * da_pct_sales
        out.append(ebit * (1.0 - tax_rate) + da - rv)
    return out


def pv_cash_flows(cash_flows: List[float], r: float, survival: Optional[List[float]] = None) -> float:
    """PV of cash flows with optional survival probabilities per period."""
    if not _num(r) or r <= -1:
        raise ValueError("r must be numeric and > -100%")
    if not isinstance(cash_flows, list) or len(cash_flows) == 0:
        raise ValueError("cash_flows must be a non-empty list")
    if survival is not None:
        if len(survival) != len(cash_flows):
            raise ValueError("survival must match cash_flows length")
        for p in survival:
            if not _num(p) or not (0.0 <= p <= 1.0):
                raise ValueError("survival probabilities must be in [0,1]")
    pv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        if not _num(cf):
            raise ValueError("cash_flows contains non-numeric values")
        p = 1.0 if survival is None else survival[t - 1]
        pv += (p * cf) / ((1.0 + r) ** t)
    return pv


def terminal_value_perpetuity(cf_next: float, r: float, g: float) -> float:
    """TV = CF_{n+1} / (r - g)."""
    if not all(_num(v) for v in [cf_next, r, g]):
        raise ValueError("inputs must be numeric")
    if r <= g:
        raise ValueError("Need r > g for finite TV")
    return cf_next / (r - g)


def vc_method_pre_money(
    exit_metric: float,
    exit_multiple: float,
    years_to_exit: int,
    target_irr: float,
    new_investment: float,
    expected_dilution: float = 0.0
) -> Dict[str, float]:
    """VC method: compute post-money and pre-money today. expected_dilution is fraction (e.g., 0.25)."""
    for name, v in {
        "exit_metric": exit_metric, "exit_multiple": exit_multiple, "years_to_exit": years_to_exit,
        "target_irr": target_irr, "new_investment": new_investment, "expected_dilution": expected_dilution
    }.items():
        if name in {"years_to_exit"}:
            continue
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if years_to_exit <= 0:
        raise ValueError("years_to_exit must be > 0")
    if target_irr <= -1:
        raise ValueError("target_irr must be > -100%")
    if exit_metric < 0 or exit_multiple <= 0 or new_investment < 0:
        raise ValueError("exit_metric must be >=0, exit_multiple >0, new_investment >=0")
    if not (0.0 <= expected_dilution < 1.0):
        raise ValueError("expected_dilution must be in [0,1)")

    exit_equity_value = exit_metric * exit_multiple
    post_money_today = exit_equity_value / ((1.0 + target_irr) ** years_to_exit)

    # Adjust for expected dilution between now and exit (investor's economic claim diluted)
    # If dilution is expected, today's post-money that maps to exit is effectively lower.
    post_money_today_adj = post_money_today * (1.0 - expected_dilution)

    pre_money_today = post_money_today_adj - new_investment
    ownership = 0.0 if post_money_today_adj == 0 else new_investment / post_money_today_adj

    return {
        "exit_equity_value": exit_equity_value,
        "post_money_today": post_money_today_adj,
        "pre_money_today": pre_money_today,
        "investor_ownership": ownership
    }


# Example usage: DCF-style (simple) + VC method check

# Driver-based revenue forecast
start_customers = 50_000
years = 6
customer_growth = [0.80, 0.60, 0.45, 0.35, 0.25, 0.20]
arpu = [120, 140, 165, 190, 215, 240]
churn = [0.18, 0.15, 0.12, 0.10, 0.09, 0.08]
revenues = forecast_revenue_from_customers(start_customers, customer_growth, arpu, churn)

# Margin ramp from -25% to 18% over 6 years
op_margins = linear_ramp(-0.25, 0.18, years)

fcff = fcff_from_revenue_path(
    revenues=revenues,
    op_margins=op_margins,
    tax_rate=0.24,
    da_pct_sales=0.04,
    sales_to_capital=2.2
)

# Survival probabilities (declining early risk, improving later)
survival = [0.85, 0.82, 0.80, 0.79, 0.78, 0.78]

wacc = 0.13
pv_stage1 = pv_cash_flows(fcff, wacc, survival=survival)

# Terminal value: assume CF in year 7 at steady-state margin 18%, g=3%
g = 0.03
rev6 = revenues[-1]
rev7 = rev6 * (1.0 + g)
ebit7 = rev7 * 0.18
reinv7 = (rev7 - rev6) / 2.2
da7 = rev7 * 0.04
fcff7 = ebit7 * (1.0 - 0.24) + da7 - reinv7
tv6 = terminal_value_perpetuity(fcff7, wacc, g)
pv_tv = (survival[-1] * tv6) / ((1.0 + wacc) ** years)

ev_dcf = pv_stage1 + pv_tv
print(f"DCF-style EV (survival-weighted): ${ev_dcf/1e6:,.1f}m")

# VC method check: exit on EV/Sales of 6x on year-6 revenue (equity proxy for simplicity)
exit_metric = revenues[-1]  # revenue
exit_multiple = 6.0
vc = vc_method_pre_money(
    exit_metric=exit_metric,
    exit_multiple=exit_multiple,
    years_to_exit=6,
    target_irr=0.35,
    new_investment=40_000_000,
    expected_dilution=0.25
)
print(f"VC pre-money today: ${vc['pre_money_today']/1e6:,.1f}m, ownership: {vc['investor_ownership']:.1%}")
```

---

### Valuation Impact
Why this matters:
- Start-up value is often a leveraged outcome of a few assumptions: time to scale, steady-state margin, reinvestment efficiency, and survival. Small changes here can swing value dramatically.
- The VC method is widely used in practice; it enforces discipline on exits and required returns, but can embed unrealistic exit multiples or ignore dilution if used carelessly.
- Connecting valuation to drivers (customers, ARPU, retention) makes forecasts auditable and highlights which assumptions are doing the work.

Impact on multiples:
- Applying EV/ARR or EV/Sales requires explicit assumptions about retention and margin ramp; a “peer multiple” is a proxy for those assumptions.
- Exit multiples should be checked against what the DCF implies at terminal (implied EV/Sales or EV/EBITDA).

Impact on DCF inputs:
- High discount rates alone do not represent failure well; survival-weighted cash flows make failure explicit and separable from operating risk.
- Reinvestment closure (sales-to-capital) prevents the “high growth, no capital” mistake.

Practical adjustments:
```python
def implied_exit_multiple(exit_value: float, exit_metric: float) -> Optional[float]:
    """Compute implied exit multiple; returns None if metric <= 0."""
    if not all(_num(v) for v in [exit_value, exit_metric]):
        raise ValueError("inputs must be numeric")
    if exit_metric <= 0:
        return None
    return exit_value / exit_metric
```

---

### Quality of Earnings Flags (Start-Ups)
⚠️ “Adjusted EBITDA” excludes core recurring costs (SBC, ongoing R&D) to manufacture profitability.  
⚠️ Growth driven by heavy discounting; ARPU or gross margin deteriorates as revenue rises.  
⚠️ CAC rising and payback period lengthening; retention weakening (cohort decay).  
⚠️ Capitalisation of costs (software dev, commissions) inflates current earnings and delays expense recognition.  
⚠️ Dilution ignored (option pool + future rounds) when converting to per-share value.  
✅ Green flag: improving cohorts/retention, stable or improving unit economics, transparent reconciliation of adjusted metrics.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS | ARR quality, retention, SBC | Forecast ARR by cohorts; use EV/ARR with net retention; treat SBC consistently. |
| Consumer apps | Monetisation uncertainty | Driver-based (MAU → ARPU) and scenario weight; caution with user multiples. |
| Biotech | Binary outcomes | Scenario/probability weighting; option-like valuation; treat R&D as value driver. |
| Hardware | Working capital/capex | Explicit reinvestment and scale efficiencies; avoid SaaS-like multiples. |
| Fintech | Regulation + loss risk | Risk adjustments and capital needs; watch unit economics and credit losses. |

---

### Practical Comparison Tables

#### 1) Start-up valuation approaches
| Approach | Best when | Key outputs | Main risk |
|---|---|---|---|
| Driver-based DCF (survival-weighted) | Drivers are measurable | EV, implied margins | Overconfidence in long-term ramps |
| VC method | Exit markets and IRR targets drive pricing | Pre-money, ownership | Unrealistic exit multiple/ignoring dilution |
| Relative (EV/ARR, EV/Sales) | Mature peer set exists | Quick range | Implicit margins and reinvestment ignored |
| Scenario tree | Binary outcomes | Probability-weighted value | Probabilities subjective |

#### 2) Dilution checklist
| Source of dilution | How to treat | Common mistake |
|---|---|---|
| Existing option pool | TSM or option value claim | ignore because “non-cash” |
| Future option grants | add expected pool increase | assume static pool |
| Future funding rounds | scenario-based ownership path | ignore future capital need |
| Convertibles/SAFEs | model conversion at cap/discount | treat as simple debt |

---

### Real-World Example
Scenario: You value a start-up using a driver-based forecast (customers, ARPU, churn) and a margin ramp, then compare against a VC method valuation based on an EV/Sales exit multiple and a 35% target IRR (with 25% expected dilution).

```python
# Outputs printed by the example code above:
# - DCF-style EV (survival-weighted)
# - VC pre-money and ownership %
```
Interpretation: If the VC pre-money is far below the DCF, either (a) the DCF assumes an aggressive margin ramp/reinvestment efficiency, (b) failure risk is understated, or (c) the VC assumptions (exit multiple, IRR, dilution) are conservative. Use the gap to identify which assumptions need tightening or better evidence.
