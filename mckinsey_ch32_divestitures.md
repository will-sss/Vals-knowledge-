# McKinsey Valuation - Key Concepts for Valuation

## Chapter 32: Divestitures (Carve-outs, Value Realisation, Standalone Costs, Separation Adjustments)

### Core Concept
Divestitures (asset sales, business unit disposals, spin-offs) can create value by improving strategic focus, reallocating capital, reducing complexity, and enabling the market to value parts more appropriately. The practical valuation job is to (1) value the asset/business on a standalone basis, (2) quantify separation impacts (TSAs, stranded costs, dis-synergies), (3) evaluate proceeds net of taxes and transaction costs, and (4) compare against keeping the asset (opportunity cost and capital allocation).

See also: Chapter 19 on valuation by parts (SOTP); Chapter 31 on M&A (deal value creation); Chapter 16 on enterprise-to-equity bridge (cash proceeds and claims); Chapter 20 on taxes (tax leakage on disposals).

---

### Core Concept: Three Numbers Drive the Decision
1) **Standalone value of the divested business** (to a buyer or as an independent entity).  
2) **Net proceeds to seller** (after tax, fees, working capital true-ups, and debt repayment conditions).  
3) **Value impact on remaining company** (stranded costs, lost synergies, improved focus, redeployment of proceeds).  

A divestiture creates value when the net proceeds plus the value of the remaining company after separation exceeds the value of the combined company before separation.

---

### Formula/Methodology

#### 1) Value creation test (seller perspective)
```text
Value Created = (Value_RemainCo_after + Net Proceeds) − Value_Combined_before
```

Where:
- Value_RemainCo_after is the value of the continuing business after removing the divested segment and accounting for stranded costs/dis-synergies and any benefits.
- Net Proceeds is cash received net of tax leakage and transaction costs.

#### 2) Net proceeds (headline to cash)
```text
Net Proceeds = Sale Price
           − Transaction Fees
           − Separation / TSA Costs (cash)
           − Taxes on Gain
           ± Working Capital / Debt-like true-ups
           − Mandatory Debt Repayment (if covenant/structure requires)
```

#### 3) Tax on gain (simplified)
```text
Tax on Gain = Tax Rate × max(0, Sale Price − Tax Basis − Selling Costs)
```
Actual tax depends on jurisdiction, structure, and available tax attributes (NOLs, exemptions).

#### 4) Stranded costs and dis-synergies
Stranded costs are overhead costs that remain after a unit is sold. Dis-synergies are lost benefits (e.g., procurement scale). These reduce RemainCo cash flows and must be modelled explicitly.

A simple PV of perpetual stranded costs:
```text
PV(Stranded Costs) = Cost_1 / (WACC − g)
```
Where:
- Cost_1 = stranded cost in year 1 after separation (after any mitigation)
- g = growth in stranded costs (often inflation-like)

---

### Python Implementation
```python
from typing import List, Dict, Optional

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def pv_perpetuity(cashflow_1: float, wacc: float, g: float = 0.0) -> float:
    """PV = CF1 / (WACC - g)."""
    if not _num(cashflow_1):
        raise ValueError("cashflow_1 must be numeric")
    if not _num(wacc) or wacc <= 0:
        raise ValueError("wacc must be numeric and > 0")
    if not _num(g) or g < 0:
        raise ValueError("g must be numeric and >= 0")
    if wacc <= g:
        raise ValueError("wacc must be greater than g")
    return cashflow_1 / (wacc - g)


def tax_on_gain(sale_price: float, tax_basis: float, tax_rate: float, selling_costs: float = 0.0) -> float:
    """Tax = tax_rate * max(0, sale_price - tax_basis - selling_costs)."""
    for name, v in {"sale_price": sale_price, "tax_basis": tax_basis, "tax_rate": tax_rate, "selling_costs": selling_costs}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if sale_price < 0 or tax_basis < 0 or selling_costs < 0:
        raise ValueError("sale_price, tax_basis, selling_costs must be >= 0")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    gain = max(0.0, sale_price - tax_basis - selling_costs)
    return gain * tax_rate


def net_proceeds(sale_price: float,
                 transaction_fees: float,
                 separation_costs: float,
                 tax: float,
                 wc_true_up: float = 0.0,
                 debt_true_up: float = 0.0,
                 mandatory_debt_repayment: float = 0.0) -> float:
    """Compute net proceeds to seller."""
    for name, v in {
        "sale_price": sale_price,
        "transaction_fees": transaction_fees,
        "separation_costs": separation_costs,
        "tax": tax,
        "wc_true_up": wc_true_up,
        "debt_true_up": debt_true_up,
        "mandatory_debt_repayment": mandatory_debt_repayment
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if sale_price < 0:
        raise ValueError("sale_price must be >= 0")
    for name, v in {
        "transaction_fees": transaction_fees,
        "separation_costs": separation_costs,
        "tax": tax,
        "mandatory_debt_repayment": mandatory_debt_repayment
    }.items():
        if v < 0:
            raise ValueError(f"{name} must be >= 0")

    return sale_price - transaction_fees - separation_costs - tax + wc_true_up + debt_true_up - mandatory_debt_repayment


def value_created(value_remainco_after: float, net_proceeds_value: float, value_combined_before: float) -> float:
    """Value created = (RemainCo after + net proceeds) - combined before."""
    for name, v in {
        "value_remainco_after": value_remainco_after,
        "net_proceeds_value": net_proceeds_value,
        "value_combined_before": value_combined_before
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return (value_remainco_after + net_proceeds_value) - value_combined_before


# Example usage
sale_price = 2_800_000_000
fees = 70_000_000
separation = 180_000_000
tax = tax_on_gain(sale_price, tax_basis=1_600_000_000, tax_rate=0.24, selling_costs=fees)

np = net_proceeds(
    sale_price=sale_price,
    transaction_fees=fees,
    separation_costs=separation,
    tax=tax,
    wc_true_up=-40_000_000,        # e.g., working capital deficit at close
    debt_true_up=0.0,
    mandatory_debt_repayment=600_000_000
)

# PV of stranded costs affecting RemainCo
stranded_cost_annual = 55_000_000
pv_stranded = pv_perpetuity(cashflow_1=stranded_cost_annual, wacc=0.09, g=0.02)

value_combined_before = 12_500_000_000
value_remainco_after = 10_100_000_000 - pv_stranded  # illustrative: base remainco value minus stranded cost PV

vc = value_created(value_remainco_after, np, value_combined_before)

print(f"Tax on gain: ${tax/1e6:.0f}M")
print(f"Net proceeds: ${np/1e9:.2f}B")
print(f"PV stranded costs: ${pv_stranded/1e9:.2f}B")
print(f"Value created: ${vc/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- Divestitures can unlock value by enabling a more appropriate multiple/DCF valuation for each business, but separation effects (stranded costs, lost synergies) can erase benefits if ignored.
- Net proceeds are often much lower than headline price after tax and costs; valuation decisions must focus on net proceeds and opportunity cost.
- Proceeds deployment matters: debt repayment reduces risk (potentially WACC), while share repurchases can concentrate value but depend on buyback price vs intrinsic value.

Impact on multiples:
- Carve-outs often re-rate because “pure play” multiples differ; model the implied re-rating and compare to peer segments.
- Ensure metrics are standalone and include appropriate corporate allocations; otherwise multiples will be biased.

Impact on DCF inputs:
- RemainCo forecasts must include stranded costs and any dis-synergies.
- If proceeds reduce leverage, reflect WACC changes consistently and avoid double-counting the benefit (e.g., both in cash flows and in discount rate adjustments).

Practical adjustments:
```python
def remainco_adjusted_value(base_remainco_value: float, pv_stranded_costs: float, pv_dis_synergies: float = 0.0, pv_benefits: float = 0.0) -> float:
    """Adjust RemainCo value for separation impacts."""
    for name, v in {
        "base_remainco_value": base_remainco_value,
        "pv_stranded_costs": pv_stranded_costs,
        "pv_dis_synergies": pv_dis_synergies,
        "pv_benefits": pv_benefits
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return base_remainco_value - pv_stranded_costs - pv_dis_synergies + pv_benefits
```

---

### Quality of Earnings / Divestiture Red Flags
⚠️ Standalone EBITDA excludes corporate costs that will remain (overstates buyer value, understates seller stranded costs).  
⚠️ Separation costs and TSAs assumed “one-off” but extend materially beyond plan.  
⚠️ Working capital and debt-like true-ups not modelled (closing adjustments can be large).  
⚠️ Tax leakage ignored or assumed minimal without structure analysis (gain taxes, trapped cash).  
⚠️ RemainCo margin assumed unchanged despite lost scale benefits (procurement, shared services).  
✅ Clear bridge from headline price to net proceeds and a quantified stranded-cost plan with mitigation timeline.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Industrials | Shared plants/procurement | Model dis-synergies and stranded overhead; include TSAs and capex to separate |
| Consumer | Brand and channel entanglement | Standalone forecasting must reflect marketing and distribution separation |
| Technology | Shared platform/data | Identify duplicated infrastructure costs; consider IP licensing and ongoing service agreements |
| Energy | Asset retirement obligations | Ensure buyer/seller allocation of environmental liabilities and decommissioning provisions |
| Financials | Regulatory capital and ring-fencing | Model capital impacts and restricted dividend capacity; value on equity/capital framework |

---

### Practical Comparison Tables

#### 1) Divestiture Net Proceeds Bridge
| Line item | Direction | Notes |
|---|---:|---|
| Sale price | + | Headline consideration |
| Fees | − | Advisors, legal, financing |
| Separation/TSA costs | − | One-time cash costs |
| Taxes on gain | − | Depends on basis and structure |
| Working capital true-up | ± | Closing adjustment |
| Debt-like true-ups | ± | Pensions, leases, provisions allocation |
| Mandatory debt repayment | − | Covenants / structure |
| **Net proceeds** |  | Cash available for redeployment |

#### 2) Separation Impact Checklist (RemainCo)
| Item | Where it shows up | What to do |
|---|---|---|
| Stranded corporate costs | FCFF / margins | Time-phase mitigation; include PV drag |
| Lost synergies | FCFF | Model procurement/scale impacts |
| Shared IT/platform | Capex + opex | Add duplication costs and transition capex |
| Contracts and pricing | Revenue/margins | Re-price service agreements; model churn risk |
| Employee/benefits | Opex/provisions | Include severance and retention costs |

---

### Real-World Example (Headlines vs Net Proceeds)

Scenario: A company sells a unit for $2.8B. After fees, separation costs, taxes, working capital true-up, and mandatory debt repayment, only $1.9B is available. Meanwhile, stranded costs reduce RemainCo value by $0.8B PV. Compute value creation vs keeping the unit.

```python
sale_price = 2_800_000_000
fees = 70_000_000
separation = 180_000_000
tax = tax_on_gain(sale_price, tax_basis=1_600_000_000, tax_rate=0.24, selling_costs=fees)

net = net_proceeds(
    sale_price=sale_price,
    transaction_fees=fees,
    separation_costs=separation,
    tax=tax,
    wc_true_up=-40_000_000,
    mandatory_debt_repayment=600_000_000
)

pv_stranded = pv_perpetuity(cashflow_1=55_000_000, wacc=0.09, g=0.02)

base_remainco_value_after = 10_100_000_000
remainco_value = base_remainco_value_after - pv_stranded

combined_before = 12_500_000_000
vc = value_created(remainco_value, net, combined_before)

print(f"Net proceeds available: ${net/1e9:.2f}B")
print(f"RemainCo value after stranded costs: ${remainco_value/1e9:.2f}B")
print(f"Value created vs combined: ${vc/1e9:.2f}B")
```

Interpretation:
- Net proceeds can be materially below headline sale price; deployment and true-ups matter.
- Stranded costs can dominate economics; separation planning is a valuation driver.
- The value-creation test avoids “feel-good” disposals that reduce earnings but do not increase shareholder value.
