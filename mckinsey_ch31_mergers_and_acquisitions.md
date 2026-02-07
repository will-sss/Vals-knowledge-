# McKinsey Valuation - Key Concepts for Valuation

## Chapter 31: Mergers and Acquisitions (Synergies, Deal Valuation, Accretion/Dilution, Value Creation Tests)

### Core Concept
M&A valuation is about whether a transaction creates value for the acquirer’s shareholders after paying the purchase price, funding structure costs, taxes, and integration risk. The practical workflow is: value the target standalone, value synergies separately (and conservatively), quantify the premium paid, then test value creation using NPV, IRR, accretion/dilution, and pro forma leverage/coverage under downside cases.

See also: Chapter 16 on moving from enterprise value to value per share (claims and bridge); Chapter 17 on analysing results (sensitivities); Chapter 19 on valuation by parts (break-ups/carve-outs); Chapter 33 on capital structure (funding choices).

---

### Core Concept: Separate the Three Distinct Valuations
1) **Target standalone value** (what it is worth as-is).  
2) **Synergy value** (what it is worth to *this* buyer).  
3) **Price paid** (what you actually pay, including premiums, fees, and financing effects).  

Value creation requires:
- Synergies that are real, timed, and achievable
- Premium not exceeding synergy value (after tax, after costs, risk-adjusted)
- Funding structure not destroying value (excess leverage / dilution / cost of capital impact)

---

### Formula/Methodology

#### 1) Deal NPV (to acquirer shareholders)
```text
Deal NPV = PV(Synergies after tax) − Premium Paid − Transaction Costs − Integration Costs (PV)
```
Where:
- Premium Paid = Purchase price − Target standalone value (at announcement, consistent basis)
- PV discount rate should reflect synergy risk (often higher than base WACC for revenue synergies)

#### 2) Maximum price justified (synergy-based)
```text
Max Purchase Price = Standalone Value + PV(Synergies after tax) − PV(Costs)
```
If price > max price, the deal destroys value (unless strategic options value is explicitly justified).

#### 3) Value per share test (economic, not accounting)
```text
Δ Value per share = (Acquirer value with deal − Acquirer value without deal) / Shares_outstanding
```
Use DCF-based value, not only EPS accretion.

#### 4) Accretion/Dilution (accounting screen; not sufficient)
```text
Pro forma EPS = Pro forma Net Income / Pro forma Shares
Accretion (%) = (Pro forma EPS / Acquirer EPS) − 1
```
Interpretation:
- EPS accretion can occur even when value is destroyed (e.g., buying low P/E with debt) if risk/leverage rises or synergies are overstated.

#### 5) Synergy categories (model explicitly)
```text
Cost synergies: opex/cogs reductions, procurement, footprint rationalisation
Revenue synergies: cross-sell, price uplift, distribution expansion (higher risk)
Financial synergies: tax shields, lower WACC (often overstated), cash redeployment
```
Model synergies as incremental FCFF:
- Incremental EBIT × (1 − tax) minus incremental reinvestment (capex + NWC) and one-time integration costs.

---

### Python Implementation
```python
from typing import List, Dict, Optional

def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def pv_cashflows(cashflows: List[float], rate: float, t0: int = 1) -> float:
    """PV of cashflows at t=t0.."""
    if not isinstance(cashflows, list) or len(cashflows) == 0:
        raise ValueError("cashflows must be a non-empty list")
    if not _num(rate) or rate <= 0:
        raise ValueError("rate must be numeric and > 0")
    if not isinstance(t0, int) or t0 < 0:
        raise ValueError("t0 must be int >= 0")
    total = 0.0
    for i, cf in enumerate(cashflows):
        if not _num(cf):
            raise ValueError("cashflows must be numeric")
        t = t0 + i
        total += cf / ((1.0 + rate) ** t)
    return total


def deal_npv(pv_synergies: float,
             premium_paid: float,
             pv_transaction_costs: float = 0.0,
             pv_integration_costs: float = 0.0) -> float:
    """Deal NPV = PV(synergies) - premium - PV(costs)."""
    for name, v in {
        "pv_synergies": pv_synergies,
        "premium_paid": premium_paid,
        "pv_transaction_costs": pv_transaction_costs,
        "pv_integration_costs": pv_integration_costs
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return pv_synergies - premium_paid - pv_transaction_costs - pv_integration_costs


def max_price_justified(standalone_value: float, pv_synergies: float, pv_costs: float) -> float:
    """Max price = standalone + PV(synergies) - PV(costs)."""
    for name, v in {"standalone_value": standalone_value, "pv_synergies": pv_synergies, "pv_costs": pv_costs}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    return standalone_value + pv_synergies - pv_costs


def accretion(proforma_eps: float, acquirer_eps: float) -> Optional[float]:
    """Accretion = proforma_eps / acquirer_eps - 1. Returns None if acquirer_eps <= 0."""
    if not _num(proforma_eps) or not _num(acquirer_eps):
        raise ValueError("inputs must be numeric")
    if acquirer_eps <= 0:
        return None
    return (proforma_eps / acquirer_eps) - 1.0


def proforma_eps(acquirer_net_income: float,
                 target_net_income: float,
                 after_tax_synergy: float,
                 financing_cost_after_tax: float,
                 new_shares: int,
                 existing_shares: int) -> float:
    """Compute pro forma EPS (simple)."""
    for name, v in {
        "acquirer_net_income": acquirer_net_income,
        "target_net_income": target_net_income,
        "after_tax_synergy": after_tax_synergy,
        "financing_cost_after_tax": financing_cost_after_tax
    }.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if not isinstance(new_shares, int) or new_shares < 0:
        raise ValueError("new_shares must be int >= 0")
    if not isinstance(existing_shares, int) or existing_shares <= 0:
        raise ValueError("existing_shares must be int > 0")
    shares = existing_shares + new_shares
    ni = acquirer_net_income + target_net_income + after_tax_synergy - financing_cost_after_tax
    return ni / shares


# Example usage: PV of synergies and deal NPV
# (Illustrative numbers)
synergy_cashflows_after_tax = [120_000_000, 160_000_000, 180_000_000, 200_000_000, 200_000_000]
synergy_discount = 0.11  # often higher for revenue synergies
pv_syn = pv_cashflows(synergy_cashflows_after_tax, synergy_discount, t0=1)

standalone_value = 5_500_000_000
purchase_price = 6_600_000_000
premium = purchase_price - standalone_value

pv_tx_costs = 80_000_000
pv_integration = 250_000_000

npv = deal_npv(pv_syn, premium, pv_transaction_costs=pv_tx_costs, pv_integration_costs=pv_integration)
max_price = max_price_justified(standalone_value, pv_syn, pv_costs=pv_tx_costs + pv_integration)

print(f"PV synergies: ${pv_syn/1e9:.2f}B")
print(f"Premium paid: ${premium/1e9:.2f}B")
print(f"Deal NPV: ${npv/1e9:.2f}B")
print(f"Max price justified: ${max_price/1e9:.2f}B")
```

---

### Valuation Impact
Why this matters:
- M&A often fails due to overpaid premiums, overstated synergies, and under-modelled integration risk. A disciplined valuation separates these explicitly.
- Even “EPS accretive” deals can destroy value if risk increases or synergies are uncertain; value-based tests prevent accounting traps.
- Deal valuation feeds directly into purchase price allocation assumptions and post-deal performance measurement (synergy tracking, ROIC of acquisition).

Impact on multiples:
- A premium is often justified implicitly by a higher multiple. Convert the premium to required synergies and required ROIC to assess plausibility.
- For multiple-based bids, cross-check implied target EV/EBITDA vs peers and vs the buyer’s implied terminal multiple.

Impact on DCF inputs:
- Synergies should be modelled as incremental FCFF with explicit timing, taxes, and reinvestment, and discounted at an appropriate risk level.
- Financing effects should be reflected consistently: either via WACC changes (careful) or via APV-style tax shield modelling.

Practical adjustments:
```python
def synergy_fcff_incremental(ebit_synergy: float, tax_rate: float, delta_nwc: float, capex: float, one_time_cost: float = 0.0) -> float:
    """Incremental FCFF from synergies in a period."""
    if not _num(ebit_synergy) or not _num(tax_rate) or not _num(delta_nwc) or not _num(capex) or not _num(one_time_cost):
        raise ValueError("inputs must be numeric")
    if not (0.0 <= tax_rate < 1.0):
        raise ValueError("tax_rate must be in [0,1)")
    if capex < 0:
        raise ValueError("capex must be >= 0")
    return ebit_synergy * (1.0 - tax_rate) - delta_nwc - capex - one_time_cost
```

---

### Quality of Earnings / Deal Red Flags
⚠️ “Synergies” are mostly revenue synergies with weak customer evidence or long payback.  
⚠️ Synergy plan ignores reinvestment needed to achieve revenue uplift (salesforce, capex, working capital).  
⚠️ Premium justified by “multiple expansion” rather than cash flow improvements.  
⚠️ Integration costs treated as non-recurring but recur in multiple deals or years.  
⚠️ Pro forma earnings rely on aggressive “purchase accounting” add-backs or cost capitalisation.  
✅ Synergies are bottom-up, time-phased, risk-adjusted, and tracked with owners and milestones.

---

### Sector-Specific Considerations

| Sector | Key issue | Typical treatment |
|---|---|---|
| Software | Retention and product roadmap risk | Discount revenue synergies higher; stress churn and integration disruption |
| Healthcare | Regulatory and pipeline uncertainty | Scenario-weighted cash flows; longer integration timelines |
| Industrials | Footprint and procurement synergies | Cost synergies more credible; model one-time restructuring cash costs |
| Financials | Capital and regulatory constraints | Value with capital/ROE framework; ensure regulatory capital impacts are modelled |
| Consumer | Brand dilution and channel conflict | Stress volume and pricing impacts; treat revenue synergies cautiously |

---

### Practical Comparison Tables

#### 1) Synergy Credibility Checklist
| Synergy type | Evidence required | Risk level | Typical modelling notes |
|---|---|---|---|
| Cost synergies | Headcount/site plans, contracts | Medium | Time-phase; include severance and exit costs |
| Procurement | Supplier overlap and terms | Medium | Model price reductions; include renegotiation timing |
| Revenue synergies | Customer pipeline, win rates | High | Model ramp and churn risk; higher discount rate |
| Financial (tax/WACC) | Tax structure, capital market data | Medium | Avoid overstating; ensure consistency with WACC/APV |

#### 2) Deal Value-Creation Tests (Minimum Set)
| Test | Pass condition | Why |
|---|---|---|
| Deal NPV | NPV > 0 | Value creation after paying premium and costs |
| Max price | Price ≤ max justified | Discipline on bid ceiling |
| Downside case | NPV stays acceptable | Robustness to integration risk |
| Leverage/coverage | Within tolerances | Avoids distress risk and rating impact |
| ROIC on acquisition | Exceeds WACC over time | Economic profitability of deployed capital |

---

### Real-World Example (Synergy PV + Accretion Screen)

Scenario: A buyer pays a premium and expects synergies over 5 years. Calculate PV synergies, deal NPV, max price, then compute a simple EPS accretion screen.

```python
# PV synergy value
synergy_cashflows_after_tax = [120_000_000, 160_000_000, 180_000_000, 200_000_000, 200_000_000]
pv_syn = pv_cashflows(synergy_cashflows_after_tax, rate=0.11, t0=1)

standalone_value = 5_500_000_000
purchase_price = 6_600_000_000
premium = purchase_price - standalone_value

pv_costs = 80_000_000 + 250_000_000
npv = deal_npv(pv_syn, premium, pv_transaction_costs=80_000_000, pv_integration_costs=250_000_000)
max_price = max_price_justified(standalone_value, pv_syn, pv_costs=pv_costs)

print(f"Deal NPV: ${npv/1e9:.2f}B")
print(f"Max price justified: ${max_price/1e9:.2f}B")

# EPS accretion screen (simplified)
acquirer_eps = 2.40
acquirer_ni = 2_400_000_000
target_ni = 420_000_000

after_tax_synergy_year1 = 120_000_000
financing_cost_after_tax = 180_000_000  # after-tax interest cost or foregone earnings

pro_eps = proforma_eps(acquirer_ni, target_ni, after_tax_synergy_year1, financing_cost_after_tax, new_shares=30_000_000, existing_shares=1_000_000_000)
acc = accretion(pro_eps, acquirer_eps)

print(f"Pro forma EPS: ${pro_eps:.2f}")
print(f"Accretion: {acc:.1%}" if acc is not None else "Accretion: n/a")
```

Interpretation:
- If deal NPV is negative but EPS is accretive, the deal is likely value-destructive despite “headline” accretion.
- Use max price and downside scenarios to set a disciplined bid ceiling and structure protection (earn-outs, contingent value, walk-away thresholds).
