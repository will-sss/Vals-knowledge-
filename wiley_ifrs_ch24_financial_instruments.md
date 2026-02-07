# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 24: Financial Instruments (IFRS 9, classification, ECL, hedging, FVOCI/FVTPL impacts)

### Core Concept
IFRS 9 governs classification and measurement of financial assets/liabilities, impairment using expected credit losses (ECL), and hedge accounting. For valuation, IFRS 9 affects earnings volatility (FVTPL vs amortised cost), capital and equity (FVOCI reserves), credit loss provisioning (quality of earnings), and comparability of leverage and interest metrics across firms—especially banks, lenders, and asset-heavy corporates with significant financial assets.

### Formula/Methodology

#### 1) Financial asset classification (decision rules)
```text
Classify financial assets based on:
A) Business model for managing the asset (hold to collect / hold to collect and sell / other), AND
B) Contractual cash flow characteristics test (SPPI: solely payments of principal and interest)

Outcomes:
1) Amortised cost: business model = hold to collect AND SPPI passes
2) FVOCI (debt): business model = hold to collect & sell AND SPPI passes
3) FVTPL: all other cases (including non-SPPI)
Equity investments:
- Default FVTPL, optional irrevocable election FVOCI (no recycling to P&L on disposal; gains/losses stay in OCI)
```

#### 2) Effective interest rate (EIR) and amortised cost
```text
EIR: rate that exactly discounts expected future cash flows to the gross carrying amount at initial recognition.

Amortised cost roll-forward (simplified):
Opening gross carrying amount
+ Interest income (EIR × opening gross)
- Cash received
± Fees/transaction costs amortisation
= Closing gross carrying amount
```

#### 3) Expected credit loss (ECL) model (impairment)
```text
ECL is a probability-weighted estimate of credit losses, discounted at EIR.

12-month ECL (Stage 1): expected credit losses from default events possible within 12 months
Lifetime ECL (Stages 2 & 3): expected losses over the remaining life of the instrument

ECL (common approximation):
ECL = PD × LGD × EAD
Where:
PD = probability of default
LGD = loss given default (1 - recovery rate)
EAD = exposure at default

Discounting:
Discount factor = 1 / (1 + EIR)^t
```

#### 4) Staging (credit deterioration)
```text
Stage 1: no significant increase in credit risk (SICR) since origination -> 12-month ECL
Stage 2: SICR since origination -> lifetime ECL (interest still on gross)
Stage 3: credit-impaired -> lifetime ECL (interest on net carrying amount in many cases)
```

#### 5) Fair value measurement effects (earnings and equity)
```text
FVTPL: fair value changes -> profit or loss (earnings volatility)
FVOCI (debt): FV changes -> OCI, interest + ECL -> P&L; on disposal, FVOCI reserve is recycled to P&L
FVOCI (equity election): FV changes -> OCI; no recycling to P&L on disposal (dividends may go to P&L if meeting criteria)
```

#### 6) Hedge accounting (high-level mechanics)
```text
Goal: align accounting with risk management where hedging relationship is documented and effective.

Fair value hedge:
- Hedged item fair value changes -> P&L
- Hedging instrument fair value changes -> P&L (offset)

Cash flow hedge:
- Effective portion of hedging instrument -> OCI (cash flow hedge reserve)
- Reclassify to P&L when hedged cash flows affect P&L (basis adjustment or recycling)

Net investment hedge:
- Similar to cash flow hedge; effective portion in OCI (translation reserve-related)
```

---

### Practical Application for Valuation Analysts

#### 1) Normalise earnings for measurement category
FVTPL positions can create material non-operating P&L swings (especially where “investment income” is mixed with operating earnings).
Actions:
- Reconcile “operating EBITDA/EBIT” vs fair value gains/losses.
- For multiple-based valuation, consider using “pre-fair-value” operating metrics where the market is not valuing trading book mark-to-market in an EV/EBITDA framework.

#### 2) Understand whether FVOCI is “real” or “paper”
FVOCI debt:
- OCI moves with rates/credit spreads; recycled on disposal.
Actions:
- Treat FVOCI reserve as sensitivity indicator rather than core performance.
- For credit-focused businesses, examine ECL and credit quality rather than OCI volatility.

FVOCI equity election:
- Gains/losses stay in OCI permanently (no recycling).
Actions:
- Be careful interpreting net income for holding company valuations; total comprehensive income may be more informative.

#### 3) ECL is a key quality-of-earnings and cycle indicator
ECL is forward-looking and can move sharply with macro overlays, model updates, or portfolio mix.
Actions:
- Split ECL into: volume/mix, macro overlay, model recalibration, and write-offs.
- Identify “management overlays” and assess persistence.
- For lenders, build credit cost assumptions explicitly in forecasts.

#### 4) Debt covenants and leverage can be affected by IFRS 9
FVTPL liabilities (rare) and own-credit adjustments can move earnings/equity.
Actions:
- Check covenant definitions; many use “frozen GAAP/IFRS” or adjust for fair value movements.
- Ensure net debt and interest expense are on a consistent basis.

#### 5) Hedge accounting reduces earnings volatility—but watch the reserves
Cash flow hedge reserves can accumulate and then unwind into P&L when hedged items occur.
Actions:
- Track hedge reserve roll-forward; incorporate into forecast interest expense or COGS where relevant.
- For commodity hedgers, validate that “hedged” prices align with forecast assumptions.

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

def ecl_pd_lgd_ead(pd: float, lgd: float, ead: float, discount_rate: float = 0.0, t_years: float = 1.0) -> float:
    """
    Compute (discounted) expected credit loss using PD × LGD × EAD.

    Args:
        pd: probability of default (0-1)
        lgd: loss given default (0-1)
        ead: exposure at default (currency)
        discount_rate: discount rate (EIR) as decimal (>= -1)
        t_years: time horizon for discounting in years (>= 0)

    Returns:
        float: ECL amount (currency)

    Raises:
        ValueError: if inputs outside bounds.
    """
    PD = _num(pd, "pd")
    LGD = _num(lgd, "lgd")
    EAD = _num(ead, "ead")
    r = _num(discount_rate, "discount_rate")
    t = _num(t_years, "t_years")

    if not (0.0 <= PD <= 1.0):
        raise ValueError("pd must be between 0 and 1.")
    if not (0.0 <= LGD <= 1.0):
        raise ValueError("lgd must be between 0 and 1.")
    if EAD < 0:
        raise ValueError("ead must be >= 0.")
    if r <= -1.0:
        raise ValueError("discount_rate must be > -1.0.")
    if t < 0:
        raise ValueError("t_years must be >= 0.")

    undiscounted = PD * LGD * EAD
    discount_factor = 1.0 / ((1.0 + r) ** t) if t > 0 else 1.0
    return undiscounted * discount_factor

def stage_ecl(stage: int, pd_12m: float, pd_lifetime: float, lgd: float, ead: float, eir: float, avg_life_years: float) -> float:
    """
    Compute ECL based on IFRS 9 staging.

    Args:
        stage: 1, 2, or 3
        pd_12m: 12-month PD
        pd_lifetime: lifetime PD (over remaining life)
        lgd: loss given default
        ead: exposure at default
        eir: effective interest rate for discounting
        avg_life_years: average remaining life for discounting lifetime ECL

    Returns:
        float: stage-appropriate ECL

    Raises:
        ValueError: invalid stage.
    """
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3.")
    if stage == 1:
        return ecl_pd_lgd_ead(pd_12m, lgd, ead, discount_rate=eir, t_years=1.0)
    # Stage 2/3 lifetime ECL (simplified): discount over average remaining life
    return ecl_pd_lgd_ead(pd_lifetime, lgd, ead, discount_rate=eir, t_years=avg_life_years)

def classify_financial_asset(business_model: str, sppi_pass: bool, is_equity: bool = False, equity_fvoci_election: bool = False) -> str:
    """
    Simplified IFRS 9 classification helper.

    Args:
        business_model: 'htc' (hold to collect), 'htcs' (hold to collect and sell), 'other'
        sppi_pass: SPPI test result
        is_equity: True if equity investment
        equity_fvoci_election: if is_equity, whether FVOCI election applied

    Returns:
        str classification: 'amortised_cost', 'fvoci_debt', 'fvtpl', 'fvoci_equity'

    Raises:
        ValueError: invalid business model.
    """
    bm = business_model.lower().strip()
    if is_equity:
        return "fvoci_equity" if equity_fvoci_election else "fvtpl"
    if bm not in {"htc", "htcs", "other"}:
        raise ValueError("business_model must be 'htc', 'htcs', or 'other'.")
    if bm == "htc" and sppi_pass:
        return "amortised_cost"
    if bm == "htcs" and sppi_pass:
        return "fvoci_debt"
    return "fvtpl"

def fair_value_change_to_pnl_or_oci(classification: str, fair_value_change: float) -> Dict[str, float]:
    """
    Route fair value change to P&L or OCI based on classification.

    Args:
        classification: 'fvtpl', 'fvoci_debt', 'fvoci_equity', 'amortised_cost'
        fair_value_change: period change in fair value

    Returns:
        dict: {'pnl': x, 'oci': y}

    Raises:
        ValueError: invalid classification.
    """
    c = classification
    d = _num(fair_value_change, "fair_value_change")
    if c == "fvtpl":
        return {"pnl": d, "oci": 0.0}
    if c in {"fvoci_debt", "fvoci_equity"}:
        return {"pnl": 0.0, "oci": d}
    if c == "amortised_cost":
        # fair value changes not recognised (unless disclosed); return zeros for accounting impact
        return {"pnl": 0.0, "oci": 0.0}
    raise ValueError("Invalid classification.")

# Example usage
asset_class = classify_financial_asset(business_model="htcs", sppi_pass=True, is_equity=False)
print("Classification:", asset_class)

ecl1 = stage_ecl(stage=1, pd_12m=0.02, pd_lifetime=0.08, lgd=0.45, ead=100_000_000, eir=0.06, avg_life_years=3.0)
ecl2 = stage_ecl(stage=2, pd_12m=0.02, pd_lifetime=0.08, lgd=0.45, ead=100_000_000, eir=0.06, avg_life_years=3.0)
print(f"Stage 1 ECL: ${ecl1/1e6:.2f}M")
print(f"Stage 2 ECL: ${ecl2/1e6:.2f}M")

fv_route = fair_value_change_to_pnl_or_oci(asset_class, fair_value_change=-12_000_000)
print("FV impact:", fv_route)
```

---

### Valuation Impact
Why this matters:
- Classification drives where volatility shows up: P&L (FVTPL) vs OCI (FVOCI). If you use EBITDA or net income as a valuation anchor, measurement category can materially skew your view.
- ECL provisions are forward-looking and can embed management overlays; they are central to forecasting credit costs for lenders and for corporates with material receivables/loans.
- Hedge accounting can smooth earnings, but hedge reserves (OCI) can unwind into P&L and should be understood in forecasts.

Impact on multiples:
- EV/EBITDA: fair value gains/losses and ECL can contaminate operating metrics; normalise where market practice values core operations.
- P/E and P/B: for financials, these often incorporate credit costs; ensure consistent treatment of ECL and fair value movements.

Impact on DCF inputs:
- Forecast credit losses explicitly (PD/LGD/EAD style assumptions) for lending/receivable-heavy models.
- Align cash flows with accounting: ECL is non-cash at recognition but predicts future write-offs; do not treat as pure “add-back” without considering credit cash losses.

Practical adjustments:
```python
def normalize_profit_for_fvtpl(reported_profit: float, fvtpl_gains_losses: float) -> float:
    """
    Remove FVTPL gains/losses to approximate 'core' profit.

    Args:
        reported_profit: profit after tax (or operating profit, define consistently)
        fvtpl_gains_losses: net FVTPL movement included in reported profit (positive increases profit)

    Returns:
        float: normalized profit
    """
    rp = _num(reported_profit, "reported_profit")
    gl = _num(fvtpl_gains_losses, "fvtpl_gains_losses")
    return rp - gl
```

---

### Quality of Earnings Flags
⚠️ Large, recurring FVTPL gains/losses included in “underlying” KPIs without reconciliation.  
⚠️ Material ECL “management overlays” or sudden model changes that reduce provisions near reporting dates.  
⚠️ Stage migration spikes (Stage 1 → Stage 2) without clear credit quality narrative.  
⚠️ High proportion of FVOCI reserves masking economic losses that may recycle on disposal (debt FVOCI).  
✅ Clear disclosures of classification buckets, ECL movements (PD/LGD/EAD drivers), and hedge effectiveness.

---

### Sector-Specific Considerations

| Sector | Key IFRS 9 issue | Typical valuation handling |
|---|---|---|
| Banks / Lenders | ECL staging, P&L sensitivity, P/B valuation | Model credit costs cycle; focus on ROE, cost of equity, and capital |
| Insurers / Asset managers | FVOCI/FVTPL portfolio volatility | Separate fee-based earnings from investment mark-to-market where relevant |
| Industrials | Trade receivables ECL | Assess ageing, SICR triggers; incorporate bad debt and credit insurance |
| Energy / Commodities | Hedge accounting (cash flow hedges) | Track OCI hedge reserves and unwind into revenue/COGS assumptions |

---

### Real-World Example
Scenario: Lender has $5bn loan book, Stage 1 PD 1.5%, Stage 2 PD lifetime 6.0%, LGD 40%, EIR 7%. Stage mix drives profit volatility.

```python
loan_book = 5_000_000_000
stage1_share = 0.85
stage2_share = 0.15

ecl_stage1 = stage_ecl(stage=1, pd_12m=0.015, pd_lifetime=0.06, lgd=0.40, ead=loan_book * stage1_share, eir=0.07, avg_life_years=3.0)
ecl_stage2 = stage_ecl(stage=2, pd_12m=0.015, pd_lifetime=0.06, lgd=0.40, ead=loan_book * stage2_share, eir=0.07, avg_life_years=3.0)

total_ecl = ecl_stage1 + ecl_stage2
print(f"Total ECL allowance: ${total_ecl/1e6:.1f}M")
print(f"ECL as % of loan book: {total_ecl/loan_book:.2%}")
```

Interpretation: Stage migration (credit deterioration) can increase allowances materially; valuation should reflect expected through-the-cycle credit costs, not just current-quarter provisioning.

See also: Chapter 25 (fair value) for valuation techniques and inputs; Chapter 23 (foreign currency) for FX risk on financial instruments; Chapter 18 (provisions) for non-financial provisions vs ECL; Chapter 26 (income taxes) for deferred tax on FVOCI and hedge reserves.
