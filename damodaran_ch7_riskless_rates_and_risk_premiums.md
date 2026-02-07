# Damodaran Investment Valuation (2023) - Key Concepts for Valuation

## Chapter 7: Riskless Rates and Risk Premiums (Risk-Free Rate, Equity Risk Premium, Country Risk, Currency Consistency)

### Core Concept
Riskless rates and risk premiums are the foundation of discount rates in valuation: they determine the baseline time value of money and the market price of risk. Practically, you must (1) choose a **risk-free rate consistent with currency and cash-flow timing**, (2) select and justify an **equity risk premium (ERP)**, and (3) adjust for **country risk** and other systematic risk components when valuing firms with material exposure outside mature markets. fileciteturn3file2

See also: Chapter 8 (Risk parameters and costs of financing), Chapter 15 (WACC and APV approaches), Chapter 12 (Terminal value: long-run inputs must align with risk-free/ERP choices).

---

### Formula/Methodology

#### 1) Cost of equity (CAPM baseline)
```text
k_e = R_f + β × ERP
Where:
R_f = risk-free rate (currency-consistent)
β   = exposure to market risk (equity beta)
ERP = equity risk premium for the market (mature-market ERP + any incremental premia)
```
Practical rule:
- If cash flows are in **USD**, use a **USD risk-free rate** and **USD-consistent ERP**.
- If cash flows are in **local currency**, use a **local-currency risk-free rate** and **ERP consistent with that currency/market**.

#### 2) Risk-free rate selection (what “risk-free” means operationally)
Risk-free rate should be:
- **Default-free** (or near) for the currency
- **Matched to duration** of cash flows (or use a term structure)

Common implementation:
```text
Risk-free rate proxy = Yield on government bond in same currency (long-term for long-horizon valuations)
```
If the government bond is not default-free (emerging markets), consider:
- Using a **default-free proxy** (e.g., USD/Euro) + inflation/currency adjustments, or
- Backing out a **default spread** and removing it to approximate a risk-free local rate (methodology varies). fileciteturn3file2

#### 3) Equity risk premium (ERP)
ERP is the market compensation for bearing equity risk (over risk-free). In practice you will use one of:
- **Historical ERP**: average equity excess return over long history
- **Implied ERP**: solve for ERP that makes market price equal to PV of expected cash flows
- **Survey/consensus ERP**: practitioner estimates (useful cross-check)

Generic implied ERP setup:
```text
Market Level: Price = PV(Expected Cash Flows discounted at R_f + ERP)
Solve ERP such that PV = Price
```
Practical note: implied ERP is “live” and moves with market prices and expected cash flows. fileciteturn3file2

#### 4) Country risk premium (CRP) and exposure weighting
When valuing companies exposed to country risk beyond mature markets, a practical approach is:
```text
ERP_total = ERP_mature + CRP_exposure
CRP_exposure = Σ (Revenue_weight_c × CRP_c)
```
Where:
- Revenue_weight_c = share of revenues (or operating exposure) in country c
- CRP_c = incremental country risk premium for country c (often tied to sovereign default spreads, scaled for equity risk)

A common scaling heuristic:
```text
CRP_equity ≈ Default Spread × (σ_equity / σ_bond)
```
Where:
- Default Spread is sovereign spread over a default-free benchmark
- σ_equity and σ_bond are volatilities of equity index and sovereign bond (or proxies)

Use caution: scaling is noisy; treat as a range and run sensitivities. fileciteturn3file2

#### 5) Currency consistency (non-negotiable)
```text
Nominal cash flows (in currency X) must be discounted at nominal rates in currency X.
Real cash flows must be discounted at real rates.
Do not mix real/nominal or currency-inconsistent Rf/ERP.
```
If you switch currency, switch both:
- Risk-free rate
- Inflation assumptions
- Equity risk premium/country risk treatment

---

### Python Implementation
```python
from typing import Dict, List, Optional, Tuple


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def capm_cost_of_equity(rf: float, beta: float, erp: float) -> float:
    """k_e = rf + beta * erp"""
    for name, v in {"rf": rf, "beta": beta, "erp": erp}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if rf <= -1:
        raise ValueError("rf must be > -100%")
    if erp < 0:
        raise ValueError("erp must be >= 0")
    return rf + beta * erp


def equity_risk_premium_total(mature_erp: float, crp_exposure: float = 0.0) -> float:
    """ERP_total = mature_erp + crp_exposure."""
    for name, v in {"mature_erp": mature_erp, "crp_exposure": crp_exposure}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if mature_erp < 0:
        raise ValueError("mature_erp must be >= 0")
    if crp_exposure < 0:
        raise ValueError("crp_exposure must be >= 0")
    return mature_erp + crp_exposure


def country_risk_premium_scaled(default_spread: float, sigma_equity: float, sigma_bond: float) -> float:
    """CRP ≈ default_spread * (sigma_equity / sigma_bond)."""
    for name, v in {"default_spread": default_spread, "sigma_equity": sigma_equity, "sigma_bond": sigma_bond}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if default_spread < 0:
        raise ValueError("default_spread must be >= 0")
    if sigma_equity <= 0 or sigma_bond <= 0:
        raise ValueError("sigmas must be > 0")
    return default_spread * (sigma_equity / sigma_bond)


def exposure_weighted_crp(exposures: List[Tuple[str, float, float]]) -> float:
    """
    Exposure-weighted CRP.

    Args:
        exposures: list of (country, weight, crp_country) where weight is revenue/exposure share in [0,1].

    Returns:
        float: CRP exposure as decimal.

    Raises:
        ValueError: invalid weights or missing data.
    """
    if not isinstance(exposures, list) or len(exposures) == 0:
        raise ValueError("exposures must be a non-empty list")
    total_w = 0.0
    total = 0.0
    for c, w, crp in exposures:
        if not _num(w) or not (0.0 <= w <= 1.0):
            raise ValueError("weights must be in [0,1]")
        if not _num(crp) or crp < 0:
            raise ValueError("crp must be >= 0")
        total_w += w
        total += w * crp
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"exposure weights must sum to 1 (got {total_w})")
    return total


def nominal_rate_from_real(real_rate: float, inflation: float) -> float:
    """Nominal ≈ (1+real)*(1+inflation) - 1."""
    for name, v in {"real_rate": real_rate, "inflation": inflation}.items():
        if not _num(v):
            raise ValueError(f"{name} must be numeric")
    if real_rate <= -1 or inflation <= -1:
        raise ValueError("real_rate and inflation must be > -100%")
    return (1.0 + real_rate) * (1.0 + inflation) - 1.0


# Example usage: build a currency-consistent cost of equity with country exposure
rf_usd = 0.042
beta = 1.15
mature_erp = 0.050

# Country CRP example: two countries, exposure-weighted
exposures = [
    ("MatureMarket", 0.70, 0.000),
    ("EmergingA",     0.30, 0.030),
]
crp_exposure = exposure_weighted_crp(exposures)
erp_total = equity_risk_premium_total(mature_erp, crp_exposure=crp_exposure)
ke = capm_cost_of_equity(rf_usd, beta, erp_total)

print(f"CRP exposure: {crp_exposure:.1%}")
print(f"Total ERP:    {erp_total:.1%}")
print(f"Cost of equity (USD): {ke:.1%}")

# Currency consistency demo: converting real to nominal
real_rf = 0.015
usd_infl = 0.025
nominal_rf = nominal_rate_from_real(real_rf, usd_infl)
print(f"Nominal risk-free (approx): {nominal_rf:.1%}")
```

---

### Valuation Impact
Why this matters:
- Risk-free and ERP assumptions set the **level of discount rates**; small changes can create large value swings when terminal value is a big share of total value. fileciteturn3file2  
- Country risk affects both **discount rates** and **cash flow risk**: if exposure is material, ignoring CRP will systematically overvalue businesses operating in riskier environments.
- Currency inconsistency (discounting local currency cash flows with USD rates, or mixing real and nominal) is a common, high-impact modelling error.

Impact on multiples:
- Higher implied discount rates (via higher ERP/CRP) should correspond to lower valuation multiples for otherwise similar growth/ROIC profiles.
- If peers differ in country exposure, multiples must be risk-adjusted (directly via regressions or indirectly via adjusted discount rates and DCF cross-checks).

Impact on DCF inputs:
- Use a **currency-consistent** Rf and ERP; align inflation assumptions with the currency used for cash flows and terminal growth.
- Run sensitivities: ERP (±1–2%), CRP range, and risk-free curve shifts, particularly if the valuation is terminal-value heavy.

Practical adjustments:
```python
def cost_of_equity_with_crp(rf: float, beta: float, mature_erp: float, crp_exposure: float) -> float:
    """k_e = rf + beta * (mature_erp + crp_exposure)."""
    erp_total = equity_risk_premium_total(mature_erp, crp_exposure)
    return capm_cost_of_equity(rf, beta, erp_total)
```

---

### Quality of Earnings Flags (Risk/Discount-Rate Signals)
⚠️ Material emerging-market revenues with no CRP adjustment (discount rate too low).  
⚠️ “Risk-free rate” chosen from a different currency than the model cash flows.  
⚠️ Using a short-term bill rate to discount long-horizon cash flows without a term structure rationale.  
⚠️ ERP input changes year-to-year without documented method (ad hoc tuning).  
⚠️ Terminal growth set above nominal risk-free or above long-run nominal GDP for the valuation currency (inconsistent with macro constraints).  
✅ Green flag: explicit documentation of currency, risk-free tenor, ERP method (historical vs implied), and country exposure weights with sensitivity tables.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Global consumer | Geographic revenue mix | Compute exposure-weighted CRP using revenue shares by region/country. |
| Commodities | Country + commodity cyclicality | Treat CRP as one input; also run commodity price scenarios and stress cash flows. |
| Financials | Sovereign-bank linkage | Country risk can transmit directly; consider higher CRP and stress sovereign exposure. |
| Infrastructure | Contracted cash flows but political risk | Use CRP and scenario analysis (regulatory/political outcomes). |

---

### Practical Comparison Tables

#### 1) Risk-free selection checklist
| Question | Decision rule | Common mistake |
|---|---|---|
| What currency are cash flows in? | Use Rf in same currency | Discounting local cash flows at USD Rf |
| What horizon drives value? | Use long-term Rf or term structure | Using 3m bills for 10y+ value |
| Is the sovereign default-free? | If not, adjust/remove default spread | Treating risky sovereign yield as risk-free |

#### 2) ERP/CRP implementation choices
| Input | Practical approach | When to be cautious |
|---|---|---|
| ERP | Use one method consistently; implied ERP for “live” market | During bubbles/crashes; implied ERP can swing sharply |
| CRP | Scale from default spread; weight by exposure | If exposures are poorly disclosed; treat as range |
| Currency inflation | Keep nominal/real consistent | Mixing real growth with nominal discount rates |

---

### Real-World Example
Scenario: A USD DCF values a company with 30% exposure to a higher-risk country. You compute an exposure-weighted CRP and update k_e. Then you sanity-check terminal growth vs risk-free rate consistency.

```python
rf_usd = 0.042
beta = 1.15
mature_erp = 0.050

crp_exposure = exposure_weighted_crp([
    ("MatureMarket", 0.70, 0.000),
    ("EmergingA",    0.30, 0.030),
])

ke = cost_of_equity_with_crp(rf_usd, beta, mature_erp, crp_exposure)
print(f"Cost of equity with CRP: {ke:.1%}")

terminal_g = 0.03
if terminal_g >= rf_usd:
    raise ValueError("Terminal growth should generally be below the nominal risk-free rate in the valuation currency.")
print("Terminal growth check passed.")
```

Interpretation: The CRP adjustment raises the discount rate, reducing intrinsic value unless cash flows are correspondingly more resilient. Terminal assumptions must be internally consistent: a high terminal growth rate paired with a low risk-free/ERP set is usually a modelling error or an implicit “optimism premium”.
