# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 23: Foreign Currency (IAS 21, FX translation, functional currency, hedging impacts on valuation)

### Core Concept
IAS 21 determines how to account for foreign currency transactions and how to translate the results and financial position of foreign operations. For valuation, FX affects comparability of historical performance, the mapping from local-currency cash flows to a valuation currency, and equity movements through translation reserves (OCI). Misunderstanding functional currency and translation mechanics can lead to inconsistent margins, ROIC, and leverage analysis across geographies.

### Formula/Methodology

#### 1) Key definitions (practical)
```text
Functional currency: currency of the primary economic environment in which the entity operates (drives pricing, costs, financing).

Presentation currency: currency in which financial statements are presented (can differ from functional).

Foreign operation: a subsidiary/branch/associate/joint arrangement whose activities are based in a different currency environment.
```

#### 2) Foreign currency transactions (functional currency accounting)
```text
At initial recognition:
Record transaction in functional currency using spot FX rate at transaction date.

At subsequent measurement:
- Monetary items (cash, receivables, payables, loans): retranslate at closing rate; FX gains/losses in P&L.
- Non-monetary items at historical cost (PPE at cost, inventory at cost): keep historical rate (no retranslation).
- Non-monetary items at fair value: translate using the rate at the date fair value is determined; FX effects follow where the fair value change is recognised (P&L or OCI).
```

#### 3) Translating a foreign operation into presentation currency (consolidation)
```text
Income statement:
- Translate income and expenses at average rate (or transaction-date rates if practical).

Balance sheet:
- Translate assets and liabilities at closing rate.
- Translate equity components at historical rates (share capital often at issuance-date rates).

Translation difference:
- Recognise in OCI (foreign currency translation reserve, FCTR), as part of equity.
```

#### 4) Disposal of a foreign operation (recycling)
```text
On disposal (loss of control/significant influence):
Reclassify cumulative translation differences (FCTR) from OCI to P&L (recycling),
consistent with the extent of the disposal.
```

#### 5) Valuation currency consistency (DCF principle)
```text
Rule: discount rate and cash flows must be in the same currency.

If cash flows are forecast in local currency:
- Use a local-currency discount rate (incorporating local inflation/FX expectations), OR
- Convert cash flows to valuation currency using expected FX rates and use valuation-currency discount rate.

Never mix: local-currency cash flows with USD/GBP discount rate without FX conversion.
```

---

### Practical Application for Valuation Analysts

#### 1) Determine functional currency before doing anything else
Analyst checks:
- Where are sales prices set (currency)?
- Where are costs incurred (currency)?
- Financing currency and cash retention currency?
Practical impact:
- If an entity has a “mismatched” functional currency, transaction FX gains/losses can create volatile P&L.

#### 2) Separate operating performance from FX translation noise
Common pitfalls:
- Revenue growth in presentation currency can be driven by FX rather than volume/price.
- Margin changes can be FX mix effects rather than operating improvements.
Practical approach:
- Analyse both “constant currency” and “reported currency”.
- Recreate constant-currency growth by holding FX rates constant to a base period.

#### 3) Treatment of translation reserves (FCTR) in valuation
FCTR is not “cash” and does not represent distributable earnings in many regimes.
Valuation actions:
- Do not treat translation reserve as operating income; it is equity movement through OCI.
- When valuing equity, FCTR remains part of equity; but for per-share value, focus on cash flow generating ability rather than OCI swings.
- For M&A, be aware of recycling to P&L on disposal which can create one-off gains/losses.

#### 4) Net debt and FX: avoid leverage misreads
Monetary liabilities denominated in foreign currency create FX gains/losses and can change net debt.
Valuation actions:
- Identify FX-denominated debt and hedges.
- Consider whether debt is naturally hedged by foreign-currency cash flows.
- For EV bridge, use net debt in valuation currency at closing rate; if a big move, explain sensitivity.

#### 5) Hyperinflation and extreme FX regimes
If IAS 29 (hyperinflation) applies, translation mechanics change materially.
Valuation actions:
- Use real vs nominal consistency; treat inflation and discount rate currency explicitly.
- Avoid using raw historical multiples without normalising for inflation/FX distortions.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def translate_is_average_rate(is_items: Dict[str, float], avg_rate: float) -> Dict[str, float]:
    """
    Translate income statement items using an average FX rate.

    Args:
        is_items: dict of line item -> amount in functional currency
        avg_rate: presentation per functional (e.g., USD per EUR)

    Returns:
        dict: translated items in presentation currency

    Raises:
        ValueError: invalid rate.
    """
    r = _num(avg_rate, "avg_rate")
    if r <= 0:
        raise ValueError("avg_rate must be > 0.")
    return {k: _num(v, f"is_items[{k}]") * r for k, v in is_items.items()}

def translate_bs_closing_rate(bs_assets: Dict[str, float], bs_liabs: Dict[str, float], closing_rate: float) -> Dict[str, Dict[str, float]]:
    """
    Translate balance sheet assets and liabilities at closing FX rate.

    Args:
        bs_assets: dict of asset line -> amount (functional)
        bs_liabs: dict of liability line -> amount (functional)
        closing_rate: presentation per functional

    Returns:
        dict with translated 'assets' and 'liabilities'

    Raises:
        ValueError: invalid rate.
    """
    r = _num(closing_rate, "closing_rate")
    if r <= 0:
        raise ValueError("closing_rate must be > 0.")
    assets_t = {k: _num(v, f"bs_assets[{k}]") * r for k, v in bs_assets.items()}
    liabs_t = {k: _num(v, f"bs_liabs[{k}]") * r for k, v in bs_liabs.items()}
    return {"assets": assets_t, "liabilities": liabs_t}

def fx_gain_loss_monetary(opening_fc: float, closing_fc: float, opening_rate: float, closing_rate: float) -> float:
    """
    FX gain/loss on a monetary item in functional currency measured at amortised cost (simplified).

    For a foreign-currency monetary item held in functional currency accounting:
    - If item is denominated in foreign currency, remeasure at closing rate; gain/loss is difference.

    This function is for a simple exposure measured in foreign currency units:
      Value_presentation_open = opening_fc * opening_rate
      Value_presentation_close = closing_fc * closing_rate
      FX P&L approx = (closing_fc * closing_rate) - (opening_fc * opening_rate)

    Args:
        opening_fc: opening amount in foreign currency units
        closing_fc: closing amount in foreign currency units
        opening_rate: functional per foreign (or choose convention consistently)
        closing_rate: same convention

    Returns:
        float: FX gain/loss in functional currency units based on chosen convention

    Raises:
        ValueError: invalid rates.
    """
    ofc = _num(opening_fc, "opening_fc")
    cfc = _num(closing_fc, "closing_fc")
    r0 = _num(opening_rate, "opening_rate")
    r1 = _num(closing_rate, "closing_rate")
    if r0 <= 0 or r1 <= 0:
        raise ValueError("Rates must be > 0.")
    return (cfc * r1) - (ofc * r0)

def constant_currency_growth(current_revenue: float, prior_revenue: float, current_rate: float, base_rate: float) -> float:
    """
    Compute constant-currency revenue growth by restating current period revenue at a base FX rate.

    Convention:
      - Revenue in functional currency -> translate to presentation
      - If current revenue is already translated at current_rate, approximate underlying functional:
          functional = current_revenue / current_rate
        then restate at base_rate:
          restated = functional * base_rate

    Args:
        current_revenue: reported current period revenue in presentation currency
        prior_revenue: reported prior period revenue in presentation currency
        current_rate: FX used in current period translation
        base_rate: FX to restate current revenue (e.g., prior period avg)

    Returns:
        float: constant-currency growth rate

    Raises:
        ValueError: invalid inputs or prior_revenue=0.
    """
    cur = _num(current_revenue, "current_revenue")
    prev = _num(prior_revenue, "prior_revenue")
    r_cur = _num(current_rate, "current_rate")
    r_base = _num(base_rate, "base_rate")
    if r_cur <= 0 or r_base <= 0:
        raise ValueError("Rates must be > 0.")
    if prev == 0:
        raise ValueError("prior_revenue must be non-zero.")
    functional = cur / r_cur
    restated = functional * r_base
    return (restated - prev) / prev

# Example usage
is_fc = {"revenue": 1_000.0, "ebit": 150.0}
is_usd = translate_is_average_rate(is_fc, avg_rate=1.10)  # USD per EUR example
print("Translated IS (USD):", is_usd)

assets_fc = {"cash": 200.0, "ppe": 800.0}
liabs_fc = {"debt": 500.0, "payables": 150.0}
bs_usd = translate_bs_closing_rate(assets_fc, liabs_fc, closing_rate=1.12)
print("Translated BS (USD):", bs_usd)

# Constant-currency growth
g_cc = constant_currency_growth(current_revenue=1_100.0, prior_revenue=1_050.0, current_rate=1.10, base_rate=1.05)
print(f"Constant-currency revenue growth: {g_cc:.1%}")
```

---

### Valuation Impact
Why this matters:
- FX can create apparent growth/decline that is purely translation; valuation should anchor on underlying volume/price dynamics and local-currency margins.
- Discount rates and cash flows must be currency-consistent; mismatches create systematic valuation errors.
- FX translation reserves (OCI) can be large but are not operating performance; they matter mainly on disposal (recycling) and in equity analysis context.

Impact on multiples:
- EV/EBITDA and P/E can swing due to FX; peers with different currency mixes require constant-currency analysis.
- Market multiples often embed currency risk; avoid naïve cross-region comparisons without adjusting for inflation/FX regimes.

Impact on DCF inputs:
- Growth: forecast in local currencies where drivers are local; then convert using expected FX or use local discount rates.
- WACC: ensure risk-free rate, equity risk premium, and inflation assumptions align with the currency of cash flows.
- Terminal value: long-run FX assumptions can dominate value for emerging market exposures.

Practical adjustments:
```python
def convert_cashflows(cashflows: List[float], fx_rates: List[float]) -> List[float]:
    """
    Convert a series of local-currency cash flows into valuation currency using expected FX rates.

    Args:
        cashflows: local currency cash flows
        fx_rates: valuation currency per local currency for each period

    Raises:
        ValueError: mismatched lengths or invalid rates.
    """
    if len(cashflows) != len(fx_rates):
        raise ValueError("cashflows and fx_rates must have the same length.")
    out = []
    for i, (cf, r) in enumerate(zip(cashflows, fx_rates)):
        c = _num(cf, f"cashflows[{i}]")
        rr = _num(r, f"fx_rates[{i}]")
        if rr <= 0:
            raise ValueError("fx rate must be > 0.")
        out.append(c * rr)
    return out
```

---

### Quality of Earnings Flags
⚠️ Large recurring FX gains/losses included in “operating profit” or adjusted EBIT without clear reconciliation.  
⚠️ Functional currency choice appears inconsistent with revenue/cost drivers (potential earnings management).  
⚠️ Significant FX-denominated debt without disclosed hedging policy or natural hedge.  
⚠️ Large translation reserve movements driving equity changes without narrative (risk of disposal-related P&L shocks).  
✅ Transparent constant-currency disclosures and clear separation of transaction FX vs translation FX effects.

---

### Sector-Specific Considerations

| Sector | Key FX issue | Typical valuation handling |
|---|---|---|
| Consumer multinationals | Translation impacts on revenue growth | Constant-currency growth and local margin analysis; scenario FX |
| Commodities | Sales in USD, costs in local | Natural hedge; model local cost inflation and FX pass-through |
| Financials | FX on capital and RWAs | Align valuation currency with reporting and regulatory capital; careful with OCI reserves |
| Emerging markets | FX volatility + inflation | Currency-consistent DCF; stress tests; consider sovereign risk premium |

---

### Real-World Example
Scenario: Global business reports 8% revenue growth in USD, but USD strengthened vs EUR. You want constant-currency growth to assess true operating momentum.

```python
reported_current_usd = 5_400_000_000
reported_prior_usd = 5_000_000_000
avg_rate_current = 1.08  # USD per EUR
avg_rate_prior = 1.00    # base

cc_growth = constant_currency_growth(
    current_revenue=reported_current_usd,
    prior_revenue=reported_prior_usd,
    current_rate=avg_rate_current,
    base_rate=avg_rate_prior
)
print(f"Constant-currency growth (restated at prior FX): {cc_growth:.1%}")
```

Interpretation: If constant-currency growth is materially lower than reported, the “growth” is FX translation-driven and should not be extrapolated into terminal value assumptions.

See also: Chapter 6 (cash flows) for translation and cash conversion; Chapter 24 (financial instruments) for FX risk management and hedge accounting interactions; Chapter 25 (fair value) for FX on fair-valued items; Chapter 26 (income taxes) for FX on deferred taxes in foreign operations.
