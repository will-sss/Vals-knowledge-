# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 6: Statement of Cash Flows (IAS 7, FCF construction, operating vs financing classification)

### Core Concept
IAS 7 explains how cash flows are classified into operating, investing, and financing activities and how cash flow statements reconcile to cash movements. For valuation, cash flows are the bridge between accounting earnings and intrinsic value: you need a consistent free cash flow (FCF) definition and you must understand classification choices (especially interest, taxes, leases, and working capital) to avoid double-counting or misreading cash conversion.

### Formula/Methodology

#### 1) Cash flow categories
```text
Net Change in Cash = CFO + CFI + CFF
Where:
CFO = cash flows from operating activities
CFI = cash flows from investing activities
CFF = cash flows from financing activities
```

#### 2) Free cash flow (common enterprise DCF definition)
```text
Free Cash Flow (FCF) = NOPAT + D&A - ΔNet Working Capital - Capex
Where:
NOPAT = EBIT × (1 - Tax rate)
D&A = depreciation + amortization (non-cash)
ΔNet Working Capital = change in operating working capital (increase is cash outflow)
Capex = cash paid for property, plant and equipment and other operating capex
```

Alternative “cash-flow statement” build (useful when you trust IAS 7 cash lines):
```text
FCF (from cash flow statement) = CFO (pre-interest, pre-tax) - Maintenance Capex
or
FCF = CFO - Capex
Adjustments:
- If CFO includes interest or tax, ensure consistency with WACC/discounting and NOPAT definition.
- If lease principal is in financing, treat it like debt service (do not subtract again in operating FCF unless you also adjust EBITDA).
```

#### 3) Working capital from cash flows (indirect method)
```text
CFO (indirect) = Profit before tax
                + Non-cash items (D&A, impairments, provisions, SBC, FV movements)
                ± Working capital movements (Δreceivables, Δinventory, Δpayables, etc.)
                - Income taxes paid (if presented within CFO)
```

#### 4) Interest and dividends classification (IAS 7 policy choice)
```text
IAS 7 allows policy choices (must be applied consistently):
- Interest paid: Operating or Financing
- Interest received: Operating or Investing
- Dividends received: Operating or Investing
- Dividends paid: Financing (commonly), but IAS 7 permits Operating in some cases
```

---

### Practical Application for Valuation Analysts

#### 1) Choose and enforce a single FCF definition
For enterprise DCF with WACC discounting:
- Target a pre-financing cash flow: operating after tax, before financing flows.
- Ensure interest paid and lease principal are treated consistently (either include as financing and exclude from FCF, or include in FCF and adjust discount rate and capital structure logic accordingly).

Recommended enterprise FCF policy (typical):
- Start from NOPAT
- Add back non-cash items
- Subtract ΔNWC and Capex
- Exclude interest paid/received and dividends
- Treat lease principal as financing (debt-like); if you adjust EBITDA for IFRS 16, keep lease principal out of operating FCF

#### 2) Indirect method: validate cash conversion and spot earnings quality issues
Key checks:
- CFO consistently below EBITDA can indicate aggressive revenue recognition, capitalised costs, or working-capital drag.
- Large add-backs (impairment, provisions) can mask weak cash generation if they recur.
- “Other working capital” swings may hide reclassifications or supply chain finance.

#### 3) IAS 7 policy variability: comparability adjustments in peer analysis
If peers classify interest paid differently:
- Reclassify to a consistent basis for cash conversion metrics.
- Avoid comparing “CFO” across companies without verifying classification.

#### 4) Leases (IFRS 16) interaction
After IFRS 16, lease payments split into:
- Interest portion (often in CFO or CFF depending on policy)
- Principal portion (typically in CFF)
This can mechanically increase CFO versus pre-IFRS 16 (because principal moves to financing), so cash conversion metrics must be adjusted for comparability.

---

### Python Implementation
```python
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def compute_fcf_from_nopat(
    ebit: float,
    tax_rate: float,
    depreciation: float,
    amortization: float,
    delta_nwc: float,
    capex: float
) -> float:
    """
    Enterprise FCF build: NOPAT + D&A - ΔNWC - Capex.

    Args:
        ebit: earnings before interest and tax
        tax_rate: effective operating tax rate (0..1)
        depreciation: period depreciation expense (non-cash proxy)
        amortization: period amortization expense (non-cash proxy)
        delta_nwc: change in net working capital (increase = cash outflow, so subtract)
        capex: cash paid for capex (positive number)

    Returns:
        float: free cash flow (enterprise)

    Raises:
        ValueError: invalid inputs.
    """
    e = _num(ebit, "ebit")
    t = _num(tax_rate, "tax_rate")
    if not (0.0 <= t <= 1.0):
        raise ValueError("tax_rate must be between 0 and 1.")
    d = _num(depreciation, "depreciation")
    a = _num(amortization, "amortization")
    nwc = _num(delta_nwc, "delta_nwc")
    cx = _num(capex, "capex")
    if cx < 0:
        raise ValueError("capex should be a positive cash outflow amount.")
    nopat = e * (1.0 - t)
    return nopat + d + a - nwc - cx

def compute_fcf_from_cashflow(cfo: float, capex: float, include_interest: bool = False, interest_paid: float = 0.0) -> float:
    """
    Simple FCF build from cash flow statement: FCF = CFO - Capex.
    Optionally remove interest paid if CFO includes it and you want pre-financing FCF.

    Args:
        cfo: cash flow from operations (as reported)
        capex: cash paid for capex (positive number)
        include_interest: if False, removes interest_paid from CFO (pre-financing basis)
        interest_paid: cash interest paid (positive number)

    Returns:
        float: FCF

    Raises:
        ValueError: invalid inputs.
    """
    o = _num(cfo, "cfo")
    cx = _num(capex, "capex")
    if cx < 0:
        raise ValueError("capex should be positive.")
    i = _num(interest_paid, "interest_paid")
    if i < 0:
        raise ValueError("interest_paid should be positive.")
    adj_cfo = o if include_interest else (o + i)  # if CFO is after interest, add back to get pre-financing
    return adj_cfo - cx

def reclassify_interest_policy(cf: Dict[str, Any], interest_policy: str) -> Dict[str, float]:
    """
    Reclassify interest paid between CFO and CFF to standardise cash flow categories.

    Args:
        cf: dict with keys:
            - cfo, cfi, cff
            - interest_paid (cash, positive number)
            - interest_in_cfo (bool) whether interest is currently included in CFO
        interest_policy: "interest_in_financing" or "interest_in_operating"

    Returns:
        dict: adjusted cfo/cff and unchanged cfi.

    Raises:
        ValueError: invalid inputs or policy.
    """
    if interest_policy not in {"interest_in_financing", "interest_in_operating"}:
        raise ValueError("interest_policy must be 'interest_in_financing' or 'interest_in_operating'.")

    cfo = _num(cf.get("cfo"), "cfo")
    cfi = _num(cf.get("cfi"), "cfi")
    cff = _num(cf.get("cff"), "cff")
    ip = _num(cf.get("interest_paid", 0.0), "interest_paid")
    in_cfo = cf.get("interest_in_cfo", None)
    if not isinstance(in_cfo, bool):
        raise ValueError("interest_in_cfo must be boolean.")

    # If interest is currently in CFO and we want it in financing: move from CFO -> CFF
    if in_cfo and interest_policy == "interest_in_financing":
        cfo_adj = cfo + ip
        cff_adj = cff - ip
    # If interest is currently in financing and we want it in operating: move from CFF -> CFO
    elif (not in_cfo) and interest_policy == "interest_in_operating":
        cfo_adj = cfo - ip
        cff_adj = cff + ip
    else:
        cfo_adj, cff_adj = cfo, cff

    return {"cfo": cfo_adj, "cfi": cfi, "cff": cff_adj}

def cash_reconciliation(cfo: float, cfi: float, cff: float) -> float:
    """Net change in cash = CFO + CFI + CFF."""
    return _num(cfo, "cfo") + _num(cfi, "cfi") + _num(cff, "cff")

# Example usage
ebit = 420.0
tax_rate = 0.24
dep = 60.0
amort = 15.0
delta_nwc = 25.0
capex = 90.0

fcf1 = compute_fcf_from_nopat(ebit, tax_rate, dep, amort, delta_nwc, capex)
print(f"FCF (NOPAT build): ${fcf1:.1f}m")

cf = {"cfo": 260.0, "cfi": -120.0, "cff": -110.0, "interest_paid": 35.0, "interest_in_cfo": True}
cf_std = reclassify_interest_policy(cf, "interest_in_financing")
net_change = cash_reconciliation(cf_std["cfo"], cf_std["cfi"], cf_std["cff"])
print(cf_std)
print(f"Net change in cash: ${net_change:.1f}m")
```

---

### Valuation Impact
Why this matters:
- Cash flows validate whether earnings are real: persistent gaps between profit and CFO are often where valuation risk hides.
- FCF is the primary driver of DCF valuation; classification inconsistencies (interest, leases, taxes) can cause systematic mispricing.
- Working-capital movements determine reinvestment needs; they also highlight aggressive revenue recognition or supplier financing.

Impact on multiples:
- Cash conversion (CFO/EBITDA, FCF/EBITDA) supports screening of “high quality” earnings peers.
- Misclassified interest can make CFO look stronger and distort cash conversion vs peers.

Impact on DCF inputs:
- ΔNWC and Capex feed reinvestment rates and ROIC-based growth logic.
- Lease classification affects EBITDA and debt; keep FCF consistent with WACC and capital structure.

Practical adjustments:
```python
def cash_conversion(cfo: float, ebitda: float) -> float:
    """
    CFO/EBITDA cash conversion.

    Raises:
        ValueError: ebitda near zero.
    """
    o = float(cfo)
    e = float(ebitda)
    if not np.isfinite(o) or not np.isfinite(e):
        raise ValueError("inputs must be finite.")
    if abs(e) < 1e-12:
        raise ValueError("ebitda must not be zero for conversion ratio.")
    return o / e
```

---

### Quality of Earnings Flags
⚠️ CFO persistently below EBITDA without clear growth/working-capital explanation (possible aggressive revenue or capitalised costs).  
⚠️ Large recurring “non-cash” add-backs (provisions, impairments) used to explain weak CFO.  
⚠️ “Other working capital” swings that dominate CFO (risk of reclassifications or supply chain finance).  
⚠️ Post-IFRS 16 CFO uplift used to claim improved performance without lease-adjusted comparatives.  
✅ Strong, stable CFO/EBITDA with transparent working-capital disclosures and consistent interest classification.

---

### Sector-Specific Considerations

| Sector | Key cash flow issue | Typical valuation treatment |
|---|---|---|
| Technology / SaaS | Deferred revenue drives CFO (timing benefit) | Analyse billings vs revenue; treat deferred revenue as operating liability in invested capital |
| Retail / Consumer | Inventory seasonality | Use average working capital; normalise for seasonality and one-off stock build/release |
| Real Estate | Capex vs development spend | Separate maintenance vs growth capex; consider project cash flows and NAV |
| Banking | Cash flow statement less useful for enterprise FCF | Use sector-specific valuation (dividends, FCFE, excess capital frameworks) |

---

### Real-World Example
Scenario: Two peers show identical EBITDA margins, but one has much lower CFO due to receivables build. You want to reflect cash quality in peer selection.

```python
peer_a = {"cfo": 180.0, "ebitda": 300.0}
peer_b = {"cfo": 260.0, "ebitda": 300.0}

conv_a = cash_conversion(peer_a["cfo"], peer_a["ebitda"])
conv_b = cash_conversion(peer_b["cfo"], peer_b["ebitda"])

print(f"Peer A CFO/EBITDA: {conv_a:.0%}")
print(f"Peer B CFO/EBITDA: {conv_b:.0%}")
```

Interpretation: If Peer A’s weak conversion is structural (credit terms, collection issues), its EBITDA multiple may deserve a discount versus Peer B.

See also: Chapter 4 (balance sheet) for working-capital definitions; Chapter 22 (leases) for lease cash flow classification effects; Chapter 26 (income taxes) for tax cash vs tax expense differences.
