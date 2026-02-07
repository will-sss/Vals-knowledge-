# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 7: Accounting Policies, Changes in Accounting Estimates and Errors (IAS 8, comparability, restatements, forecasting hygiene)

### Core Concept
IAS 8 governs how entities select and apply accounting policies, how they account for changes in estimates, and how they correct errors. For valuation, IAS 8 is critical because changes can break trend comparability and contaminate forecasting inputs: you must detect and normalise restatements, ensure consistent definitions across periods, and understand whether changes reflect new information (estimate) or corrected misstatement (error).

### Formula/Methodology

#### 1) Retrospective restatement vs prospective change
```text
Accounting policy change (IAS 8): generally retrospective (restate comparatives) unless impracticable.
Change in accounting estimate: prospective (affects current and future periods only).
Prior period error: retrospective restatement (correct prior periods) unless impracticable.
```

#### 2) Comparative integrity (valuation time series)
```text
Comparable time series requires:
- Same recognition and measurement policies across periods, or
- Restated historicals to a consistent basis, with documented adjustments.
```

#### 3) Common valuation-relevant effects
```text
EBITDA, EBIT, NOPAT, and FCF can shift due to:
- Revenue recognition policy changes
- Capitalisation vs expensing policy changes
- Depreciation method / useful lives (estimate)
- Provision measurement updates (estimate)
- Error corrections (misclassifications, omissions, mismeasurement)
```

---

### Practical Application for Valuation Analysts

#### 1) Build a “policy change log” and force consistency before forecasting
Minimum steps:
- Identify any IAS 8 disclosures: policy changes, estimate changes, errors, and the quantified impacts by line item.
- Capture which periods are restated (comparatives, opening equity).
- Create a “constant policy” historical series used for growth rates, margins, working-capital turns, and capex intensity.

Practical rule:
- If the company provides restated comparatives, use restated series; if not, create your own adjustment series using disclosed quantification (or treat as a forecast break with explicit caveat).

#### 2) Differentiate estimate vs policy for forecasting treatment
- **Estimates** (prospective): update forecast assumptions (e.g., new useful life implies different future depreciation).
- **Policy changes** (retrospective): rebuild history to keep ratios consistent, then forecast based on the new policy basis.
- **Errors**: treat as data quality red flag; increase scrutiny on other lines and control environment.

#### 3) Watch for “silent” comparability breaks
Even without formal IAS 8 disclosures, comparability can break when:
- Presentation changes reclassify costs between line items (e.g., distribution moved into cost of sales).
- New segments or discontinued operations change the base.
- Acquisitions change accounting bases (IFRS 3 fair value step-ups affect D&A).

---

### Python Implementation
A lightweight approach in Python-in-Excel: (i) detect discontinuities, (ii) apply disclosed adjustments, (iii) re-run KPIs.

```python
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def _finite(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def apply_restatement(
    df: pd.DataFrame,
    restatement: Dict[str, Dict[str, float]],
    period_col: str = "period"
) -> pd.DataFrame:
    """
    Apply disclosed restatement adjustments to historical financials.

    Args:
        df: DataFrame with a period column and numeric financial columns.
        restatement: dict of {period: {column_name: delta}} where delta is added to reported to get restated.
        period_col: name of period identifier column.

    Returns:
        DataFrame: restated financials.

    Raises:
        ValueError: invalid inputs.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame.")
    if period_col not in df.columns:
        raise ValueError(f"df must contain '{period_col}' column.")
    if not isinstance(restatement, dict):
        raise ValueError("restatement must be a dict period -> {col: delta}.")

    out = df.copy()
    # Ensure numeric columns for adjusted items
    for p, deltas in restatement.items():
        if not isinstance(deltas, dict):
            raise ValueError(f"restatement for period {p} must be a dict of column -> delta.")
        for col, delta in deltas.items():
            if col not in out.columns:
                raise ValueError(f"Column '{col}' not found in df.")
            if not pd.api.types.is_numeric_dtype(out[col]):
                raise ValueError(f"Column '{col}' must be numeric for restatement.")
            _ = _finite(delta, f"restatement delta for {p}:{col}")

    # Apply adjustments row-wise
    for i, row in out.iterrows():
        p = row[period_col]
        if p in restatement:
            for col, delta in restatement[p].items():
                out.at[i, col] = float(out.at[i, col]) + float(delta)

    return out

def discontinuity_flags(series: pd.Series, z_threshold: float = 3.5) -> pd.DataFrame:
    """
    Detect potential discontinuities using robust z-scores on period-to-period changes.

    Args:
        series: numeric time series indexed by period order.
        z_threshold: threshold for robust z-score flag.

    Returns:
        DataFrame: with change, robust_z, and flag.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("series must be numeric.")
    if len(series) < 3:
        raise ValueError("series needs at least 3 observations.")

    changes = series.diff()
    median = np.nanmedian(changes.values)
    mad = np.nanmedian(np.abs(changes.values - median))
    if mad < 1e-12:
        mad = 1e-12  # prevent division by zero

    robust_z = 0.6745 * (changes - median) / mad
    flag = np.abs(robust_z) >= float(z_threshold)

    return pd.DataFrame({"change": changes, "robust_z": robust_z, "flag": flag})

def compute_margin(revenue: float, metric: float) -> float:
    """metric margin = metric / revenue with denominator guard."""
    r = _finite(revenue, "revenue")
    m = _finite(metric, "metric")
    if abs(r) < 1e-12:
        raise ValueError("revenue must not be zero for margin.")
    return m / r

# Example usage
hist = pd.DataFrame(
    {
        "period": ["2022", "2023", "2024"],
        "revenue": [1_000.0, 1_080.0, 1_150.0],
        "ebitda": [220.0, 250.0, 290.0],
        "opex": [780.0, 830.0, 860.0],
    }
)

# Disclosed: 2023 opex was reclassified; EBITDA increased by 8, opex decreased by 8.
restatement = {"2023": {"ebitda": 8.0, "opex": -8.0}}
hist_restated = apply_restatement(hist, restatement)

print(hist_restated)

# Flag potential breaks in EBITDA
flags = discontinuity_flags(hist_restated.set_index("period")["ebitda"])
print(flags)

# Recompute margin after restatement
m_2023 = compute_margin(hist_restated.loc[hist_restated["period"] == "2023", "revenue"].iloc[0],
                        hist_restated.loc[hist_restated["period"] == "2023", "ebitda"].iloc[0])
print(f"2023 EBITDA margin (restated): {m_2023:.1%}")
```

---

### Valuation Impact
Why this matters:
- Trend inputs (growth, margins, cash conversion) are only meaningful on a consistent basis; IAS 8 changes can introduce false signals.
- Peer comparisons require consistent definitions; policy changes can make a company look like an outlier when it is just accounting.
- Restatements and errors are governance signals that may justify higher risk premia or scenario weighting.

Impact on multiples:
- A policy change that moves costs out of operating expenses can inflate EBITDA and reduce EV/EBITDA mechanically.
- Errors corrected in revenue recognition can change run-rate sales and invalidate revenue multiples.

Impact on DCF inputs:
- Forecast margins and reinvestment rates must be calibrated on restated historicals.
- Depreciation useful life changes affect future D&A and capex maintenance logic.

Practical adjustments:
```python
def restated_series(reported: pd.Series, delta: pd.Series) -> pd.Series:
    """
    Restated = reported + delta; aligns indices and fills missing deltas with 0.
    """
    if not isinstance(reported, pd.Series) or not isinstance(delta, pd.Series):
        raise ValueError("reported and delta must be pandas Series.")
    d = delta.reindex(reported.index).fillna(0.0)
    out = reported.astype(float) + d.astype(float)
    return out
```

---

### Quality of Earnings Flags
⚠️ Frequent “immaterial” reclassifications that cumulatively change EBITDA materially.  
⚠️ Restatements driven by revenue recognition errors or aggressive estimates (high governance risk).  
⚠️ Large estimate changes that improve earnings (e.g., extending useful lives) without strong economic justification.  
⚠️ Disclosures claim “impracticable” restatement without quantified impact (limits comparability).  
✅ Clear, quantified IAS 8 note with line-by-line impacts and restated comparatives.

---

### Sector-Specific Considerations

| Sector | Common IAS 8 sensitivity | Typical valuation treatment |
|---|---|---|
| Technology / SaaS | Revenue recognition and contract cost capitalisation policy changes | Restate ARR/revenue and cost lines; ensure consistent CAC payback and margin history |
| Industrials | Useful lives, depreciation methods, provisions estimates | Treat as estimate changes; update forecast D&A and maintenance capex assumptions |
| Real Estate | Investment property measurement and classification changes | Separate recurring cash earnings from FV effects; consistent NAV framework |
| Financials | Classification and measurement policy changes under IFRS 9 | Rebuild comparatives where possible; adjust peer metrics consistently |

---

### Real-World Example
Scenario: A company changes an estimate: extends useful lives, reducing depreciation by $15m in 2024 (prospective). You want to ensure forecast D&A is aligned.

```python
dep_2023 = 60.0
dep_2024_reported = 45.0  # after useful-life extension
delta_dep = dep_2024_reported - dep_2023

print(f"Depreciation change: ${delta_dep:.1f}m")

# Forecast policy: carry forward the new run-rate if justified, but cross-check maintenance capex.
```

Interpretation: If D&A drops due to longer lives, check whether maintenance capex realistically also drops; otherwise EBITDA/EBIT may look better without real cash benefit.

See also: Chapter 3 (presentation) for reclassification effects; Chapter 9 (PPE) and Chapter 11 (intangibles) for useful lives and amortisation policies; Chapter 20 (revenue) for policy-driven top-line changes.
