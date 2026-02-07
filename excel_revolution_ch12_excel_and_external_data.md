# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 12: Excel and External Data (Refresh, Normalization, Data Governance for Valuation Inputs)

### Core Concept
Chapter 12 covers how Excel connects to external data sources and how to manage refresh, cleaning, and integration so models stay consistent over time. In valuation, external data (market caps, prices, FX, peer multiples, macro rates, transactions, KPI exports) can quickly degrade model integrity if units, dates, identifiers, and definitions are not controlled. The practical goal is to build an input layer that is **repeatable, auditable, and resilient**: clear “as-of” dates, stable identifiers, validation gates, and versioned assumptions.

See also: Chapter 7 (pandas cleaning/joins), Chapter 8 (automation), Chapter 13 (templates/add-ins), McKinsey chapters on reorganised statements and multiples.

---

### Formula/Methodology

#### 1) Data integrity rules for valuation inputs
```text
Golden rules:
- One source of truth per field (avoid mixing providers without flags)
- Explicit as-of date for market data and FX
- Stable identifiers (ticker + exchange, ISIN, company_id)
- Unit normalization (currency, $m vs $, percentage vs decimal)
- Validation gates before outputs are used
```

#### 2) Currency normalization
```text
Value_USD = Value_Local / FX(Local per USD)
or
Value_USD = Value_Local × FX(USD per Local)

Where:
FX must be explicitly defined (direction matters)
As-of date for FX must match valuation date (or be disclosed)
```

#### 3) Market-derived fields for EV and multiples
```text
Enterprise Value (EV) = Market Cap + Total Debt - Cash + Minority Interest + Preferred Equity - Non-operating Assets
Where:
All components must be at the same date and in the same currency/unit
```

#### 4) Time alignment (critical for comps)
```text
Multiple uses:
EV (as-of date) / Metric (LTM or FY period)

Rule:
EV date and Metric period must be disclosed and consistent across peers.
```

---

### Python Implementation
The patterns below are designed for Python-in-Excel constraints (no network calls). Assume external data is already landed in Excel tables via Power Query, manual paste, or exports; Python is used to validate, normalize, and produce clean “model-ready” tables.

```python
from typing import Dict, Any, Sequence, Optional, Union, Tuple
import numpy as np
import pandas as pd

Number = Union[int, float]

def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str = "df") -> None:
    if df is None or df.empty:
        raise ValueError(f"{name} must be non-empty.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def clean_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    require_columns(df, cols, "df")
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="raise")
    return out

def normalize_units(df: pd.DataFrame, value_col: str, unit_col: str, unit_map: Dict[str, float]) -> pd.DataFrame:
    """
    Normalize values to base units using a unit multiplier.

    Example unit_map:
        {"$": 1.0, "$m": 1_000_000.0, "$bn": 1_000_000_000.0}

    Args:
        df: input table
        value_col: numeric value column
        unit_col: column containing unit labels
        unit_map: mapping from unit label -> multiplier to base unit

    Returns:
        DataFrame with normalized_value column

    Raises:
        ValueError: missing units or invalid unit labels.
    """
    require_columns(df, [value_col, unit_col], "df")
    out = clean_numeric(df, [value_col]).copy()
    if out[unit_col].isna().any():
        raise ValueError(f"{unit_col} contains missing unit labels.")
    unknown = sorted(set(out[unit_col].astype(str)) - set(unit_map.keys()))
    if unknown:
        raise ValueError(f"Unknown unit labels: {unknown}. Provide in unit_map.")
    mult = out[unit_col].astype(str).map(unit_map).astype(float)
    out["normalized_value"] = out[value_col].astype(float) * mult
    return out

def normalize_currency(
    df: pd.DataFrame,
    value_col: str,
    currency_col: str,
    fx_table: pd.DataFrame,
    fx_from_col: str = "ccy",
    fx_rate_col: str = "usd_per_ccy",
    asof_col: Optional[str] = None,
    fx_asof_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert values to USD using an FX table with explicit direction.

    Convention:
        usd_value = local_value * usd_per_ccy

    Args:
        df: table with values and currency codes
        fx_table: table with fx rates (usd_per_ccy)

    Raises:
        ValueError: missing currencies, missing fx rates, or as-of mismatch if provided.
    """
    require_columns(df, [value_col, currency_col], "df")
    require_columns(fx_table, [fx_from_col, fx_rate_col], "fx_table")
    out = clean_numeric(df, [value_col]).copy()

    fx = fx_table.copy()
    fx[fx_from_col] = fx[fx_from_col].astype(str)
    fx[fx_rate_col] = pd.to_numeric(fx[fx_rate_col], errors="raise")

    out[currency_col] = out[currency_col].astype(str)
    merged = pd.merge(out, fx[[fx_from_col, fx_rate_col] + ([fx_asof_col] if fx_asof_col else [])],
                      left_on=currency_col, right_on=fx_from_col, how="left")
    if merged[fx_rate_col].isna().any():
        missing = merged.loc[merged[fx_rate_col].isna(), currency_col].unique().tolist()
        raise ValueError(f"Missing FX rates for currencies: {missing}")

    if asof_col and fx_asof_col:
        merged[asof_col] = pd.to_datetime(merged[asof_col], errors="raise")
        merged[fx_asof_col] = pd.to_datetime(merged[fx_asof_col], errors="raise")
        # Soft check: require exact match; if not, flag
        merged["fx_asof_match"] = (merged[asof_col] == merged[fx_asof_col])

    merged["value_usd"] = merged[value_col].astype(float) * merged[fx_rate_col].astype(float)
    return merged.drop(columns=[fx_from_col])

def validate_asof_consistency(df: pd.DataFrame, asof_cols: Sequence[str], max_unique: int = 1) -> None:
    """
    Ensure a table uses a single as-of date (or a small number).
    Useful for preventing mixed market data snapshots.

    Raises:
        ValueError: if too many distinct dates.
    """
    for c in asof_cols:
        if c not in df.columns:
            raise ValueError(f"Missing as-of column: {c}")
        dt = pd.to_datetime(df[c], errors="raise")
        n = dt.nunique()
        if n > max_unique:
            raise ValueError(f"{c} has {n} distinct dates; expected <= {max_unique}.")

def compute_ev_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EV using standardized columns (already USD and normalized units).

    Required columns:
        market_cap_usd, total_debt_usd, cash_usd
    Optional:
        minority_interest_usd, preferred_equity_usd, non_operating_assets_usd
    """
    req = ["market_cap_usd", "total_debt_usd", "cash_usd"]
    require_columns(df, req, "df")
    out = clean_numeric(df, req).copy()

    for opt in ["minority_interest_usd", "preferred_equity_usd", "non_operating_assets_usd"]:
        if opt not in out.columns:
            out[opt] = 0.0
        out[opt] = pd.to_numeric(out[opt], errors="raise")

    out["ev_usd"] = (
        out["market_cap_usd"] + out["total_debt_usd"] - out["cash_usd"]
        + out["minority_interest_usd"] + out["preferred_equity_usd"]
        - out["non_operating_assets_usd"]
    )
    return out

# Example usage (pretend these were loaded from Excel tables)
company = pd.DataFrame({
    "ticker": ["AAA", "BBB"],
    "as_of": ["2026-02-06", "2026-02-06"],
    "market_cap": [2.5, 1.8],
    "market_cap_unit": ["$bn", "$bn"],
    "currency": ["USD", "EUR"],
    "total_debt": [0.9, 0.4],
    "total_debt_unit": ["$bn", "$bn"],
    "cash": [0.2, 0.15],
    "cash_unit": ["$bn", "$bn"]
})

fx = pd.DataFrame({
    "ccy": ["USD", "EUR"],
    "usd_per_ccy": [1.0, 1.08],
    "fx_asof": ["2026-02-06", "2026-02-06"]
})

unit_map = {"$": 1.0, "$m": 1_000_000.0, "$bn": 1_000_000_000.0}

# Normalize units for each component
mc = normalize_units(company, "market_cap", "market_cap_unit", unit_map).rename(columns={"normalized_value": "market_cap_base"})
debt = normalize_units(company, "total_debt", "total_debt_unit", unit_map).rename(columns={"normalized_value": "total_debt_base"})
cash = normalize_units(company, "cash", "cash_unit", unit_map).rename(columns={"normalized_value": "cash_base"})

tmp = company[["ticker", "as_of", "currency"]].copy()
tmp["market_cap_base"] = mc["market_cap_base"]
tmp["total_debt_base"] = debt["total_debt_base"]
tmp["cash_base"] = cash["cash_base"]

# Convert to USD (direction: usd_value = local * usd_per_ccy)
tmp = normalize_currency(tmp, "market_cap_base", "currency", fx, asof_col="as_of", fx_asof_col="fx_asof").rename(columns={"value_usd": "market_cap_usd"})
tmp = normalize_currency(tmp, "total_debt_base", "currency", fx, asof_col="as_of", fx_asof_col="fx_asof").rename(columns={"value_usd": "total_debt_usd"})
tmp = normalize_currency(tmp, "cash_base", "currency", fx, asof_col="as_of", fx_asof_col="fx_asof").rename(columns={"value_usd": "cash_usd"})

validate_asof_consistency(tmp, ["as_of"], max_unique=1)

ev_tbl = compute_ev_table(tmp)
print(ev_tbl[["ticker", "market_cap_usd", "total_debt_usd", "cash_usd", "ev_usd"]])
```

---

### Valuation Impact
Why this matters:
- **Comps integrity:** EV and multiples are extremely sensitive to unit/currency errors and mixed as-of dates.
- **Forecast baselines:** external historical exports often contain missing periods, restatements, and inconsistent sign conventions.
- **Governance and repeatability:** a standard external-data intake process reduces model risk and improves review speed.

Impact on multiples:
- Ensures peer EV and denominators are comparable (same currency, same unit, consistent timing).
- Allows robust filtering (exclude stale market data, missing FX, negative denominators) before computing medians.

Impact on DCF inputs:
- FX normalization matters for multi-currency forecasts and terminal value comparisons.
- As-of consistency supports reconciling enterprise value outputs to market reference points (sanity checks).

Practical adjustments:
```python
def flag_stale_data(df: pd.DataFrame, asof_col: str, valuation_date: str, max_age_days: int = 5) -> pd.DataFrame:
    """Flag rows where as-of date is older than allowed tolerance."""
    if asof_col not in df.columns:
        raise ValueError("Missing asof_col.")
    out = df.copy()
    out[asof_col] = pd.to_datetime(out[asof_col], errors="raise")
    vd = pd.to_datetime(valuation_date, errors="raise")
    out["data_age_days"] = (vd - out[asof_col]).dt.days
    out["is_stale"] = out["data_age_days"] > int(max_age_days)
    return out
```

---

### Quality of Earnings Flags
⚠️ Red flag: external exports with mixed fiscal calendars or “LTM” labels that are not consistently computed.  
⚠️ Red flag: stale prices/FX (as-of drift) used in peer multiples.  
⚠️ Red flag: manual edits to imported tables without audit trail (breaks reproducibility).  
✅ Green flag: explicit intake checks (units, FX direction, as-of, identifiers) and flagged exceptions before valuation outputs update.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | KPI exports (ARR, NRR) from BI tools | Standardize definitions and as-of; validate that KPI tie-outs are consistent across periods |
| Banking | Multiple data sources (reg filings, market feeds) | Strong identifier control; separate regulatory vs accounting metrics; strict as-of alignment |
| Real Estate | Property datasets and appraisals | Version and timestamp appraisal inputs; separate market prices vs appraised values |
| Industrials | Multi-entity ERP exports | Enforce consistent chart-of-accounts mapping and currency translation rules |

---

### Real-World Example
Scenario: A comps dashboard pulls market caps and FX daily via Power Query and pulls EBITDA from financial statement exports monthly. Python validates as-of dates, converts all values to USD, and refuses to compute medians if FX is missing or if the table mixes dates.

```text
Workflow:
1) Land raw tables in Excel (Power Query / exports)
2) Python: validate identifiers, units, FX direction, as-of dates
3) Python: output model-ready tables (USD, base units)
4) Excel: compute dashboards, valuation ranges, pack outputs
```

Interpretation: A controlled external-data layer prevents silent compounding errors and improves confidence in peer ranges and valuation sanity checks.

See also: Chapter 7 (pandas joins/cleaning), Chapter 8 (automation), Chapter 13 (templates/add-ins).
