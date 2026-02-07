# Excel Revolution: Python with VBA in Excel - Key Concepts for Valuation

## Chapter 13: Boosting Efficiency with Templates and Add-ons (Standardized Models, Controls, QA, Reuse)

### Core Concept
Chapter 13 is about building reusable Excel assets—templates, standard tabs, add-ins, and lightweight tooling—to reduce repetitive work and increase model quality. For valuation, templates matter because they standardize definitions (EV bridge, QoE adjustments, forecast layout, sensitivity blocks), enforce consistent checks, and make outputs comparable across engagements. Add-ins and reusable modules should support transparency: they should accelerate execution without obscuring logic.

See also: Chapter 11 (reporting), Chapter 8 (automation pipelines), Chapter 9 (VBA orchestration), McKinsey chapters on reorganizing statements and using multiples.

---

### Formula/Methodology

#### 1) Template architecture for valuation models
```text
Recommended structure:
- Inputs (assumptions, scenario switches, data dictionary)
- Historical financials (as reported + reclassified)
- Adjustments (QoE, non-operating items, IFRS/GAAP normalizations)
- Forecast drivers (revenue, margins, reinvestment, working capital)
- DCF / Multiples / Cross-checks
- Outputs (tables, charts, pack-ready exhibits)
- Checks (reconciliation, constraint flags, error log)
```

#### 2) “Model QA” design (tests built into the template)
```text
Examples:
- Balance sheet ties: Assets = Liabilities + Equity
- EV bridge ties: EV -> Equity -> Per-share
- WACC > g constraint for perpetuity TV
- Terminal value share of EV threshold (e.g., flag >80%)
- Denominator gates (no EV/EBITDA when EBITDA <= 0 unless justified)
```

#### 3) Standardization of definitions (reduces comparability risk)
```text
Normalized EBITDA = Reported EBITDA + Addbacks - Deductions
Equity Value = Enterprise Value - Net Debt - Preferred - Minority + Non-operating Assets
Per-share Value = Equity Value / Diluted Shares
```

---

### Python Implementation
The goal is to make template outputs reproducible. In Python-in-Excel, you can implement a lightweight “checks engine” that reads key figures and returns a compact issues log.

```python
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

def require_finite(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def check_close(a: float, b: float, tol_abs: float = 1e-6, tol_rel: float = 1e-6) -> bool:
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= (tol_abs + tol_rel * denom)

def build_issue(issue: str, severity: str, detail: str) -> Dict[str, str]:
    if severity not in {"INFO", "WARN", "ERROR"}:
        raise ValueError("severity must be INFO/WARN/ERROR.")
    return {"severity": severity, "issue": issue, "detail": detail}

def valuation_checks(inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Run standardized template QA checks.

    Expected keys (extend as needed):
        assets, liabilities, equity
        ev, equity_value, net_debt
        wacc, terminal_g
        terminal_value, enterprise_value
        ebitda, shares_diluted

    Returns:
        DataFrame: issues log with severity.
    """
    issues: List[Dict[str, str]] = []

    # Balance sheet tie
    if all(k in inputs for k in ["assets", "liabilities", "equity"]):
        A = require_finite(inputs["assets"], "assets")
        L = require_finite(inputs["liabilities"], "liabilities")
        E = require_finite(inputs["equity"], "equity")
        if not check_close(A, L + E, tol_abs=1e-2, tol_rel=1e-6):
            issues.append(build_issue(
                "Balance sheet does not tie",
                "ERROR",
                f"Assets ({A:,.2f}) != Liabilities+Equity ({(L+E):,.2f})"
            ))

    # EV bridge sanity: Equity = EV - NetDebt (simplified check)
    if all(k in inputs for k in ["ev", "equity_value", "net_debt"]):
        EV = require_finite(inputs["ev"], "ev")
        EQ = require_finite(inputs["equity_value"], "equity_value")
        ND = require_finite(inputs["net_debt"], "net_debt")
        expected = EV - ND
        if not check_close(EQ, expected, tol_abs=1e-2, tol_rel=1e-6):
            issues.append(build_issue(
                "EV to equity bridge mismatch",
                "WARN",
                f"Equity ({EQ:,.2f}) != EV-NetDebt ({expected:,.2f}) (check MI/Pref/NOA)"
            ))

    # WACC > g constraint
    if all(k in inputs for k in ["wacc", "terminal_g"]):
        w = require_finite(inputs["wacc"], "wacc")
        g = require_finite(inputs["terminal_g"], "terminal_g")
        if w <= g:
            issues.append(build_issue(
                "Perpetuity constraint violated",
                "ERROR",
                f"WACC ({w:.4f}) must be > terminal g ({g:.4f})"
            ))

    # Terminal value dominance
    if all(k in inputs for k in ["terminal_value", "enterprise_value"]):
        tv = require_finite(inputs["terminal_value"], "terminal_value")
        ev = require_finite(inputs["enterprise_value"], "enterprise_value")
        if abs(ev) > 1e-12:
            share = tv / ev
            if share > 0.80:
                issues.append(build_issue(
                    "Terminal value dominates EV",
                    "WARN",
                    f"TV/EV = {share:.1%} (review long-run assumptions)"
                ))

    # Multiple denominator gate
    if "ebitda" in inputs:
        e = require_finite(inputs["ebitda"], "ebitda")
        if e <= 0:
            issues.append(build_issue(
                "EBITDA non-positive",
                "WARN",
                f"EBITDA = {e:,.2f}; avoid EV/EBITDA unless justified (consider EV/Revenue)"
            ))

    # Per-share sanity
    if all(k in inputs for k in ["equity_value", "shares_diluted"]):
        eq = require_finite(inputs["equity_value"], "equity_value")
        sh = require_finite(inputs["shares_diluted"], "shares_diluted")
        if sh <= 0:
            issues.append(build_issue(
                "Invalid diluted shares",
                "ERROR",
                f"shares_diluted must be > 0 (got {sh:,.2f})"
            ))

    if not issues:
        issues.append(build_issue("No issues detected", "INFO", "All checks passed."))

    return pd.DataFrame(issues)

# Example usage
inputs = {
    "assets": 5_000_000_000,
    "liabilities": 3_100_000_000,
    "equity": 1_900_000_000,
    "ev": 5_200_000_000,
    "net_debt": 1_300_000_000,
    "equity_value": 3_850_000_000,  # intentionally mismatched vs EV-ND to trigger WARN
    "wacc": 0.10,
    "terminal_g": 0.03,
    "terminal_value": 4_100_000_000,
    "enterprise_value": 5_200_000_000,
    "ebitda": 420_000_000,
    "shares_diluted": 770_000_000
}
issues = valuation_checks(inputs)
print(issues)
```

---

### Valuation Impact
Why this matters:
- Templates reduce “model drift” and enforce a consistent valuation narrative (how we got from accounts to value).
- Built-in checks catch common errors early (broken links, mismatched EV bridge, terminal value misuse).
- Standard layouts shorten review time and improve comparability across deals, sectors, and analysts.

Impact on multiples:
- Standard peer-table structures (filters, median/IQR, outlier rules) reduce selection bias and produce defensible ranges.
- Denominator gates prevent accidental use of inappropriate multiples.

Impact on DCF inputs:
- Standard driver trees and constraints improve forecast discipline and reduce “story-first” assumption creep.
- Cross-checks (implied multiple from perpetuity TV) become routine rather than optional.

Practical adjustments:
```python
def standardize_percent(x: float) -> float:
    """
    Convert Excel-style percent inputs to decimals if needed.
    If x > 1.0, treat as percent (e.g., 10 => 0.10).
    """
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("percent input must be finite.")
    return v / 100.0 if v > 1.0 else v
```

---

### Quality of Earnings Flags
⚠️ Red flag: templates reused without updating definitions (e.g., cash includes restricted cash in some models but not others).  
⚠️ Red flag: hidden add-in logic changes outputs without logging (version risk).  
⚠️ Red flag: inconsistent treatment of leases, SBC, restructuring costs across engagements.  
✅ Green flag: template includes an explicit “Definitions & Adjustments” tab and an auto-generated audit log of changes.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology | SBC and KPI definitions | Template includes SBC treatment toggle + KPI dictionary; consistent ARR/NRR definitions |
| Banking | Regulatory vs accounting metrics | Separate templates for bank valuations; avoid forcing EV/EBITDA structures |
| Real Estate | NAV and property roll-ups | Include standardized property table schema and cap-rate sensitivity blocks |
| Industrials | Cyclicality and segment mix | Include mid-cycle normalization blocks and segment bridge templates |

---

### Real-World Example
Scenario: A firm has a “Valuation Pack Template” with standardized tables: (i) reported→normalized EBITDA bridge, (ii) EV→equity bridge, (iii) DCF summary + sensitivities, (iv) peer multiples with outlier rules. Python generates an issues log; VBA buttons refresh data and export exhibits.

```text
Template workflow:
1) Paste/refresh inputs
2) Run validation checks (issues log)
3) Refresh comps and outputs
4) Export pack
```

Interpretation: Templates and add-ons compress cycle time while improving consistency and auditability across valuation deliverables.

See also: Chapter 11 (reporting), Chapter 8 (automation), Chapter 9 (orchestration).
