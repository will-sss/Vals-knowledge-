# VBA Finance - Learn VBA Programming: For Finance & Accounting

## Chapter 5: Error Handling (Production-Grade Traps, Validation, Debugging)

### Core Concept
In finance models, errors are rarely “just code issues”: they become wrong valuation outputs (misstated cash flows, broken sensitivities, incorrect debt/lease adjustments) that can propagate into client deliverables. Robust VBA error handling is about **containing failures**, **preserving data integrity**, and **making errors diagnosable** (what failed, where, why, and what the user should do next).

### VBA Implementation

#### Pattern: Centralised Error Handler (On Error GoTo)
**Use case**: Any macro that touches assumptions, pulls external data, or writes outputs (DCF refresh, comp update, reporting pack build). Ensures consistent messaging, cleanup, and logging.

**VBA Code**:
```vba
Option Explicit

Sub RunValuationUpdate()
    ' Purpose: End-to-end refresh for a valuation workbook (inputs -> calculations -> outputs)
    ' Context: Use before saving a client pack or exporting outputs

    On Error GoTo ErrorHandler

    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual

    ' --- Main logic ---
    Call ValidateAssumptionsRange(ThisWorkbook.Worksheets("Inputs").Range("B2:B50"))
    Call RefreshExternalData()            ' e.g., Capital IQ / Bloomberg plug-ins
    Call RecalcAndCheckOutputs()          ' sanity checks on key KPIs
    Call WriteReportTables()              ' update output tables used in slides

Cleanup:
    ' Always restore Excel state
    Application.Calculation = xlCalculationAutomatic
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    Exit Sub

ErrorHandler:
    Call LogError("RunValuationUpdate", Err.Number, Err.Description)
    MsgBox "Update failed: " & Err.Description & vbCrLf & _
           "Check Inputs and try again. If persistent, see the 'ErrorLog' sheet.", _
           vbCritical, "Valuation Update Error"
    Resume Cleanup
End Sub

Private Sub RefreshExternalData()
    ' Stub: replace with your actual refresh calls
    ' Example: Application.Run "CIQ.RefreshAll"  ' depends on your add-in
End Sub

Private Sub RecalcAndCheckOutputs()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("Outputs")

    ws.Calculate

    ' Example sanity checks
    If ws.Range("EBITDA_Last12M").Value < 0 Then
        Err.Raise vbObjectError + 101, "RecalcAndCheckOutputs", _
                  "EBITDA is negative. Confirm normalisation items and revenue inputs."
    End If

    If ws.Range("EV_Implied").Value <= 0 Then
        Err.Raise vbObjectError + 102, "RecalcAndCheckOutputs", _
                  "Implied enterprise value is non-positive. Check WACC/terminal value inputs."
    End If
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def run_valuation_update(inputs: pd.DataFrame) -> dict:
    # Python alternative to VBA orchestration.
    # Use when: heavy data cleaning, large tables, reproducible transformations.
    required = {"name", "value"}
    if not required.issubset(inputs.columns):
        raise ValueError(f"inputs must include columns {required}")

    if inputs["value"].isna().any():
        bad = inputs.loc[inputs["value"].isna(), "name"].tolist()
        raise ValueError(f"Missing assumptions: {bad}")

    # Example sanity logic (replace with real calculations)
    wacc = float(inputs.loc[inputs["name"] == "WACC", "value"].iloc[0])
    if not (0 < wacc < 0.5):
        raise ValueError("WACC must be between 0 and 0.50 (as decimal).")

    return {"status": "ok", "wacc": wacc}
```

---

#### Pattern: Validate Inputs Before Calculation (Data Integrity Guardrails)
**Use case**: Prevent user errors in assumptions (percentages typed as 8 instead of 0.08, negative growth in terminal period, empty tax rates). This is the single highest-leverage control in client-facing models.

**VBA Code**:
```vba
Option Explicit

Public Sub ValidateAssumptionsRange(ByVal rng As Range)
    On Error GoTo ErrorHandler

    Dim c As Range
    For Each c In rng.Cells
        If IsEmpty(c.Value) Then
            Err.Raise vbObjectError + 201, "ValidateAssumptionsRange", _
                      "Blank input at " & c.Address(External:=True)
        End If

        If IsError(c.Value) Then
            Err.Raise vbObjectError + 202, "ValidateAssumptionsRange", _
                      "Excel error (" & CStr(c.Text) & ") at " & c.Address(External:=True)
        End If

        ' Example rule: percentages must be decimals unless explicitly marked otherwise
        If InStr(1, c.NumberFormat, "%") > 0 Then
            If c.Value > 1 Then
                Err.Raise vbObjectError + 203, "ValidateAssumptionsRange", _
                          "Percentage input > 100% at " & c.Address(External:=True) & _
                          ". Enter as decimal (e.g., 0.08 for 8%)."
            End If
        End If
    Next c

    Exit Sub

ErrorHandler:
    Call LogError("ValidateAssumptionsRange", Err.Number, Err.Description)
    Err.Raise Err.Number, Err.Source, Err.Description ' bubble up to caller
End Sub
```

**Integration with Valuation Work**
- DCF models: prevents invalid WACC, terminal growth, tax rates, and margin assumptions from flowing into FCFF/TV.
- Comparable analysis: prevents #N/A from Bloomberg/CapIQ fields contaminating medians and quartiles.
- Data preparation: enforces “no blanks, no errors” standards before exporting to slides.

**Best Practices for Finance**
- ⚠️ Avoid: letting macros “fix” bad inputs silently (creates audit risk).
- ✅ Do: fail fast with specific cell addresses and required format.
- ✅ Do: validate *units* (decimals vs percentages, $ vs thousands).

---

#### Pattern: Handling Excel Errors (#N/A, #VALUE!) During Imports
**Use case**: When pulling market data into tables used for multiples (EV/EBITDA, EV/Revenue) and peer sets. You want controlled treatment: drop, impute, or flag.

**VBA Code**:
```vba
Option Explicit

Public Function SafeValue(ByVal v As Variant, Optional ByVal defaultValue As Double = 0) As Double
    ' Purpose: Convert potentially error/empty values to a numeric safe output
    ' Context: Use when reading imported data into calculations
    On Error GoTo ErrorHandler

    If IsError(v) Or IsEmpty(v) Then
        SafeValue = defaultValue
    ElseIf IsNumeric(v) Then
        SafeValue = CDbl(v)
    Else
        SafeValue = defaultValue
    End If
    Exit Function

ErrorHandler:
    SafeValue = defaultValue
End Function

Sub CleanImportedMultiples()
    On Error GoTo ErrorHandler

    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("Comps")

    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row

    Dim r As Long
    For r = 2 To lastRow
        ws.Cells(r, "F").Value = SafeValue(ws.Cells(r, "F").Value, defaultValue:=0) ' EV/EBITDA
        ws.Cells(r, "G").Value = SafeValue(ws.Cells(r, "G").Value, defaultValue:=0) ' EV/Revenue
    Next r

    Exit Sub

ErrorHandler:
    Call LogError("CleanImportedMultiples", Err.Number, Err.Description)
    MsgBox "Comps cleaning failed: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd
import numpy as np

def clean_imported_multiples(df: pd.DataFrame) -> pd.DataFrame:
    # Replace invalid values with NaN, then apply finance rules.
    for col in ["ev_ebitda", "ev_revenue"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")  # invalid -> NaN

    # Typical finance rule: treat 0 as missing for multiples
    df.loc[df["ev_ebitda"] == 0, "ev_ebitda"] = np.nan
    df.loc[df["ev_revenue"] == 0, "ev_revenue"] = np.nan
    return df
```

---

#### Pattern: Currency Normalisation (Consistent Units for Reporting Packs)
**Use case**: Imported financials arrive in different currencies/units (GBP thousands, USD millions). Before valuation comparisons, standardise to a base currency and consistent unit scale.

**Formula/Methodology**
```text
Base_Currency_Value = Local_Value × FX_Rate_to_Base
Scaled_Value = Base_Currency_Value × Unit_Scale
```
Where:
- Local_Value = value as presented (e.g., 120.5 in “$m”)
- FX_Rate_to_Base = spot or average FX rate (e.g., USD->USD = 1.0; EUR->USD = 1.08)
- Unit_Scale = factor to convert units (e.g., millions -> 1,000,000)

**VBA Code**:
```vba
Option Explicit

Public Function NormalizeCurrency( _
    ByVal valueInLocalUnits As Double, _
    ByVal fxToUSD As Double, _
    ByVal unitScale As Double) As Double
    On Error GoTo ErrorHandler

    If fxToUSD <= 0 Then Err.Raise vbObjectError + 301, "NormalizeCurrency", "FX rate must be > 0."
    If unitScale <= 0 Then Err.Raise vbObjectError + 302, "NormalizeCurrency", "Unit scale must be > 0."

    NormalizeCurrency = valueInLocalUnits * fxToUSD * unitScale
    Exit Function

ErrorHandler:
    Call LogError("NormalizeCurrency", Err.Number, Err.Description)
    Err.Raise Err.Number, Err.Source, Err.Description
End Function
```

---

#### Pattern: Logging Errors to a Dedicated Sheet (Auditability)
**Use case**: Client models need traceability. A hidden or protected `ErrorLog` sheet provides a lightweight audit trail: timestamp, procedure, error number, message, workbook/user.

**VBA Code**:
```vba
Option Explicit

Public Sub LogError(ByVal procName As String, ByVal errNum As Long, ByVal errDesc As String)
    On Error Resume Next ' logging should not crash the original error flow

    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("ErrorLog")

    Dim nextRow As Long
    nextRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row + 1

    ws.Cells(nextRow, "A").Value = Now
    ws.Cells(nextRow, "B").Value = Environ$("Username")
    ws.Cells(nextRow, "C").Value = procName
    ws.Cells(nextRow, "D").Value = errNum
    ws.Cells(nextRow, "E").Value = errDesc
    ws.Cells(nextRow, "F").Value = ThisWorkbook.Name
End Sub
```

**Best Practices for Finance**
- ✅ Include the *procedure name* and *user* to support troubleshooting across teams.
- ✅ Keep log write operations minimal (no loops over cells) to avoid slowing large models.
- ⚠️ Avoid: writing logs to external files in locked-down client environments.

---

#### Pattern: Debugging Workflow for Financial Models (Breakpoints, Watches, Assertions)
**Use case**: Diagnosing why a model output broke (negative EV, circular reference, unexpected NWC sign). Use VBA IDE tools in a consistent workflow.

**Practical checklist (VBA IDE)**
- Breakpoints: stop before writing outputs; step through line-by-line.
- Watches: monitor key variables (WACC, terminal value, debt bridge).
- Immediate Window: print intermediate results (`Debug.Print`).
- Assertions: raise errors when invariants break.

**VBA Snippet (Assertions)**
```vba
Option Explicit

Private Sub AssertPositive(ByVal x As Double, ByVal label As String)
    If x <= 0 Then
        Err.Raise vbObjectError + 401, "AssertPositive", label & " must be > 0."
    End If
End Sub

Sub ExampleChecks()
    On Error GoTo ErrorHandler

    Dim wacc As Double: wacc = 0.1
    Call AssertPositive(wacc, "WACC")

    Exit Sub
ErrorHandler:
    Call LogError("ExampleChecks", Err.Number, Err.Description)
    MsgBox Err.Description, vbCritical
End Sub
```

---

### Integration with Valuation Work

**How this connects to valuation**
- **DCF**: input validation prevents invalid discounting/growth; assertions prevent negative enterprise values from being “accepted”.
- **Multiples/Comps**: controlled handling of missing values prevents biased medians and misleading quartiles.
- **Reporting**: predictable failure modes stop half-updated output tables from being copied into decks.

**Example workflow (monthly comps refresh + report table build)**
```vba
Option Explicit

Sub MonthlyCompsPack()
    On Error GoTo ErrorHandler

    Application.ScreenUpdating = False

    Call RefreshExternalData
    Call CleanImportedMultiples
    Call BuildCompsSummaryTable

Cleanup:
    Application.ScreenUpdating = True
    Exit Sub

ErrorHandler:
    Call LogError("MonthlyCompsPack", Err.Number, Err.Description)
    MsgBox "Comps pack failed: " & Err.Description, vbCritical
    Resume Cleanup
End Sub

Private Sub BuildCompsSummaryTable()
    ' Stub: aggregate cleaned comps into median/mean/quartiles for reporting
End Sub
```

---

### Performance Considerations (Finance Models with 10k+ Rows)
| Technique | VBA Approach | When to Use |
|---|---|---|
| Bulk reads/writes | Read ranges into Variant arrays; write back once | Large tables (10k+ rows) |
| Reduce volatility | Disable ScreenUpdating/Events; set Manual calculation | Any refresh macro |
| Avoid cell-by-cell | Operate in-memory; use arrays/dictionaries | Cleaning, mapping, consolidations |
| Logging | One-row append per run, not per cell | Traceability without slowdown |

**VBA vs Python-in-Excel vs Power Query**
- Use **VBA** for workbook orchestration, UI controls, and add-in integration.
- Use **Python-in-Excel** for heavy transformations and reproducible cleaning/analytics.
- Use **Power Query** for repeatable ETL from files/databases where available.

---

### Real-World Example
Scenario: Automating a recurring valuation refresh where external data may return `#N/A`, assumptions may be missing, and outputs must pass sanity checks.

**VBA Code**:
```vba
Option Explicit

Sub ProductionValuationRefresh()
    On Error GoTo ErrorHandler

    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual

    ' 1) Validate inputs
    Call ValidateAssumptionsRange(ThisWorkbook.Worksheets("Inputs").Range("B2:B60"))

    ' 2) Refresh data + clean
    Call RefreshExternalData
    Call CleanImportedMultiples

    ' 3) Recalculate and check key outputs
    Call RecalcAndCheckOutputs

    MsgBox "Refresh complete. Outputs updated and validated.", vbInformation

Cleanup:
    Application.Calculation = xlCalculationAutomatic
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    Exit Sub

ErrorHandler:
    Call LogError("ProductionValuationRefresh", Err.Number, Err.Description)
    MsgBox "Refresh failed: " & Err.Description & vbCrLf & _
           "See 'ErrorLog' for details.", vbCritical
    Resume Cleanup
End Sub
```

Output: Updated `Outputs` tables (EV, equity value per share, ROIC/WACC checks) and a populated `ErrorLog` for audit/troubleshooting.

---

**Suggested filename**: `vba_finance_ch5_error_handling.md`
