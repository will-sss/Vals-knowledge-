# VBA Finance - Financial Modeling

## Chapter 3: Financial Modeling (DCF Automation, Sensitivities, Scenarios, Custom Functions)

### Core Concept
This chapter focuses on using VBA to make financial models (especially DCFs) repeatable, less error-prone, and faster to maintain. The core theme is separating **inputs**, **calculations**, and **outputs**, then automating recalculation, scenario switching, and reporting so models can be refreshed reliably under time pressure.

### VBA Implementation

#### Pattern: Model Architecture for Finance (Inputs → Engine → Outputs)
**Use case**: Building DCF models where assumptions change frequently (client updates, market shocks) and outputs must remain consistent across scenarios.

**VBA Code**:
```vba
Option Explicit

' Purpose: Enforce consistent model structure and prevent silent range breaks.
' Context: DCF models with changing templates across deals/clients.

Public Sub ValidateModelRanges()
    On Error GoTo ErrorHandler
    
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("Model")
    
    ' Named ranges expected to exist (Inputs)
    Dim requiredNames As Variant
    requiredNames = Array("inp_Revenue0", "inp_RevGrowth", "inp_EBITDAMargin", "inp_TaxRate", _
                          "inp_NWCpctRev", "inp_CapexpctRev", "inp_WACC", "inp_g", "inp_Shares")
    
    Dim i As Long, nm As String
    For i = LBound(requiredNames) To UBound(requiredNames)
        nm = CStr(requiredNames(i))
        If Not NameExists(nm) Then
            Err.Raise vbObjectError + 1001, "ValidateModelRanges", "Missing named range: " & nm
        End If
    Next i
    
    ' Outputs expected to exist
    Dim requiredOutputs As Variant
    requiredOutputs = Array("out_EV", "out_EquityValue", "out_ValuePerShare")
    
    For i = LBound(requiredOutputs) To UBound(requiredOutputs)
        nm = CStr(requiredOutputs(i))
        If Not NameExists(nm) Then
            Err.Raise vbObjectError + 1002, "ValidateModelRanges", "Missing output named range: " & nm
        End If
    Next i
    
    Exit Sub
    
ErrorHandler:
    MsgBox "Model validation failed: " & Err.Description, vbCritical
End Sub

Private Function NameExists(ByVal nameText As String) As Boolean
    On Error GoTo NotFound
    Dim n As Name
    Set n = ThisWorkbook.Names(nameText)
    NameExists = True
    Exit Function
NotFound:
    NameExists = False
End Function
```

**Python-in-Excel Alternative (when applicable)**:
```python
import numpy as np

def validate_inputs(d: dict, required: list) -> None:
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Missing required inputs: {missing}")
```

### Integration with Valuation Work
- **DCF models**: Consistent inputs/outputs reduce model risk and speed up scenario iteration.
- **Comparable analysis**: Standardised output blocks make it easier to reconcile implied multiples vs DCF.
- **Data preparation**: Named ranges and validation support plug-and-play templates across clients.

---

#### Pattern: DCF Engine Macro (Forecast → FCF → PV → EV → Equity Value)
**Use case**: Recalculating a DCF after updating assumptions and producing a clean set of outputs for reporting.

**VBA Code**:
```vba
Option Explicit

' Purpose: Compute a simple multi-year DCF and write key outputs.
' Context: Valuation model refresh after new trading update or revised assumptions.

Public Sub RunDCF()
    On Error GoTo ErrorHandler
    
    Call ValidateModelRanges
    
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("Model")
    
    Dim rev0 As Double, g As Double, ebitdaM As Double
    Dim tax As Double, nwcPct As Double, capexPct As Double
    Dim wacc As Double, tg As Double, shares As Double
    
    rev0 = CDbl(Range("inp_Revenue0").Value)
    g = CDbl(Range("inp_RevGrowth").Value)
    ebitdaM = CDbl(Range("inp_EBITDAMargin").Value)
    tax = CDbl(Range("inp_TaxRate").Value)
    nwcPct = CDbl(Range("inp_NWCpctRev").Value)
    capexPct = CDbl(Range("inp_CapexpctRev").Value)
    wacc = CDbl(Range("inp_WACC").Value)
    tg = CDbl(Range("inp_g").Value)
    shares = CDbl(Range("inp_Shares").Value)
    
    If rev0 < 0 Then Err.Raise vbObjectError + 1101, "RunDCF", "Revenue0 cannot be negative."
    If wacc <= tg Then Err.Raise vbObjectError + 1102, "RunDCF", "WACC must be greater than terminal growth rate (g)."
    If shares <= 0 Then Err.Raise vbObjectError + 1103, "RunDCF", "Shares must be > 0."
    
    Dim nYears As Long
    nYears = 5
    
    Dim t As Long
    Dim rev() As Double, fcf() As Double, pv() As Double
    ReDim rev(1 To nYears)
    ReDim fcf(1 To nYears)
    ReDim pv(1 To nYears)
    
    For t = 1 To nYears
        If t = 1 Then
            rev(t) = rev0 * (1 + g)
        Else
            rev(t) = rev(t - 1) * (1 + g)
        End If
        
        ' Highly simplified: treat EBITDA*(1-tax) as proxy for NOPAT and subtract reinvestment
        Dim ebitda As Double, nopat As Double
        ebitda = rev(t) * ebitdaM
        nopat = ebitda * (1 - tax)
        
        Dim reinvest As Double
        reinvest = (rev(t) * capexPct) + (rev(t) * nwcPct) ' proxy
        
        fcf(t) = nopat - reinvest
        pv(t) = fcf(t) / ((1 + wacc) ^ t)
    Next t
    
    Dim tv As Double, pvTV As Double
    tv = (fcf(nYears) * (1 + tg)) / (wacc - tg)
    pvTV = tv / ((1 + wacc) ^ nYears)
    
    Dim ev As Double
    ev = pvTV
    For t = 1 To nYears
        ev = ev + pv(t)
    Next t
    
    ' Equity bridge: keep simple; adjust in your template with net debt & non-operating items
    Dim equityValue As Double, vps As Double
    equityValue = ev
    vps = equityValue / shares
    
    Range("out_EV").Value = ev
    Range("out_EquityValue").Value = equityValue
    Range("out_ValuePerShare").Value = vps
    
    Exit Sub
    
ErrorHandler:
    MsgBox "DCF failed: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import numpy as np

def dcf_value(rev0, growth, ebitda_margin, tax, nwc_pct, capex_pct, wacc, g_terminal, shares, n_years=5):
    if rev0 < 0:
        raise ValueError("rev0 cannot be negative")
    if shares <= 0:
        raise ValueError("shares must be > 0")
    if wacc <= g_terminal:
        raise ValueError("wacc must be > terminal growth rate")

    rev = np.zeros(n_years)
    fcf = np.zeros(n_years)
    pv = np.zeros(n_years)

    for t in range(n_years):
        rev[t] = (rev0 * (1 + growth)) if t == 0 else rev[t-1] * (1 + growth)
        ebitda = rev[t] * ebitda_margin
        nopat = ebitda * (1 - tax)
        reinvest = rev[t] * capex_pct + rev[t] * nwc_pct
        fcf[t] = nopat - reinvest
        pv[t] = fcf[t] / ((1 + wacc) ** (t + 1))

    tv = (fcf[-1] * (1 + g_terminal)) / (wacc - g_terminal)
    pv_tv = tv / ((1 + wacc) ** n_years)
    ev = pv.sum() + pv_tv
    vps = ev / shares
    return {"ev": ev, "equity_value": ev, "value_per_share": vps}
```

### Valuation Impact
- **DCF inputs**: Automating the DCF reduces manual range errors and ensures consistent application of WACC, tax, reinvestment, and terminal value logic.
- **Cross-checks**: Automated outputs can feed implied multiples (EV/EBITDA) vs comps for reasonableness.
- **Sensitivity discipline**: Macro-based runs make it easier to produce base/bull/bear outputs with consistent mechanics.

### Quality of Earnings Flags
⚠️ EBITDA margin “improvements” driven by capitalising costs (software dev, commissions) rather than operating efficiency.  
⚠️ FCF uplift from one-off NWC release (unsustainable) being treated as recurring.  
✅ Clean reconciliation from EBITDA → NOPAT → FCF with explicit reinvestment drivers (Capex + NWC).

---

#### Pattern: Scenario Manager (Bull/Base/Bear Switch)
**Use case**: Switching assumption sets while keeping output ranges identical (valuation committee, client options).

**VBA Code**:
```vba
Option Explicit

' Purpose: Apply named input sets for scenarios.
' Context: Quick switching between bull/base/bear without overwriting model structure.

Public Sub ApplyScenario(ByVal scenarioName As String)
    On Error GoTo ErrorHandler
    
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("Scenarios")
    
    Dim tbl As ListObject
    Set tbl = ws.ListObjects("tblScenarios") ' columns: Scenario, RevGrowth, EBITDAMargin, WACC, g
    
    Dim found As Range
    Set found = tbl.ListColumns("Scenario").DataBodyRange.Find(What:=scenarioName, LookAt:=xlWhole)
    If found Is Nothing Then
        Err.Raise vbObjectError + 1201, "ApplyScenario", "Scenario not found: " & scenarioName
    End If
    
    Dim r As Long
    r = found.Row - tbl.DataBodyRange.Row + 1
    
    Range("inp_RevGrowth").Value = CDbl(tbl.ListColumns("RevGrowth").DataBodyRange.Cells(r, 1).Value)
    Range("inp_EBITDAMargin").Value = CDbl(tbl.ListColumns("EBITDAMargin").DataBodyRange.Cells(r, 1).Value)
    Range("inp_WACC").Value = CDbl(tbl.ListColumns("WACC").DataBodyRange.Cells(r, 1).Value)
    Range("inp_g").Value = CDbl(tbl.ListColumns("g").DataBodyRange.Cells(r, 1).Value)
    
    Call RunDCF
    
    Exit Sub
    
ErrorHandler:
    MsgBox "Scenario apply failed: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
def apply_scenario(inputs: dict, scenarios: dict, name: str) -> dict:
    if name not in scenarios:
        raise ValueError(f"Unknown scenario: {name}")
    updated = inputs.copy()
    updated.update(scenarios[name])
    return updated
```

---

#### Pattern: Sensitivity Table Automation (2D Grid)
**Use case**: Producing WACC vs terminal growth (or margin vs growth) grids used in valuation packs.

**VBA Code**:
```vba
Option Explicit

' Purpose: Create a 2D sensitivity table by iterating two inputs.
' Context: Client outputs need WACC vs g sensitivity grid with consistent formatting.

Public Sub BuildSensitivityTable()
    On Error GoTo ErrorHandler
    
    Dim waccVals As Variant, gVals As Variant
    waccVals = Array(0.08, 0.09, 0.1, 0.11, 0.12)
    gVals = Array(0.01, 0.02, 0.025, 0.03)
    
    Dim outWs As Worksheet
    Set outWs = ThisWorkbook.Worksheets("Outputs")
    
    Dim topLeft As Range
    Set topLeft = outWs.Range("B5") ' header cell
    
    Dim i As Long, j As Long
    
    ' Headers
    topLeft.Value = "WACC \\ g"
    For j = LBound(gVals) To UBound(gVals)
        topLeft.Offset(0, j + 1).Value = gVals(j)
    Next j
    
    For i = LBound(waccVals) To UBound(waccVals)
        topLeft.Offset(i + 1, 0).Value = waccVals(i)
    Next i
    
    ' Populate
    Dim baseWACC As Double, baseG As Double
    baseWACC = CDbl(Range("inp_WACC").Value)
    baseG = CDbl(Range("inp_g").Value)
    
    For i = LBound(waccVals) To UBound(waccVals)
        Range("inp_WACC").Value = waccVals(i)
        For j = LBound(gVals) To UBound(gVals)
            Range("inp_g").Value = gVals(j)
            Call RunDCF
            topLeft.Offset(i + 1, j + 1).Value = CDbl(Range("out_ValuePerShare").Value)
        Next j
    Next i
    
    ' Restore
    Range("inp_WACC").Value = baseWACC
    Range("inp_g").Value = baseG
    Call RunDCF
    
    Exit Sub
    
ErrorHandler:
    MsgBox "Sensitivity build failed: " & Err.Description, vbCritical
End Sub
```

### Best Practices for Finance
⚠️ Avoid: Hard-coded cell addresses in VBA for client models (breaks when rows/columns shift).  
✅ Do: Use named ranges + table objects (ListObjects) for stable references.  
⚠️ Avoid: Running iterative loops without restoring base assumptions.  
✅ Do: Cache baseline inputs and restore them at the end of the macro.  

### Performance Considerations
| Approach | Best for | Typical bottleneck | Practical tips |
|---|---|---|---|
| VBA | Interactive models, workbook-to-workbook automation | Frequent sheet recalculation | Turn off screen updating; minimise `.Select`; batch writes |
| Python-in-Excel | Table transforms, vectorised calculations, stats | Data conversion between Excel ↔ Python | Use arrays/DataFrames; reduce calls; return compact outputs |
| Power Query | Repeatable ETL, large imports | Refresh time | Keep query steps minimal; push heavy transforms into PQ |

### Sector-Specific Applications
| Financial Task | VBA approach | When to use |
|---|---|---|
| DCF refresh + report tables | Macro to run forecast + write outputs | Recurring valuation updates |
| Multi-scenario outputs | Scenario table + ApplyScenario | IC packs / stress testing |
| Sensitivity grids | 2D iteration macro | Client-facing sensitivity pages |
| Custom WACC/ROIC functions | UDFs with validation | Standardising models across teams |

### Real-World Example
Scenario: Automating a monthly DCF refresh and exporting key outputs to a reporting sheet.

**VBA Code**:
```vba
Option Explicit

Public Sub MonthlyValuationRefresh()
    On Error GoTo ErrorHandler
    
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    
    ' 1) Validate structure
    Call ValidateModelRanges
    
    ' 2) Apply base scenario and run DCF
    Call ApplyScenario("Base")
    
    ' 3) Write summary outputs
    Dim outWs As Worksheet
    Set outWs = ThisWorkbook.Worksheets("Outputs")
    
    outWs.Range("B2").Value = "Enterprise Value ($)"
    outWs.Range("C2").Value = CDbl(Range("out_EV").Value)
    
    outWs.Range("B3").Value = "Equity Value ($)"
    outWs.Range("C3").Value = CDbl(Range("out_EquityValue").Value)
    
    outWs.Range("B4").Value = "Value per Share ($)"
    outWs.Range("C4").Value = CDbl(Range("out_ValuePerShare").Value)
    
    ' 4) Build sensitivity grid
    Call BuildSensitivityTable
    
Cleanup:
    Application.ScreenUpdating = True
    Application.EnableEvents = True
    Exit Sub
    
ErrorHandler:
    MsgBox "Monthly refresh failed: " & Err.Description, vbCritical
    Resume Cleanup
End Sub
```

Output: Updated EV, equity value, and value per share on the reporting sheet, plus a refreshed WACC vs g sensitivity table.

Integration points:
- Capital IQ / Bloomberg: Use their Excel add-ins to populate the **Inputs** sheet; then run `MonthlyValuationRefresh` to compute and publish outputs.
- Power Query: Use PQ to stage and clean raw CSV exports; VBA can trigger refresh then read the cleaned table for model inputs.
- Python-in-Excel: Use Python for heavy statistical tasks (beta estimation, regressions) and pass results back to named ranges used by VBA macros.

---
**Suggested filename**: `vba_finance_ch3_financial_modeling.md`
