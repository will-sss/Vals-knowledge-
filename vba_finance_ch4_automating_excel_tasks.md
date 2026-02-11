# VBA Finance - Automating Excel Tasks

## Chapter 4: Automating Excel Tasks (Event-Driven Models, Refresh Scheduling, Consolidation, Email Reporting)

### Core Concept
Automation makes financial models repeatable and less error-prone by turning manual, multi-step workflows into controlled routines. In valuation and reporting, the highest leverage automations are: (i) refresh pipelines for market/comps data, (ii) event-driven checks that prevent invalid assumptions, (iii) consolidation across workbooks/entities, and (iv) scheduled distribution of outputs.

### VBA Implementation

#### Pattern: Event-driven model integrity checks (Worksheet_Change + validation)
**Use case**: Guardrails in client-facing DCFs and LBOs—prevent negative tax rates, impossible growth assumptions, or mismatched units (e.g., % vs decimals).

**VBA Code**:
```vba
Private Sub Worksheet_Change(ByVal Target As Range)
    ' Purpose: Validate key valuation inputs when a user edits assumptions
    ' Context: Prevents silent model breaks in DCF / multiples worksheets

    On Error GoTo ErrorHandler
    Application.EnableEvents = False

    Dim rng As Range
    Set rng = Intersect(Target, Range("B2:B20")) ' example: assumptions range
    If rng Is Nothing Then GoTo SafeExit

    Dim c As Range
    For Each c In rng.Cells
        If IsEmpty(c.Value) Then GoTo NextCell

        ' Example validation rules
        If c.Address = "$B$2" Then ' tax rate input (decimal)
            If c.Value < 0 Or c.Value > 0.6 Then
                MsgBox "Tax rate must be between 0.0 and 0.60 (decimal).", vbExclamation
                c.ClearContents
            End If
        End If

        If c.Address = "$B$3" Then ' terminal growth input (decimal)
            If c.Value < -0.05 Or c.Value > 0.06 Then
                MsgBox "Terminal growth should be a realistic long-run range (e.g., -5% to 6%).", vbExclamation
                c.ClearContents
            End If
        End If

        If c.Address = "$B$4" Then ' WACC input (decimal)
            If c.Value <= 0 Or c.Value > 0.5 Then
                MsgBox "WACC must be > 0 and typically < 0.50 (decimal).", vbExclamation
                c.ClearContents
            End If
        End If

NextCell:
    Next c

SafeExit:
    Application.EnableEvents = True
    Exit Sub

ErrorHandler:
    Application.EnableEvents = True
    MsgBox "Error validating inputs: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def validate_assumptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Python alternative to VBA input validation.

    Use when:
      - Inputs are stored as a table (structured references)
      - You want an auditable validation report (flags + messages)
      - You prefer deterministic validation independent of worksheet events
    """
    required = {"assumption", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")

    out = df.copy()
    out["is_valid"] = True
    out["message"] = ""

    rules = {
        "tax_rate": (0.0, 0.60, "Tax rate must be between 0.0 and 0.60 (decimal)."),
        "terminal_growth": (-0.05, 0.06, "Terminal growth range should be -5% to 6% (decimal)."),
        "wacc": (1e-9, 0.50, "WACC must be > 0 and typically < 0.50 (decimal)."),
    }

    for name, (lo, hi, msg) in rules.items():
        mask = out["assumption"].astype(str).str.lower().eq(name) & ~out["value"].between(lo, hi)
        out.loc[mask, "is_valid"] = False
        out.loc[mask, "message"] = msg

    return out
```

**Integration with Valuation Work**
- DCF models: prevents invalid drivers (growth, margins, tax, WACC) and reduces review time.
- Comparable analysis: ensures multiples are in consistent units (x vs %) and removes accidental text entries.
- Data preparation: standardises input structure (assumptions table) enabling automated scenario runs.

**Best Practices for Finance**
- ⚠️ Avoid: hard-coding cell addresses across sheets; name critical ranges or use structured tables.
- ✅ Do: store assumptions in a single “Inputs” table and validate against a rules dictionary (single source of truth).
- Performance: disable events + screen updating inside handlers; keep handlers small (validate only what changed).

---

#### Pattern: Scheduled refresh and recurring updates (Application.OnTime + RefreshAll)
**Use case**: Daily comp refresh before a morning meeting; weekly portfolio factsheet refresh; end-of-month valuation pack updates.

**VBA Code**:
```vba
Public NextRunTime As Date

Sub ScheduleDailyRefresh()
    ' Purpose: Schedule daily refresh of queries/pivots and update timestamp
    ' Context: Use for recurring valuation/reporting workbooks

    On Error GoTo ErrorHandler

    Dim runAt As Date
    runAt = Date + TimeValue("07:30:00") ' local time 07:30

    ' If it's already past the time today, schedule for tomorrow
    If Now > runAt Then runAt = runAt + 1

    NextRunTime = runAt
    Application.OnTime EarliestTime:=NextRunTime, Procedure:="RunRefreshPipeline", Schedule:=True

    Exit Sub
ErrorHandler:
    MsgBox "Error scheduling refresh: " & Err.Description, vbCritical
End Sub

Sub CancelScheduledRefresh()
    On Error Resume Next
    If NextRunTime <> 0 Then
        Application.OnTime EarliestTime:=NextRunTime, Procedure:="RunRefreshPipeline", Schedule:=False
    End If
End Sub

Sub RunRefreshPipeline()
    ' Purpose: Refresh connections, pivots, and calculate model
    On Error GoTo ErrorHandler

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    ThisWorkbook.RefreshAll
    Application.CalculateFull

    With ThisWorkbook.Sheets("Dashboard")
        .Range("B1").Value = "Last refreshed:"
        .Range("C1").Value = Now
        .Range("C1").NumberFormat = "yyyy-mm-dd hh:mm"
    End With

    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True

    ' Reschedule next run (daily)
    ScheduleDailyRefresh
    Exit Sub

ErrorHandler:
    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
    MsgBox "Refresh pipeline failed: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def refresh_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Python-in-Excel can't schedule background tasks the same way as VBA.
    Use Python to transform refreshed data after Excel connections update.
    """
    if df is None or df.empty:
        return pd.DataFrame({"error": ["No data received. Refresh connection/table first."]})

    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out
```

**Integration with Valuation Work**
- Run refresh first, then run standardised “post-refresh” transforms (currency, mapping, KPIs) to ensure repeatability.

**Best Practices for Finance**
- ⚠️ Avoid: scheduling on a workbook that may be closed—OnTime requires Excel running.
- ✅ Do: include a visible “last refreshed” stamp and log failures to a “RunLog” sheet.

---

#### Pattern: Multi-workbook consolidation (entity roll-ups, comps packs, portfolio reporting)
**Use case**: Group valuations (multiple subsidiaries), consolidation of management accounts, or merging weekly portfolio extracts.

**VBA Code**:
```vba
Sub ConsolidateWorkbooks()
    ' Purpose: Combine same-structure tables across multiple files into a master sheet
    ' Context: Roll-ups for valuation packs (e.g., multiple entities)

    On Error GoTo ErrorHandler
    Application.ScreenUpdating = False

    Dim folderPath As String
    folderPath = ThisWorkbook.Path & "\Inputs\"
    If Dir(folderPath, vbDirectory) = "" Then
        MsgBox "Inputs folder not found: " & folderPath, vbExclamation
        GoTo SafeExit
    End If

    Dim master As Worksheet
    Set master = ThisWorkbook.Sheets("MasterData")
    master.Cells.ClearContents

    ' Header setup
    master.Range("A1:E1").Value = Array("entity", "period", "revenue", "ebitda", "net_debt")

    Dim fileName As String
    Dim wb As Workbook, ws As Worksheet
    Dim nextRow As Long: nextRow = 2

    fileName = Dir(folderPath & "*.xlsx")
    Do While fileName <> ""
        Set wb = Workbooks.Open(folderPath & fileName, ReadOnly:=True)
        Set ws = wb.Sheets("KeyMetrics")

        Dim lastRow As Long
        lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row

        Dim r As Long
        For r = 2 To lastRow
            master.Cells(nextRow, 1).Value = ws.Range("A1").Value ' entity name
            master.Cells(nextRow, 2).Value = ws.Cells(r, 1).Value ' period
            master.Cells(nextRow, 3).Value = ws.Cells(r, 2).Value ' revenue
            master.Cells(nextRow, 4).Value = ws.Cells(r, 3).Value ' ebitda
            master.Cells(nextRow, 5).Value = ws.Cells(r, 4).Value ' net debt
            nextRow = nextRow + 1
        Next r

        wb.Close SaveChanges:=False
        fileName = Dir
    Loop

SafeExit:
    Application.ScreenUpdating = True
    Exit Sub

ErrorHandler:
    Application.ScreenUpdating = True
    MsgBox "Consolidation error: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def consolidate_tables(tables: list) -> pd.DataFrame:
    """
    Python alternative: consolidate multiple same-structure tables.
    """
    if not tables:
        raise ValueError("No tables provided for consolidation.")

    cols0 = list(tables[0].columns)
    for i, t in enumerate(tables):
        if list(t.columns) != cols0:
            raise ValueError(f"Schema mismatch at index {i}: {list(t.columns)} vs {cols0}")

    return pd.concat(tables, ignore_index=True)
```

**Best Practices for Finance**
- ⚠️ Avoid: opening dozens of large workbooks repeatedly in one session without cleanup.
- ✅ Do: consolidate once, then run all analysis off the master dataset (single refresh point).

---

#### Pattern: Outlook email distribution (send valuation packs / dashboards)
**Use case**: Monthly reporting packs, investment committee decks, or daily KPI snapshots.

**VBA Code**:
```vba
Sub EmailValuationPack()
    ' Purpose: Email a valuation pack to recipients (link or attachment)
    ' Context: Use for recurring reporting; ensure approvals before sending externally

    On Error GoTo ErrorHandler

    Dim OutlookApp As Object
    Dim MailItem As Object
    Set OutlookApp = CreateObject("Outlook.Application")
    Set MailItem = OutlookApp.CreateItem(0)

    Dim recipients As String
    recipients = "team@company.com" ' replace with internal DL

    Dim subjectText As String
    subjectText = "Valuation Pack - " & Format(Date, "yyyy-mm-dd")

    Dim bodyText As String
    bodyText = "Hi all," & vbCrLf & vbCrLf & _
               "Please find attached the latest valuation pack." & vbCrLf & _
               "Key items updated: comps, WACC, scenario outputs." & vbCrLf & vbCrLf & _
               "Regards,"

    With MailItem
        .To = recipients
        .Subject = subjectText
        .Body = bodyText

        ' Optional attachment (ensure file exists)
        Dim filePath As String
        filePath = ThisWorkbook.Path & "\Outputs\ValuationPack.pdf"
        If Dir(filePath) <> "" Then
            .Attachments.Add filePath
        Else
            MsgBox "Attachment not found: " & filePath, vbExclamation
        End If

        .Display ' Use .Send only in controlled environments
    End With

    Exit Sub

ErrorHandler:
    MsgBox "Email error: " & Err.Description, vbCritical
End Sub
```

**Integration with Valuation Work**
- Suitable for internal distribution. For external clients, add workflow controls: approvals, watermarking, and version locks.

**Best Practices for Finance**
- ⚠️ Avoid: auto-sending to external addresses without human review.
- ✅ Do: default to `.Display` and require a checklist (data refreshed, QC passed, file locked).

---

### Performance considerations
| Decision | VBA | Python-in-Excel | Power Query |
|---|---|---|---|
| Trigger actions from user edits (events) | ✅ Best | ⚠️ Not event-driven | ❌ |
| Transform large tabular data (10k–1M rows) | ⚠️ Slower, fragile | ✅ Strong | ✅ Strong |
| Repeatable imports from files | ✅ OK | ⚠️ No file I/O | ✅ Best |
| Robust audit trail of transforms | ⚠️ Manual | ✅ Good | ✅ Very good |
| Distribution via Outlook | ✅ Native | ❌ | ❌ |

---

### Sector-Specific Applications
| Financial Task | VBA Approach | When to Use |
|---|---|---|
| DCF model input controls | Worksheet_Change validation | Shared models; junior analyst inputs |
| Comparable updates | RefreshAll + mapping macros | Weekly/monthly comp packs |
| Multi-entity consolidation | Loop workbooks + master sheet | Holding companies / groups |
| IC reporting | Export + Outlook email | Regular governance cycles |
| Lease schedule updates | Macros to rebuild schedules | Repeatable IFRS 16 / lease analytics |

---

### Real-World Example
Scenario: Automating a weekly comparable update pack (refresh data, clean, refresh pivots, stamp time, export PDF, open email draft).

**VBA Code**:
```vba
Sub WeeklyCompPackPipeline()
    On Error GoTo ErrorHandler

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    ' 1) Refresh sources
    ThisWorkbook.RefreshAll
    Application.CalculateFull

    ' 2) Basic data cleaning example (replace errors in a table range)
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets("CompsData")
    ws.Range("A:Z").Replace What:="#N/A", Replacement:="", LookAt:=xlPart
    ws.Range("A:Z").Replace What:="#DIV/0!", Replacement:="", LookAt:=xlPart

    ' 3) Refresh pivots/charts
    Dim p As PivotTable
    For Each p In ThisWorkbook.Sheets("CompsPivot").PivotTables
        p.RefreshTable
    Next p

    ' 4) Stamp refresh
    With ThisWorkbook.Sheets("Dashboard")
        .Range("B1").Value = "Last refreshed:"
        .Range("C1").Value = Now
        .Range("C1").NumberFormat = "yyyy-mm-dd hh:mm"
    End With

    ' 5) Export PDF (assumes a sheet named "Pack" is the report)
    Dim pdfPath As String
    pdfPath = ThisWorkbook.Path & "\Outputs\WeeklyCompPack_" & Format(Date, "yyyymmdd") & ".pdf"
    ThisWorkbook.Sheets("Pack").ExportAsFixedFormat Type:=xlTypePDF, Filename:=pdfPath, Quality:=xlQualityStandard

    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True

    ' 6) Draft email (internal)
    Call EmailPackInternal(pdfPath)

    Exit Sub

ErrorHandler:
    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
    MsgBox "Pipeline failed: " & Err.Description, vbCritical
End Sub

Sub EmailPackInternal(ByVal filePath As String)
    On Error GoTo ErrorHandler

    Dim OutlookApp As Object, MailItem As Object
    Set OutlookApp = CreateObject("Outlook.Application")
    Set MailItem = OutlookApp.CreateItem(0)

    With MailItem
        .To = "team@company.com"
        .Subject = "Weekly Comp Pack - " & Format(Date, "yyyy-mm-dd")
        .Body = "Hi all," & vbCrLf & vbCrLf & _
                "Attached is the refreshed comp pack." & vbCrLf & _
                "Please review outliers and recent corporate actions." & vbCrLf & vbCrLf & _
                "Regards,"
        If Dir(filePath) <> "" Then .Attachments.Add filePath
        .Display
    End With

    Exit Sub
ErrorHandler:
    MsgBox "Email draft failed: " & Err.Description, vbCritical
End Sub
```

**Output**: A refreshed comps pack PDF in /Outputs plus an Outlook draft email with the attachment.

**Integration points**
- Capital IQ Excel Plugin: refresh workbook connections first; then run cleaning/mapping macros.
- Bloomberg Terminal: similarly refresh formula-linked sheets; isolate Bloomberg fields in a dedicated input sheet for easier QC.
- Power Query: use PQ for heavy file ingestion/joins; use VBA for orchestration and output packaging.

---
**Suggested filename**: `vba_finance_ch4_automating_excel_tasks.md`
