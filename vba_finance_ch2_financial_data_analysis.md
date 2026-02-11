# VBA Finance - Financial Data Analysis

## Chapter 2: Financial Data Analysis (Import, Clean, Normalize, Report)

### Core Concept
Financial data analysis in Excel becomes production-grade when import, cleaning, standardisation, and reporting are automated end-to-end. VBA enables repeatable ingestion from common finance sources (CSV exports, databases, APIs), deterministic cleaning (errors, types, duplicates), and dynamic reporting (pivots, monthly filters, refresh-on-open).

### VBA Implementation

#### Pattern: Import CSV to a Staging Sheet (QueryTables)
**Use case**: Daily/weekly transaction dumps, price histories, exported comps, Capital IQ / Bloomberg “Export to CSV” outputs.

**VBA Code**:
```vba
Sub ImportCSVToStaging()
    ' Purpose: Import a CSV file into a staging sheet using QueryTables
    ' Context: Use for recurring transaction/pricing/comps exports

    On Error GoTo ErrorHandler

    Dim ws As Worksheet
    Dim csvPath As String
    Dim qt As QueryTable

    Set ws = ThisWorkbook.Sheets("Staging_CSV")
    csvPath = "C:\Data\Transactions.csv" ' <-- update

    ' Clear prior import
    ws.Cells.Clear

    ' Delete existing QueryTables to avoid duplicates
    For Each qt In ws.QueryTables
        qt.Delete
    Next qt

    ' Import
    With ws.QueryTables.Add(Connection:="TEXT;" & csvPath, Destination:=ws.Range("A1"))
        .TextFileParseType = xlDelimited
        .TextFileCommaDelimiter = True
        .TextFileConsecutiveDelimiter = False
        .TextFilePlatform = xlWindows
        .Refresh BackgroundQuery:=False
    End With

    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def import_csv_to_dataframe(path: str) -> pd.DataFrame:
    """
    Use when: you want cleaner parsing control, faster transforms, easier type handling.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV not found at: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}") from e
    return df
```

---

#### Pattern: Pull Latest Market Data from a Database (ADODB + SQL)
**Use case**: Internal pricing tables, warehouse extracts, “latest available” prices/FX, daily close pulls for valuation models.

**VBA Code**:
```vba
Sub PullLatestPricesFromDatabase()
    ' Purpose: Pull latest prices into a sheet from a SQL database
    ' Context: Use when your org stores pricing/FX data centrally

    On Error GoTo ErrorHandler

    Dim conn As Object
    Dim rs As Object
    Dim connectionString As String
    Dim sql As String

    Set conn = CreateObject("ADODB.Connection")
    Set rs = CreateObject("ADODB.Recordset")

    connectionString = "Provider=SQLOLEDB;" & _
                       "Data Source=YourServerName;" & _
                       "Initial Catalog=YourDatabaseName;" & _
                       "User ID=YourUserID;" & _
                       "Password=YourPassword;"

    sql = "SELECT Symbol, LastTradePrice " & _
          "FROM StockPrices " & _
          "WHERE TradeDate = (SELECT MAX(TradeDate) FROM StockPrices)"

    conn.Open connectionString
    rs.Open sql, conn

    With ThisWorkbook.Sheets("Staging_DB")
        .Cells.Clear
        .Range("A1").Value = "Symbol"
        .Range("B1").Value = "LastTradePrice"
        .Range("A2").CopyFromRecordset rs
    End With

    rs.Close
    conn.Close

    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
    On Error Resume Next
    If Not rs Is Nothing Then If rs.State = 1 Then rs.Close
    If Not conn Is Nothing Then If conn.State = 1 Then conn.Close
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def pull_latest_prices(df_prices: pd.DataFrame, date_col: str = "TradeDate") -> pd.DataFrame:
    """
    Use when: you already have a pricing table in-memory (e.g., from an Excel range) and want the latest slice.
    """
    if df_prices is None or not isinstance(df_prices, pd.DataFrame):
        raise ValueError("df_prices must be a pandas DataFrame")
    if date_col not in df_prices.columns:
        raise ValueError(f"Missing required column: {date_col}")

    d = df_prices.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if d[date_col].isna().all():
        raise ValueError("All dates are invalid; cannot determine latest TradeDate")

    latest = d[date_col].max()
    return d.loc[d[date_col] == latest].reset_index(drop=True)
```

---

#### Pattern: Retrieve Data from an API (HTTP + JSON Parse)
**Use case**: Lightweight “live” snapshots (prices, FX, risk-free rates), or where a data vendor exposes REST endpoints.

**VBA Code**:
```vba
Sub RetrieveDataFromAPI()
    ' Purpose: Make an HTTP GET request, parse JSON, and write to a sheet
    ' Context: Use for real-time / near-real-time market data snapshots

    On Error GoTo ErrorHandler

    Dim http As Object
    Dim url As String
    Dim response As String

    Set http = CreateObject("MSXML2.XMLHTTP")
    url = "https://api.example.com/financialdata?apikey=YourApiKey" ' <-- update

    http.Open "GET", url, False
    http.Send

    response = http.responseText

    ' Parse JSON via ScriptControl (works in many Windows Excel installs)
    Dim parser As Object
    Dim json As Object
    Dim i As Long
    Dim ws As Worksheet

    Set parser = CreateObject("ScriptControl")
    parser.Language = "JScript"
    Set json = parser.Eval("(" & response & ")")

    Set ws = ThisWorkbook.Sheets("Staging_API")
    ws.Cells.Clear
    ws.Range("A1").Value = "Symbol"
    ws.Range("B1").Value = "Price"

    For i = 0 To json.stocks.Length - 1
        ws.Cells(i + 2, 1).Value = json.stocks(i).symbol
        ws.Cells(i + 2, 2).Value = json.stocks(i).price
    Next i

    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def parse_api_json_to_df(payload: dict, key: str = "stocks") -> pd.DataFrame:
    """
    Use when: you already have JSON content (e.g., from a connector) and just need robust parsing.
    Note: In Python-in-Excel, external HTTP may be restricted; prefer vendor add-ins or Power Query for fetch.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if key not in payload or not isinstance(payload[key], list):
        raise ValueError(f"payload must contain a list under key '{key}'")
    return pd.DataFrame(payload[key])
```

---

#### Pattern: Clean Imported Financial Data (Errors, Types, Duplicates)
**Use case**: Remove #N/A/#VALUE! cells, flag invalid rows, stabilise downstream formulas and pivot sources.

**VBA Code**:
```vba
Sub FinancialErrorCheckingAndFlagging()
    ' Purpose: Identify spreadsheet errors and flag cells for review
    ' Context: Use after imports (CSV/API/DB) and before reporting

    On Error GoTo ErrorHandler

    Dim ws As Worksheet
    Dim rng As Range

    Set ws = ThisWorkbook.Sheets("Staging_CSV")

    For Each rng In ws.UsedRange
        If IsError(rng.Value) Then
            Debug.Print "Error at " & rng.Address & "; Type: " & rng.Text
            rng.Interior.Color = RGB(255, 215, 0)
            On Error Resume Next
            rng.Comment.Delete
            On Error GoTo 0
            rng.Comment.Add Text:="Review error before reporting"
        End If
    Next rng

    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd
import numpy as np

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use when: you want systematic treatment of missing/error-like values and robust typing.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.drop_duplicates()
    return d
```

---

#### Pattern: Currency Normalisation (Standardise to One Currency)
**Use case**: Comparable company tables spanning reporting currencies; consolidating multi-currency line items into a single valuation currency.

**VBA Code**:
```vba
Sub StandardizeCurrencyData()
    ' Purpose: Convert values to a standard currency and apply consistent formatting
    ' Context: Use when imported data contains mixed currencies

    On Error GoTo ErrorHandler

    Dim ws As Worksheet
    Dim targetCurrency As String
    Dim cell As Range

    Set ws = ThisWorkbook.Sheets("Staging_FX")
    targetCurrency = "USD" ' <-- set valuation currency

    For Each cell In ws.Range("C2:C1000") ' assuming values in column C
        If IsNumeric(cell.Value) Then
            ' Example: call a conversion function (placeholder)
            cell.Value = ConvertCurrencyToStandard(cell.Value, cell.NumberFormat, targetCurrency)
            cell.NumberFormat = "Currency"
        End If
    Next cell

    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
End Sub

Function ConvertCurrencyToStandard(value As Double, sourceFormat As String, targetCurrency As String) As Double
    ' Purpose: Convert currency values to target currency
    ' Context: Replace with live FX logic (sheet rates, DB, API, or vendor add-in)

    Dim conversionRate As Double
    conversionRate = 1.2 ' placeholder
    ConvertCurrencyToStandard = value * conversionRate
End Function
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def standardize_currency(values: pd.Series, fx_rate: float) -> pd.Series:
    """
    Use when: you have FX rates available (sheet table or vendor add-in output) and want vectorised conversion.
    """
    if values is None:
        raise ValueError("values cannot be None")
    if fx_rate <= 0:
        raise ValueError("fx_rate must be positive")
    s = pd.to_numeric(values, errors="coerce")
    return s * fx_rate
```

---

#### Pattern: Dynamic Reporting (Pivot Refresh + Current Month Filter)
**Use case**: Monthly reporting packs, portfolio rollups, management dashboards, recurring client update decks where pivots must always reflect the latest import.

**VBA Code**:
```vba
Sub UpdatePivotTable_CurrentMonth()
    ' Purpose: Refresh pivot and filter to current month
    ' Context: Use in reporting workbooks that must remain current

    On Error GoTo ErrorHandler

    Dim ws As Worksheet
    Dim pt As PivotTable
    Dim currentMonth As String

    Set ws = ThisWorkbook.Sheets("Report")
    Set pt = ws.PivotTables("FinancePivot")

    pt.RefreshTable
    pt.PivotCache.EnableRefresh = True

    currentMonth = Format(Now, "mmmm yyyy")

    With pt.PivotFields("Date")
        .ClearAllFilters
        .CurrentPage = currentMonth
    End With

    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
End Sub
```

**Python-in-Excel Alternative (when applicable)**:
```python
import pandas as pd

def monthly_summary(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Use when: you want reproducible reporting logic without pivot state issues.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if date_col not in df.columns:
        raise ValueError(f"Missing required column: {date_col}")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["month"] = d[date_col].dt.to_period("M").astype(str)
    current_month = pd.Timestamp.today().to_period("M").astype(str)
    return d.loc[d["month"] == current_month].reset_index(drop=True)
```

---

### Integration with Valuation Work

#### How this connects to valuation
- **DCF models**: Automate refresh of inputs (risk-free rates, betas, peer multiples, FX), then trigger recalculation and export of key outputs to a reporting sheet.
- **Comparable analysis**: Import comps exports, clean missing values, standardise currencies, and rebuild pivot-based multiple tables (EV/EBITDA, EV/Revenue, leverage, growth).
- **Data preparation**: Stage raw pulls → validate/flag errors → standardise formats → load a clean “Model_Input” table used by downstream sheets.

#### Example workflow: Automating a comparable company data update
```vba
Sub UpdateCompsWorkflow()
    ' Purpose: End-to-end comps update: import -> clean -> report refresh
    ' Context: Run daily/weekly in valuation teams maintaining comp dashboards

    On Error GoTo ErrorHandler

    Call ImportCSVToStaging
    Call FinancialErrorCheckingAndFlagging
    Call StandardizeCurrencyData
    Call UpdatePivotTable_CurrentMonth

    MsgBox "Comps workflow complete.", vbInformation
    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
End Sub
```

---

### Best Practices for Finance

#### ⚠️ Avoid
- Hardcoding ranges (breaks when data grows/shrinks).
- Leaving stale QueryTables / pivot caches (silently duplicates or reports old data).
- Ignoring IsError() checks (errors propagate into model outputs and client exhibits).
- Building imports directly into model sheets (always use a staging layer).

#### ✅ Do
- Maintain a strict flow: **Staging → Clean → Model_Input → Report**.
- Use explicit sheet names and table structures (avoid relying on ActiveSheet).
- Log errors to the Immediate Window and highlight cells for review.
- Separate configuration (paths, URLs, currencies, refresh rules) into a “Config” sheet.

#### Performance considerations
- **VBA**: Fast for workbook-native automation (pivots, formatting, multi-sheet operations). Best when the output must live inside Excel and preserve Excel objects.
- **Python-in-Excel**: Best for heavy transforms (10k+ rows), merges/joins, robust typing, and repeatable report logic without pivot state.
- **Power Query**: Best for structured ingestion pipelines and repeatable data shaping; use VBA as the orchestration layer when needed.

---

### Sector-Specific Applications

| Financial Task | VBA Approach | When to Use |
|---|---|---|
| Comps refresh (weekly) | CSV import + cleaning + pivot refresh | Repeated market multiple updates |
| FX standardisation | Conversion function + format enforcement | Multi-currency peer sets |
| Portfolio price snapshot | API pull + write to staging | Quick daily monitoring |
| Monthly reporting pack | Pivot refresh + month filter + export | Management/IC reporting cadence |
| Data quality gate | IsError scan + highlighting + comments | Client-facing model assurance |

---

### Real-World Example

Scenario: Automating a monthly reporting pack refresh from CSV exports (transactions + prices), with error checks, currency standardisation, and pivot refresh.

```vba
Sub MonthlyReportRefresh()
    ' Purpose: Monthly pack refresh: import multiple CSVs, validate, standardise, refresh reporting
    ' Context: Finance teams producing recurring packs with tight timelines

    On Error GoTo ErrorHandler

    ' 1) Import transactions
    Call ImportCSVToStaging

    ' 2) Validate / flag issues
    Call FinancialErrorCheckingAndFlagging

    ' 3) Standardise currency values (if applicable)
    Call StandardizeCurrencyData

    ' 4) Refresh pivots / reporting
    Call UpdatePivotTable_CurrentMonth

    ' 5) Optional: save a timestamped copy
    Dim outPath As String
    outPath = "C:\Reports\MonthlyPack_" & Format(Now, "yyyymmdd_hhnn") & ".xlsx"
    ThisWorkbook.SaveCopyAs outPath

    MsgBox "Monthly report refresh complete.", vbInformation
    Exit Sub

ErrorHandler:
    MsgBox "Error: " & Err.Description, vbCritical
End Sub
```

Output: Updated staging tables (clean), refreshed pivot-based report tables filtered to the current month, and a timestamped workbook copy for distribution.

Integration points:
- **Capital IQ Excel Plugin / Bloomberg Excel**: Treat exports as inputs (CSV) or build staging sheets that accept vendor formulas, then use VBA to “values-only” snapshot before reporting.
- **Power Query**: If ingestion is stable and repeatable, push CSV ingestion + shaping to Power Query, then run VBA only to refresh pivots, apply filters, and export.

---

**Suggested filename**: `vba_finance_ch2_financial_data_analysis.md`
