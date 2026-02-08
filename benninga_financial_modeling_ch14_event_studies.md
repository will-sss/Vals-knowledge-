# Financial Modeling (Benninga, 5th ed., 2022) - Key Concepts for Valuation

## Chapter 14: Event Studies (abnormal returns, CAR, estimation windows)

### Core Concept
Event studies measure how security prices react to a defined event (earnings, M&A announcement, regulatory change) by separating “normal” expected returns from abnormal returns attributable to the event. In valuation, event studies are used to infer market-perceived value impact (e.g., expected synergies, reassessment of risk/growth), validate narrative assumptions, and benchmark how similar events have historically moved peers. Done poorly (bad windows, confounded events), they produce misleading “impact” estimates.

### Formula/Methodology

#### 1) Market model for expected return (estimation window)
```text
Ri,t = αi + βi × Rm,t + εi,t
```

Where:
- Ri,t = security i return at time t
- Rm,t = market return at time t
- αi, βi = estimated from a pre-event estimation window
- εi,t = residual (unexpected component)

#### 2) Abnormal return (AR)
```text
ARi,t = Ri,t - (αi + βi × Rm,t)
```

#### 3) Cumulative abnormal return (CAR) over an event window [t1, t2]
```text
CARi(t1,t2) = Σ_{t=t1..t2} ARi,t
```

#### 4) Multi-firm average abnormal return (AAR) and cumulative average abnormal return (CAAR)
For N firms:
```text
AARt = (1/N) × Σ_{i=1..N} ARi,t
CAAR(t1,t2) = Σ_{t=t1..t2} AARt
```

#### 5) Simple constant-mean expected return (fallback)
```text
E[Ri] = mean(Ri,t) over estimation window
ARi,t = Ri,t - E[Ri]
```

---

### Practical Application (How to run an event study that’s usable in valuation)

#### Step 1: Define the event precisely
- Event date t0: first public disclosure time (not filing date if already leaked).
- If intra-day timing matters and you only have daily returns, treat the “event day” as the first full trading day after the announcement.

#### Step 2: Choose windows (and document them)
Typical:
- Estimation window: [-250, -30] trading days before t0 (avoid contamination).
- Event window: [-1, +1] for immediate reaction; also test [-2, +2] and [0, +5] for delayed absorption.

Rule: keep event window short to reduce confounding news.

#### Step 3: Estimate α and β in the estimation window
- Use OLS regression of Ri on Rm.
- Use the same market index across all firms.
- Remove obvious data errors/outliers only if clearly non-economic.

#### Step 4: Compute AR and CAR
- Compute expected returns using α, β.
- AR = actual - expected
- CAR = sum of AR over event window

#### Step 5: Control for confounding events (critical in practice)
Exclude or flag events where:
- Earnings release and M&A announcement are in the same window.
- Broader market shock dominates (e.g., macro announcements).
- Multiple firm-specific announcements occur close together.

#### Step 6: Translate market reaction to valuation narrative
- Positive CAR on acquisition announcement suggests expected value creation (synergies > premium + execution risk), but does not prove it.
- Negative CAR on guidance change may indicate reassessment of growth, margins, or risk (inputs you should revisit in DCF).

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def _check_series(s: pd.Series, name: str) -> None:
    if not isinstance(s, pd.Series):
        raise ValueError(f"{name} must be a pandas Series.")
    if s.empty:
        raise ValueError(f"{name} must not be empty.")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be indexed by dates (DatetimeIndex).")

def align_returns(stock: pd.Series, market: pd.Series) -> pd.DataFrame:
    """Align stock and market returns on common dates."""
    _check_series(stock, "stock")
    _check_series(market, "market")
    df = pd.concat({"stock": stock, "market": market}, axis=1).dropna(how="any")
    if df.shape[0] < 60:
        raise ValueError("Need at least 60 aligned observations for a basic event study.")
    return df

def ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """OLS with intercept: y = a + b*x."""
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("y and x must be 1D arrays.")
    if len(y) != len(x):
        raise ValueError("y and x must have the same length.")
    if len(y) < 30:
        raise ValueError("Need at least 30 observations to estimate alpha/beta.")
    if np.allclose(np.var(x), 0.0):
        raise ValueError("Market return variance is ~0; cannot estimate beta.")
    X = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coeff[0]), float(coeff[1])

def event_study_market_model(
    stock: pd.Series,
    market: pd.Series,
    event_date: pd.Timestamp,
    estimation_window: Tuple[int, int] = (-250, -30),
    event_window: Tuple[int, int] = (-1, 1),
) -> Dict[str, Any]:
    """Run a single-firm market-model event study (daily/periodic returns).

    Args:
        stock: stock return series (decimal), indexed by date
        market: market return series (decimal), indexed by date
        event_date: event date (Timestamp). Must exist in aligned index or be in-range.
        estimation_window: (start, end) offsets in trading days relative to event_date (negative numbers)
        event_window: (start, end) offsets in trading days relative to event_date

    Returns:
        dict with alpha, beta, AR series, CAR scalar, and metadata

    Raises:
        ValueError: invalid inputs or insufficient data.
    """
    df = align_returns(stock, market)

    if not isinstance(event_date, pd.Timestamp):
        raise ValueError("event_date must be a pandas Timestamp.")
    if event_date < df.index.min() or event_date > df.index.max():
        raise ValueError("event_date must fall within the date range of the data.")

    # Find nearest trading day (if event_date not in index)
    if event_date not in df.index:
        # choose next trading day after event_date (conservative for after-hours announcements)
        idx = df.index.searchsorted(event_date)
        if idx >= len(df.index):
            raise ValueError("event_date is after last trading day in data.")
        t0 = df.index[idx]
    else:
        t0 = event_date

    est_start, est_end = estimation_window
    ev_start, ev_end = event_window
    if est_start >= est_end:
        raise ValueError("estimation_window start must be < end.")
    if ev_start > ev_end:
        raise ValueError("event_window start must be <= end.")

    # Convert offsets to positions
    pos0 = df.index.get_loc(t0)
    est_pos_start = pos0 + est_start
    est_pos_end = pos0 + est_end
    ev_pos_start = pos0 + ev_start
    ev_pos_end = pos0 + ev_end

    if est_pos_start < 0 or est_pos_end <= 0:
        raise ValueError("estimation window extends before available data.")
    if ev_pos_start < 0 or ev_pos_end >= len(df.index):
        raise ValueError("event window extends beyond available data.")
    if est_pos_end - est_pos_start < 30:
        raise ValueError("estimation window must contain at least 30 observations.")

    est_df = df.iloc[est_pos_start:est_pos_end + 1]
    ev_df = df.iloc[ev_pos_start:ev_pos_end + 1]

    alpha, beta = ols_alpha_beta(est_df["stock"].values.astype(float), est_df["market"].values.astype(float))
    expected = alpha + beta * ev_df["market"].values.astype(float)
    ar = ev_df["stock"].values.astype(float) - expected
    ar_s = pd.Series(ar, index=ev_df.index, name="AR")
    car = float(ar_s.sum())

    return {
        "event_trading_day": t0,
        "alpha": alpha,
        "beta": beta,
        "AR": ar_s,
        "CAR": car,
        "estimation_window": estimation_window,
        "event_window": event_window,
        "n_est": int(len(est_df)),
        "n_event": int(len(ev_df)),
    }

# Example usage (synthetic; replace with real returns)
np.random.seed(42)
dates = pd.date_range("2022-01-01", periods=300, freq="B")
market = pd.Series(np.random.normal(0.0003, 0.01, size=len(dates)), index=dates, name="Rm")
# Stock with beta ~1.1; add event shock at t0
stock = 0.0002 + 1.1 * market + np.random.normal(0.0, 0.012, size=len(dates))
stock = pd.Series(stock.values, index=dates, name="Ri")

t0 = dates[250]
# inject event effect on day 0 and +1
stock.loc[t0] += 0.03
stock.loc[dates[251]] += 0.01

res = event_study_market_model(stock, market, event_date=t0, estimation_window=(-200, -30), event_window=(-1, 1))
print("Event day:", res["event_trading_day"].date())
print(f"alpha={res['alpha']:.5f}, beta={res['beta']:.2f}, CAR={res['CAR']:.2%}")
print(res["AR"].round(4))
```

---

### Valuation Impact
Why this matters:
- Event reactions provide an external “market check” on value impact assumptions (synergies, growth, risk).
- For M&A, the acquirer’s CAR can help infer whether markets believe the premium is justified.

Impact on multiples:
- Events can re-rate multiples (P/E, EV/EBITDA) by changing perceived growth, risk, or profitability trajectory.
- Peer event studies can support defensible multiple expansion/contraction assumptions.

Impact on DCF inputs:
- A negative CAR after guidance can indicate higher perceived risk or lower growth, suggesting revisiting revenue/margin forecasts or discount rates.
- Regulatory events can shift terminal assumptions (growth, ROIC persistence).

Comparability issues across companies:
- Different liquidity, disclosure quality, and market coverage can change measured AR/CAR.
- Overlapping events cause noisy estimates; do not compare CARs without checking for confounding news.

Practical adjustments:
```python
def implied_value_change_from_car(market_cap: float, car: float) -> float:
    """Approximate market value change from CAR on event window."""
    mc = _num(market_cap, "market_cap")
    c = _num(car, "car")
    if mc < 0:
        raise ValueError("market_cap must be >= 0.")
    return mc * c
```

---

### Quality of Earnings Flags
⚠️ Event window includes other major announcements (earnings, guidance, litigation) — CAR is not attributable.  
⚠️ Using filing date rather than first disclosure date (timing error).  
⚠️ Very long event windows used to “find” an effect (p-hacking risk).  
✅ Clear pre-registered windows, confounding checks, and robustness across alternative windows ([-1,+1], [-2,+2], [0,+5]).

---

### Sector-Specific Considerations

| Sector | Typical events | Key issue | Typical handling |
|---|---|---|---|
| Banks | capital raises, stress test results | market moves reflect macro + regulation | use short windows; include control events |
| Pharma | trial readouts, FDA decisions | leakage and pre-positioning common | consider [-5,+5] sensitivity; check pre-trends |
| Tech | product launches, guidance resets | high volatility; low R² on market model | use factor model if available; interpret cautiously |
| Utilities | rate cases, regulatory rulings | announcements can be anticipated | choose precise event dates; include expectation proxies |

---

### Real-World Example
Scenario: Estimate whether an acquisition announcement created or destroyed value for the acquirer.

Steps:
1) Estimate α and β using a clean pre-event window.
2) Compute CAR over [-1,+1] and [0,+5].
3) Translate CAR into approximate market value change and compare with announced premium and expected synergies.

```python
market_cap = 8_000_000_000  # $8.0B
car = res["CAR"]
value_change = implied_value_change_from_car(market_cap, car)
print(f"Approx market value change: ${value_change/1e6:.0f}M")
```

Interpretation: If the acquirer’s market value drop roughly matches (or exceeds) the premium paid, markets may be skeptical that synergies will exceed the price and execution risk.

See also: Chapter 13 (beta estimation), Chapter 12 (covariance matrix) for the underlying risk model; M&A chapters for interpreting synergy/premium logic.
