# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 11: Survival Analysis and Censored Data (Kaplan–Meier, Hazard, Cox PH, Time-to-Event Forecasting)

### Core Concept
Survival analysis models **time until an event** (default, churn, sale/acquisition, failure) when many observations are **censored** (event has not occurred yet by the end of the dataset). In valuation, this supports **probability-weighted scenarios** (e.g., probability a private company remains independent), **customer lifetime value (CLV)**, **expected distress costs**, and **timing of exits**—all of which can materially change terminal value assumptions and risk overlays.

See also: Chapter 4 (classification for event/no-event), Chapter 5 (validation), Damodaran Chapter 33 (scenario/simulations, conceptually).

---

### Formula/Methodology

#### 1) Key objects: survival, density, hazard
Survival function (probability surviving beyond time t):
```text
S(t) = Pr(T > t)
```
CDF and density:
```text
F(t) = Pr(T ≤ t) = 1 − S(t)
f(t) = dF(t) / dt
```
Hazard (instantaneous event rate conditional on survival):
```text
h(t) = lim_{Δt→0} Pr(t ≤ T < t+Δt | T ≥ t) / Δt
```
Relationship between hazard and survival:
```text
S(t) = exp( − ∫_0^t h(u) du )
```

Censoring:
- Observed time is:
```text
Y = min(T, C)
```
- Event indicator:
```text
δ = 1 if T ≤ C else 0
```
Where C is censoring time.

#### 2) Kaplan–Meier (non-parametric survival estimator)
At distinct event times t1 < t2 < ...:
```text
Ŝ(t) = Π_{t_i ≤ t} (1 − d_i / n_i)
```
Where:
- d_i = number of events at time t_i
- n_i = number at risk just before t_i

Practical use:
- Direct estimate of survival curve without covariates.
- Easy way to compute expected survival at a horizon (e.g., 3-year independence probability).

#### 3) Cox proportional hazards (semi-parametric with covariates)
Model hazard as:
```text
h(t | x) = h0(t) × exp(β' x)
```
Where:
- h0(t) = baseline hazard (unspecified)
- exp(β' x) scales the hazard multiplicatively

Hazard ratio (HR) between two covariate vectors x_a and x_b:
```text
HR = h(t | x_a) / h(t | x_b) = exp(β' (x_a − x_b))
```
Interpretation:
- HR > 1 ⇒ higher hazard (event sooner)
- HR < 1 ⇒ lower hazard (event later)

Key assumption:
- Proportional hazards: HR is constant over time.

#### 4) Expected time / expected loss with survival
If you have survival curve S(t) over discrete periods:
```text
Expected time-to-event ≈ Σ_{k=0..K-1} S(t_k) × Δt
```
Probability event occurs within horizon H:
```text
Pr(T ≤ H) = 1 − S(H)
```

Valuation link for independence / churn:
- Expected customer life (in years) informs CLV and implied growth.
- Independence probability informs scenario weights for “remain private” vs “exit” paths.

---

### Python Implementation
```python
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# NOTE: lifelines is not allowed per your constraints. This implementation is:
# - Kaplan–Meier from scratch (numpy/pandas)
# - Cox proportional hazards via scikit-learn style optimisation is non-trivial without extra libs,
#   so we provide a practical alternative: discrete-time hazard model (logistic regression by period),
#   which works well in Excel/Python constraints and is easy to validate.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _finite_1d(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf).")
    return a


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """
    Kaplan–Meier survival estimator.

    Args:
        times: observed times Y = min(T, C), non-negative
        events: event indicator δ (1=event observed, 0=censored)

    Returns:
        DataFrame with columns:
          time, n_at_risk, d_events, survival

    Raises:
        ValueError: invalid inputs (negative times, non-binary events, mismatch).
    """
    t = _finite_1d(times, "times")
    e = np.asarray(events).ravel()
    if e.size != t.size:
        raise ValueError("events length must match times length.")
    if np.any(t < 0):
        raise ValueError("times must be non-negative.")
    u = np.unique(e)
    if not set(u.tolist()).issubset({0, 1}):
        raise ValueError("events must be binary (0/1).")
    e = e.astype(int)

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    e = e[order]

    # Unique event times (exclude censored-only times)
    event_times = np.unique(t[e == 1])
    if event_times.size == 0:
        # No events observed: survival stays 1
        return pd.DataFrame({"time": [0.0], "n_at_risk": [int(t.size)], "d_events": [0], "survival": [1.0]})

    survival = 1.0
    rows = []
    n = t.size

    for ti in event_times:
        # At risk just before ti: those with time >= ti
        n_at_risk = int(np.sum(t >= ti))
        d_events = int(np.sum((t == ti) & (e == 1)))
        if n_at_risk <= 0:
            raise ValueError("n_at_risk became non-positive; check inputs.")
        survival *= (1.0 - d_events / n_at_risk)
        rows.append((float(ti), n_at_risk, d_events, float(survival)))

    out = pd.DataFrame(rows, columns=["time", "n_at_risk", "d_events", "survival"])
    # Add a t=0 anchor
    out = pd.concat([pd.DataFrame({"time": [0.0], "n_at_risk": [n], "d_events": [0], "survival": [1.0]}), out],
                    ignore_index=True)
    return out


def survival_at(km_df: pd.DataFrame, t: float) -> float:
    """
    Step-function survival at time t using KM table.
    """
    if not np.isfinite(t) or t < 0:
        raise ValueError("t must be finite and non-negative.")
    if km_df is None or km_df.empty:
        raise ValueError("km_df must be a non-empty KM table.")
    # Take last survival where time <= t
    sub = km_df[km_df["time"] <= t]
    if sub.empty:
        return 1.0
    return float(sub.iloc[-1]["survival"])


def prob_event_within(km_df: pd.DataFrame, horizon: float) -> float:
    """
    Probability event occurs by horizon: 1 - S(horizon)
    """
    s = survival_at(km_df, horizon)
    return float(1.0 - s)


def expected_time_discrete(km_df: pd.DataFrame, max_time: float, step: float = 1.0) -> float:
    """
    Approximate expected time-to-event using discrete integration of S(t).

    Args:
        km_df: Kaplan–Meier table
        max_time: upper integration limit
        step: discretization step (e.g., 1 year)

    Returns:
        expected time approx

    Raises:
        ValueError: invalid args.
    """
    if not np.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be > 0.")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be > 0.")
    grid = np.arange(0.0, max_time + 1e-12, step)
    s_vals = np.array([survival_at(km_df, float(g)) for g in grid], dtype=float)
    # Riemann sum: sum S(t_k) * step
    return float(np.sum(s_vals[:-1] * step))


def to_person_period(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
    feature_cols: List[str],
    max_period: Optional[int] = None
) -> pd.DataFrame:
    """
    Expand subject-level survival data to person-period format for discrete-time hazard modeling.

    Each subject contributes rows for periods 1..T_i, where T_i is observed time (rounded up).
    Event occurs at final period if event=1 else none.

    Args:
        df: columns include id, time, event, features
        time_col: observed time in periods (can be non-integer; will be ceil)
        event_col: 1 if event observed at time, else 0
        max_period: cap periods for expansion

    Returns:
        DataFrame with columns: id, period, y_event, features...

    Raises:
        ValueError: input validation failures.
    """
    if df is None or df.empty:
        raise ValueError("df must be non-empty.")
    if id_col not in df.columns or time_col not in df.columns or event_col not in df.columns:
        raise ValueError("Required columns missing.")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")

    out_rows = []
    for _, row in df.iterrows():
        sid = row[id_col]
        t = row[time_col]
        ev = int(row[event_col])
        if not np.isfinite(t) or t < 0:
            raise ValueError("time must be finite and non-negative.")
        Ti = int(np.ceil(float(t)))
        if max_period is not None:
            Ti = min(Ti, int(max_period))
        if Ti == 0:
            # immediate censoring/event at time 0 contributes no periods
            continue

        for p in range(1, Ti + 1):
            y_event = 1 if (ev == 1 and p == Ti) else 0
            rec = {"id": sid, "period": p, "y_event": y_event}
            for c in feature_cols:
                rec[c] = float(row[c])
            out_rows.append(rec)

    return pd.DataFrame(out_rows)


def fit_discrete_time_hazard(
    pp: pd.DataFrame,
    feature_cols: List[str],
    period_col: str = "period",
    y_col: str = "y_event",
    C: float = 1.0
) -> Pipeline:
    """
    Fit a discrete-time hazard model using logistic regression on person-period data.

    Model:
        logit(Pr(event at period p | survived to p)) = α_p + β' x
    Implementation:
        include period as a feature (numeric) or use dummies externally if needed.

    Args:
        pp: person-period DataFrame
        feature_cols: covariates to include (excluding period)
        C: inverse regularization strength for LogisticRegression

    Returns:
        sklearn Pipeline: StandardScaler -> LogisticRegression

    Raises:
        ValueError: input validation.
    """
    if pp is None or pp.empty:
        raise ValueError("pp must be non-empty.")
    for c in feature_cols:
        if c not in pp.columns:
            raise ValueError(f"Missing feature: {c}")
    if period_col not in pp.columns or y_col not in pp.columns:
        raise ValueError("Missing period/y columns.")
    if C <= 0 or not np.isfinite(C):
        raise ValueError("C must be finite and > 0.")

    X = pp[[period_col] + feature_cols].to_numpy(dtype=float)
    y = pp[y_col].to_numpy(dtype=int)
    if np.any(~np.isfinite(X)):
        raise ValueError("Features contain non-finite values.")
    u = np.unique(y)
    if not set(u.tolist()).issubset({0, 1}):
        raise ValueError("y_event must be binary 0/1.")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(C=C, solver="lbfgs", max_iter=2000))
    ])
    pipe.fit(X, y)
    return pipe


def predict_survival_discrete(
    model: Pipeline,
    x: Dict[str, float],
    horizon: int,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Predict survival curve S(p) for p=1..horizon from a discrete hazard model.

    For each period p:
        h_p = Pr(event at p | survived to p)
        S_p = Π_{k=1..p} (1 - h_k)

    Args:
        model: fitted hazard model
        x: dict of covariates (feature_cols)
        horizon: max period
        feature_cols: list of covariate keys used in the model (excluding period)

    Returns:
        DataFrame with columns: period, hazard, survival

    Raises:
        ValueError: invalid inputs.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    for c in feature_cols:
        if c not in x or not np.isfinite(x[c]):
            raise ValueError(f"Missing or non-finite covariate: {c}")

    S = 1.0
    rows = []
    for p in range(1, horizon + 1):
        Xp = np.array([[float(p)] + [float(x[c]) for c in feature_cols]], dtype=float)
        # predict_proba returns columns [Pr(0), Pr(1)]
        hp = float(model.predict_proba(Xp)[0, 1])
        hp = float(np.clip(hp, 0.0, 1.0))
        S *= (1.0 - hp)
        rows.append((p, hp, S))
    return pd.DataFrame(rows, columns=["period", "hazard", "survival"])


# Example usage A: independence (private bank stays independent) using Kaplan–Meier
times = np.array([1.2, 2.5, 3.0, 4.7, 5.0, 6.1, 7.3, 7.9, 8.0], dtype=float)  # years observed
events = np.array([1,   1,   0,   1,   0,   0,   1,   0,   0], dtype=int)       # 1=sold, 0=censored

km = kaplan_meier(times, events)
print(km.head())

print("S(5y):", survival_at(km, 5.0))
print("Pr(sold within 5y):", prob_event_within(km, 5.0))
print("Expected time (0-10y approx):", expected_time_discrete(km, max_time=10.0, step=1.0))

# Example usage B: discrete-time hazard with covariates (size, ROE, capital ratio)
df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "time": [3.2, 2.0, 5.5, 4.1, 1.3],          # years
    "event": [1, 0, 1, 0, 1],                   # sold/acquired observed?
    "log_assets": [10.5, 9.8, 11.2, 10.1, 9.5],
    "roe": [0.10, 0.07, 0.14, 0.08, 0.12],
    "capital_ratio": [0.12, 0.10, 0.13, 0.11, 0.09],
})

pp = to_person_period(df, id_col="id", time_col="time", event_col="event",
                      feature_cols=["log_assets", "roe", "capital_ratio"], max_period=10)
haz_model = fit_discrete_time_hazard(pp, feature_cols=["log_assets", "roe", "capital_ratio"], C=1.0)

profile = {"log_assets": 10.2, "roe": 0.09, "capital_ratio": 0.11}
pred_curve = predict_survival_discrete(haz_model, profile, horizon=10, feature_cols=["log_assets", "roe", "capital_ratio"])
print(pred_curve.head())
print("Predicted Pr(event within 5y):", 1.0 - float(pred_curve.loc[pred_curve["period"] == 5, "survival"].iloc[0]))
```

---

### Valuation Impact
Why this matters:
- **Scenario weights with timing:** Survival curves give probabilities **by horizon** (e.g., probability of sale within 3/5/10 years). This is superior to a single “PD” because timing affects discounting and terminal value.
- **CLV and churn timing:** Expected customer life and hazard rates translate directly into CLV and sustainable growth assumptions.
- **Distress and default costs:** Hazard-based distress timing can drive expected distress costs and inform cost of debt (rather than adding arbitrary bps).

Impact on multiples:
- For comp selection and multiples, survival-style models can segment firms by hazard (e.g., high acquisition likelihood) which can change observed multiples (control premium expectations).

Impact on DCF inputs:
- **Explicit scenario timing**: if an exit (sale) is likely within 3–5 years, scenario valuation can replace an overly long terminal value horizon.
- **Terminal value realism**: high hazard of failure/exit reduces the probability-weighted weight on a long steady-state.

Practical adjustments:
```python
import numpy as np

def scenario_weight_from_survival(survival_at_h: float) -> dict:
    """
    Convert survival probability into scenario weights.

    Args:
        survival_at_h: S(H) in [0,1], probability event has NOT happened by horizon H

    Returns:
        dict: {"no_event_by_H": S(H), "event_by_H": 1-S(H)}
    """
    if not np.isfinite(survival_at_h):
        raise ValueError("survival_at_h must be finite.")
    if not (0.0 <= survival_at_h <= 1.0):
        raise ValueError("survival_at_h must be in [0,1].")
    return {"no_event_by_H": float(survival_at_h), "event_by_H": float(1.0 - survival_at_h)}
```

---

### Quality of Earnings Flags
⚠️ **Red flag: censoring is informative** — if weaker firms drop out of data (unobserved) or reporting stops after stress, KM estimates are biased upward (survival overstated).  
⚠️ **Red flag: definition drift** — “event” changes (sale vs merger vs minority stake) makes hazard non-comparable across time/peers.  
⚠️ **Red flag: time leakage in covariates** — using covariates measured after the start of the risk window inflates performance.  
⚠️ **Red flag: ignoring regime shifts** — hazard changes materially across rate cycles; validate pre/post shocks.  
✅ **Green flag: consistent event definition** and time-aligned covariates; survival curves validated by cohort.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banking | M&A / loss of independence timing | Model sale hazard with capital ratio, ROE, size; time-split validation |
| SaaS | Customer churn is time-to-event | Survival/hazard preferred over one-shot classification; use retention cohorts |
| Real estate | Defaults depend on refinance windows | Use discrete-time hazard by maturity buckets; include rates/DSCR |
| Industrials | Failure risk is cyclic | Use regime covariates and scenario-based hazards |

---

### Real-World Example
Scenario: Estimate probability a private bank remains independent over 5 years (KM), then extend with covariates using a discrete-time hazard model that is implementable in Python-in-Excel constraints.

```python
# The example code above:
# - computes KM survival S(t) and Pr(event within horizon)
# - expands data to person-period format
# - fits a discrete-time hazard model (logistic)
# - predicts survival curve and event probability by horizon
```

Interpretation:
- Use KM when you only have time and event/censoring.
- Use discrete-time hazard with covariates when you need drivers and segmentation, while remaining within constrained Python environments.

See also: Chapter 4 (classification), Chapter 6 (regularization), Chapter 5 (validation)
