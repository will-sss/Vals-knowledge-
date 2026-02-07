# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 10: Deep Learning (Neural Networks, Training, Regularization, Practical Forecasting)

### Core Concept
Deep learning uses neural networks to learn complex, non-linear mappings from drivers to outcomes (e.g., revenue, churn, default risk) and to combine many weak signals into a single prediction. In valuation, deep learning is most useful when relationships are highly non-linear and interaction-heavy (customer behaviour, pricing, credit risk, text features), but it requires strict validation to avoid overfitting and leakage—otherwise forecasts become overconfident and distort DCF inputs and scenario weights.

See also: Chapter 8 (tree-based methods), Chapter 6 (regularization), Chapter 5 (cross-validation), Chapter 4 (classification).

---

### Formula/Methodology

#### 1) Single hidden-layer neural network (regression)
```text
ŷ = β0 + Σ_{m=1..M} βm × g( a_m0 + Σ_{j=1..p} a_mj x_j )
```
Where:
- x_j = input features (drivers)
- g(·) = activation function (e.g., ReLU, tanh)
- a_mj = weights from inputs to hidden unit m
- βm = weights from hidden units to output
- M = number of hidden units

#### 2) ReLU activation (common in deep nets)
```text
ReLU(z) = max(0, z)
```

#### 3) Logistic output for binary classification
```text
p = Pr(Y=1 | X) = 1 / (1 + exp(−s))
```
Where s is the network score (final layer output before sigmoid).

#### 4) Loss functions (what training minimises)
Regression (MSE):
```text
Loss = (1/n) × Σ (y_i − ŷ_i)^2
```
Classification (log loss):
```text
Loss = −(1/n) × Σ [ y_i log(p_i) + (1 − y_i) log(1 − p_i) ]
```

#### 5) Weight decay (L2 regularization)
```text
Regularized Loss = Loss + λ × Σ w^2
```
Where:
- w = network weights (typically excluding bias terms)
- λ controls shrinkage (reduces variance / overfitting)

#### 6) Early stopping (practical overfitting control)
Rule:
- Stop training when validation loss stops improving for a patience window.
- Use the best validation checkpoint weights.

#### 7) Data scaling (critical)
Neural nets are sensitive to feature scale. Standardize inputs:
```text
x_standardized = (x − mean(x)) / std
```

---

### Python Implementation
```python
from typing import Dict, Tuple, Optional
import numpy as np

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array.")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN/Inf).")
    return X


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")
    if np.any(~np.isfinite(y.astype(float))):
        raise ValueError("y contains non-finite values (NaN/Inf).")
    return y


def train_val_split_time_order(n: int, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-ordered train/validation split (no shuffling) to reduce leakage in forecasting contexts.

    Args:
        n: number of observations
        val_frac: fraction for validation in (0,1)

    Returns:
        (train_idx, val_idx)

    Raises:
        ValueError: invalid n or val_frac.
    """
    if not isinstance(n, int) or n <= 2:
        raise ValueError("n must be an integer > 2.")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0,1).")
    n_val = max(1, int(round(n * val_frac)))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)
    if train_idx.size == 0:
        raise ValueError("val_frac too large: no training samples.")
    return train_idx, val_idx


def fit_mlp_regressor(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (32, 16),
    alpha: float = 1e-4,
    max_iter: int = 5000,
    random_state: int = 0
) -> Pipeline:
    """
    Fit a feedforward neural network for regression with scaling and early stopping.

    Args:
        X: predictors (n, p)
        y: response (n,)
        hidden_layer_sizes: network architecture
        alpha: L2 regularization strength
        max_iter: training iterations cap
        random_state: seed

    Returns:
        sklearn Pipeline: StandardScaler -> MLPRegressor

    Raises:
        ValueError: invalid inputs/hyperparameters.
    """
    X = _as_2d(X)
    y = _as_1d(y).astype(float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and >= 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,     # uses internal validation split
        n_iter_no_change=30,
        validation_fraction=0.2,
        random_state=random_state
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", model)
    ])
    pipe.fit(X, y)
    return pipe


def fit_mlp_classifier(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (32, 16),
    alpha: float = 1e-4,
    max_iter: int = 5000,
    random_state: int = 0
) -> Pipeline:
    """
    Fit a feedforward neural network for binary classification with scaling and early stopping.

    Returns:
        sklearn Pipeline: StandardScaler -> MLPClassifier
    """
    X = _as_2d(X)
    y = _as_1d(y)
    # enforce binary labels
    u = np.unique(y)
    if u.size != 2:
        raise ValueError("y must be binary for this classifier example.")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and >= 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.2,
        random_state=random_state
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", model)
    ])
    pipe.fit(X, y)
    return pipe


def safe_predict(pipe: Pipeline, X_new: np.ndarray) -> np.ndarray:
    """
    Predict with defensive checks.

    Args:
        pipe: fitted sklearn Pipeline
        X_new: new predictors (m, p)

    Returns:
        np.ndarray: predictions

    Raises:
        ValueError: if X_new invalid.
    """
    X_new = _as_2d(X_new)
    pred = pipe.predict(X_new)
    return np.asarray(pred)


def safe_predict_proba(pipe: Pipeline, X_new: np.ndarray) -> np.ndarray:
    """
    Predict probabilities for class 1 for binary classifier.

    Returns:
        np.ndarray of shape (m,) probabilities
    """
    X_new = _as_2d(X_new)
    if not hasattr(pipe, "predict_proba"):
        raise ValueError("Pipeline model does not support predict_proba.")
    p = pipe.predict_proba(X_new)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("predict_proba returned unexpected shape.")
    return p[:, 1]


# Example A (DCF driver): predict next-year EBITDA margin under non-linear interactions
X_reg = np.array([
    [0.60, 0.06, 0.30],   # util, input_infl, pricing_power
    [0.65, 0.05, 0.35],
    [0.70, 0.04, 0.40],
    [0.75, 0.03, 0.45],
    [0.80, 0.02, 0.55],
    [0.85, 0.01, 0.60],
    [0.55, 0.07, 0.25],
    [0.90, 0.00, 0.70],
], dtype=float)

y_reg = np.array([0.10, 0.11, 0.13, 0.14, 0.16, 0.17, 0.09, 0.18], dtype=float)

reg_pipe = fit_mlp_regressor(X_reg, y_reg, hidden_layer_sizes=(16, 8), alpha=1e-3, max_iter=5000, random_state=1)
yhat_reg = safe_predict(reg_pipe, X_reg)
print("Train MSE:", mean_squared_error(y_reg, yhat_reg))

X_new_reg = np.array([[0.78, 0.025, 0.50]], dtype=float)
pred_margin = float(safe_predict(reg_pipe, X_new_reg)[0])
print(f"Predicted EBITDA margin: {pred_margin:.2%}")

# Example B (risk overlay): predict distress probability (binary) and compute expected loss proxy
X_clf = np.array([
    [3.0, 2.5,  0.04],   # interest_cover, leverage, growth
    [1.2, 4.0, -0.02],
    [2.0, 3.2,  0.01],
    [0.8, 5.0, -0.05],
    [4.5, 2.0,  0.06],
    [1.0, 4.5, -0.01],
], dtype=float)

y_clf = np.array([0, 1, 0, 1, 0, 1], dtype=int)

clf_pipe = fit_mlp_classifier(X_clf, y_clf, hidden_layer_sizes=(8, 4), alpha=1e-3, max_iter=5000, random_state=1)
p_hat = safe_predict_proba(clf_pipe, X_clf)

print("Train log loss:", log_loss(y_clf, np.column_stack([1 - p_hat, p_hat])))
print("Train AUC:", roc_auc_score(y_clf, p_hat))

X_new_clf = np.array([[1.5, 3.8, 0.00]], dtype=float)
pd = float(safe_predict_proba(clf_pipe, X_new_clf)[0])
print(f"Predicted PD: {pd:.2%}")
```

---

### Valuation Impact
Why this matters:
- **Forecasting with complex drivers:** Deep learning can capture non-linearities that materially change DCF outputs (e.g., margins accelerating only after a utilisation threshold).
- **Risk scoring as valuation overlay:** Probability outputs can drive scenario weights (distress vs refinance success), expected loss, and valuation haircuts.
- **Text and alternative data:** Neural nets underpin embeddings (e.g., business-description similarity), potentially improving comparable selection and segment classification.

Impact on multiples:
- Model-derived “quality” or “risk” scores can support multiple selection (premium/discount), but should be used as *evidence*, not a mechanical multiple engine.
- Avoid circularity: do not train models directly on valuation outcomes that already incorporate your judgement.

Impact on DCF inputs:
- Use deep learning outputs to triangulate growth/margin assumptions and set uncertainty bands (base/bear/bull).
- Where model risk is high, blend model outputs with simpler models and analyst priors.

Practical adjustments:
```python
import numpy as np

def blend_forecasts(model_forecast: float, simple_forecast: float, weight: float) -> float:
    """
    Blend a complex model forecast with a simpler benchmark to reduce model risk.

    Args:
        model_forecast: neural net output
        simple_forecast: benchmark (e.g., linear model or historical average)
        weight: in [0,1], higher means trust neural net more

    Returns:
        float: blended forecast
    """
    if not (np.isfinite(model_forecast) and np.isfinite(simple_forecast)):
        raise ValueError("Forecasts must be finite.")
    if not (0.0 <= weight <= 1.0):
        raise ValueError("weight must be in [0,1].")
    return float(weight * model_forecast + (1.0 - weight) * simple_forecast)
```

---

### Quality of Earnings Flags
⚠️ **Red flag: leakage via future information** — deep nets will exploit leakage aggressively and look “accurate” in-sample, producing biased valuation inputs.  
⚠️ **Red flag: unstable predictions** — small input changes produce large output swings; indicates overfitting or poor scaling.  
⚠️ **Red flag: no time-based validation** for forecasting — random splits inflate performance when data is autocorrelated.  
⚠️ **Red flag: black-box outputs used directly** — using a single model to set terminal margins or long-run growth without economic justification.  
✅ **Green flag: consistent performance across regimes** (pre/post shocks) and outputs that align with business logic and capacity constraints.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS / Consumer Internet | User behaviour is non-linear; high dimensional KPIs | Use early stopping and strong regularization; prefer time-based splits; blend with cohort logic |
| Banking / Credit | Rare events; calibration critical | Focus on probability calibration; stress-test under rate regimes; use expected-loss framing |
| Industrials | Threshold effects (utilisation) | Use neural nets or trees; validate across cycles; constrain outputs with guardrails |
| Real Estate | Non-linear price sensitivity to rates | Use model for scenario sensitivity, not as sole valuation driver |

---

### Real-World Example
Scenario: Predict next-year EBITDA margin from utilisation, input inflation, and pricing power, then use the result to set a base-case margin in a DCF. Separately, predict distress probability to inform scenario weights.

```python
# The example code above:
# - trains an MLPRegressor to forecast EBITDA margin
# - trains an MLPClassifier to estimate PD (distress probability)
# - prints training metrics and a scenario prediction
```

Interpretation:
- Use the neural net forecast as a triangulation point and cap/blend with a simpler benchmark where data is limited.
- Treat PD outputs as inputs to scenario-weighted valuation (distress haircut) only if probabilities are calibrated and validated out-of-sample.

See also: Chapter 5 (resampling for validation), Chapter 6 (regularization), Chapter 11 (survival analysis for time-to-event)
