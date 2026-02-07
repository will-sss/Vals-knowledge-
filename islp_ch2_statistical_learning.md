# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 2: Statistical Learning (Prediction vs Inference, Model Accuracy, Bias–Variance, KNN)

### Core Concept
Statistical learning links inputs (predictors) to outputs (responses) to support **prediction** (forecasting future outcomes) and **inference** (understanding drivers). In valuation work, these tools help quantify relationships between operating drivers (e.g., volume, price, churn, default risk) and value-relevant outcomes (revenue, margins, cash flows, credit losses), while controlling for overfitting and data leakage.

### Formula/Methodology
**General supervised learning set-up**
```text
Y = f(X) + ε
```
Where:
- Y = response (e.g., revenue growth, default indicator, margin)
- X = predictors (drivers/features, e.g., customer count, ARPU, macro factors)
- f(·) = unknown relationship to estimate
- ε = error term (noise, unobserved factors)

**Regression accuracy: Mean Squared Error (MSE)**
```text
Training MSE = (1/n) × Σ (y_i - ŷ_i)^2
```
Where:
- n = number of observations
- y_i = true response
- ŷ_i = predicted response from the model

**Train vs test error (why it matters)**
```text
Test MSE = average of (y_0 - ŷ_0)^2 over unseen (test) observations
```
Key application rule:
- Optimise for **test** error (out-of-sample performance), not training error.

**Bias–variance decomposition (regression)**
```text
Expected Test MSE(x0) = Variance( ŷ(x0) ) + Bias( ŷ(x0) )^2 + Variance(ε)
```
Where:
- Variance(ε) = irreducible error (cannot be eliminated by better modelling)
- Increasing model flexibility tends to ↓ bias and ↑ variance (trade-off)

**Classification accuracy: Error rate**
```text
Training Error Rate = (1/n) × Σ I(y_i ≠ ŷ_i)
Test Error Rate = average of I(y_0 ≠ ŷ_0) on unseen data
```
Where:
- I(·) = indicator function returning 1 if condition true else 0

**K-Nearest Neighbors (KNN) classifier: class probability estimate**
```text
Pr(Y = j | X = x0) = (1/K) × Σ_{i in N0} I(y_i = j)
```
Where:
- K = number of neighbours
- N0 = set of K training observations closest to x0 (under a chosen distance metric)

### Python Implementation
```python
from typing import Iterable, Tuple, Union, Optional
import numpy as np


def _to_1d_float_array(x: Union[Iterable[float], np.ndarray], name: str) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf).")
    return arr


def mse(y_true: Union[Iterable[float], np.ndarray],
        y_pred: Union[Iterable[float], np.ndarray]) -> float:
    """
    Mean Squared Error for regression.

    Args:
        y_true: True values (same length as y_pred).
        y_pred: Predicted values.

    Returns:
        float: MSE.

    Raises:
        ValueError: if inputs are empty, mismatched length, or contain NaN/Inf.
    """
    yt = _to_1d_float_array(y_true, "y_true")
    yp = _to_1d_float_array(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    err = yt - yp
    return float(np.mean(err * err))


def classification_error_rate(y_true: Union[Iterable, np.ndarray],
                              y_pred: Union[Iterable, np.ndarray]) -> float:
    """
    Error rate for classification (proportion misclassified).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        float: Misclassification rate in [0, 1].

    Raises:
        ValueError: if inputs are empty or mismatched length.
    """
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    if yt.size == 0 or yp.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(yt != yp))


def train_test_split_index(n: int, test_size: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate train/test indices without sklearn (Python-in-Excel friendly).

    Args:
        n: Number of observations.
        test_size: Fraction in (0, 1).
        seed: Random seed.

    Returns:
        (train_idx, test_idx)

    Raises:
        ValueError: if n invalid or test_size out of range.
    """
    if not isinstance(n, int) or n <= 1:
        raise ValueError("n must be an integer > 1.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_size)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if train_idx.size == 0:
        raise ValueError("test_size too large: no training samples left.")
    return train_idx, test_idx


def knn_predict_proba(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      k: int = 5,
                      classes: Optional[np.ndarray] = None) -> np.ndarray:
    """
    KNN class probability estimates using Euclidean distance.

    Args:
        X_train: shape (n_train, p)
        y_train: shape (n_train,) labels
        X_test: shape (n_test, p)
        k: positive integer, number of neighbours
        classes: optional sorted unique classes; if None inferred from y_train

    Returns:
        np.ndarray: shape (n_test, n_classes) probabilities summing to 1 per row

    Raises:
        ValueError: on shape mismatch, invalid k, NaN/Inf
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of columns (features).")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train rows must match y_train length.")
    if np.any(~np.isfinite(X_train)) or np.any(~np.isfinite(X_test)):
        raise ValueError("X_train/X_test contain non-finite values (NaN/Inf).")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > X_train.shape[0]:
        raise ValueError("k cannot exceed number of training observations.")

    if classes is None:
        classes = np.unique(y_train)
    else:
        classes = np.asarray(classes)

    # Distances: (n_test, n_train)
    # For large data, consider chunking to reduce memory usage in Python-in-Excel.
    diffs = X_test[:, None, :] - X_train[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))

    # Indices of k nearest neighbours for each test point
    nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

    # Compute probabilities
    proba = np.zeros((X_test.shape[0], classes.size), dtype=float)
    for i in range(X_test.shape[0]):
        neigh_labels = y_train[nn_idx[i]]
        for j, c in enumerate(classes):
            proba[i, j] = np.mean(neigh_labels == c)

    # Numerical safety: ensure rows sum to 1
    row_sums = proba.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Invalid probability computation (row sum <= 0).")
    proba = proba / row_sums
    return proba


def knn_predict_class(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      k: int = 5) -> np.ndarray:
    """
    KNN class prediction (argmax of probability).

    Returns:
        np.ndarray: predicted labels
    """
    classes = np.unique(y_train)
    proba = knn_predict_proba(X_train, y_train, X_test, k=k, classes=classes)
    return classes[np.argmax(proba, axis=1)]


# Example usage (valuation context): classify customers into "churn" / "no churn"
X_train = np.array([[0.10, 1.2], [0.30, 0.7], [0.25, 0.9], [0.05, 1.5]])  # e.g., usage drop, complaints count
y_train = np.array([1, 0, 0, 1])  # 1=churn, 0=retain
X_test = np.array([[0.20, 1.0], [0.02, 1.8]])

pred = knn_predict_class(X_train, y_train, X_test, k=3)
print("Predicted classes:", pred.tolist())
```

### Valuation Impact
Why this matters:
- **DCF inputs (growth, margins, reinvestment):** Statistical learning provides disciplined ways to map drivers to forecast assumptions and quantify uncertainty. Overfitting inflates “confidence” and can bias cash-flow forecasts.
- **Risk adjustments (default, distress, churn):** Classification models support probability-of-default, customer churn, or covenant breach signals that feed into WACC/credit spreads, scenario weighting, and cash-flow haircut assumptions.
- **Comparability & model risk:** Different modelling flexibility levels can produce materially different forward metrics (e.g., “adjusted EBITDA” forecasts). Without robust out-of-sample validation, comparability across companies and deals breaks down.

Practical adjustments (valuation modelling hygiene):
```python
import numpy as np

def shrink_forecast_to_prior(forecast: float, prior: float, weight: float) -> float:
    """
    Simple forecast shrinkage to reduce overfitting risk.

    Args:
        forecast: model output (e.g., revenue growth forecast).
        prior: base-case/prior belief (e.g., long-run industry growth).
        weight: in [0, 1], higher = trust model more.

    Returns:
        float: shrunk forecast.

    Raises:
        ValueError: if weight out of bounds or inputs non-finite.
    """
    if not np.isfinite(forecast) or not np.isfinite(prior):
        raise ValueError("forecast and prior must be finite.")
    if not (0.0 <= weight <= 1.0):
        raise ValueError("weight must be between 0 and 1.")
    return weight * forecast + (1.0 - weight) * prior
```

### Quality of Earnings Flags
⚠️ **Red flag: data leakage** — using future information (or revised/restated data not available at forecast date) inflates apparent accuracy and leads to overly aggressive forecasts.  
⚠️ **Red flag: overfitting to one-off periods** — e.g., post-shock spikes, temporary pricing, or exceptional working capital movements; strong training fit but weak test performance.  
⚠️ **Red flag: target contamination** — predictors directly contain the outcome (e.g., using “reported EBITDA” to predict “adjusted EBITDA”).  
✅ **Green flag: stable out-of-sample performance** — similar error metrics across time splits and stress periods; clear driver logic consistent with business reality.

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Technology / SaaS | Churn and expansion drive revenue; high noise in early cohorts | Use classification for churn + regression for net retention; validate on time-based splits (cohorts) |
| Banking / Credit | Default events rare; class imbalance | Use probabilistic outputs; calibrate probabilities; avoid accuracy as sole metric; validate out-of-sample |
| Retail / Consumer | Seasonality and promotions distort signals | Engineer seasonal features; compare models on holdout seasons; monitor stability across promotions |
| Industrials / Cyclicals | Regime shifts (macro cycles) | Use scenario-based evaluation; avoid overfitting to one cycle; include macro predictors |

### Real-World Example
Scenario: Build a simple revenue-growth forecast model and check for overfitting by comparing training and test MSE.

```python
import numpy as np

def linear_prediction(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Basic linear predictor y = X @ beta

    Raises:
        ValueError: if shapes mismatch or inputs non-finite.
    """
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[1] != beta.shape[0]:
        raise ValueError("X columns must match beta length.")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(beta)):
        raise ValueError("X and beta must be finite.")
    return X @ beta

# Example data (drivers -> growth)
# Features: [price_change, volume_change, churn_change]
X = np.array([
    [0.02, 0.05, -0.01],
    [0.00, 0.04,  0.00],
    [0.03, 0.02, -0.02],
    [-0.01, 0.01, 0.03],
    [0.01, 0.03, 0.00],
    [0.02, 0.00, 0.02],
], dtype=float)

y = np.array([0.08, 0.05, 0.07, 0.01, 0.04, 0.02], dtype=float)  # revenue growth

# Reuse helper from above (copy into one cell in Python-in-Excel if needed)
def train_test_split_index(n: int, test_size: float = 0.33, seed: int = 42):
    if not isinstance(n, int) or n <= 1:
        raise ValueError("n must be an integer > 1.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_size)))
    return idx[n_test:], idx[:n_test]

def mse(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0 or yp.size == 0:
        raise ValueError("Inputs must not be empty.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("Length mismatch.")
    if np.any(~np.isfinite(yt)) or np.any(~np.isfinite(yp)):
        raise ValueError("Inputs must be finite.")
    return float(np.mean((yt - yp) ** 2))

train_idx, test_idx = train_test_split_index(n=len(y), test_size=0.33, seed=42)
X_tr, y_tr = X[train_idx], y[train_idx]
X_te, y_te = X[test_idx], y[test_idx]

# Pretend beta came from a fit elsewhere (keep example lightweight)
beta = np.array([1.5, 0.8, -1.2])

yhat_tr = linear_prediction(X_tr, beta)
yhat_te = linear_prediction(X_te, beta)

train_mse = mse(y_tr, yhat_tr)
test_mse = mse(y_te, yhat_te)

print(f"Training MSE: {train_mse:.6f}")
print(f"Test MSE:     {test_mse:.6f}")
print("Overfitting flag:", "YES" if test_mse > 2 * train_mse else "NO (not obvious)")

# Simple valuation interpretation: shrink next-year growth toward a base case if test error is high
base_case_growth = 0.03
model_growth_next_year = float(np.mean(yhat_te))

def shrink_forecast_to_prior(forecast: float, prior: float, weight: float) -> float:
    if not np.isfinite(forecast) or not np.isfinite(prior):
        raise ValueError("forecast and prior must be finite.")
    if not (0.0 <= weight <= 1.0):
        raise ValueError("weight must be between 0 and 1.")
    return weight * forecast + (1.0 - weight) * prior

shrunk_growth = shrink_forecast_to_prior(model_growth_next_year, base_case_growth, weight=0.6)
print(f"Model growth (next year):  {model_growth_next_year:.2%}")
print(f"Shrunk growth (next year): {shrunk_growth:.2%}")
```

Interpretation:
- If **test MSE** is materially higher than **training MSE**, the model is likely capturing noise. In valuation, treat the output as directional and normalise (shrink) forecasts or use simpler specifications.

See also: Chapter 5 (Cross-Validation), Chapter 3 (Linear Regression), Chapter 4 (Classification), Chapter 12 (Unsupervised Learning)
