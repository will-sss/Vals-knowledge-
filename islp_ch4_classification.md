# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 4: Classification (Logistic Regression, LDA/QDA, Naive Bayes, ROC/AUC, Calibration, Imbalance)

### Core Concept
Classification predicts a **category** (e.g., default vs non-default, churn vs retain, fraud vs normal) and often outputs a **probability** used for decision thresholds. In valuation and credit work, calibrated probabilities feed **scenario weights**, **expected loss**, **distress haircuts**, and **customer lifetime value (CLV)** adjustments. The key practical issues are class imbalance, threshold choice, probability calibration, and preventing leakage.

See also: Chapter 2 (KNN), Chapter 3 (linear regression), Chapter 5 (cross-validation), Chapter 11 (survival analysis for time-to-event).

---

### Formula/Methodology

#### 1) Logistic regression (probability model)
Binary outcome Y ∈ {0,1}. Logistic model:
```text
p(X) = Pr(Y=1 | X) = 1 / (1 + exp(−(β0 + β'X)))
```
Log-odds (logit) is linear:
```text
log(p(X) / (1 − p(X))) = β0 + β'X
```
Where:
- p(X) = probability of class 1 (e.g., default)
- β = coefficients (driver impacts on log-odds)

Decision rule with threshold τ:
```text
Predict 1 if p(X) ≥ τ, else 0
```

#### 2) Confusion matrix and key rates
```text
TP = predicted 1 and true 1
FP = predicted 1 and true 0
TN = predicted 0 and true 0
FN = predicted 0 and true 1
```
Metrics:
```text
Accuracy = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)
Recall (TPR) = TP / (TP + FN)
Specificity (TNR) = TN / (TN + FP)
FPR = FP / (FP + TN)
```

#### 3) ROC curve and AUC (ranking quality)
ROC plots TPR vs FPR as τ varies.
```text
AUC = area under ROC curve (0.5 random, 1.0 perfect)
```
Practical rule:
- AUC measures ranking ability, not calibration.

#### 4) LDA / QDA (generative classifiers)
Assume each class has a multivariate normal distribution:
```text
X | Y=k ~ Normal(μ_k, Σ_k)
```
LDA: shared covariance Σ across classes (Σ_k = Σ).  
QDA: class-specific covariance (Σ_k differs).

LDA discriminant (two-class intuition):
```text
δ_k(x) = x' Σ^(-1) μ_k − 0.5 μ_k' Σ^(-1) μ_k + log(π_k)
Assign class with max δ_k(x)
```
Where:
- μ_k = mean vector for class k
- Σ = pooled covariance
- π_k = prior probability of class k

Practical use:
- LDA can work well with small samples and correlated predictors.
- QDA is more flexible but higher variance (needs more data).

#### 5) Naive Bayes (conditional independence approximation)
```text
Pr(Y=k | X) ∝ π_k × Π_j Pr(X_j | Y=k)
```
Assumes conditional independence of features given class.

#### 6) Expected loss and decision thresholds (valuation/credit)
For a probability model p and costs:
```text
Expected Cost(τ) = C_FN × FN(τ) + C_FP × FP(τ)
Choose τ that minimises expected cost.
```
In credit:
```text
Expected Loss (EL) = PD × LGD × EAD
```
Where:
- PD = probability of default (classification output)
- LGD = loss given default
- EAD = exposure at default

---

### Python Implementation
```python
from typing import Dict, Tuple, Optional
import numpy as np


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array.")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN/Inf).")
    return X


def _as_binary(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")
    u = np.unique(y)
    if u.size > 2:
        raise ValueError("y must be binary (contain at most 2 unique values).")
    # Map to {0,1} if needed
    if set(u.tolist()) == {0, 1}:
        return y.astype(int)
    # if labels like {-1, 1} or {"N","Y"} map deterministically
    mapping = {u[0]: 0, u[-1]: 1}
    return np.vectorize(mapping.get)(y).astype(int)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid."""
    z = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(z)):
        raise ValueError("z contains non-finite values.")
    # Avoid overflow: clip z
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Binary log loss."""
    y_true = _as_binary(y_true)
    p = np.asarray(p, dtype=float).ravel()
    if p.shape[0] != y_true.shape[0]:
        raise ValueError("Length mismatch between y_true and p.")
    if np.any(~np.isfinite(p)):
        raise ValueError("p contains non-finite values.")
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def logistic_fit_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.1, iters: int = 2000, l2: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Fit logistic regression via gradient descent (dependency-light; Python-in-Excel friendly).

    Args:
        X: predictors, shape (n, p)
        y: binary labels (0/1), shape (n,)
        lr: learning rate
        iters: iterations
        l2: L2 penalty strength (ridge), applied to non-intercept coefficients

    Returns:
        dict with keys: beta (p+1,), p_hat (n,), loss (scalar)

    Raises:
        ValueError: invalid inputs, non-finite values, etc.
    """
    X = _as_2d(X)
    y = _as_binary(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if not (lr > 0 and iters > 0):
        raise ValueError("lr and iters must be positive.")
    if l2 < 0:
        raise ValueError("l2 must be >= 0.")

    n, p = X.shape
    Xd = np.column_stack([np.ones(n, dtype=float), X])
    beta = np.zeros(p + 1, dtype=float)

    for _ in range(iters):
        z = Xd @ beta
        p_hat = sigmoid(z)
        grad = (Xd.T @ (p_hat - y)) / n
        # L2 penalty (exclude intercept)
        grad[1:] += (l2 / n) * beta[1:]
        beta -= lr * grad

    p_hat = sigmoid(Xd @ beta)
    loss = np.array([log_loss(y, p_hat)], dtype=float)
    return {"beta": beta, "p_hat": p_hat, "loss": loss}


def predict_proba_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predict probabilities with fitted beta."""
    X = _as_2d(X)
    beta = np.asarray(beta, dtype=float).ravel()
    if beta.size != X.shape[1] + 1:
        raise ValueError("beta length must be p+1 (including intercept).")
    Xd = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    return sigmoid(Xd @ beta)


def confusion_counts(y_true: np.ndarray, p_hat: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    """Compute TP, FP, TN, FN at a given threshold."""
    y_true = _as_binary(y_true)
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    if p_hat.shape[0] != y_true.shape[0]:
        raise ValueError("Length mismatch.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0,1].")
    y_pred = (p_hat >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    """Compute common metrics from confusion counts."""
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    total = tp + fp + tn + fn
    if total == 0:
        raise ValueError("Empty confusion matrix.")
    def safe_div(a, b):
        return float(a / b) if b > 0 else float("nan")
    return {
        "accuracy": safe_div(tp + tn, total),
        "precision": safe_div(tp, tp + fp),
        "recall_tpr": safe_div(tp, tp + fn),
        "specificity_tnr": safe_div(tn, tn + fp),
        "fpr": safe_div(fp, fp + tn),
    }


def roc_curve(y_true: np.ndarray, p_hat: np.ndarray, n_thresholds: int = 101) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROC curve points (FPR, TPR) over thresholds.
    Returns (thresholds, fpr, tpr).
    """
    y_true = _as_binary(y_true)
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    if p_hat.shape[0] != y_true.shape[0]:
        raise ValueError("Length mismatch.")
    if n_thresholds < 3:
        raise ValueError("n_thresholds must be >= 3.")
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    fpr = np.zeros_like(thresholds)
    tpr = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        c = confusion_counts(y_true, p_hat, threshold=float(t))
        m = metrics_from_counts(c)
        fpr[i] = m["fpr"]
        tpr[i] = m["recall_tpr"]
    return thresholds, fpr, tpr


def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUC using trapezoidal rule (assumes fpr sorted)."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.size != tpr.size or fpr.size < 2:
        raise ValueError("fpr and tpr must be same length >= 2.")
    if np.any(~np.isfinite(fpr)) or np.any(~np.isfinite(tpr)):
        raise ValueError("fpr/tpr must be finite.")
    # Sort by fpr to be safe
    idx = np.argsort(fpr)
    return float(np.trapz(tpr[idx], fpr[idx]))


def brier_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """Brier score for calibration (mean squared prob error)."""
    y_true = _as_binary(y_true)
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    if p_hat.shape[0] != y_true.shape[0]:
        raise ValueError("Length mismatch.")
    if np.any(~np.isfinite(p_hat)):
        raise ValueError("p_hat must be finite.")
    return float(np.mean((p_hat - y_true) ** 2))


def expected_loss(pd: float, lgd: float, ead: float) -> float:
    """
    Expected Loss = PD * LGD * EAD

    Args:
        pd: probability of default in [0,1]
        lgd: loss given default in [0,1]
        ead: exposure at default (currency)

    Returns:
        float: expected loss (currency)

    Raises:
        ValueError: if inputs out of range or non-finite.
    """
    for name, v in {"pd": pd, "lgd": lgd, "ead": ead}.items():
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite.")
    if not (0.0 <= pd <= 1.0):
        raise ValueError("pd must be in [0,1].")
    if not (0.0 <= lgd <= 1.0):
        raise ValueError("lgd must be in [0,1].")
    return float(pd * lgd * ead)


# Example usage (valuation/credit context): predict distress / default probability
# Features: [interest_cover, leverage, revenue_growth]
X = np.array([
    [3.0, 2.5,  0.04],
    [1.2, 4.0, -0.02],
    [2.0, 3.2,  0.01],
    [0.8, 5.0, -0.05],
    [4.5, 2.0,  0.06],
    [1.0, 4.5, -0.01],
], dtype=float)

y = np.array([0, 1, 0, 1, 0, 1], dtype=int)  # 1=distress event

fit = logistic_fit_gd(X, y, lr=0.2, iters=5000, l2=0.5)
p_hat = fit["p_hat"]
print("Training log loss:", float(fit["loss"][0]))

# Evaluate ROC/AUC
thr, fpr, tpr = roc_curve(y, p_hat, n_thresholds=101)
auc = auc_trapz(fpr, tpr)
print(f"AUC: {auc:.3f}")
print(f"Brier score (calibration): {brier_score(y, p_hat):.4f}")

# Choose threshold by cost trade-off (example: FN is 5x more costly than FP)
def choose_threshold_by_cost(y_true, p, c_fn: float, c_fp: float, n_thresholds: int = 101) -> float:
    if c_fn <= 0 or c_fp <= 0:
        raise ValueError("Costs must be positive.")
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        c = confusion_counts(y_true, p, float(t))
        cost = c_fn * c["FN"] + c_fp * c["FP"]
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t

t_star = choose_threshold_by_cost(y, p_hat, c_fn=5.0, c_fp=1.0)
print("Cost-optimal threshold:", round(t_star, 3))

# Expected loss example
pd = float(np.mean(p_hat))
el = expected_loss(pd=pd, lgd=0.55, ead=200_000_000)  # $200m exposure
print(f"Portfolio PD: {pd:.2%}, Expected loss: ${el/1e6:,.1f}m")
```

---

### Valuation Impact
Why this matters:
- **Scenario weighting:** Probabilities from classification (e.g., distress, success vs failure) provide disciplined scenario weights in DCF, especially for binary outcomes (approval, turnaround, refinancing).
- **Credit spreads and WACC:** Default probability affects cost of debt and can justify higher spreads or explicit distress scenarios (rather than arbitrarily increasing discount rates).
- **Cash-flow haircuts:** High churn/default probability can justify conservative revenue growth, higher bad debt, and higher working-capital needs.
- **Deal structuring:** Classification supports earn-outs, contingent payments, and covenants by quantifying downside likelihood.

Impact on multiples:
- A high risk score may justify discounting peer multiples or using lower percentile multiples.
- Use probabilities to segment comps into “healthy” vs “distressed” for cleaner comparability.

Impact on DCF inputs:
- Use PD to drive scenario cash flow haircuts and to modify terminal assumptions (shorter competitive advantage period if churn risk high).
- Calibration matters: if probabilities are miscalibrated, expected loss and scenario weights are wrong even if AUC is strong.

Practical adjustments:
```python
import numpy as np

def probability_to_scenario_weights(p_event: float, min_p: float = 0.0, max_p: float = 1.0) -> dict:
    """
    Convert a single event probability into two-scenario weights (event vs no-event).

    Args:
        p_event: probability in [0,1]
        min_p/max_p: optional clipping bounds

    Returns:
        dict: {"event": p, "no_event": 1-p}
    """
    if not np.isfinite(p_event):
        raise ValueError("p_event must be finite.")
    p = float(np.clip(p_event, min_p, max_p))
    if not (0.0 <= p <= 1.0):
        raise ValueError("p_event must be in [0,1] after clipping.")
    return {"event": p, "no_event": 1.0 - p}
```

---

### Quality of Earnings Flags
⚠️ **Red flag: misclassified “one-offs”** — if distress events are driven by non-recurring accounting noise, the model will not generalise and can bias valuation risk overlays.  
⚠️ **Red flag: label leakage** — using variables that are outcomes of distress (e.g., covenant breach already recorded) to predict distress.  
⚠️ **Red flag: ignoring imbalance** — rare events (defaults) can make accuracy look high even if the model is useless; use AUC, precision/recall, and calibrated probabilities.  
⚠️ **Red flag: uncalibrated probabilities** — good AUC but poor Brier score; scenario weights and expected loss become unreliable.  
✅ **Green flag: stable AUC + good calibration** across time splits and stress periods; thresholds chosen by explicit cost logic.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| Banking / Credit | Rare default events | Use calibrated PDs; evaluate with precision/recall, AUC; consider economic stress splits |
| SaaS / Subscriptions | Churn is time-to-event | Prefer survival analysis for hazard/retention; classification for near-term churn flags |
| Retail | Fraud/chargebacks | Cost-based thresholds; monitor drift with seasonality |
| Industrials | Covenant-driven distress | Include leverage/coverage; avoid post-default variables; prefer time-based validation |

---

### Real-World Example
Scenario: You build a distress classifier using leverage and coverage ratios and then use the output to compute expected loss and to choose a threshold based on misclassification costs.

```python
# The example code above prints:
# - training log loss
# - ROC AUC (ranking)
# - Brier score (calibration)
# - cost-optimal threshold
# - portfolio expected loss in $m
```

Interpretation:
- If AUC is good but Brier score is weak, treat probabilities cautiously: use them for ranking but calibrate before using as scenario weights or PD in expected loss.

See also: Chapter 5 (cross-validation), Chapter 6 (regularization), Chapter 11 (survival analysis)
