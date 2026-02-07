# An Introduction to Statistical Learning (ISLP) - Key Concepts for Valuation

## Chapter 8: Tree-Based Methods (Decision Trees, Random Forests, Gradient Boosting, Feature Importance)

### Core Concept
Tree-based methods model **non-linear relationships and interactions** without requiring explicit feature engineering (e.g., different margin behaviour at high utilisation, churn risk jumping when pricing exceeds a threshold). In valuation, trees are practical for **forecasting with regime changes**, **risk flagging** (distress, covenant breach), and **driver discovery** when linear models underfit or assumptions break. The main risks are overfitting (single trees), leakage, and unstable importance measures if data is limited.

See also: Chapter 3 (linear regression), Chapter 6 (regularization), Chapter 5 (cross-validation), Chapter 2 (bias–variance).

---

### Formula/Methodology

#### 1) Regression trees (piecewise constant predictions)
A regression tree partitions feature space into regions R1..RM and predicts the average y in each region:
```text
ŷ(x) = average(y_i) for all i such that x ∈ R_m
```

Split criterion (regression): choose split that minimises SSE:
```text
Choose split (j, s) to minimise:
SSE = Σ_{x_i ∈ R1(j,s)} (y_i − ȳ_R1)^2 + Σ_{x_i ∈ R2(j,s)} (y_i − ȳ_R2)^2
```
Where:
- j = feature index
- s = split threshold
- R1 / R2 = left/right partitions induced by the split
- ȳ_R = mean response in region R

Split criterion (classification): reduce impurity.
Common impurity measures:
```text
Gini = Σ_k p_k (1 − p_k) = 1 − Σ_k p_k^2
Entropy = − Σ_k p_k log(p_k)
Misclassification = 1 − max_k p_k
```
Where p_k is the fraction of class k in a node.

#### 2) Pruning (control complexity)
Large trees overfit. Prune using cost-complexity:
```text
Minimise:  RSS(T) + α × |T|
```
Where:
- RSS(T) = residual sum of squares of tree T
- |T| = number of terminal nodes (leaves)
- α controls tree size (higher α ⇒ smaller tree)

Select α by cross-validation.

#### 3) Bagging and Random Forests (variance reduction)
Bagging:
- Fit many trees on bootstrap samples and average predictions.

Random forest:
- Same as bagging but at each split consider only a random subset of m features (mtry).
Practical effect:
- De-correlates trees ⇒ stronger variance reduction than bagging when predictors are correlated.

Out-of-bag (OOB) error:
- Each tree predicts on observations not included in its bootstrap sample; aggregate for validation without a separate test set.

#### 4) Boosting (bias reduction with additive trees)
Gradient boosting fits trees sequentially to residuals (or gradients).
Additive model:
```text
F_M(x) = Σ_{m=1..M} ν × h_m(x)
```
Where:
- h_m(x) = shallow tree at iteration m
- ν = learning rate (shrinkage)
- M = number of trees

Practical tuning:
- Lower ν usually improves generalisation but requires larger M.

#### 5) Feature importance (interpretability)
Common approach: impurity decrease / split gain aggregated across trees (forest/boosting).
Valuation caution:
- Importance can be biased toward high-cardinality or noisy features; validate with stability checks and domain logic.

---

### Python Implementation
```python
from typing import Dict, Tuple, Optional, List
import numpy as np

# NOTE: This implementation uses scikit-learn (allowed per your constraints).
# In Python-in-Excel: ensure scikit-learn is available in your environment.

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from sklearn.model_selection import KFold


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


def tree_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    max_depth_grid: List[Optional[int]],
    min_samples_leaf_grid: List[int],
    k: int = 5,
    seed: int = 0
) -> Dict[str, float]:
    """
    Cross-validate a regression tree hyperparameter grid and return best params by CV MSE.

    Args:
        X: predictors (n, p)
        y: response (n,)
        max_depth_grid: list of max_depth values (None allowed)
        min_samples_leaf_grid: list of min_samples_leaf integers
        k: folds
        seed: random seed

    Returns:
        dict: best_max_depth, best_min_samples_leaf, best_cv_mse

    Raises:
        ValueError: invalid inputs.
    """
    X = _as_2d(X)
    y = _as_1d(y).astype(float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if k < 2 or k > len(y):
        raise ValueError("k must be between 2 and n.")
    if not max_depth_grid or not min_samples_leaf_grid:
        raise ValueError("Parameter grids must be non-empty.")
    for msl in min_samples_leaf_grid:
        if not isinstance(msl, int) or msl <= 0:
            raise ValueError("min_samples_leaf values must be positive integers.")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    best = {"best_max_depth": None, "best_min_samples_leaf": None, "best_cv_mse": float("inf")}

    for md in max_depth_grid:
        for msl in min_samples_leaf_grid:
            mses = []
            for tr_idx, te_idx in kf.split(X):
                model = DecisionTreeRegressor(max_depth=md, min_samples_leaf=msl, random_state=seed)
                model.fit(X[tr_idx], y[tr_idx])
                pred = model.predict(X[te_idx])
                mses.append(mean_squared_error(y[te_idx], pred))
            cv_mse = float(np.mean(mses))
            if cv_mse < best["best_cv_mse"]:
                best.update({"best_max_depth": md, "best_min_samples_leaf": float(msl), "best_cv_mse": cv_mse})
    return best


def fit_random_forest_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    max_features: str = "sqrt",
    seed: int = 0
) -> RandomForestRegressor:
    """
    Fit a RandomForestRegressor with defensive checks.

    Returns:
        fitted RandomForestRegressor
    """
    X = _as_2d(X)
    y = _as_1d(y).astype(float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be > 0.")
    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be > 0.")

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf


def fit_gb_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    seed: int = 0
) -> GradientBoostingRegressor:
    """
    Fit GradientBoostingRegressor (shallow trees with shrinkage).

    Returns:
        fitted GradientBoostingRegressor
    """
    X = _as_2d(X)
    y = _as_1d(y).astype(float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be > 0.")
    if not (0.0 < learning_rate <= 1.0):
        raise ValueError("learning_rate must be in (0,1].")
    if max_depth <= 0:
        raise ValueError("max_depth must be > 0.")

    gb = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed
    )
    gb.fit(X, y)
    return gb


def feature_importance_table(importances: np.ndarray, feature_names: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """
    Create a sorted feature importance table.

    Returns:
        list of (feature_name, importance) sorted descending
    """
    importances = np.asarray(importances, dtype=float).ravel()
    if importances.size == 0:
        raise ValueError("importances must not be empty.")
    if np.any(~np.isfinite(importances)):
        raise ValueError("importances must be finite.")
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(importances.size)]
    if len(feature_names) != importances.size:
        raise ValueError("feature_names length must match importances length.")
    pairs = list(zip(feature_names, importances.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


# Example usage (valuation context): forecast EBITDA margin under non-linear effects
# Features: [utilisation, input_inflation, pricing_power_index, backlog_growth]
X = np.array([
    [0.60, 0.06, 0.30, 0.02],
    [0.65, 0.05, 0.35, 0.03],
    [0.70, 0.04, 0.40, 0.04],
    [0.75, 0.03, 0.45, 0.02],
    [0.80, 0.02, 0.55, 0.01],
    [0.85, 0.01, 0.60, 0.00],
    [0.55, 0.07, 0.25, 0.01],
    [0.90, 0.00, 0.70, 0.05],
], dtype=float)

y = np.array([0.10, 0.11, 0.13, 0.14, 0.16, 0.17, 0.09, 0.18], dtype=float)  # EBITDA margin

# 1) Choose a small tree by CV (interpretability)
best = tree_regression_cv(X, y, max_depth_grid=[2, 3, 4, None], min_samples_leaf_grid=[1, 2, 3], k=4, seed=1)
print("Best tree params:", best)

tree = DecisionTreeRegressor(max_depth=int(best["best_max_depth"]) if best["best_max_depth"] is not None else None,
                             min_samples_leaf=int(best["best_min_samples_leaf"]), random_state=1)
tree.fit(X, y)

# 2) Fit random forest (robust non-linear baseline)
rf = fit_random_forest_regression(X, y, n_estimators=300, max_depth=None, min_samples_leaf=2, seed=1)
rf_imp = feature_importance_table(rf.feature_importances_, ["util", "infl", "pricing", "backlog"])
print("RF feature importance:", rf_imp)

# 3) Fit gradient boosting (stronger predictive power with tuning)
gb = fit_gb_regression(X, y, n_estimators=500, learning_rate=0.05, max_depth=2, seed=1)
gb_imp = feature_importance_table(gb.feature_importances_, ["util", "infl", "pricing", "backlog"])
print("GB feature importance:", gb_imp)

# Predict next-year EBITDA margin under a scenario
X_new = np.array([[0.78, 0.025, 0.50, 0.03]], dtype=float)
print("Tree pred:", float(tree.predict(X_new)[0]))
print("RF pred:", float(rf.predict(X_new)[0]))
print("GB pred:", float(gb.predict(X_new)[0]))
```

---

### Valuation Impact
Why this matters:
- **Non-linear drivers:** Trees capture thresholds (e.g., margin jumps above 75% utilisation; churn spikes above a price threshold), improving forecast realism versus linear extrapolation.
- **Scenario modelling:** Tree models naturally support scenario inputs (macro regimes) and can identify “break points” for bear/bull cases.
- **Risk overlays:** Classification forests/boosting support distress, fraud, covenant breach, or customer churn risk scoring that feeds expected loss, scenario weights, and valuation haircuts.

Impact on multiples:
- Tree-based models can be used to explain why a company’s multiple differs from peers by capturing interactions (e.g., high growth only rewarded when churn is low).
- Use as evidence, not as a mechanical “multiple generator”; avoid model-driven circularity.

Impact on DCF inputs:
- Use predicted margins/growth to triangulate base-case; compare to management plan and industry structure.
- Use model sensitivity to identify key value drivers and structure scenario analysis around them.

Practical adjustments:
```python
import numpy as np

def blend_model_with_management(model_forecast: float, mgmt_forecast: float, model_weight: float) -> float:
    """
    Blend model forecast with management forecast (triangulation).

    Args:
        model_forecast: model output (e.g., margin)
        mgmt_forecast: management case
        model_weight: in [0,1]

    Returns:
        blended forecast
    """
    if not (np.isfinite(model_forecast) and np.isfinite(mgmt_forecast)):
        raise ValueError("Forecasts must be finite.")
    if not (0.0 <= model_weight <= 1.0):
        raise ValueError("model_weight must be in [0,1].")
    return float(model_weight * model_forecast + (1.0 - model_weight) * mgmt_forecast)
```

---

### Quality of Earnings Flags
⚠️ **Red flag: leakage via “post-outcome” features** — e.g., using revised EBITDA, covenant breach flags, or post-period collections to predict distress.  
⚠️ **Red flag: single deep tree** — very low training error but unstable out-of-sample; prune or move to ensembles.  
⚠️ **Red flag: unstable feature importance** — if importance rankings change wildly across resamples, do not over-interpret drivers.  
⚠️ **Red flag: small samples** — forests/boosting can overfit without careful cross-validation.  
✅ **Green flag: time-based validation** and stable performance across regimes; drivers align with operational logic.

---

### Sector-Specific Considerations
| Sector | Key Issue | Typical Treatment |
|---|---|---|
| SaaS | Interactions between growth, churn, and CAC | Boosting for prediction; interpret via partial dependence or scenario probes |
| Banking | Non-linear default risk at high leverage | Forests/GBMs for PD models; calibrate probabilities; monitor drift |
| Industrials | Margin regime shifts with utilisation | Regression trees for interpretability; forests for robustness |
| Retail | Promotions create non-linear demand | Trees capture thresholds; validate across seasons |

---

### Real-World Example
Scenario: EBITDA margin depends non-linearly on utilisation and pricing power (threshold effects). A tree/forest/boosted model provides more realistic forecasts and identifies key driver breakpoints.

```python
# The example code above:
# - selects tree complexity by CV
# - fits random forest and gradient boosting
# - prints feature importance
# - predicts next-year margin under a scenario
```

Interpretation:
- If tree/forest outputs materially exceed management forecasts, check for leakage and overfitting before using in valuation.
- Use model-driven breakpoints to define scenarios (e.g., utilisation < 70% bear case, > 80% bull case).

See also: Chapter 6 (regularization), Chapter 5 (cross-validation), Chapter 12 (unsupervised learning for segment discovery)
