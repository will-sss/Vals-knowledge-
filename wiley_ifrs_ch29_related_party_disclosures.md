# Wiley Interpretation and Application of IFRS Standards (2023) - Key Concepts for Valuation

## Chapter 29: Related Party Disclosures (IAS 24, transactions, transfer pricing, QoE risk)

### Core Concept
IAS 24 requires disclosures of relationships and transactions with related parties (e.g., parent entities, subsidiaries, associates, joint ventures, key management personnel, and entities under common control). For valuation, related party arrangements can distort reported revenue, margins, working capital, and cash flows, and may signal governance risk or non-arm’s-length pricing. A valuation analyst uses IAS 24 disclosures to normalise earnings and identify off-market support (or leakage) that will not persist under a market participant view.

### Formula/Methodology

#### 1) Related party definition (practical test)
```text
A party is related if it has:
- control or joint control of the entity, or significant influence, or
- is a member of the same group, or
- is a key management person (KMP) or a close family member of KMP, or
- is an entity controlled/jointly controlled by KMP/close family member, or
- is an associate or joint venture of the entity or of a group member, or
- is a post-employment benefit plan for employees of the entity (or related group).

Key goal:
Identify relationships that could influence transactions and terms.
```

#### 2) What must be disclosed (analyst checklist)
```text
Disclose:
- parent and ultimate controlling party
- relationships with related parties
- nature and amounts of related party transactions
- outstanding balances (receivables/payables/loans) and terms
- provisions for doubtful debts related to those balances
- expenses recognised for bad debts/written off related to related parties
- KMP compensation (often broken into short-term, post-employment, share-based, other)

Transactions include:
- sales/purchases of goods and services
- transfers of assets, IP licensing
- leases
- guarantees/collateral
- loans and equity contributions
- management fees and cost sharing
```

#### 3) Normalisation principle for valuation
```text
Normalised metric = Reported metric - Off-market effect

Off-market effect examples:
- Related-party sales priced above/below market
- Management fees that are non-market (too high/too low)
- Non-recurring support (waived rent, injected working capital, subsidised financing)
- One-way cash leakage (excessive dividends, related-party procurement)

Objective:
Restate to market participant terms and sustainable standalone economics.
```

---

### Practical Application for Valuation Analysts

#### 1) Convert IAS 24 disclosures into a “related-party dependency map”
Create a table capturing:
- Related party counterparty
- Relationship type (parent, fellow subsidiary, KMP-controlled entity, JV, etc.)
- Transaction type (sales, purchases, management fees, loans, leases, guarantees)
- Annual P&L effect (revenue/expense)
- Balance sheet exposure (receivable, payable, loan, guarantee)
- Terms (pricing basis, maturity, interest rate, collateral)
- Whether transaction is recurring and arm’s-length

Valuation use-cases:
- Remove dependency on captive related-party revenue if not transferable post-transaction.
- Adjust gross margin if transfer pricing is off-market.
- Reclassify loans/guarantees in enterprise-to-equity bridge.

#### 2) Identify “standalone” vs “group-supported” economics
Common patterns:
- Parent provides cheap funding (below market interest) → EBITDA looks fine but FCFE is overstated.
- Group provides shared services at low cost → operating margins are overstated (future costs higher).
- Entity provides services to group with pricing set for tax reasons → revenue and margins not market-based.

Practical steps:
- Benchmark related-party pricing to external comparables (market rates, third-party contracts).
- Where pricing is unknown, build a sensitivity around “market fee” assumptions.

#### 3) Treat related-party loans and guarantees as debt-like items
Adjust EV-to-equity bridge:
- Add related-party loans to net debt (unless clearly equity-like and subordinated with no repayment intent).
- Consider guarantees: they are contingent liabilities; treat as debt-like if likely to be called.

#### 4) Cross-check with other disclosures
Related-party risk often shows up elsewhere:
- Segment disclosures (IFRS 8): major customers that may be related parties.
- Leases (IFRS 16): leases with parent/KMP entities (off-market rents).
- Business combinations: post-acquisition “continuing involvement” or earn-outs with sellers.

---

### Python Implementation
```python
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

def _num(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is missing.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v

def related_party_dependency_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a structured related-party table for valuation review.

    Required keys per row:
        - counterparty
        - relationship
        - transaction_type
        - pnl_effect (positive for revenue, negative for expense)
        - balance_exposure (positive for receivable/asset, negative for payable/liability)

    Returns:
        DataFrame with basic validation and helper flags.

    Raises:
        ValueError: missing keys or invalid numbers.
    """
    if not rows:
        raise ValueError("rows must not be empty.")
    df = pd.DataFrame(rows)

    req = {"counterparty", "relationship", "transaction_type", "pnl_effect", "balance_exposure"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Validate numeric fields
    df["pnl_effect"] = df["pnl_effect"].apply(lambda x: _num(x, "pnl_effect"))
    df["balance_exposure"] = df["balance_exposure"].apply(lambda x: _num(x, "balance_exposure"))

    df["is_revenue"] = df["pnl_effect"] > 0
    df["is_expense"] = df["pnl_effect"] < 0
    df["net_pnl_effect"] = df["pnl_effect"]
    df["net_balance_exposure"] = df["balance_exposure"]
    return df

def normalize_for_off_market_pricing(
    reported_amount: float,
    market_amount: float
) -> float:
    """
    Replace a reported related-party revenue/expense with a market-equivalent amount.

    Args:
        reported_amount: amount in statements (currency)
        market_amount: analyst estimate of market-equivalent amount (currency)

    Returns:
        float: adjustment to apply to reported metric (market - reported)

    Raises:
        ValueError: invalid inputs.
    """
    rep = _num(reported_amount, "reported_amount")
    mkt = _num(market_amount, "market_amount")
    return mkt - rep

def adjusted_ebitda(reported_ebitda: float, adjustments: List[float]) -> float:
    """
    Apply a list of EBITDA adjustments (positive increases EBITDA, negative decreases).

    Raises:
        ValueError: invalid input.
    """
    e = _num(reported_ebitda, "reported_ebitda")
    adj = 0.0
    for i, a in enumerate(adjustments):
        adj += _num(a, f"adjustments[{i}]")
    return e + adj

def adjust_net_debt_for_related_party_loans(net_debt: float, related_party_loans: float) -> float:
    """
    Add related-party loans to net debt (debt-like treatment).

    Args:
        net_debt: existing net debt (positive means debt > cash)
        related_party_loans: outstanding related party loans (positive number)

    Returns:
        float: adjusted net debt

    Raises:
        ValueError: negative loans.
    """
    nd = _num(net_debt, "net_debt")
    rpl = _num(related_party_loans, "related_party_loans")
    if rpl < 0:
        raise ValueError("related_party_loans must be >= 0.")
    return nd + rpl

# Example usage
rows = [
    {"counterparty": "ParentCo", "relationship": "parent", "transaction_type": "management fee",
     "pnl_effect": -12_000_000, "balance_exposure": -3_000_000, "terms": "monthly fee, no benchmark disclosed"},
    {"counterparty": "GroupDistCo", "relationship": "fellow subsidiary", "transaction_type": "sales",
     "pnl_effect": 45_000_000, "balance_exposure": 8_000_000, "terms": "transfer pricing basis not disclosed"},
]

df = related_party_dependency_table(rows)
print(df[["counterparty", "transaction_type", "pnl_effect", "balance_exposure"]])

# Normalise management fee to market estimate
fee_adjustment = normalize_for_off_market_pricing(reported_amount=-12_000_000, market_amount=-18_000_000)
# If expense becomes more negative, EBITDA decreases by 6m
new_ebitda = adjusted_ebitda(200_000_000, [fee_adjustment])
print(f"EBITDA after management fee normalisation: ${new_ebitda/1e6:.1f}M")

# Add related-party loans to net debt
adj_net_debt = adjust_net_debt_for_related_party_loans(net_debt=500_000_000, related_party_loans=120_000_000)
print(f"Adjusted net debt: ${adj_net_debt/1e6:.1f}M")
```

---

### Valuation Impact
Why this matters:
- Related-party terms can inflate (or depress) profitability and cash conversion versus an arm’s-length standalone business.
- Captive revenue or subsidised costs can make multiples misleading unless normalised.
- Related-party loans and guarantees can be hidden leverage that reduces equity value and increases risk.

Impact on multiples:
- EV/EBITDA: margins may be artificially high/low if transfer pricing or management fees are off-market.
- P/E: net income can be distorted by non-market interest, tax structuring, or non-recurring support.

Impact on DCF inputs:
- Forecasts must reflect sustainable standalone pricing, costs, working capital terms, and financing costs.
- Remove non-recurring related-party support when estimating long-run margins and cash taxes.

Comparability issues:
- Group entities may be optimised for tax/transfer pricing rather than standalone profitability.
- Different disclosure depth across companies can bias peer comparisons; apply a disclosure risk haircut in cases with limited data.

Practical adjustments:
```python
def valuation_adjustment_for_related_party_revenue(reported_revenue: float, rp_revenue: float, sustainable_share: float) -> float:
    """
    Reduce reported revenue for non-sustainable related-party revenue.

    Args:
        reported_revenue: total revenue (currency)
        rp_revenue: related-party revenue within total (currency)
        sustainable_share: 0-1 estimate of portion that is sustainable post-transaction

    Returns:
        float: adjusted revenue

    Raises:
        ValueError: invalid inputs.
    """
    rr = _num(reported_revenue, "reported_revenue")
    rp = _num(rp_revenue, "rp_revenue")
    s = _num(sustainable_share, "sustainable_share")
    if rr < 0 or rp < 0:
        raise ValueError("revenues must be >= 0.")
    if rp > rr:
        raise ValueError("rp_revenue cannot exceed reported_revenue.")
    if not (0.0 <= s <= 1.0):
        raise ValueError("sustainable_share must be between 0 and 1.")
    return rr - rp * (1.0 - s)
```

---

### Quality of Earnings Flags
⚠️ High proportion of revenue from a single related party or “group companies” without long-term contracts.  
⚠️ Significant management fees/royalties paid to parent with no benchmark or explanation of services provided.  
⚠️ Related-party receivables growing faster than sales (collection risk; potential profit overstatement).  
⚠️ Non-market financing (interest-free or below-market loans) masking true cost of capital and cash flow risk.  
⚠️ Related-party guarantees and commitments not quantified or described clearly.  
✅ Clear disclosure of nature, amounts, and terms, with evidence (or strong rationale) that pricing is arm’s-length.

---

### Sector-Specific Considerations

| Sector | Common related-party issue | Typical valuation handling |
|---|---|---|
| Real estate | properties leased from KMP-controlled entities | normalise rent to market; treat lease liabilities consistently |
| Private equity roll-ups | management fees to sponsor/affiliate | assess sustainability post-deal; normalise to market services |
| Emerging markets groups | transfer pricing and shared procurement | sensitivity on margins and FX; disclosure risk adjustments |
| Financial services | intra-group funding and guarantees | treat funding as debt-like; assess liquidity backstops and contagion risk |

---

### Real-World Example
Scenario: A company reports $300m revenue. $90m is sales to a fellow subsidiary at unknown pricing. Analyst believes only 40% would remain if the company were sold to a third party. Management fee to parent is $10m but market estimate is $16m.

```python
reported_rev = 300_000_000
rp_rev = 90_000_000
adj_rev = valuation_adjustment_for_related_party_revenue(reported_rev, rp_rev, sustainable_share=0.40)
print(f"Adjusted revenue: ${adj_rev/1e6:.1f}M")

reported_ebitda = 60_000_000
fee_adj = normalize_for_off_market_pricing(reported_amount=-10_000_000, market_amount=-16_000_000)
adj_ebitda = adjusted_ebitda(reported_ebitda, [fee_adj])
print(f"Adjusted EBITDA: ${adj_ebitda/1e6:.1f}M")
```

Interpretation: Related-party normalisation can materially reduce sustainable revenue and margins, lowering EV/EBITDA value and DCF cash flows.

See also: Chapter 20 (revenue) for customer concentration and contract terms; Chapter 22 (leases) for related-party leases; Chapter 24 (financial instruments) for related-party loans/guarantees; Chapter 28 (segments) to reconcile major customers and intersegment activity.
