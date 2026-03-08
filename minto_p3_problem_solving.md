# Minto Pyramid Principle - Logic in Problem Solving

## Part 3: Defining and Structuring Financial Problems

### Core Concept

Before analyzing any financial problem, you must define it properly. Most analysts jump straight to number-crunching without establishing what question they're actually answering. The Problem Definition Framework ensures you understand the situation, identify what changed (complication), and articulate the specific question requiring analysis. This structured approach prevents wasted effort on irrelevant analysis and ensures your findings directly address stakeholder concerns.

### The Problem Definition Framework (SCQA in Analysis)

Every financial problem has four elements that must be explicitly stated:

```
Starting Point → Disturbing Event → R1/R2 → Question

Where:
- Starting Point: The stable situation before the problem
- Disturbing Event: What changed or was discovered
- R1 (Result): Undesired outcome that occurred
- R2 (Result): Desired outcome we want instead
- Question: What we must decide or discover
```

#### Element 1: Starting Point / Opening Scene

**Definition**: The stable, factual situation before the problem arose.

**Financial Example - Valuation Assignment**:
```
Starting Point:
"Private equity firm XYZ has been approached by the founders 
of TechCo (B2B SaaS, $150M ARR, growing 40%) who are seeking 
a minority investor. The firm typically targets 3.0x MOIC and 
25% IRR on minority deals."
```

**Board Decision Example**:
```
Starting Point:
"Our European manufacturing division generates €500M revenue 
with 12% EBITDA margins. We acquired it five years ago for 
€600M as part of our international expansion strategy."
```

**Rule**: State only facts that both you and the reader agree are true. No interpretation yet.

#### Element 2: Disturbing Event

**Definition**: Something happened or was discovered that disrupts the starting point.

**Types of Disturbing Events**:

**1. External Change**:
```
"A strategic buyer has made an unsolicited offer to acquire 
the division for €800M (12x EBITDA), 30% above our internal 
valuation of €600M."
```

**2. Internal Discovery**:
```
"Financial review reveals the division's ROIC (6%) has fallen 
below our WACC (8%) for three consecutive years, destroying 
economic value despite positive accounting earnings."
```

**3. Opportunity Identified**:
```
"Market analysis shows consolidation in the sector, with 
three competitors acquired at 14-16x EBITDA in the past year, 
suggesting potential undervaluation of our division."
```

**4. New Requirement**:
```
"The board has mandated portfolio optimization, requiring all 
divisions to achieve ROIC > WACC + 300bps or face divestment 
consideration within 12 months."
```

#### Element 3: R1 (Undesired Result) or R2 (Desired Result)

**Two scenarios exist**:

**Scenario 1: R1 - Something bad happened**
```
Disturbing Event: "ROIC fell below WACC"
R1 (Undesired): "We are destroying shareholder value"
Question: "How do we fix this?"
```

**Scenario 2: R2 - We want to achieve something**
```
Disturbing Event: "Received acquisition offer"
R2 (Desired): "We want to maximize shareholder value"
Question: "Should we sell or hold?"
```

**Financial Examples**:

**Private Equity Deal (R2 - Desired Result)**:
```
Starting Point: Seeking minority investment opportunity
Disturbing Event: TechCo founders offering 40% stake for $200M
R2: We want to deploy capital at >25% IRR
Question: Does this deal meet our return hurdle?
```

**Credit Decision (R1 - Undesired Result)**:
```
Starting Point: Company requests $100M revolving credit facility
Disturbing Event: Leverage is 4.5x (our limit: 3.5x)
R1: Lending would violate credit policy
Question: Can we structure the facility to mitigate risk?
```

#### Element 4: The Question

**The question must be one that**:
- The reader actually cares about answering
- Directly follows from R1 or R2
- Can be answered with analysis

**Common Question Patterns in Finance**:

| **Situation Type** | **Typical Question** |
|-------------------|---------------------|
| **Valuation** | "What is fair value?" / "What should we pay?" |
| **Investment Decision** | "Should we buy/sell/hold?" |
| **Strategic Choice** | "Should we acquire or build organically?" |
| **Problem Diagnosis** | "Why is ROIC declining?" / "What's driving margin compression?" |
| **Action Planning** | "How do we improve returns?" |
| **Risk Assessment** | "What could cause this investment to fail?" |

### Converting Problem Definition to Introduction

Once you've defined the problem, convert it directly to your document introduction:

**Problem Definition**:
```
Starting Point: Division generates €500M revenue, 12% EBITDA margin
Disturbing Event: Strategic buyer offers €800M (12x EBITDA)
R2: Want to maximize shareholder value  
Question: Should we sell or retain?
```

**Document Introduction (SCQA)**:
```
Situation: Our European division generates €500M revenue with 
12% EBITDA margins, representing 15% of company EBITDA.

Complication: We've received an unsolicited offer of €800M 
(12x EBITDA) from a strategic buyer—30% above our internal 
valuation of €600M.

Question: Should we accept the offer and realize immediate 
value, or retain the division for long-term value creation?

Answer: Recommend accepting the offer. Immediate value 
realization outweighs uncertain long-term upside given 
execution risks and superior capital redeployment opportunities.
```

### Real-Life Example: Restructuring a Valuation Problem

**Initial (Unclear) Request**:
```
"Can you value Company X? The owners want to sell and we're 
considering buying them."
```

**Problem**: This doesn't define what question you're actually answering.

**Problem Definition Process**:

**1. Identify Starting Point**:
```
Q: "What's the current situation?"
A: "Company X is a family-owned distributor doing $50M revenue. 
The founding generation is retiring and wants liquidity. We're 
a strategic buyer in the same sector."
```

**2. Identify Disturbing Event**:
```
Q: "What changed?"
A: "The owners approached us about a sale. They've given us 
exclusivity for 60 days to make an offer."
```

**3. Identify R2 (what we want)**:
```
Q: "What do we want to achieve?"
A: "We want to acquire them if value creation exceeds our 
15% IRR hurdle."
```

**4. Determine the Question**:
```
Q: "What do you need to know?"
A: "What's the maximum price we can pay while achieving 15% IRR?"
```

**Proper Problem Definition**:
```
Starting Point: Company X (family-owned distributor, $50M revenue) 
is available for acquisition. We're a strategic buyer with 15% IRR 
hurdle for acquisitions.

Disturbing Event: Owners seeking liquidity and have granted us 
60-day exclusivity.

R2: We want to acquire if we can achieve 15% returns including 
synergies.

Question: What is the maximum price we should offer to achieve 
our 15% IRR hurdle?
```

**Now the analysis path is clear**:
```
To answer the question, we need to:
1. Forecast standalone cash flows
2. Quantify available synergies (revenue + cost)
3. Build acquisition model with synergies
4. Solve for price that yields 15% IRR
5. Add risk-adjusted sensitivity ranges
```

### Structuring the Analysis

Once the problem is defined, structure your analysis using frameworks that pre-organize thinking into pyramid form.

#### Framework 1: Physical Structure Analysis

**When to use**: Analyzing organizations, processes, systems with definable components.

**Example: Diagnosing Margin Decline**

**Problem**: "Why are gross margins declining?"

**Physical Structure** (Income Statement):
```
Revenue
  ├── Volume (units sold)
  └── Price (per unit)

minus

Cost of Goods Sold  
  ├── Variable costs (per unit)
  └── Fixed costs (total)

equals

Gross Profit
```

**Analysis Framework**:
```
Margin declined 300bps (from 40% to 37%)

Investigate each component:
1. Revenue side
   1.1 Volume trend: Declining 5% annually
   1.2 Price trend: Flat (no pricing power)
   
2. COGS side  
   2.1 Variable cost/unit: Up 12% (raw materials)
   2.2 Fixed costs: Up 8% (new facility)
   
Conclusion: Margin pressure driven by unfavorable cost trends 
outpacing volume decline impact.
```

**Pyramid Structure Emerges**:
```
Margin decline driven by COGS inflation, not revenue pressure
├── Variable costs up 12% (commodity prices)
│   └── Evidence: Raw material basket up 15% industry-wide
├── Fixed costs up 8% (new facility)
│   └── Evidence: Depreciation increased $5M annually
└── Revenue factors neutral
    ├── Volume down 5% (market share stable)
    └── Price flat (competitive market)
```

#### Framework 2: Cause-and-Effect Analysis

**When to use**: Understanding why something happened.

**Example: Why Did the Company Miss Earnings?**

**Cause-Effect Chain**:
```
Root Cause → Intermediate Effect → Final Result

Sales team turnover → Lost key accounts → Revenue miss
       ↓                    ↓                ↓
    (70% churn)      (Top 3 accounts)   (-15% vs. plan)
```

**Analysis Structure**:
```
Company missed revenue by 15% due to customer attrition
├── Lost 3 of top 10 accounts (30% of revenue)
│   ├── Причина: Account manager departures
│   │   └── Evidence: 70% sales team turnover in Q2
│   └── Результат: $12M revenue lost
│       └── Evidence: These accounts = $40M annual ARR
└── Remaining accounts: on-track
    └── Evidence: Retention 95% ex-affected accounts
```

**Question/Answer Flow**:
```
Statement: "We missed revenue targets"
    ↓ (Why?)
"Lost key accounts"
    ↓ (Why?)
"High sales team turnover"
    ↓ (Why?)
"Compensation structure non-competitive"
```

#### Framework 3: Classification of Causes

**When to use**: Complex problems with multiple contributing factors.

**Example: M&A Integration Risks**

**Classification Framework**:
```
Integration Risks
├── Financial Risks
│   ├── Debt covenant compliance
│   ├── Working capital drain
│   └── Earnout disputes
├── Operational Risks  
│   ├── Systems integration complexity
│   ├── Customer retention during transition
│   └── Key employee departures
└── Strategic Risks
    ├── Synergies not realized
    ├── Cultural mismatch
    └── Market share loss to competitors
```

**Pyramid Structure**:
```
Integration carries elevated risk requiring mitigation plan
├── Financial risks moderate (manageable)
│   ├── Covenant headroom: 1.5x buffer
│   └── Working capital: $10M contingency reserved
├── Operational risks high (require active management)
│   ├── Customer retention: 15% at risk ($20M ARR)
│   ├── Key employee retention: 5 critical roles
│   └── Systems: 18-month integration timeline
└── Strategic risks material (threatening synergies)
    ├── Synergies: $30M target, 40% at risk
    └── Culture: Significant differences in structure
```

### Applying Frameworks to Valuation Problems

#### Diagnostic Framework: DCF Sensitivity Analysis

**Problem**: "What drives valuation sensitivity?"

**Framework**: Isolate DCF components and test
```
Enterprise Value
├── PV of Forecast Period CF
│   ├── Revenue growth rate (test: 8%/10%/12%)
│   ├── EBITDA margin (test: 20%/22%/24%)
│   └── Working capital efficiency
├── Terminal Value
│   ├── Perpetuity growth (test: 2%/3%/4%)
│   └── Exit multiple (test: 9x/10x/11x)
└── WACC
    ├── Cost of equity (test: 10%/11%/12%)
    └── Capital structure (test: 60/40, 70/30)
```

**Analysis Output (Pyramid)**:
```
Valuation most sensitive to terminal value assumptions
├── Terminal value = 75% of enterprise value
│   ├── Perpetuity growth: ±1% = ±$150M EV
│   └── Exit multiple: ±1x = ±$180M EV
├── Forecast period sensitivities moderate
│   ├── Revenue growth: ±2% = ±$50M EV
│   └── EBITDA margin: ±200bps = ±$60M EV
└── WACC impact meaningful
    └── ±100bps = ±$120M EV
```

**Conclusion**: "Focus valuation negotiation on exit assumptions, not near-term forecasts."

#### Diagnostic Framework: Comparable Company Selection

**Problem**: "Which companies are truly comparable?"

**Framework**: Multi-dimensional screening
```
Comparability Criteria
├── Business Model Similarity
│   ├── Product/service overlap (>70%)
│   ├── Customer segment (B2B vs B2C)
│   └── Revenue model (subscription, transactional, etc.)
├── Financial Characteristics  
│   ├── Size (revenue within 0.5x - 2x)
│   ├── Growth profile (±10% CAGR)
│   └── Profitability stage (profitable vs. growth)
└── Geographic Exposure
    ├── Currency exposure (USD, EUR, emerging)
    └── Regulatory environment similarity
```

**Analysis Structure**:
```
Final comp set: 6 companies (from initial 25)

Selection logic:
├── Business model: 12 companies passed
│   └── Criteria: >70% revenue from same product categories
├── Financial profile: 8 companies passed  
│   ├── Size: $500M - $2B revenue
│   ├── Growth: 20-30% CAGR
│   └── Margins: 15-25% EBITDA
└── Geographic: 6 companies passed
    └── Criteria: >50% revenue from developed markets

Excluded companies:
├── 10 too small (<$500M revenue)
├── 5 wrong business model (hardware vs. software)
└── 4 emerging market exposure (>50% revenue)
```

### Structuring Recommendations: Action Steps Framework

**Problem**: "How do we improve ROIC?"

**Framework**: Decompose ROIC drivers
```
ROIC = NOPAT / Invested Capital
     = (NOPAT/Sales) × (Sales/IC)
     = Margin × Capital Turnover
```

**Action Plan Structure**:
```
Improve ROIC from 8% to 12% through three initiatives

1. Margin expansion (8% → 10% = +250bps ROIC)
   1.1 Pricing: Implement 3% annual increases (adds 180bps)
   1.2 Procurement: Consolidate suppliers (saves 2% COGS = 70bps)
   
2. Capital efficiency (adds +150bps ROIC)
   2.1 Working capital: Reduce DSO from 60 to 45 days
   2.2 Fixed assets: Close underutilized facility (frees $50M)
   
3. Portfolio optimization (adds +100bps ROIC)
   3.1 Divest low-ROIC division (6% ROIC, $100M IC)
   3.2 Redeploy capital to high-ROIC segment (15% ROIC)

Total impact: +500bps ROIC improvement
Timeline: 18-24 months
Investment required: $10M (restructuring costs)
```

### Common Mistakes in Problem Structuring

#### Mistake 1: Jumping to Analysis Without Defining the Problem

**❌ Wrong Approach**:
```
Analyst: "I calculated a DCF valuation of $450M for Company X."

Executive: "Okay... should we buy them or not?"

Analyst: "Um, that depends on what you want to pay?"
```

**Problem**: Analysis didn't answer a decision-oriented question.

**✅ Correct Approach**:
```
Define problem first:
- Starting Point: We're evaluating acquisition of Company X
- Disturbing Event: They're asking $500M
- R2: We want to acquire if value >3x MOIC
- Question: Does Company X at $500M meet our return hurdle?

Then analyze:
- DCF fair value: $450M
- Required synergies at $500M: $75M (to hit 3x MOIC)
- Available synergies: $100M (revenue + cost)
- Conclusion: Yes, deal meets hurdle with $25M cushion
```

#### Mistake 2: Analyzing Wrong Components

**❌ Wrong Structure**:
```
Problem: "Why did revenue miss forecast by 15%?"

Analysis:
1. Economy slowed (GDP growth 2% vs. forecast 3%)
2. Competitor launched new product
3. Our sales team had high turnover
4. We reduced marketing spend
```

These are mixed (external vs. internal factors). Cannot properly prioritize.

**✅ Correct Structure**:
```
Revenue miss driven primarily by internal execution issues

1. Sales execution breakdown (-12% impact)
   1.1 Sales team turnover: 70% (lost key accounts)
   1.2 Marketing cuts: -30% spend (pipeline down 40%)
   
2. External factors secondary (-3% impact)
   2.1 Economy: GDP 2% vs. 3% (modest headwind)
   2.2 Competition: New product launch (affected 1 segment)
```

Now we know where to focus remediation efforts.

#### Mistake 3: Confusing Symptoms with Root Causes

**❌ Surface Analysis**:
```
Problem: Low profitability

Symptoms listed as causes:
1. Gross margins declining
2. Operating expenses high
3. Revenue growth slowing
```

These are symptoms, not root causes.

**✅ Root Cause Analysis**:
```
Low profitability driven by three root causes:

1. Pricing pressure (causing margin decline)
   └── Root cause: Undifferentiated product (commoditization)
   
2. Overhead bloat (causing high opex)
   └── Root cause: Acquired companies not integrated
   
3. Market share loss (causing slow growth)
   └── Root cause: Product quality issues (defect rate up 3x)
```

### Integration with Pyramid Writing

**Problem Definition Framework → SCQA Introduction**:
```
Starting Point      → Situation
Disturbing Event    → Complication  
R1 or R2            → (Implicit in Complication)
Question            → Question
```

**Analytical Framework → Pyramid Body**:
```
Framework components → Second-level pyramid groups
Detailed analysis    → Third-level supporting evidence
```

**Complete Example**:

**Problem Definition**:
```
Starting Point: Portfolio company has 8% EBITDA margin (peers: 15%)
Disturbing Event: New CEO tasked with margin improvement
R2: Want to achieve peer-level margins within 24 months
Question: What initiatives will close the 700bps gap?
```

**Pyramid Structure**:
```
We can achieve 15% EBITDA margin through three initiatives

1. Pricing optimization (+300bps margin)
   1.1 Implement value-based pricing (currently cost-plus)
   1.2 Reduce promotional discounting (20% → 10% of sales)
   1.3 Timeline: 12 months, Low execution risk
   
2. Procurement transformation (+250bps)
   2.1 Consolidate supplier base (1,200 → 400 suppliers)
   2.2 Implement global sourcing (currently local)
   2.3 Timeline: 18 months, Medium risk (implementation)
   
3. Footprint optimization (+150bps)
   3.1 Close 3 underutilized facilities (15% capacity)
   3.2 Consolidate into remaining 5 plants
   3.3 Timeline: 24 months, High risk (restructuring)

Total: +700bps margin improvement
Investment: $25M (restructuring costs)
Payback: 18 months
```

---

**Suggested filename**: `minto_p3_problem_solving.md`
