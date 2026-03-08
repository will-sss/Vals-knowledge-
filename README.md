# Vals Knowledge Base

**Knowledge repository powering a financial valuation copilot agent.** This repo contains curated chapter summaries and source PDFs from five authoritative textbooks, purpose-built to give an AI agent deep, defensible expertise in valuation, accounting, statistical learning, Python-in-Excel implementation, and professional financial communication.

---

## What This Is

This is the knowledge base for a copilot agent designed to assist with **private company valuation** — specifically comparable company analysis, DCF modeling, multiples-based valuation, and the technical tooling (Python-in-Excel, Power Query, VBA) that supports these workflows in practice.

The agent's core use case: a private company has no GICS classification. The agent helps find public comparables using NLP-based similarity (TF-IDF + LSA on business descriptions), validates the selection using valuation theory, produces defensible outputs suitable for investment committees, and structures communication in clear, pyramid-based frameworks.

---

## Knowledge Architecture

The repo is organized around **five pillars** that cover the full stack of valuation work:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VALS KNOWLEDGE BASE                                   │
├──────────────┬──────────────┬──────────────┬───────────────┬─────────────────┤
│  VALUATION   │  ACCOUNTING  │  STATISTICAL │ IMPLEMENTATION│  COMMUNICATION  │
│  THEORY      │  STANDARDS   │  LEARNING    │  & TOOLING    │  & STRUCTURE    │
│              │              │              │               │                 │
│  Damodaran   │  Wiley IFRS  │  ISLP        │  Excel        │  Minto Pyramid  │
│  Investment  │  Interpre-   │  (James,     │  Revolution   │  Principle      │
│  Valuation   │  tation &    │   Witten,    │  (Strauss &   │  (Barbara Minto)│
│  (2023)      │  Application │   Hastie,    │   Van Der     │                 │
│              │  (2023)      │   Tibshirani)│   Post, 2024) │                 │
│  McKinsey    │              │  (2023)      │               │                 │
│  Valuation   │              │              │               │                 │
│  (7th, 2020) │              │              │               │                 │
│              │              │              │               │                 │
│  Benninga    │              │              │               │                 │
│  Financial   │              │              │               │                 │
│  Modeling    │              │              │               │                 │
│  (5th, 2022) │              │              │               │                 │
│              │              │              │               │                 │
│  Penman FSA  │              │              │               │                 │
│  (5th, 2013) │              │              │               │                 │
└──────────────┴──────────────┴──────────────┴───────────────┴─────────────────┘
```

### Why These Five?

| Pillar | Purpose | When the Agent Uses It |
|--------|---------|------------------------|
| **Valuation Theory** (Damodaran + McKinsey + Benninga + Penman) | First-principles reasoning about *what* to value and *how* | Selecting appropriate multiples, building DCFs, adjusting for private company discounts, terminal value estimation, cost of capital, portfolio optimization |
| **Accounting Standards** (Wiley IFRS) | Understanding *what the numbers mean* across jurisdictions | Normalizing financials for comparability — lease adjustments (IFRS 16), revenue recognition (IFRS 15), impairment, consolidation, fair value measurement |
| **Statistical Learning** (ISLP) | Mathematical foundations for the similarity engine and future ML features | TF-IDF/LSA theory, regression for multiple prediction, clustering for peer discovery, model selection, cross-validation |
| **Implementation** (Excel Revolution) | Practical execution in the user's environment | Python-in-Excel patterns, VBA automation, Power Query ETL, pandas workflows, formula engineering |
| **Communication** (Minto Pyramid Principle) | Structuring analysis and presenting findings clearly | Investment memos, valuation reports, board presentations, problem definition, logical argumentation |

---

## File Inventory

### Source PDFs (6)

Full textbook PDFs for deep reference when chapter summaries need expansion:

| File | Source |
|------|--------|
| `Aswath Damodaran - Investment Valuation, University Edition (2023).pdf` | Damodaran, NYU Stern |
| `McKinsey & Company Inc. - Valuation, 7th University Edition (2020).pdf` | Koller, Goedhart, Wessels |
| `Simon Benninga & Tal Mofkadi - Financial Modeling, Fifth Edition (2022).pdf` | MIT Press |
| `Penman - Financial Statement Analysis (5th Edition, 2013).pdf` | McGraw-Hill |
| `[Wiley] PKF International - Wiley 2023 IFRS Interpretation and Application (2023).pdf` | PKF International |
| `Gareth James et al. - An Introduction to Statistical Learning with Python (2023).pdf` | James, Witten, Hastie, Tibshirani |
| `Strauss, Van Der Post - Excel Revolution: Python with VBA in Excel (2024).pdf` | Reactive Publishing |
| `Barbara Minto - The Minto Pyramid Principle (2010).pdf` | Financial Times Prentice Hall |

### Chapter Summaries (95+ markdown files)

Pre-extracted, structured summaries optimized for agent retrieval. Naming convention: `{source_prefix}_ch{number}_{topic_slug}.md`

#### NEW: Minto Pyramid Principle — Professional Communication (4 files)

| File | Topic | Application |
|------|-------|-------------|
| `minto_p1_pyramid_structure.md` | SCQA framework, vertical/horizontal logic, document structure | Investment memos, valuation reports, board papers |
| `minto_p2_logical_thinking.md` | Time/structure/degree ordering, summarizing groups, inductive leaps | Organizing due diligence findings, financial analysis |
| `minto_p3_problem_solving.md` | Problem definition framework, analytical structures, root cause analysis | Defining valuation scope, diagnosing margin issues, M&A analysis |
| `minto_p4_presentation.md` | Slide design, storyboarding, prose structure, visual hierarchy | Investment committee presentations, client deliverables |

**Why Minto matters for valuation work**: Clear communication is as critical as correct numbers. The Minto framework ensures:
- Investment recommendations are structured persuasively (SCQA: Situation-Complication-Question-Answer)
- Complex analyses are presented logically (pyramid structure with supporting evidence)
- Valuation reports answer questions as they arise in the reader's mind
- Presentation slides tell a coherent story from titles alone

#### Damodaran — Investment Valuation (18 chapters)

*[Previous Damodaran content remains unchanged]*

#### McKinsey — Valuation (14 chapters)

*[Previous McKinsey content remains unchanged]*

#### Benninga — Financial Modeling (23 chapters)

| File | Topic | Valuation Relevance |
|------|-------|---------------------|
| `benninga_financial_modeling_ch2_corporate_valuation_overview.md` | Valuation framework, FCF models | Foundation for DCF |
| `benninga_financial_modeling_ch3_wacc.md` | WACC calculation, capital structure | Discount rate estimation |
| `benninga_financial_modeling_ch4_pro_forma_dcf_valuation.md` | Building pro forma models, DCF mechanics | Practical DCF implementation |
| `benninga_financial_modeling_ch6_financial_analysis_of_leasing.md` | Lease vs. buy analysis | IFRS 16 context |
| `benninga_financial_modeling_ch7_bond_duration.md` | Duration, convexity, immunization | Fixed income valuation |
| `benninga_financial_modeling_ch8_modeling_term_structure.md` | Yield curves, forward rates | Interest rate modeling |
| `benninga_financial_modeling_ch9_default_adjusted_expected_bond_returns.md` | Credit spreads, default probability | Credit analysis |
| `benninga_financial_modeling_ch10_portfolio_models_introduction.md` | Mean-variance optimization | Portfolio construction |
| `benninga_financial_modeling_ch11_efficient_portfolios_and_efficient_frontier.md` | Efficient frontier, Sharpe ratio | Portfolio optimization |
| `benninga_financial_modeling_ch12_variance_covariance_matrix.md` | Covariance estimation, correlation | Risk modeling |
| `benninga_financial_modeling_ch13_estimating_betas_and_security_market_line.md` | Beta calculation, SML | CAPM implementation |
| `benninga_financial_modeling_ch14_event_studies.md` | Abnormal returns, event windows | Market impact analysis |
| `benninga_financial_modeling_ch15_black_litterman_portfolio_optimization.md` | Black-Litterman model | Incorporating views |
| `benninga_financial_modeling_ch16_introduction_to_options.md` | Option basics, payoffs | Derivatives foundation |
| `benninga_financial_modeling_ch17_binomial_option_pricing_model.md` | Binomial trees, risk-neutral valuation | Option pricing |
| `benninga_financial_modeling_ch18_black_scholes_model.md` | Black-Scholes formula, assumptions | Option valuation |
| `benninga_financial_modeling_ch19_option_greeks.md` | Delta, gamma, vega, theta | Risk management |
| `benninga_financial_modeling_ch20_real_options.md` | Real options valuation, flexibility value | Strategic valuation |
| `benninga_financial_modeling_ch21_generating_and_using_random_numbers.md` | Random number generation, distributions | Simulation setup |
| `benninga_financial_modeling_ch22_intro_to_monte_carlo_methods.md` | Monte Carlo basics, convergence | Probabilistic valuation |
| `benninga_financial_modeling_ch24_monte_carlo_simulations_for_investments.md` | Investment analysis via simulation | Risk assessment |
| `benninga_financial_modeling_ch25_value_at_risk_var.md` | VaR calculation, backtesting | Risk quantification |
| `benninga_financial_modeling_ch26_replicating_options_and_option_strategies.md` | Synthetic positions, hedging | Derivatives strategies |
| `benninga_financial_modeling_ch27_monte_carlo_option_pricing.md` | Simulating option prices | Complex derivatives |
| Plus: Ch 28-31 on Excel techniques (data tables, matrices, functions, arrays) |

#### Penman — Financial Statement Analysis (7 chapters)

| File | Topic | Valuation Relevance |
|------|-------|---------------------|
| `penman_fsa_ch10_analysis_balance_sheet_income_statement.md` | Financial statement structure, quality of earnings | Understanding reported numbers |
| `penman_fsa_ch11_analysis_cash_flow_statement.md` | Cash flow analysis, accruals | FCFF derivation |
| `penman_fsa_ch12_analysis_profitability.md` | ROIC, ROE decomposition, DuPont analysis | Performance assessment |
| `penman_fsa_ch13_growth_and_sustainable_earnings.md` | Sustainable growth, earnings quality | Forecasting framework |
| `penman_fsa_ch14_value_of_operations_and_enterprise_multiples.md` | Enterprise value, operating multiples | Valuation multiples |

#### Wiley IFRS — Interpretation and Application (22 chapters)

*[Previous IFRS content remains unchanged]*

#### ISLP — Introduction to Statistical Learning with Python (8 chapters)

*[Previous ISLP content remains unchanged]*

#### Excel Revolution — Python with VBA in Excel (11 chapters)

*[Previous Excel Revolution content remains unchanged]*

---

## How the Agent Should Use This

### Retrieval Strategy

The chapter summaries are structured for **semantic search retrieval**. When a user asks a question, the agent should:

1. **Identify the pillar(s)** — Is this a valuation theory question? An accounting treatment question? A technical implementation question? A communication/structure question?
2. **Retrieve relevant chapters** — Use file names and topic descriptions to pull the right summaries
3. **Cross-reference across pillars** — A question about "how to structure an investment memo recommending an acquisition" touches Damodaran (ch25), McKinsey (ch31), and Minto (p1, p3)
4. **Fall back to PDFs** — If the summary doesn't have enough detail, reference the full PDF

### Example Query Routing

| User Question | Primary Source | Supporting Sources |
|---------------|----------------|-------------------|
| "How do I calculate WACC for a private company?" | Damodaran ch8, ch24; Benninga ch3 | McKinsey ch15 |
| "Should I adjust for IFRS 16 leases in my comps?" | Wiley IFRS ch22 | McKinsey ch11, ch18 |
| "How do I structure an investment memo?" | Minto p1 (SCQA), p4 (presentation) | Minto p2 (logical ordering) |
| "Help me define this valuation problem before starting analysis" | Minto p3 (problem definition) | Damodaran ch2 (approaches) |
| "Why is LSA better than TF-IDF alone?" | ISLP ch12 | ISLP ch2 (bias-variance) |
| "How do I read Excel named ranges in Python?" | Excel Revolution ch6 | ch5 (setup) |
| "What multiple should I use for a money-losing SaaS company?" | Damodaran ch20, ch22 | McKinsey ch24; Minto p2 (organizing findings) |
| "Create a storyboard for my M&A board presentation" | Minto p4 (slide design) | Damodaran ch25; McKinsey ch31 |
| "How to handle negative EBITDA in peer selection?" | Damodaran ch22 | McKinsey ch18, ch12 |
| "Build a Monte Carlo simulation for this investment" | Benninga ch22, ch24 | Damodaran ch33 |

---

## Companion Tools

This knowledge base powers a copilot agent that works alongside:

* **Company Similarity Engine (v9.3+)** — Python-in-Excel tool using TF-IDF (bigrams) + LSA to find comparable public companies from business descriptions. 5-cell architecture: Utilities → Engine → Orchestrator → Loader → Execution, using a dict registry pattern injected via `builtins._SIM_ENGINE`. Features 4-level GICS hierarchy boosting, region filtering, keyword boosting, and configurable matching strategies. Scores on 0–100 scale. See `SimilarityEngine_PythonGuide_v1.docx` for foundational documentation (note: guide covers v8 architecture; v9.3+ adds the Loader cell and dict registry pattern).
* **Excel Valuation Workbook** — The primary workspace where comparables are identified, financials normalized, and multiples applied.

---

## Contributing

To add new knowledge:

1. **Extract chapter summaries** as structured markdown files following the naming convention: `{source_prefix}_ch{number}_{topic_slug}.md`
2. **Include the source PDF** for full reference
3. **Update this README** with the new entries in the appropriate pillar table

Priority areas for expansion:

* US GAAP codification (ASC topics) for dual GAAP/IFRS coverage
* Sector-specific valuation guides (SaaS, healthcare, energy, financial institutions)
* Advanced NLP techniques (transformers, embeddings) for future similarity engine versions
* Power BI / DAX reference for dashboard builds
* Additional professional communication frameworks (SCQA variants, consulting frameworks)

---

*Last updated: March 2026*
