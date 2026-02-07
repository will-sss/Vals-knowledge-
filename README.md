# Vals Knowledge Base

**Knowledge repository powering a financial valuation copilot agent.** This repo contains curated chapter summaries and source PDFs from four authoritative textbooks, purpose-built to give an AI agent deep, defensible expertise in valuation, accounting, statistical learning, and Python-in-Excel implementation.

---

## What This Is

This is the knowledge base for a copilot agent designed to assist with **private company valuation** — specifically comparable company analysis, DCF modeling, multiples-based valuation, and the technical tooling (Python-in-Excel, Power Query, VBA) that supports these workflows in practice.

The agent's core use case: a private company has no GICS classification. The agent helps find public comparables using NLP-based similarity (TF-IDF + LSA on business descriptions), validates the selection using valuation theory, and produces defensible outputs suitable for investment committees.

---

## Knowledge Architecture

The repo is organized around **four pillars** that cover the full stack of valuation work:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VALS KNOWLEDGE BASE                              │
├─────────────────┬─────────────────┬──────────────┬─────────────────┤
│   VALUATION     │   ACCOUNTING    │  STATISTICAL │  IMPLEMENTATION │
│   THEORY        │   STANDARDS     │  LEARNING    │  & TOOLING      │
│                 │                 │              │                 │
│  Damodaran      │  Wiley IFRS     │  ISLP        │  Excel          │
│  Investment     │  Interpretation │  (James,     │  Revolution     │
│  Valuation      │  & Application  │   Witten,    │  (Strauss &     │
│  (2023)         │  (2023)         │   Hastie,    │   Van Der Post) │
│                 │                 │   Tibshirani)│                 │
│  McKinsey       │                 │  (2023)      │                 │
│  Valuation      │                 │              │                 │
│  (7th Ed, 2020) │                 │              │                 │
└─────────────────┴─────────────────┴──────────────┴─────────────────┘
```

### Why These Four?

| Pillar | Purpose | When the Agent Uses It |
|--------|---------|----------------------|
| **Valuation Theory** (Damodaran + McKinsey) | First-principles reasoning about *what* to value and *how* | Selecting appropriate multiples, building DCFs, adjusting for private company discounts, terminal value estimation, cost of capital |
| **Accounting Standards** (Wiley IFRS) | Understanding *what the numbers mean* across jurisdictions | Normalizing financials for comparability — lease adjustments (IFRS 16), revenue recognition (IFRS 15), impairment, consolidation, fair value measurement |
| **Statistical Learning** (ISLP) | Mathematical foundations for the similarity engine and future ML features | TF-IDF/LSA theory, regression for multiple prediction, clustering for peer discovery, model selection, cross-validation |
| **Implementation** (Excel Revolution) | Practical execution in the user's environment | Python-in-Excel patterns, VBA automation, Power Query ETL, pandas workflows, formula engineering |

---

## File Inventory

### Source PDFs (5)

Full textbook PDFs for deep reference when chapter summaries need expansion:

| File | Source |
|------|--------|
| `Aswath Damodaran - Investment Valuation, University Edition (2023).pdf` | Damodaran, NYU Stern |
| `McKinsey & Company Inc. - Valuation, 7th University Edition (2020).pdf` | Koller, Goedhart, Wessels |
| `[Wiley] PKF International - Wiley 2023 IFRS Interpretation and Application (2023).pdf` | PKF International |
| `Gareth James et al. - An Introduction to Statistical Learning with Python (2023).pdf` | James, Witten, Hastie, Tibshirani |
| `Strauss, Van Der Post - Excel Revolution: Python with VBA in Excel (2024).pdf` | Reactive Publishing |

### Chapter Summaries (67 markdown files)

Pre-extracted, structured summaries optimized for agent retrieval. Naming convention: `{source_prefix}_ch{number}_{topic_slug}.md`

#### Damodaran — Investment Valuation (18 chapters)

| File | Topic | Valuation Relevance |
|------|-------|-------------------|
| `damodaran_ch2_approaches_to_valuation.md` | DCF, relative valuation, option pricing frameworks | Foundation — when to use each approach |
| `damodaran_ch3_understanding_financial_statements.md` | Reading financials, accounting principles | Data quality — understanding inputs |
| `damodaran_ch4_the_basics_of_risk.md` | Risk models, CAPM, factor models | Cost of equity estimation |
| `damodaran_ch7_riskless_rates_and_risk_premiums.md` | Risk-free rate selection, ERP estimation | WACC inputs |
| `damodaran_ch8_estimating_risk_parameters_and_costs_of_financing.md` | Beta estimation, cost of debt, WACC | Discount rate construction |
| `damodaran_ch9_measuring_earnings.md` | Normalizing earnings, operating vs. non-operating | Clean EBITDA / FCFF calculation |
| `damodaran_ch12_terminal_value.md` | Gordon growth, exit multiples, fade models | Often 60-80% of DCF value |
| `damodaran_ch13_narrative_and_numbers.md` | Connecting story to valuation inputs | Sanity-checking assumptions |
| `damodaran_ch14_equity_intrinsic_value_models.md` | DDM, FCFE models | Equity valuation approaches |
| `damodaran_ch15_firm_valuation_wacc_and_apv.md` | FCFF + WACC, APV method | Enterprise value estimation |
| `damodaran_ch16_estimating_equity_value_per_share.md` | Bridge from EV to equity, dilution | Final equity value |
| `damodaran_ch17_fundamental_principles_of_relative_valuation.md` | Why multiples work, consistency rules | Comparable company analysis theory |
| `damodaran_ch18_earnings_multiples.md` | P/E, EV/EBITDA, drivers of each | Most common multiples |
| `damodaran_ch19_book_value_multiples.md` | P/B, EV/IC, Tobin's Q | Capital-intensive industries |
| `damodaran_ch20_revenue_and_sector_specific_multiples.md` | EV/Revenue, EV/subscriber, price per unit | High-growth and loss-making companies |
| `damodaran_ch22_valuing_money_losing_firms.md` | Normalizing losses, survival probability | Pre-profit companies |
| `damodaran_ch23_valuing_young_or_startup_firms.md` | Revenue build-up, option value, staging | Early-stage valuation |
| `damodaran_ch24_valuing_private_firms.md` | Illiquidity discounts, key person risk, control premiums | **Core use case** |
| `damodaran_ch25_acquisitions_and_takeovers.md` | Synergy valuation, deal pricing | M&A context |
| `damodaran_ch26_valuing_real_estate.md` | Cap rates, NOI, property valuation | Sector-specific |
| `damodaran_ch30_valuing_equity_in_distressed_firms.md` | Distress probability, option-to-default | Special situations |
| `damodaran_ch31_value_enhancement_dcf_framework.md` | Value drivers, restructuring analysis | Advisory context |
| `damodaran_ch33_probabilistic_approaches_in_valuation.md` | Simulations, decision trees, scenario analysis | Uncertainty quantification |

#### McKinsey — Valuation (14 chapters)

| File | Topic | Valuation Relevance |
|------|-------|-------------------|
| `mckinsey_ch9_growth.md` | Revenue growth decomposition, organic vs. acquired | Forecasting top line |
| `mckinsey_ch10_frameworks_for_valuation.md` | Enterprise DCF, economic profit, APV | Choosing the right framework |
| `mckinsey_ch11_reorganizing_financial_statements.md` | NOPLAT, invested capital, ROIC | **Critical** — clean inputs for multiples |
| `mckinsey_ch12_analyzing_performance.md` | ROIC trees, margin analysis, capital efficiency | Understanding comparables' quality |
| `mckinsey_ch14_estimating_continuing_value.md` | Key value driver formula, convergence | Terminal value best practices |
| `mckinsey_ch15_estimating_cost_of_capital.md` | WACC mechanics, target capital structure | Discount rate |
| `mckinsey_ch16_moving_from_ev_to_value_per_share.md` | Non-operating assets, debt bridge, minority interests | EV → equity bridge |
| `mckinsey_ch17_analyzing_the_results.md` | Sensitivity analysis, scenario testing | Validating outputs |
| `mckinsey_ch18_using_multiples.md` | Peer group selection, multiple consistency | **Core use case** — comp analysis |
| `mckinsey_ch19_valuation_by_parts.md` | Sum-of-parts, conglomerate discount | Multi-segment companies |
| `mckinsey_ch20_taxes.md` | Marginal vs. effective rates, deferred tax | NOPLAT adjustments |
| `mckinsey_ch21_nonoperating_items_provisions_reserves.md` | Pensions, provisions, restructuring charges | Cleaning financials |
| `mckinsey_ch24_measuring_performance_capital_light_businesses.md` | SaaS metrics, intangible-heavy models | Tech/services comparables |
| `mckinsey_ch26_inflation.md` | Real vs. nominal, inflation adjustments | Cross-border comparisons |
| `mckinsey_ch31_mergers_and_acquisitions.md` | Synergy types, deal structure, value creation | M&A advisory |
| `mckinsey_ch32_divestitures.md` | Carve-out valuation, stranded costs | Transaction context |
| `mckinsey_ch33_capital_structure_dividends_share_repurchases.md` | Optimal leverage, payout policy | Capital structure assumptions |

#### Wiley IFRS — Interpretation and Application (22 chapters)

| File | Topic | Valuation Relevance |
|------|-------|-------------------|
| `wiley_ifrs_ch3_presentation_financial_statements.md` | IAS 1 — structure and content | Reading international financials |
| `wiley_ifrs_ch4_statement_of_financial_position.md` | Balance sheet classification | Invested capital calculation |
| `wiley_ifrs_ch5_pnl_oci_changes_in_equity.md` | Income statement, OCI components | Understanding reported earnings |
| `wiley_ifrs_ch6_statement_of_cash_flows.md` | IAS 7 — operating, investing, financing | FCFF derivation from cash flow statement |
| `wiley_ifrs_ch7_accounting_policies_changes_estimates_errors.md` | IAS 8 — comparability across periods | Adjusting historical data |
| `wiley_ifrs_ch8_inventories.md` | IAS 2 — cost formulas, NRV | Working capital adjustments |
| `wiley_ifrs_ch9_property_plant_and_equipment.md` | IAS 16 — cost vs. revaluation model | Capital expenditure analysis |
| `wiley_ifrs_ch10_borrowing_costs.md` | IAS 23 — capitalization criteria | Adjusting reported interest |
| `wiley_ifrs_ch11_intangible_assets.md` | IAS 38 — recognition, amortization | R&D capitalization, goodwill |
| `wiley_ifrs_ch12_investment_property.md` | IAS 40 — fair value vs. cost model | Real estate comparables |
| `wiley_ifrs_ch13_impairment_and_assets_held_for_sale.md` | IAS 36, IFRS 5 — recoverable amount | Asset-based valuation floor |
| `wiley_ifrs_ch14_consolidations_joint_arrangements_associates.md` | IFRS 10, 11, IAS 28 — group accounting | Minority interests, equity method |
| `wiley_ifrs_ch15_business_combinations.md` | IFRS 3 — purchase price allocation | M&A accounting, goodwill |
| `wiley_ifrs_ch16_shareholders_equity.md` | Equity components, treasury shares | Diluted share count |
| `wiley_ifrs_ch17_share_based_payment.md` | IFRS 2 — SBC expense | Adjusting EBITDA for SBC |
| `wiley_ifrs_ch18_provisions_contingencies_events_after_reporting_period.md` | IAS 37, IAS 10 — provisions and contingencies | Hidden liabilities |
| `wiley_ifrs_ch19_employee_benefits.md` | IAS 19 — pensions, post-employment | Pension deficit in net debt |
| `wiley_ifrs_ch20_revenue_from_contracts_with_customers.md` | IFRS 15 — 5-step model | Revenue comparability across companies |
| `wiley_ifrs_ch21_government_grants.md` | IAS 20 — grant accounting | Adjusting operating income |
| `wiley_ifrs_ch22_leases.md` | IFRS 16 — right-of-use assets | **Critical** — lease-adjusted EBITDA, EV |
| `wiley_ifrs_ch23_foreign_currency.md` | IAS 21 — translation, transaction | FX normalization |
| `wiley_ifrs_ch24_financial_instruments.md` | IFRS 9 — classification, measurement, hedging | Debt valuation, derivatives |
| `wiley_ifrs_ch25_fair_value.md` | IFRS 13 — hierarchy, measurement | Fair value concepts |
| `wiley_ifrs_ch26_income_taxes.md` | IAS 12 — deferred tax | Tax normalization |
| `wiley_ifrs_ch27_earnings_per_share.md` | IAS 33 — basic and diluted EPS | Per-share metrics |
| `wiley_ifrs_ch28_operating_segments.md` | IFRS 8 — segment reporting | Sum-of-parts analysis |
| `wiley_ifrs_ch29_related_party_disclosures.md` | IAS 24 — related party transactions | Private company adjustments |

#### ISLP — Introduction to Statistical Learning with Python (8 chapters)

| File | Topic | Valuation Relevance |
|------|-------|-------------------|
| `islp_ch2_statistical_learning.md` | Bias-variance trade-off, model assessment | Foundation for ML in valuation |
| `islp_ch3_linear_regression.md` | OLS, multiple regression, diagnostics | Regression-based multiple prediction |
| `islp_ch4_classification.md` | Logistic regression, LDA, KNN | Distress prediction, industry classification |
| `islp_ch6_linear_model_selection_and_regularization.md` | Ridge, Lasso, elastic net, cross-validation | Feature selection for valuation models |
| `islp_ch8_tree_based_methods.md` | Decision trees, random forests, boosting | Non-linear multiple drivers |
| `islp_ch10_deep_learning.md` | Neural networks, CNNs, RNNs | Future capability (not in Python-in-Excel) |
| `islp_ch11_survival_analysis_and_censored_data.md` | Survival curves, hazard functions | Time-to-default, startup survival |
| `islp_ch12_unsupervised_learning.md` | PCA, K-means, hierarchical clustering | **Core** — peer group discovery, LSA theory |
| `islp_ch13_multiple_testing.md` | FWER, FDR, p-value adjustment | Statistical rigor in screening |

#### Excel Revolution — Python with VBA in Excel (9 chapters)

| File | Topic | Valuation Relevance |
|------|-------|-------------------|
| `excel_revolution_ch3_enhanced_formulas_and_functions.md` | Advanced Excel formulas | User's primary environment |
| `excel_revolution_ch4_analyzing_and_visualizing_data.md` | Charts, pivot tables, conditional formatting | Output presentation |
| `excel_revolution_ch5_introduction_to_python_in_excel.md` | Python-in-Excel setup and concepts | **Core** — similarity engine platform |
| `excel_revolution_ch6_py_function.md` | `=PY()` function, `xl()` interface | Reading/writing Excel data in Python |
| `excel_revolution_ch7_complex_excel_tasks_using_pandas.md` | DataFrame operations in Excel | Data transformation patterns |
| `excel_revolution_ch8_automating_excel_tasks_with_python.md` | Automation patterns | Batch processing, scheduled tasks |
| `excel_revolution_ch9_automation_with_macros_and_vba.md` | VBA macros, event handling | Refresh automation, chart generation |
| `excel_revolution_ch10_sophisticated_financial_equations.md` | Financial functions, modeling patterns | WACC, IRR, NPV implementation |
| `excel_revolution_ch11_financial_reporting.md` | Report generation, templates | Presentation-ready outputs |
| `excel_revolution_ch12_excel_and_external_data.md` | Power Query, external connections | Capital IQ data pipeline |
| `excel_revolution_ch13_boosting_efficiency_with_templates_and_add_ons.md` | Templates, add-ins, productivity | Workflow optimization |

---

## How the Agent Should Use This

### Retrieval Strategy

The chapter summaries are structured for **semantic search retrieval**. When a user asks a question, the agent should:

1. **Identify the pillar(s)** — Is this a valuation theory question? An accounting treatment question? A technical implementation question?
2. **Retrieve relevant chapters** — Use file names and topic descriptions to pull the right summaries
3. **Cross-reference across pillars** — A question about "how to adjust EBITDA for leases in a comp analysis" touches McKinsey (ch11, ch18), Wiley IFRS (ch22), and possibly the similarity engine implementation
4. **Fall back to PDFs** — If the summary doesn't have enough detail, reference the full PDF

### Example Query Routing

| User Question | Primary Source | Supporting Sources |
|--------------|---------------|-------------------|
| "How do I calculate WACC for a private company?" | Damodaran ch8, ch24 | McKinsey ch15 |
| "Should I adjust for IFRS 16 leases in my comps?" | Wiley IFRS ch22 | McKinsey ch11, ch18 |
| "Why is LSA better than TF-IDF alone?" | ISLP ch12 | ISLP ch2 (bias-variance) |
| "How do I read Excel named ranges in Python?" | Excel Revolution ch6 | ch5 (setup) |
| "What multiple should I use for a money-losing SaaS company?" | Damodaran ch20, ch22 | McKinsey ch24 |
| "How to handle negative EBITDA in peer selection?" | Damodaran ch22 | McKinsey ch18, ch12 |

---

## Companion Tools

This knowledge base powers a copilot agent that works alongside:

- **Company Similarity Engine (v8)** — Python-in-Excel tool using TF-IDF + LSA to find comparable public companies from business descriptions. See `SimilarityEngine_PythonGuide_v1.docx` for technical documentation.
- **Excel Valuation Workbook** — The primary workspace where comparables are identified, financials normalized, and multiples applied.

---

## Contributing

To add new knowledge:

1. **Extract chapter summaries** as structured markdown files following the naming convention: `{source_prefix}_ch{number}_{topic_slug}.md`
2. **Include the source PDF** for full reference
3. **Update this README** with the new entries in the appropriate pillar table

Priority areas for expansion:
- US GAAP codification (ASC topics) for dual GAAP/IFRS coverage
- Sector-specific valuation guides (SaaS, healthcare, energy, financial institutions)
- Advanced NLP techniques (transformers, embeddings) for future similarity engine versions
- Power BI / DAX reference for dashboard builds

---

*Last updated: February 2026*
