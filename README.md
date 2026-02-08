# Vals Knowledge Base

**Knowledge repository powering a financial valuation copilot agent.** This repo contains curated chapter summaries and source PDFs from six authoritative textbooks, purpose-built to give an AI agent deep, defensible expertise in valuation, accounting, financial modeling, financial statement analysis, statistical learning, and Python-in-Excel implementation.

---

## What This Is

This is the knowledge base for a copilot agent designed to assist with **private company valuation** — specifically comparable company analysis, DCF modeling, multiples-based valuation, and the technical tooling (Python-in-Excel, Power Query, VBA) that supports these workflows in practice.

The agent's core use case: a private company has no GICS classification. The agent helps find public comparables using NLP-based similarity (TF-IDF + LSA on business descriptions), validates the selection using valuation theory, normalizes financials using accounting and FSA expertise, and produces defensible outputs suitable for investment committees.

---

## Knowledge Architecture

The repo is organized around **six pillars** that cover the full stack of valuation work:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           VALS KNOWLEDGE BASE                                     │
├──────────────┬──────────────┬──────────────┬──────────────┬───────────┬───────────┤
│  VALUATION   │  FINANCIAL   │  FINANCIAL   │  ACCOUNTING  │ STATISTIC │ IMPLEMENT │
│  THEORY      │  MODELING    │  STATEMENT   │  STANDARDS   │ LEARNING  │ & TOOLING │
│              │              │  ANALYSIS    │              │           │           │
│ Damodaran    │ Benninga     │ Penman       │ Wiley IFRS   │ ISLP      │ Excel Rev │
│ (2023)       │ Financial    │ FSA & Sec.   │ (2023)       │ (2023)    │ (2024)    │
│              │ Modeling     │ Valuation    │              │           │           │
│ McKinsey     │ (5th, 2022)  │              │              │           │           │
│ (7th, 2020)  │              │              │              │           │           │
└──────────────┴──────────────┴──────────────┴──────────────┴───────────┴───────────┘
```

### Why These Six?

| Pillar | Purpose | When the Agent Uses It |
|--------|---------|----------------------|
| **Valuation Theory** (Damodaran + McKinsey) | First-principles reasoning about *what* to value and *how* | Selecting appropriate multiples, building DCFs, adjusting for private company discounts, terminal value estimation, cost of capital |
| **Financial Modeling** (Benninga) | Excel-based implementation of valuation, options, simulation, and portfolio models | Monte Carlo simulation for earn-outs/contingent consideration, option pricing for MIPs (binomial, Black-Scholes), WACC in Excel, sensitivity via data tables, VaR calculations, bond analysis |
| **Financial Statement Analysis** (Penman) | Bridging accounting → valuation: what drives multiples from the financials | Profitability decomposition (RNOA, ROCE), earnings quality and red flags, sustainable growth analysis, reformulating statements for valuation, enterprise multiples from fundamentals, credit risk analysis |
| **Accounting Standards** (Wiley IFRS) | Understanding *what the numbers mean* across jurisdictions | Normalizing financials for comparability — lease adjustments (IFRS 16), revenue recognition (IFRS 15), impairment, consolidation, fair value measurement |
| **Statistical Learning** (ISLP) | Mathematical foundations for the similarity engine and future ML features | TF-IDF/LSA theory, regression for multiple prediction, clustering for peer discovery, model selection, cross-validation |
| **Implementation** (Excel Revolution) | Practical execution in the user's environment | Python-in-Excel patterns, VBA automation, Power Query ETL, pandas workflows, formula engineering |

---

## File Inventory

### Source PDFs (7)

| File | Source |
|------|--------|
| `Aswath Damodaran - Investment Valuation, University Edition (2023).pdf` | Damodaran, NYU Stern |
| `McKinsey & Company Inc. - Valuation, 7th University Edition (2020).pdf` | Koller, Goedhart, Wessels |
| `Simon Benninga_Tal Mofkadi_ - Financial Modeling, Fifth Edition (2022, MIT Press).pdf` | Benninga & Mofkadi, MIT Press |
| `Stephen Penman - Financial Statement Analysis and Security Valuation.pdf` | Penman, Columbia Business School *(pending upload)* |
| `[Wiley] PKF International - Wiley 2023 IFRS Interpretation and Application (2023).pdf` | PKF International |
| `Gareth James et al. - An Introduction to Statistical Learning with Python (2023).pdf` | James, Witten, Hastie, Tibshirani |
| `Strauss, Van Der Post - Excel Revolution: Python with VBA in Excel (2024).pdf` | Reactive Publishing |

### Chapter Summaries (~112 markdown files)

Naming convention: `{source_prefix}_ch{number}_{topic_slug}.md`

---

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

---

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

---

#### Benninga — Financial Modeling, 5th Edition (27 chapters)

| File | Topic | Valuation Relevance |
|------|-------|-------------------|
| `benninga_financial_modeling_ch2_corporate_valuation_overview.md` | Valuation methods overview in Excel context | Framework for Excel-based valuation |
| `benninga_financial_modeling_ch3_wacc.md` | WACC calculation with Excel implementation | **Core** — building WACC models |
| `benninga_financial_modeling_ch4_pro_forma_dcf_valuation.md` | Pro forma financial statements, DCF in Excel | Building projection models |
| `benninga_financial_modeling_ch6_financial_analysis_of_leasing.md` | Lease vs. buy analysis, lease valuation | Lease adjustments in Excel |
| `benninga_financial_modeling_ch7_bond_duration.md` | Duration, convexity, immunization | Fixed income analysis |
| `benninga_financial_modeling_ch8_modeling_term_structure.md` | Yield curve construction, bootstrapping | Discount rate term structure |
| `benninga_financial_modeling_ch9_default_adjusted_expected_bond_returns.md` | Credit risk, default probabilities, recovery rates | **Restructuring** — creditor analysis |
| `benninga_financial_modeling_ch10_portfolio_models_introduction.md` | Mean-variance framework, diversification | Portfolio construction basics |
| `benninga_financial_modeling_ch11_efficient_portfolios_and_efficient_frontier.md` | Efficient frontier, Solver optimization | Portfolio optimization in Excel |
| `benninga_financial_modeling_ch12_variance_covariance_matrix.md` | Covariance estimation, correlation matrices | Risk model inputs |
| `benninga_financial_modeling_ch13_estimating_betas_and_security_market_line.md` | Beta regression, SML, CAPM in Excel | **Core** — cost of equity estimation |
| `benninga_financial_modeling_ch14_event_studies.md` | Abnormal returns, event windows, CAR | M&A announcement analysis |
| `benninga_financial_modeling_ch15_black_litterman_portfolio_optimization.md` | Black-Litterman model, view incorporation | Advanced portfolio optimization |
| `benninga_financial_modeling_ch16_introduction_to_options.md` | Option payoffs, put-call parity, basics | Foundation for MIP valuation |
| `benninga_financial_modeling_ch17_binomial_option_pricing_model.md` | Binomial trees, American options, early exercise | **Core** — MIP equity valuation |
| `benninga_financial_modeling_ch18_black_scholes_model.md` | Black-Scholes formula, implementation in Excel | **Core** — option-based valuation |
| `benninga_financial_modeling_ch19_option_greeks.md` | Delta, gamma, vega, theta, rho | Sensitivity of option-based valuations |
| `benninga_financial_modeling_ch20_real_options.md` | Option to expand/abandon/defer, staging | **Core** — valuing flexibility in projects |
| `benninga_financial_modeling_ch21_generating_and_using_random_numbers.md` | RNG in Excel, distributions, inverse transform | Foundation for Monte Carlo |
| `benninga_financial_modeling_ch22_intro_to_monte_carlo_methods.md` | Monte Carlo simulation principles and Excel setup | **Core** — earn-out and contingent consideration valuation |
| `benninga_financial_modeling_ch24_monte_carlo_simulations_for_investments.md` | Simulating portfolio returns, retirement planning | Investment scenario modeling |
| `benninga_financial_modeling_ch25_value_at_risk_var.md` | VaR methods: historical, parametric, Monte Carlo | Risk measurement |
| `benninga_financial_modeling_ch26_replicating_options_and_option_strategies.md` | Synthetic positions, strategy payoffs | Structuring contingent payoffs |
| `benninga_financial_modeling_ch27_monte_carlo_option_pricing.md` | MC option pricing, variance reduction, path-dependent | **Core** — complex option/MIP valuation |
| `benninga_financial_modeling_ch28_data_tables.md` | Excel data tables for sensitivity analysis | **Core** — scenario/sensitivity modeling |
| `benninga_financial_modeling_ch29_matrices.md` | Matrix operations in Excel (MMULT, MINVERSE) | Portfolio math, regression in Excel |
| `benninga_financial_modeling_ch30_excel_functions.md` | Comprehensive Excel function reference | Implementation reference |
| `benninga_financial_modeling_ch31_array_functions.md` | Dynamic arrays, LAMBDA, LET, MAP | Modern Excel patterns |

---

#### Penman — Financial Statement Analysis and Security Valuation (18 chapters)

| File | Topic | Valuation Relevance |
|------|-------|-------------------|
| `penman_fsa_ch3_how_financial_statements_are_used_in_valuation.md` | Connecting financials to valuation models | Foundation — why FSA matters for valuation |
| `penman_fsa_ch4_cash_vs_accrual_accounting_and_dcf.md` | Cash vs. accrual, why DCF alone is insufficient | Challenging cash-flow-only valuation |
| `penman_fsa_ch5_accrual_accounting_pricing_book_values.md` | Residual income, pricing book value | Book value multiples theory |
| `penman_fsa_ch6_accrual_accounting_pricing_earnings.md` | Residual earnings model, pricing P/E | Earnings multiples theory |
| `penman_fsa_ch7_valuation_and_active_investing.md` | Active investing framework, market efficiency | Applying valuation to investment decisions |
| `penman_fsa_ch8_viewing_business_through_financial_statements.md` | Reformulating statements: operating vs. financing | **Core** — separating operations from capital structure |
| `penman_fsa_ch9_analysis_statement_of_shareholders_equity.md` | Equity components, dirty surplus, comprehensive income | Equity bridge, dilution analysis |
| `penman_fsa_ch10_analysis_balance_sheet_income_statement.md` | Balance sheet and income statement analysis | **Core** — normalizing financials for comps |
| `penman_fsa_ch11_analysis_cash_flow_statement.md` | Cash flow analysis, free cash flow derivation | FCFF/FCFE calculation from reported statements |
| `penman_fsa_ch12_analysis_profitability.md` | RNOA, ROCE, DuPont decomposition, leverage effects | **Core** — what drives multiples across peer groups |
| `penman_fsa_ch13_growth_and_sustainable_earnings.md` | Sustainable vs. transitory earnings, growth decomposition | Forecasting normalized earnings for comps |
| `penman_fsa_ch14_value_of_operations_and_enterprise_multiples.md` | Enterprise P/B, enterprise P/E, RNOA-driven multiples | **Core** — connecting fundamentals to EV multiples |
| `penman_fsa_ch15_anchoring_on_financial_statements.md` | Simple forecasting from current financials | Quick valuation cross-checks |
| `penman_fsa_ch16_full_information_forecasting_and_strategy.md` | Full pro forma forecasting, business strategy analysis | Detailed DCF projection |
| `penman_fsa_ch17_creating_accounting_and_economic_value.md` | Value creation vs. value recording, accounting arbitrage | Identifying real vs. artificial earnings growth |
| `penman_fsa_ch18_quality_of_financial_statements.md` | Earnings quality, red flags, accounting manipulation | **Critical** — screening comps for accounting quality |
| `penman_fsa_ch19_analysis_of_equity_risk_and_return.md` | Fundamental risk analysis, growth-risk trade-off | Risk-adjusting multiples for peer comparison |
| `penman_fsa_ch20_analysis_of_credit_risk_and_return.md` | Credit analysis, default prediction, recovery | **Restructuring** — creditor-side analysis |

---

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

---

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

---

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

1. **Identify the pillar(s)** — valuation theory? accounting? modeling implementation? FSA?
2. **Retrieve relevant chapters** — use file names and topic descriptions
3. **Cross-reference across pillars** — most real questions span multiple pillars
4. **Fall back to PDFs** — if summaries lack sufficient detail

### Example Query Routing

| User Question | Primary Source | Supporting Sources |
|--------------|---------------|-------------------|
| "How do I calculate WACC for a private company?" | Damodaran ch8, ch24 | McKinsey ch15, Benninga ch3 |
| "Build a Monte Carlo for an earn-out" | Benninga ch22, ch21 | Damodaran ch33, Benninga ch28 |
| "What drives EV/EBITDA differences in my peer set?" | Penman ch12, ch14 | Damodaran ch18, McKinsey ch18 |
| "Price management equity under a MIP" | Benninga ch17, ch18 | Benninga ch27, ch19 |
| "Should I adjust for IFRS 16 leases in my comps?" | Wiley IFRS ch22 | McKinsey ch11, Benninga ch6 |
| "Earnings quality red flags in my comparables" | Penman ch18, ch13 | Wiley IFRS ch7 |
| "Credit analysis for a restructuring" | Penman ch20 | Benninga ch9, Damodaran ch30 |
| "Sensitivity table for my DCF" | Benninga ch28 | McKinsey ch17, Benninga ch4 |
| "How to decompose profitability for peer comparison?" | Penman ch12 | McKinsey ch12, Penman ch8 |
| "Why is LSA better than TF-IDF alone?" | ISLP ch12 | ISLP ch2 |

---

## Companion Tools

- **Company Similarity Engine (v9.2)** — Python-in-Excel tool using TF-IDF (bigrams) + LSA for comparable company discovery. 5-cell architecture with dict registry pattern via `builtins._SIM_ENGINE`. 4-level GICS boosting, scores on 0–100 scale.
- **Excel Valuation Workbook** — Primary workspace for comps, financials normalization, and multiples application.

---

## Pending Work

- **Penman ch3-9, ch15-20** — 13 chapters need markdown extraction (⏳ in table above)
- **Penman PDF** — upload to repo

---

*Last updated: February 2026*
