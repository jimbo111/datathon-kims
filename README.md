# Your Zip Code is Your Health Sentence

**Food Deserts, Geographic Inequality, and Chronic Disease in America**

Team 2Kim (Jimmy Kim + Alice Kim) | SBU AI Community Datathon 2026 | Healthcare & Wellness Track

---

## Research Question

How do food access, income, and race independently and jointly predict diabetes, obesity, and life expectancy across U.S. census tracts — and can we move beyond association toward causal identification?

## Key Findings

| Finding | Method | Result |
|---------|--------|--------|
| Food desert tracts have significantly higher diabetes | OLS + Logistic regression | OR = 1.96 (CDC benchmark), ecological OR = 5.45x |
| 44% of diabetes variance is geographic | Variance decomposition + ICC | Between-county ICC = 0.44 |
| 7.4-year life expectancy gap (richest vs poorest tracts) | Income quintile analysis | Q5 mean = 80.4 yr, Q1 = 73.0 yr |
| Race remains significant after controlling for income + food access | F-test for R² improvement | R² jumps from 0.50 to 0.60 (p < 0.001) |
| Propensity score matching confirms food desert effect | 1:1 nearest-neighbor PSM | ATT = +0.72 pp diabetes (95% CI: 0.29-1.18) |
| XGBoost agrees with OLS on top predictors | 5-fold CV, SHAP values | R² = 0.77, education + income dominate |
| Eliminating food deserts could prevent ~9,000 diabetes cases | Counterfactual simulation | Population-weighted, $86M/year in healthcare costs |
| All findings survive robustness checks | BH-FDR, LOSO CV, cluster bootstrap, definition sensitivity | 5/5 checks passed |

## Data Sources

Five federal datasets merged at the census-tract level (~27,000 tracts):

| Dataset | Source | Key Variables |
|---------|--------|---------------|
| USDA Food Access Research Atlas (2019) | ers.usda.gov | Food desert classification, low-access population shares |
| CDC PLACES (2025) | data.cdc.gov | Age-adjusted diabetes, obesity, high BP, depression prevalence |
| ACS 5-Year Estimates (2022) | api.census.gov | Income, poverty, race/ethnicity, education |
| CDC Life Expectancy (USALEEP) | cdc.gov | Life expectancy at birth by tract |
| HRSA HPSA (2026) | data.hrsa.gov | Primary care shortage designations |

## Analysis Pipeline

```
src/
  loaders/
    health_data.py        5 dataset downloaders + cleaners
    merge.py              FIPS merge -> master.parquet (27K tracts x 38 cols)
  analysis/
    health_stats.py       OLS/WLS, logistic OR, odds ratios, ICC, BH-FDR correction
    health_index.py       Health Disadvantage Index (equal + PCA weights)
    causal.py             Propensity score matching, regression discontinuity
    mediation.py          Baron & Kenny, Sobel test, bootstrap mediation
    ml_complement.py      XGBoost, SHAP, tract archetypes, counterfactual simulation
    robustness.py         FDR, pop-weighted OLS, definition sensitivity, LOSO CV
    spatial.py            Moran's I, LISA, spatial lag/error regression, GWR

notebooks/
    2kim_health_notebook.ipynb       Original datathon submission
    2kim_health_notebook_v2.ipynb    Extended analysis (15 sections, fully executed)

backend/                  FastAPI server (data + chart API endpoints)
frontend/                 React + Plotly interactive dashboard
submissions/              Presentation slides + chart images
```

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build master dataset (downloads ~100MB of federal data)
python -c "from src.loaders import build_master; build_master()"

# Run notebook
cd notebooks && jupyter notebook 2kim_health_notebook_v2.ipynb

# Dashboard
uvicorn backend.main:app --reload        # API on port 8000
cd frontend && npm install && npm run dev  # UI on port 5173
```

## Methods

- **Econometric**: OLS/WLS with HC1 robust SEs, logistic regression, partial correlation, variance decomposition, ICC (MixedLM), interaction models, Welch t-tests
- **Causal inference**: Propensity score matching (1:1 NN with caliper), regression discontinuity (triangular kernel WLS), DiD scaffold
- **Mediation**: Baron & Kenny 4-step, Sobel test, bootstrap CIs (1,000 iterations)
- **Machine learning**: XGBoost (5-fold CV), SHAP feature importance, k-means tract archetypes, counterfactual simulation
- **Robustness**: Benjamini-Hochberg FDR, population-weighted regressions, 4 food desert definitions, leave-one-state-out CV, county-cluster bootstrap
- **Spatial** (requires shapefiles): Moran's I, Local LISA, spatial lag/error models, GWR

## Limitations

1. **Ecological fallacy** — tract-level associations, not individual-level risk
2. **Correlation, not causation** — PSM and RD strengthen but do not eliminate confounding
3. **Census tract vintage mismatch** — USDA (2010) vs CDC PLACES (2020), ~85-95% FIPS match
4. **Simplified race classification** — 40% plurality threshold, White/Black/Hispanic only
5. **Temporal misalignment** — datasets span 2010-2025

## License

This project uses publicly available federal datasets. Analysis code is provided for educational and research purposes.
