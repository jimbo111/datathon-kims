# Combined TODO — Team 2Kim

**Project:** "Your Zip Code is Your Health Sentence"
**Branch:** `health-analysis` (merged from jimmy/ and alice/)

---

## Jimmy — Data Pipeline + Statistical Analysis

### Phase 1B — Data Pipeline
- [x] Download all 5 datasets (USDA, CDC PLACES, ACS, CDC Life Expectancy, HRSA)
- [x] Write loaders in `src/loaders/health_data.py`
- [x] Merge all on FIPS → master dataframe (`src/loaders/merge.py`)
- [x] Validate join quality, save to `data/processed/master.parquet`
- [x] Document schema (`data/processed/master_schema.csv`)

### Phase 2 — Food Access → Chronic Disease
- [x] OLS regressions (diabetes, obesity) + VIF check
- [x] Odds ratios + t-test
- [x] Partial correlation controlling for income
- [x] Export → `phase2_food_access_disease.json`

### Phase 3 — The Zip Code Effect
- [x] Variance decomposition (weighted between-county)
- [x] Life expectancy regression with standardized betas
- [x] Life expectancy gap by income quintile
- [x] Export → `phase3_zip_code_effect.json`

### Phase 4 — Race as a Residual Gap
- [x] Income quintile × race → diabetes matrix
- [x] Interaction model (continuous pct_black)
- [x] Residual analysis + cross-comparison
- [x] Export → `phase4_race_residual_gap.json`

### Phase 5B — Health Disadvantage Index
- [x] Equal-weighted composite index (no circularity)
- [x] Rank all tracts, export top/bottom 10%
- [x] Path diagram (bivariate associations)
- [x] Export → `phase5_health_index.json` + `health_disadvantage_index.parquet`

---

## Alice — Visualizations + Presentation

### Phase 1A — Geographic Foundation
- [x] Side-by-side choropleth: food access + diabetes rate
- [x] Choropleth: food deserts overlaid on income quintiles
- [x] Wire up choropleth to backend API
- [x] Create Chart.jsx + HealthDashboard.jsx

### Phase 2 — Food Access Visuals
- [x] Scatter with regression line: food access vs diabetes
- [x] Scatter: food access vs obesity rate
- [x] Add chart endpoints in chart_service.py

### Phase 3 — Zip Code Effect Visuals
- [x] Bar chart: life expectancy by income quintile
- [x] Add to Dashboard layout

### Phase 4 — Race Gap Visuals
- [ ] 2x2 heatmap: income quintile × race → diabetes prevalence
- [ ] Annotate: high-income Black vs. low-income white comparison

### Phase 5A — Synthesis Visuals
- [ ] Map the Health Disadvantage Index
- [ ] Top 10 worst tracts table
- [ ] Path diagram: poverty → food desert → obesity → diabetes
- [ ] Presentation slides (manual)
