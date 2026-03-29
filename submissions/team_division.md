# Team Division — Jimmy & Alice

**Project:** "Your Zip Code is Your Health Sentence"

Split by strength: **Jimmy owns data pipeline + stats**, **Alice owns visualizations + presentation**. Phases 1–2 run in parallel from day one.

---

## Jimmy — Data Engineering + Statistical Analysis

### Phase 1B — Data Pipeline (start immediately)
- [ ] Download all 5 datasets (USDA, CDC PLACES, ACS, CDC Life Expectancy, HRSA)
- [ ] Write loaders in `src/` to clean and standardize each dataset
- [ ] Merge all on FIPS census tract code → master dataframe
- [ ] Validate join quality: check for missing tracts, nulls, outliers
- [ ] Save cleaned master dataset to `data/processed/`

### Phase 2 — Food Access → Chronic Disease (stats)
- [ ] OLS regression: `diabetes_rate ~ food_access_score + median_income + pct_uninsured`
- [ ] Compute odds ratios: food desert vs. non-food-desert diabetes rates
- [ ] Confirm food access predicts diabetes independent of income
- [ ] Export regression table as JSON for frontend

### Phase 3 — The Zip Code Effect (stats)
- [ ] Variance decomposition: between-tract vs. within-tract diabetes variance
- [ ] Multivariate regression: `life_expectancy ~ income + food_access + race + education + pct_uninsured`
- [ ] Extract standardized betas — which variable dominates?
- [ ] Compute life expectancy gap (richest vs. poorest quartile tracts) per metro area

### Phase 4 — Race as a Residual Gap (stats)
- [ ] Logistic regression with interaction term: income × race → diabetes
- [ ] Residual analysis: is race still significant after controlling for income + food access?
- [ ] Build income quintile × race → diabetes prevalence matrix (feeds Alice's heatmap)

### Phase 5B — Health Disadvantage Index (synthesis)
- [ ] Build weighted composite index: food access + income + care access
- [ ] Rank all tracts, export top/bottom 10% list
- [ ] Compute summary stats table: life expectancy gap, diabetes gap, obesity gap

---

## Alice — Visualizations + Presentation

### Phase 1A — Geographic Foundation (start immediately, parallel with Jimmy)
- [ ] Side-by-side choropleth: food access score + diabetes rate (same geographic extent)
- [ ] Choropleth: food deserts overlaid on income quintiles
- [ ] Confirm visual clustering — food deserts are not random
- [ ] Wire up choropleth to backend API endpoint in `routes/charts.py`

### Phase 2 — Food Access Visuals
- [ ] Scatter with regression line: food access score vs. diabetes by tract, colored by majority race
- [ ] Scatter: food access score vs. obesity rate
- [ ] Add to Dashboard component

### Phase 3 — Zip Code Effect Visuals
- [ ] Bar chart: life expectancy by income quintile, split by race within each quintile
- [ ] Highlight the within-city gap visually (annotate worst vs. best tract in same metro)

### Phase 4 — Race Gap Visuals
- [ ] 2x2 heatmap: income × race → diabetes prevalence (Jimmy feeds the matrix)
- [ ] Annotate: high-income Black vs. low-income white comparison callout

### Phase 5A — Synthesis Visuals + Presentation
- [ ] Map the Health Disadvantage Index (Jimmy feeds the ranked data)
- [ ] Top 10 worst tracts table in dashboard
- [ ] Path diagram: poverty → food desert → obesity → diabetes (with coefficients)
- [ ] Build presentation narrative slides
- [ ] Write the opening hook and "so what" policy recommendation section

---

## Shared Checkpoints

| Checkpoint | What to sync on |
|------------|----------------|
| After Phase 1 | Confirm master dataframe schema — Alice needs column names for chart queries |
| After Phase 2 | Agree on key stats to call out (odds ratio number, R² value) |
| After Phase 4 | Review race gap finding together — sensitive framing needs alignment |
| Final day | End-to-end run of dashboard + dry run of presentation narrative |

---

## File Ownership

| Area | Owner |
|------|-------|
| `src/loaders.py` | Jimmy |
| `src/stats.py` | Jimmy |
| `backend/services/data_service.py` | Jimmy |
| `backend/routes/data.py` | Jimmy |
| `backend/services/chart_service.py` | Alice |
| `backend/routes/charts.py` | Alice |
| `frontend/src/components/` | Alice |
| `notebooks/01_eda.ipynb` | Jimmy |
| `notebooks/02_analysis.ipynb` | Jimmy |
| `submissions/` | Alice |

---

## Key Dates Reminder
- Work in parallel on Phase 1 — no blockers between Jimmy's pipeline and Alice's map scaffold
- Jimmy's regression outputs (Phases 2–4) feed Alice's visuals — communicate column names early
- Phase 5 is joint — build the index and map it together as a final integration step
