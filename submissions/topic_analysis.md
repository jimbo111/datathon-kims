# "Your Zip Code is Your Health Sentence"
### Combined Analysis: Food Deserts + Geographic Health Destiny

**Core thesis:** Where you live — not just how you live — determines your health outcomes. Food access, income, and race cluster geographically and compound into measurable chronic disease and mortality gaps.

---

## Dataset Stack

| Dataset | What it gives you | Source |
|---------|-------------------|--------|
| USDA Food Access Research Atlas | Food desert classification by census tract | ers.usda.gov |
| CDC PLACES | Chronic disease rates (diabetes, obesity, hypertension) by census tract | cdc.gov/places |
| ACS 5-Year Estimates | Income, race, education, insurance by zip/tract | census.gov |
| CDC Life Expectancy at Birth | Life expectancy by census tract | CDC NVSS |
| HRSA UDS Mapper | Primary care access by geography | hrsa.gov |

All datasets join on **FIPS county or census tract code**.

---

## Analysis Phases

### Phase 1 — Geographic Foundation
**Goal:** Show the clustering visually before any numbers.

- Choropleth: food desert tracts overlaid on income quintiles
- Key finding: food deserts cluster in low-income, majority-minority tracts — not random
- Moran's I spatial autocorrelation: confirm food access is spatially clustered
- Sets up the "it's systemic, not individual" argument

### Phase 2 — Food Access → Chronic Disease
**Goal:** Quantify the health penalty of living in a food desert.

- Scatter: food access score (continuous) vs. diabetes prevalence by tract
- Scatter: food access score vs. obesity rate by tract
- OLS regression: `diabetes_rate ~ food_access_score + median_income + pct_uninsured`
- Key result: food access predicts diabetes even after controlling for income
- Expected odds ratio: deepest food deserts have 1.4–1.8x higher diabetes rates

### Phase 3 — The Zip Code Effect
**Goal:** Show zip code explains more than individual behavior.

- Variance decomposition: between-tract vs. within-tract variance in diabetes rates
- Life expectancy gap: richest vs. poorest quartile tracts within the same metro area
- Show 10–20 year life expectancy differences within a single city
- Multivariate regression: `life_expectancy ~ income + food_access + race + education + pct_uninsured`
- Standardized betas — surface which variable has the largest coefficient

### Phase 4 — Race as a Residual Gap
**Goal:** Show that income doesn't fully explain racial health disparities — place does.

- 2x2 heatmap: income quintile × race → diabetes prevalence
- Compare high-income Black tracts vs. low-income white tracts — does income close the gap?
- Residual analysis: after controlling for income + food access, is race still significant?
- Connects zip code inequality to historical redlining

### Phase 5 — Compounding Story (Synthesis)
**Goal:** Show all factors stacking into a single narrative.

- Health Disadvantage Index: weighted composite of food access + income + care access
- Rank and map every tract
- Top 10% most disadvantaged vs. least — life expectancy gap, diabetes gap, obesity gap in one table
- Path diagram: poverty → food desert → poor diet → obesity → diabetes → early death

---

## Key Outcomes to Highlight

| Finding | Why judges care |
|---------|----------------|
| Zip code explains X% of variance in diabetes rates | Concrete, quotable stat |
| Life expectancy gap of 15–20 years within same city | Visceral inequality |
| Food desert residents have 1.5x diabetes odds at same income level | Separates food access from poverty |
| Racial health gap persists above median income | Systemic, not just economic |
| Top 50 most disadvantaged tracts identified | Actionable, specific |

---

## Visualizations to Build

1. **Side-by-side choropleth** — food access + diabetes rate (same geographic extent)
2. **Scatter with regression line** — food access score vs. diabetes by tract, colored by majority race
3. **Bar chart** — life expectancy by income quintile, split by race within each quintile
4. **Heatmap** — income × race → diabetes prevalence (2D matrix)
5. **Path diagram** — poverty → food desert → obesity → diabetes (with coefficients on each arrow)
6. **Top 10 worst tracts table** — county, state, food access score, diabetes rate, life expectancy

---

## Presentation Narrative

> "We're told obesity and diabetes are lifestyle diseases — personal choices. But when your zip code determines whether a grocery store exists within walking distance, whether you can afford fresh food, and whether a doctor is nearby, the choice was made before you were born. We quantify that sentence."

---

## Why This Wins

- **Visual:** choropleth maps are immediately compelling to any audience
- **Provocative thesis:** reframes individual health as systemic failure
- **Layered:** each phase adds a new dimension — geography, disease, race, compounding
- **Actionable:** ends with a ranked list of tracts most in need of intervention
- **Data-rich:** five public datasets, all joinable on FIPS code
