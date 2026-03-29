# Presentation Guide — Team 2Kim

**Title:** Your Zip Code is Your Health Sentence
**Track:** Healthcare & Wellness
**One-liner:** Food access, income, and race cluster geographically to create zones of compounding health disadvantage — diabetes and obesity are systemic outcomes, not personal failings.

---

## Slide Flow (suggested 8-10 minutes)

### 1. Hook (30 sec)
> "Two census tracts, 10 miles apart in the same city. One has a life expectancy of 82 years. The other: 67. Same hospitals, same weather, same state laws. The difference? One is a food desert. The other isn't."

- Show the side-by-side choropleth: food access score vs diabetes rate
- The visual clustering is immediate — no statistics needed yet

### 2. Research Question (30 sec)
**Does living in a food desert independently predict higher diabetes and obesity, even after controlling for income, insurance, and race?**

Three sub-questions:
1. How much health variance does geography explain?
2. What is the life expectancy gap between rich and poor tracts?
3. Does race still matter after controlling for income + food access?

### 3. Data (1 min)
Five federal datasets merged on census tract FIPS:

| Dataset | What it gives us |
|---------|-----------------|
| USDA Food Atlas | Food desert classification (binary + continuous) |
| CDC PLACES | Diabetes, obesity, BP, depression prevalence |
| ACS Census | Income, poverty, race, education |
| CDC USALEEP | Life expectancy at birth |
| HRSA HPSA | Primary care shortage designations |

**Key number to say:** "~72,000 census tracts across the US, merged from 5 independent federal sources."

### 4. Key Finding 1: Food Deserts → Disease (2 min)
**What to show:** Food access vs diabetes scatter plot (with OLS trend line)

**What to say:**
- Food desert tracts have **X.X% higher** mean diabetes than non-food-desert tracts (t-test p < 0.001)
- Ecological odds ratio: food desert tracts have **X.Xx** the odds of above-median diabetes (95% CI above 1)
- **This holds after controlling for poverty rate and uninsured %** (partial correlation significant)
- VIF check confirms no multicollinearity concern

### 5. Key Finding 2: The Zip Code Effect (1.5 min)
**What to show:** Life expectancy by income quintile bar chart

**What to say:**
- Between-county variance accounts for **X%** of total diabetes variance — geography matters
- Life expectancy gap between richest and poorest quintile tracts: **X.X years**
- Standardized betas show **[income/education]** is the dominant predictor of life expectancy
- This is years of life, not percentage points — the stakes are mortality

### 6. Key Finding 3: Race Persists (1.5 min)
**What to show:** Income × Race diabetes heatmap

**What to say:**
- Adding race (% Black) to the model after income + food access **still significantly improves R²**
- High-income majority-Black tracts: **X.X%** diabetes
- Low-income majority-White tracts: **X.X%** diabetes
- → Income does NOT fully close the racial gap
- This points to structural factors beyond economics (historical redlining, environmental racism)

**Framing note:** Present this as systemic/structural, not biological. The data shows *where* people live matters even at the same income level.

### 7. Health Disadvantage Index (1 min)
**What to show:** HDI choropleth map + worst tracts table

**What to say:**
- Equal-weighted composite: food access + poverty + healthcare access
- Tracts in the worst 10% have **X% higher diabetes**, **X% higher obesity**, and **X fewer years** of life expectancy than the best 10%
- These are identifiable, specific places where intervention is most needed

### 8. The Pathway (30 sec)
**What to show:** Path diagram (poverty → food desert → obesity → diabetes)

**What to say:**
- Each link is a statistically significant bivariate association
- Poverty predicts food desert status (β=X), food deserts predict obesity (β=X), obesity predicts diabetes (β=X)
- The direct poverty→diabetes path is stronger than any single indirect path — these factors compound
- *Note: this is exploratory, not formal mediation*

### 9. Limitations (1 min)
Be upfront — judges reward honesty:

1. **Ecological fallacy** — tract-level data, not individual-level
2. **Correlation ≠ causation** — observational, cross-sectional study
3. **Census tract mismatch** — USDA (2010) vs PLACES (2020) tracts, ~85-95% match
4. **Spatial autocorrelation** — neighboring tracts aren't independent, p-values are anti-conservative
5. **No population weighting** — each tract counted equally regardless of size
6. **Simplified race classification** — 40% plurality threshold, only White/Black/Hispanic + Other

### 10. So What? (30 sec)
**Policy recommendation:**
> "If diabetes and obesity are driven by where people live — by food access, income concentration, and structural racism — then interventions should target places, not just people. Bring grocery stores to food deserts. Expand Medicaid in shortage areas. Fund community health centers in the worst-HDI tracts."

End with: "We identified the tracts. The data is public. The question is whether we act on it."

---

## Numbers to Fill In

After running the pipeline, fill in these from the JSON outputs:

| Stat | Source file | Key |
|------|------------|-----|
| Diabetes mean difference (desert vs not) | `phase2_food_access_disease.json` | `ttest_diabetes.difference` |
| Odds ratio + CI | `phase2_food_access_disease.json` | `odds_ratio_diabetes.odds_ratio` |
| OLS R² (diabetes model) | `phase2_food_access_disease.json` | `ols_diabetes.r_squared` |
| Between-county variance % | `phase3_zip_code_effect.json` | `variance_decomposition.pct_between` |
| Life expectancy gap (Q5-Q1) | `phase3_zip_code_effect.json` | `life_expectancy_gap.gap_years` |
| Dominant standardized beta | `phase3_zip_code_effect.json` | `standardized_betas` (largest absolute value) |
| R² with vs without race | `phase4_race_residual_gap.json` | `residual_analysis.r2_with_race` |
| High-income Black diabetes % | `phase4_race_residual_gap.json` | `cross_comparison.high_income_black_mean_diabetes` |
| Low-income White diabetes % | `phase4_race_residual_gap.json` | `cross_comparison.low_income_white_mean_diabetes` |
| HDI decile gaps | `phase5_health_index.json` | `decile_gaps.*` |
| Path coefficients | `phase5_health_index.json` | `path_diagram.*` |

---

## Dashboard Demo Order (if showing live)

1. **Maps tab** → side-by-side food access + diabetes choropleth (the "wow" moment)
2. **Correlations tab** → scatter with trend line (the statistical proof)
3. **Race Gap tab** → heatmap (the uncomfortable truth)
4. **Synthesis tab** → HDI map → path diagram → worst tracts table (the call to action)

---

## Q&A Prep

**"Isn't this just correlation?"**
→ Yes, we say so in limitations. But the partial correlation (food access → diabetes after controlling for income) and the R² improvement from adding race suggest these aren't mere proxies.

**"Why not individual-level data?"**
→ NHANES has individual data but no geographic identifiers. Our approach identifies *which places* need intervention — that's the policy-relevant unit.

**"Why state-level choropleths if your data is tract-level?"**
→ 72K tracts don't render on a single map. The notebook has tract-level statistics; the dashboard aggregates for visual communication. Both are complementary.

**"How did you handle the 2010 vs 2020 tract mismatch?"**
→ Direct FIPS join, ~85-95% match rate. We document the coverage in the merge quality report. A crosswalk table would improve this.
