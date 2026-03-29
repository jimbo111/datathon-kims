# 10 Health Topics for Datathon — Why They Win + How to Analyze

---

## 1. Food Deserts + Chronic Disease

**Why it wins:** Geographic inequality is visual, emotional, and policy-relevant. Judges from any background get it instantly.

**Analysis:**
- USDA Food Access Atlas + CDC PLACES county-level health data
- Choropleth maps: food access score vs. diabetes/obesity rates
- Correlation: distance to grocery store → BMI/diabetes prevalence
- Regression controlling for income, race, urbanicity
- Key stat: odds ratio of diabetes in food deserts vs. non-food-deserts

---

## 2. Sleep Deprivation as a Keystone Health Driver

**Why it wins:** One variable connects obesity, mental health, diabetes, and productivity — richer story than any single-outcome analysis.

**Analysis:**
- NHANES dataset (CDC): sleep hours + BMI + mental health + glucose
- Show sleep < 6hrs as threshold where all outcomes worsen simultaneously
- Scatter matrix: sleep vs. each outcome, colored by age group
- Mediation analysis: does sleep explain the screentime → obesity link?

---

## 3. Teen Mental Health + Social Media (Post-COVID)

**Why it wins:** Timely, emotionally resonant, clear inflection point at 2020 makes the temporal story undeniable.

**Analysis:**
- CDC YRBSS (Youth Risk Behavior Survey) 2015–2023
- Interrupted time series at 2020 — did teen anxiety/depression spike beyond trend?
- Stratify by gender (girls hit harder — this contrast is striking)
- Overlay US social media adoption curves if data available

---

## 4. Zip Code as Health Destiny

**Why it wins:** Reframes individual health choices as systemic — a provocative thesis that stands out.

**Analysis:**
- ACS census data (income, education, race by zip) + CDC health outcomes
- Show zip code explains more variance in life expectancy than smoking status
- Regression: how much does moving up one income percentile change life expectancy?
- Map life expectancy gap between richest/poorest zip codes in the same city

---

## 5. Preventable ER Visits + Primary Care Deserts

**Why it wins:** Clear ROI argument — judges who care about cost will love it. Actionable policy recommendation is built in.

**Analysis:**
- CMS Hospital Compare + HRSA primary care shortage area data
- Counties with low PCP-to-population ratio → higher preventable hospitalization rates
- Cost estimate: preventable ER visit avg cost × volume = wasted dollars per county
- Choropleth overlay: PCP shortage areas vs. ER overuse rates

---

## 6. Obesity Trends by Generation (Boomers → Gen Z)

**Why it wins:** Generational framing is relatable and gives a built-in narrative arc across time.

**Analysis:**
- NHANES longitudinal data — obesity rates by birth cohort over time
- Each generation got heavier at the same age than the prior one (cohort effect)
- Decompose: diet change vs. activity change vs. sleep change contribution per generation
- Animated line chart of obesity rates by age group across decades

---

## 7. Mental Health Treatment Gap by State

**Why it wins:** Inequality angle with strong geographic variation — easy to visualize and argue for policy action.

**Analysis:**
- SAMHSA NSDUH: % with mental illness vs. % receiving treatment, by state
- Treatment gap = prevalence − treated — rank states
- Correlate gap with: insurance coverage rates, psychiatrist density, stigma proxies
- Scatter: treatment gap vs. suicide rate — is untreated mental illness lethal?

---

## 8. Antibiotic Resistance + Prescription Overprescription

**Why it wins:** Existential public health risk + clear villain (overprescription). Sophisticated topic that signals depth.

**Analysis:**
- CDC antibiotic prescribing data by state + resistance trend data
- States with highest prescription rates → correlate with resistant infection rates (lagged 2–3 years)
- Time series: resistance rates climbing as prescriptions stayed flat — show the dissociation point
- Compare US prescription rates to peer countries (US is an outlier)

---

## 9. Income × Race × Diabetes — Intersectional Analysis

**Why it wins:** Intersectionality lens is sophisticated; shows compounding disadvantage quantitatively, not just rhetorically.

**Analysis:**
- NHANES or CDC BRFSS: diabetes diagnosis by income bracket AND race
- 2x2 heatmap: income quintile × race/ethnicity → diabetes prevalence
- Show that high-income Black Americans still have higher rates than low-income white Americans (residual racial gap)
- Logistic regression with interaction term: income × race

---

## 10. Screentime + Obesity — Done Right

**Why it wins:** Common topic elevated by dissecting mechanism and type of screen — shows analytical sophistication.

**Analysis:**
- NHANES + ATUS (American Time Use Survey): screen type × physical activity × BMI
- Split passive screen (TV) vs. active screen (computer work) — different BMI profiles
- Mediation model: screen → sleep disruption → obesity (not just direct effect)
- Show confounders: income controls for much of the association — is screen time a proxy for poverty?
- Age stratification: effect is much stronger in children than adults

---

## Top 3 Picks for Winning

| Rank | Topic | Why |
|------|-------|-----|
| 1 | **Zip Code as Health Destiny** | Provocative, visual, clear thesis |
| 2 | **Sleep as Keystone Variable** | Multi-outcome richness, NHANES data is clean |
| 3 | **Teen Mental Health + Social Media** | Timely, emotional, strong 2020 breakpoint |

---

## Combined Recommendation: Topics 1 + 4

See `topic_analysis.md` for the full breakdown of the combined "Your Zip Code is Your Health Sentence" analysis.
