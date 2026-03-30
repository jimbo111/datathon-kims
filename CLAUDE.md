# CLAUDE.md — datathon-kims

## Project Overview

**"Your Zip Code is Your Health Sentence"** — SBU AI Community Datathon 2026, Healthcare & Wellness Track.
Team 2Kim (Jimmy + Alice). Won regional datathon; now upgrading for national competition.

Research question: How do food access, income, and race independently and jointly predict diabetes, obesity, and life expectancy across U.S. census tracts — and can we move beyond association toward causal identification?

## Architecture

### Data Pipeline (`src/loaders/`)
- `health_data.py` — Downloads + cleans 5 federal datasets (USDA Food Atlas, CDC PLACES, ACS, CDC USALEEP, HRSA HPSA). Multi-year USDA support (2015/2019 scaffold).
- `merge.py` — FIPS-based merge into `data/processed/master.parquet` (~27K census tracts). Includes `validate="m:1"` on all joins, row count assertions, `validate_master()` for impossible values, population-weighted coverage reporting.

### Core Analysis (`src/analysis/`) — Phases 2-5 (Regional)
- `health_stats.py` — Phase 2 (Food Access → Disease: OLS/WLS, logistic OR, ecological OR with sensitivity, Cohen's d/f², BH-FDR), Phase 3 (Zip Code Effect: variance decomposition, ICC, MixedLM, standardized + unstandardized betas, diagnostics), Phase 4 (Race Gap: interaction models for Black + Hispanic, F-test, within-quintile comparison, Welch t-test). All phases collect p-values for global BH-FDR correction.
- `health_index.py` — Phase 5: HDI with equal-weight + PCA-derived weights, Cronbach's alpha, bivariate association diagram with logistic regression for binary DV.

### National-Level Extensions (`src/analysis/`) — Phases 6a-6e
- `spatial.py` — Moran's I (global + local LISA), spatial lag/error regression (spreg), GWR. Handles W⟷y alignment via `w_subset`, removes island tracts.
- `causal.py` — Propensity score matching (1:1 NN, bootstrap ATT CIs, balance table), DiD scaffold (TWFE), regression discontinuity (triangular kernel WLS, McCrary density).
- `mediation.py` — Baron & Kenny 4-step, Sobel test, bootstrap mediation (1000 iter) for poverty→food_desert→obesity→diabetes chain.
- `ml_complement.py` — XGBoost 5-fold CV, SHAP analysis, OLS vs ML rank comparison, k-means tract archetypes, counterfactual simulation. NOTE: obesity_pct excluded from features to preserve ~22K sample size.
- `robustness.py` — BH-FDR, population-weighted OLS, 4 food desert definition sensitivity, leave-one-state-out CV, county-cluster bootstrap CIs.

### Backend (`backend/`)
- FastAPI server, CORS from `ALLOWED_ORIGINS` env var (defaults: localhost:5173, localhost:3000).
- Path traversal protection on `/data/load/{filename}`.
- Routes: `/api/data/*` and `/api/charts/*` (Plotly JSON).

### Frontend (`frontend/`)
- React 19 + Vite 8 + Plotly.js. Two views: Health Analysis (tabbed) and Data Explorer.
- Error handling on Dashboard load. Null guards on HealthSummaryTable.

### Notebooks (`notebooks/`)
- `2kim_health_notebook.ipynb` — Regional datathon submission (Phases 2-5).
- `2kim_national_notebook.ipynb` — National competition submission (Phases 2-5 + spatial, causal, mediation, ML, robustness).

## Key Data Conventions
- **Join key:** `tract_fips` (11-digit string, zero-padded).
- **Derived columns:** `is_food_desert`, `majority_race` (40% threshold), `income_quintile` (1-5).
- **Analysis exports:** `data/processed/phase{2,3,4,5}_*.json` (regional), `phase6{a-e}_*.json` (national).

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build master dataset
python -c "from src.loaders import build_master; build_master()"

# Run all analysis phases
python -c "
import pandas as pd
master = pd.read_parquet('data/processed/master.parquet')
from src.analysis.health_stats import run_all_phases
results = run_all_phases(master)
from src.analysis.health_index import build_health_disadvantage_index
build_health_disadvantage_index(master, results.get('phase2'), results.get('phase3'))

# National extensions
from src.analysis.causal import run_causal_analysis
from src.analysis.mediation import run_mediation_analysis
from src.analysis.ml_complement import run_ml_analysis
from src.analysis.robustness import run_robustness_checks
run_causal_analysis(master)
run_mediation_analysis(master)
run_ml_analysis(master, ols_results=results.get('phase2'))
run_robustness_checks(master, phase_results=results)
"

# Backend
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Environment Variables
- `OPENAI_API_KEY` — Optional, for AI-assisted features.
- `ALLOWED_ORIGINS` — Comma-separated CORS origins for backend.
- Census API key passed to `build_master(census_api_key=...)`.

## Tech Stack
- **Core:** pandas, numpy, statsmodels, scipy, rich
- **Spatial:** geopandas, libpysal, esda, spreg, mgwr
- **ML:** xgboost, shap, sklearn
- **Viz:** matplotlib, seaborn, plotly
- **Backend:** FastAPI, uvicorn, httpx
- **Frontend:** React 19, Vite 8, Plotly.js, Axios
