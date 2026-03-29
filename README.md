# Your Zip Code is Your Health Sentence

**Team 2Kim** (Jimmy + Alice) | SBU AI Community Datathon 2026 | Healthcare & Wellness Track

## Research Question

Does living in a food desert independently predict higher rates of diabetes and obesity, even after controlling for income, insurance status, and race?

## Key Findings (from 27,235 census tracts)

- Food desert tracts have **5.45x the odds** of above-median diabetes (95% CI: 4.90-6.06)
- **43.9%** of diabetes variance is between-county (geography matters)
- Life expectancy gap between best/worst HDI deciles: **~6 years**
- Race remains significant after controlling for income + food access (delta-R2=0.059)
- Health Disadvantage Index gaps: **8.6pp diabetes**, **12.2pp obesity**, **27.6pp poverty**

## Datasets

| Dataset | Source |
|---------|--------|
| USDA Food Access Research Atlas (2019) | ers.usda.gov |
| CDC PLACES (2025) | data.cdc.gov |
| ACS 5-Year Estimates (2022) | api.census.gov |
| CDC Life Expectancy (USALEEP) | ftp.cdc.gov |
| HRSA HPSA (2026) | data.hrsa.gov |

## Project Structure

```
notebooks/2kim_health_notebook.ipynb  -- Competition submission (7-section + MLA citations)
src/loaders/health_data.py            -- 5 dataset downloaders + cleaners
src/loaders/merge.py                  -- FIPS merge -> master.parquet
src/analysis/health_stats.py          -- Phase 2-4 statistical analysis
src/analysis/health_index.py          -- Health Disadvantage Index
backend/                              -- FastAPI demo server
frontend/                             -- React + Plotly dashboard
submissions/presentation_guide.md     -- Presentation outline + key stats
```

## Running

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build data (downloads ~100MB of federal data)
python -c "from src.loaders import build_master; build_master()"

# Run notebook
cd notebooks && jupyter notebook 2kim_health_notebook.ipynb

# Demo dashboard
uvicorn backend.main:app --reload  # port 8000
cd frontend && npm install && npm run dev  # port 5173
```
