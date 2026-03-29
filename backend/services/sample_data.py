"""Sample state-level health data for choropleth development.

Generates realistic-looking data matching the schema Jimmy's pipeline will produce.
Replace with real data once Jimmy delivers the merged dataset.
"""

import pandas as pd
import numpy as np

# All 50 US states + DC
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}


def generate_health_data(seed: int = 42) -> pd.DataFrame:
    """Generate sample state-level health data for map prototyping.

    Columns mirror what Jimmy's merged dataset will produce:
    - state, state_name: state code and full name
    - food_access_score: 0-100 (higher = better access)
    - pct_food_desert: % of census tracts classified as food deserts
    - diabetes_rate: % of adults with diagnosed diabetes
    - obesity_rate: % of adults with BMI >= 30
    - median_income: median household income ($)
    - pct_uninsured: % without health insurance
    - pct_minority: % non-white population
    - life_expectancy: life expectancy at birth (years)
    - pcp_per_100k: primary care physicians per 100k residents
    """
    rng = np.random.default_rng(seed)
    n = len(STATES)

    # Base poverty/disadvantage factor per state (drives correlations)
    disadvantage = rng.normal(0, 1, n)

    # Southern states tend higher disadvantage (realistic pattern)
    south = {"AL", "AR", "GA", "KY", "LA", "MS", "MO", "NC", "OK", "SC", "TN", "TX", "WV"}
    for i, st in enumerate(STATES):
        if st in south:
            disadvantage[i] += 0.6

    # Northeast/West tend lower
    affluent = {"CT", "MA", "NH", "NJ", "NY", "VT", "CA", "CO", "WA", "MN", "HI", "MD", "DC"}
    for i, st in enumerate(STATES):
        if st in affluent:
            disadvantage[i] -= 0.5

    food_access = np.clip(70 - 12 * disadvantage + rng.normal(0, 5, n), 20, 95).round(1)
    pct_food_desert = np.clip(15 + 8 * disadvantage + rng.normal(0, 3, n), 2, 45).round(1)
    diabetes_rate = np.clip(9 + 3 * disadvantage + rng.normal(0, 1.2, n), 5, 18).round(1)
    obesity_rate = np.clip(30 + 5 * disadvantage + rng.normal(0, 2, n), 20, 45).round(1)
    median_income = np.clip(62000 - 10000 * disadvantage + rng.normal(0, 5000, n), 35000, 95000).astype(int)
    pct_uninsured = np.clip(10 + 4 * disadvantage + rng.normal(0, 2, n), 3, 22).round(1)
    pct_minority = np.clip(30 + 8 * disadvantage + rng.normal(0, 10, n), 5, 70).round(1)
    life_expectancy = np.clip(78.5 - 2 * disadvantage + rng.normal(0, 1, n), 72, 83).round(1)
    pcp_per_100k = np.clip(85 - 12 * disadvantage + rng.normal(0, 10, n), 40, 140).round(1)

    return pd.DataFrame({
        "state": STATES,
        "state_name": [STATE_NAMES[s] for s in STATES],
        "food_access_score": food_access,
        "pct_food_desert": pct_food_desert,
        "diabetes_rate": diabetes_rate,
        "obesity_rate": obesity_rate,
        "median_income": median_income,
        "pct_uninsured": pct_uninsured,
        "pct_minority": pct_minority,
        "life_expectancy": life_expectancy,
        "pcp_per_100k": pcp_per_100k,
    })
