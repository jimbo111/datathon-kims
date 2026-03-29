"""Phase 5B: Health Disadvantage Index.

Builds a weighted composite index from regression coefficients,
ranks all census tracts, and exports summary stats + path diagram data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rich.console import Console

console = Console()
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_health_disadvantage_index(
    master: pd.DataFrame,
    phase2_results: dict | None = None,
    phase3_results: dict | None = None,
) -> dict:
    """Compute the Health Disadvantage Index (HDI) for every census tract.

    The index is a descriptive composite of:
    - Food access (is_food_desert, pct_low_access_1mi)
    - Income (poverty_rate)
    - Healthcare access (uninsured_pct, hpsa_shortage)

    Equal weights are used to avoid circularity (deriving weights from the same
    data used to validate the index would guarantee correlation with outcomes).
    """
    console.rule("[bold]Phase 5B: Health Disadvantage Index")
    results = {}

    # ── Define index components ──
    components = {
        "food_access": ["is_food_desert", "pct_low_access_1mi"],
        "income": ["poverty_rate"],
        "healthcare_access": ["uninsured_pct"],
    }

    # Add HPSA if available
    if "hpsa_shortage" in master.columns and master["hpsa_shortage"].notna().sum() > 100:
        components["healthcare_access"].append("hpsa_shortage")

    # Equal weights to avoid circularity (see docstring)
    weights = {k: 1.0 / len(components) for k in components}
    results["index_weights"] = weights
    results["weight_justification"] = "Equal weights used to avoid circularity with outcome variables"

    # ── Compute standardized components ──
    df = master.copy()
    component_scores = {}

    for group, cols in components.items():
        available = [c for c in cols if c in df.columns and df[c].notna().sum() > 100]
        if not available:
            continue

        # Standardize each column (higher = worse)
        z_scores = pd.DataFrame()
        for col in available:
            series = pd.to_numeric(df[col], errors="coerce")
            z = (series - series.mean()) / series.std()
            z_scores[col] = z

        # Average z-scores for this component
        component_scores[group] = z_scores.mean(axis=1)

    if not component_scores:
        console.print("[red]Not enough data to build HDI[/]")
        return results

    # ── Build composite index ──
    score_df = pd.DataFrame(component_scores)
    for group in score_df.columns:
        score_df[group] *= weights.get(group, 1.0)

    df["hdi_score"] = score_df.mean(axis=1)

    # Higher HDI = more disadvantaged
    df["hdi_percentile"] = df["hdi_score"].rank(pct=True) * 100
    df["hdi_decile"] = pd.qcut(
        df["hdi_score"].rank(method="first"), 10, labels=range(1, 11)
    ).astype("Int64")

    # ── Rank tracts ──
    ranked = df.dropna(subset=["hdi_score"]).sort_values("hdi_score", ascending=False)

    top_10pct = ranked.head(int(len(ranked) * 0.10))
    bottom_10pct = ranked.tail(int(len(ranked) * 0.10))

    top_cols = ["tract_fips", "state", "county", "hdi_score", "hdi_percentile"]
    top_cols = [c for c in top_cols if c in ranked.columns]
    if "diabetes_pct" in ranked.columns:
        top_cols.append("diabetes_pct")
    if "life_expectancy" in ranked.columns:
        top_cols.append("life_expectancy")

    results["top_10pct_worst"] = top_10pct[top_cols].head(50).to_dict(orient="records")
    results["bottom_10pct_best"] = bottom_10pct[top_cols].head(50).to_dict(orient="records")
    results["n_tracts_scored"] = int(ranked.shape[0])

    # ── Summary stats: gaps between top and bottom deciles ──
    gap_metrics = {}
    for col in ["diabetes_pct", "obesity_pct", "life_expectancy", "poverty_rate"]:
        if col in df.columns:
            top_mean = top_10pct[col].mean()
            bot_mean = bottom_10pct[col].mean()
            if pd.notna(top_mean) and pd.notna(bot_mean):
                gap_metrics[col] = {
                    "most_disadvantaged": round(top_mean, 2),
                    "least_disadvantaged": round(bot_mean, 2),
                    "gap": round(top_mean - bot_mean, 2),
                }

    results["decile_gaps"] = gap_metrics
    if gap_metrics:
        console.print("[cyan]Gaps (most vs least disadvantaged decile):[/]")
        for k, v in gap_metrics.items():
            console.print(f"  {k}: {v['most_disadvantaged']} vs {v['least_disadvantaged']} (gap: {v['gap']})")

    # ── Path diagram coefficients: poverty → food desert → obesity → diabetes ──
    path_results = _compute_path_coefficients(df)
    results["path_diagram"] = path_results

    # ── Export ──
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Save HDI as parquet
    hdi_cols = ["tract_fips", "hdi_score", "hdi_percentile", "hdi_decile"]
    hdi_cols = [c for c in hdi_cols if c in df.columns]
    for extra in ["state", "county", "diabetes_pct", "obesity_pct", "life_expectancy"]:
        if extra in df.columns:
            hdi_cols.append(extra)
    df[hdi_cols].dropna(subset=["hdi_score"]).to_parquet(
        DATA_PROCESSED / "health_disadvantage_index.parquet", index=False
    )

    # Save JSON results
    with open(DATA_PROCESSED / "phase5_health_index.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]HDI computed for {results['n_tracts_scored']:,} tracts[/]")
    console.print("[green]Exported → data/processed/health_disadvantage_index.parquet[/]")
    console.print("[green]Exported → data/processed/phase5_health_index.json[/]")

    return results


def _compute_path_coefficients(df: pd.DataFrame) -> dict:
    """Bivariate associations along a hypothesized pathway.

    NOTE: These are independent bivariate regressions, NOT a formal mediation
    analysis. Coefficients represent total bivariate associations and cannot be
    multiplied to estimate indirect effects (that would double-count shared variance).
    A formal mediation analysis would require controlling for upstream variables
    at each stage (e.g., Baron & Kenny or bootstrap-based mediation).

    Associations estimated:
    1. poverty ↔ food_desert
    2. food_desert ↔ obesity
    3. obesity ↔ diabetes
    4. poverty ↔ diabetes (total association)
    """
    paths = {}

    # Path 1: poverty_rate → is_food_desert
    cols = ["is_food_desert", "poverty_rate"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("is_food_desert ~ poverty_rate", data=clean).fit()
        paths["poverty_to_food_desert"] = {
            "coef": round(m.params.get("poverty_rate", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    # Path 2: is_food_desert → obesity_pct
    cols = ["obesity_pct", "is_food_desert"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("obesity_pct ~ is_food_desert", data=clean).fit()
        paths["food_desert_to_obesity"] = {
            "coef": round(m.params.get("is_food_desert", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    # Path 3: obesity_pct → diabetes_pct
    cols = ["diabetes_pct", "obesity_pct"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("diabetes_pct ~ obesity_pct", data=clean).fit()
        paths["obesity_to_diabetes"] = {
            "coef": round(m.params.get("obesity_pct", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    # Direct path: poverty → diabetes (total effect)
    cols = ["diabetes_pct", "poverty_rate"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("diabetes_pct ~ poverty_rate", data=clean).fit()
        paths["poverty_to_diabetes_direct"] = {
            "coef": round(m.params.get("poverty_rate", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    if paths:
        console.print("[cyan]Path diagram coefficients:[/]")
        for path, vals in paths.items():
            console.print(f"  {path}: β={vals['coef']}, R²={vals['r_squared']}")

    return paths
