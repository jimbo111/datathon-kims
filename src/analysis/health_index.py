"""Phase 5B: Health Disadvantage Index.

Builds a weighted composite index from regression coefficients,
ranks all census tracts, and exports summary stats + bivariate association diagram data.

Weighting strategies:
- Equal weights (default, avoids circularity with outcome variables)
- PCA-derived weights (data-driven, maximizes variance explained by index components)

A Cronbach's alpha reliability check is included for the composite index.
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

    Two weighting strategies are computed and compared:

    1. Equal weights — avoids circularity (deriving weights from the same data
       used to validate the index would guarantee correlation with outcomes).

    2. PCA-derived weights — first principal component loadings from the
       standardized component scores. These are data-driven but may overfit to
       the particular dataset and introduce circularity if validated against
       outcomes included in the PCA.

    A Cronbach's alpha is reported for the component scores to assess composite
    index reliability (internal consistency).
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
    weights_equal = {k: 1.0 / len(components) for k in components}
    results["index_weights"] = weights_equal
    results["weight_justification"] = "Equal weights used to avoid circularity with outcome variables"

    # ── Compute standardized components ──
    df = master.copy()
    component_scores: dict[str, pd.Series] = {}

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

    score_df = pd.DataFrame(component_scores).dropna()

    # ── Cronbach's alpha for composite reliability ──
    alpha_result = _cronbach_alpha(score_df)
    results["cronbach_alpha"] = alpha_result
    alpha_display = (
        f"{alpha_result['alpha']:.3f}"
        if not (isinstance(alpha_result["alpha"], float) and np.isnan(alpha_result["alpha"]))
        else "n/a"
    )
    console.print(
        f"  Cronbach's alpha: {alpha_display} "
        f"({alpha_result['interpretation']}) — {len(score_df.columns)} components"
    )

    # ── PCA-derived weights ──
    pca_result = _pca_weights(score_df)
    results["pca_weights"] = pca_result

    # ── Build composite index — Equal weights ──
    score_df_eq = score_df.copy()
    for group in score_df_eq.columns:
        score_df_eq[group] *= weights_equal.get(group, 1.0)
    hdi_equal = score_df_eq.mean(axis=1)

    # ── Build composite index — PCA weights ──
    pca_weights_map = pca_result.get("component_weights", {})
    if pca_weights_map:
        score_df_pca = score_df.copy()
        for group in score_df_pca.columns:
            score_df_pca[group] *= pca_weights_map.get(group, weights_equal.get(group, 1.0))
        hdi_pca = score_df_pca.sum(axis=1)
        # Normalize so scores are on same scale as equal-weight version
        hdi_pca = (hdi_pca - hdi_pca.mean()) / hdi_pca.std()
    else:
        hdi_pca = hdi_equal.copy()

    # Attach equal-weight HDI to df (primary index — matches original behaviour)
    df.loc[score_df.index, "hdi_score"] = hdi_equal
    df.loc[score_df.index, "hdi_score_pca"] = hdi_pca

    # Correlation between the two weighting approaches
    corr_eq_pca = float(hdi_equal.corr(hdi_pca))
    results["weighting_comparison"] = {
        "correlation_equal_vs_pca": round(corr_eq_pca, 4),
        "note": (
            "High correlation (>0.95) suggests weighting choice has minimal impact on tract rankings. "
            "Equal-weight HDI is used as the primary index."
        ),
    }
    console.print(f"  Equal vs PCA weight correlation: r={corr_eq_pca:.4f}")

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

    # ── Bivariate association diagram: poverty → food desert → obesity → diabetes ──
    # NOTE: These are bivariate associations, NOT path coefficients from a structural
    # equation model. Renaming to make this distinction explicit in the output.
    path_results = _compute_path_coefficients(df)
    path_results["note"] = (
        "These are bivariate associations, not path coefficients from a structural "
        "equation model. Coefficients represent marginal associations and cannot be "
        "multiplied to estimate mediated or indirect effects."
    )
    results["bivariate_association_diagram"] = path_results
    # Backward-compatible alias kept for frontend rendering
    results["path_diagram"] = path_results

    # ── Export ──
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Save HDI as parquet
    hdi_cols = ["tract_fips", "hdi_score", "hdi_score_pca", "hdi_percentile", "hdi_decile"]
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


def _cronbach_alpha(score_df: pd.DataFrame) -> dict:
    """Compute Cronbach's alpha for a DataFrame of component scores.

    Cronbach's alpha measures internal consistency (reliability) of the composite
    index: how well the components measure the same underlying construct.

    Formula: alpha = (k / (k-1)) * (1 - sum(Var_i) / Var_total)
    where k = number of components, Var_i = variance of component i,
    Var_total = variance of the sum across components.

    Interpretation:
        alpha >= 0.9: Excellent
        alpha >= 0.8: Good
        alpha >= 0.7: Acceptable
        alpha >= 0.6: Questionable
        alpha < 0.6: Poor
    """
    clean = score_df.dropna()
    k = clean.shape[1]
    if k < 2:
        return {"alpha": float("nan"), "n_components": k, "interpretation": "insufficient components"}

    item_variances = clean.var(ddof=1).sum()
    total_variance = clean.sum(axis=1).var(ddof=1)

    if total_variance == 0:
        return {"alpha": float("nan"), "n_components": k, "interpretation": "zero total variance"}

    alpha = (k / (k - 1)) * (1 - item_variances / total_variance)
    alpha = float(np.clip(alpha, -1.0, 1.0))  # alpha can be negative (indicates negative covariance)

    if alpha >= 0.9:
        interp = "excellent"
    elif alpha >= 0.8:
        interp = "good"
    elif alpha >= 0.7:
        interp = "acceptable"
    elif alpha >= 0.6:
        interp = "questionable"
    else:
        interp = "poor"

    return {
        "alpha": round(alpha, 4),
        "n_components": k,
        "interpretation": interp,
        "note": (
            "Cronbach's alpha for the HDI component scores. "
            "Measures internal consistency — how well components co-vary as a single construct. "
            "Low alpha may indicate components measure distinct dimensions (not necessarily a flaw)."
        ),
    }


def _pca_weights(score_df: pd.DataFrame) -> dict:
    """Derive weights from the first principal component of component scores.

    Uses numpy SVD on the correlation matrix (standardized components) to extract
    PC1 loadings. The squared loadings (proportional to explained variance per
    component) are normalized to sum to 1.0 for use as weights.

    Returns a dict with loadings, explained variance ratio, and normalized weights.
    """
    clean = score_df.dropna()
    k = clean.shape[1]

    if k < 2 or len(clean) < k + 1:
        return {"note": "insufficient data for PCA", "component_weights": {}}

    # Standardize (z-score) each component before PCA
    centered = clean - clean.mean()
    std_devs = clean.std(ddof=1)
    std_devs = std_devs.replace(0, np.nan)
    standardized = centered.div(std_devs).dropna(axis=1)

    if standardized.shape[1] < 2:
        return {"note": "insufficient non-zero-variance components for PCA", "component_weights": {}}

    # SVD on the data matrix (equivalent to PCA on correlation matrix)
    X = standardized.values
    n = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Explained variance per component
    explained_var = (s ** 2) / (n - 1)
    total_var = explained_var.sum()
    explained_ratio = explained_var / total_var if total_var > 0 else explained_var * 0

    # PC1 loadings (first row of Vt)
    pc1_loadings = Vt[0]
    # Ensure positive orientation (majority positive loadings = higher HDI = worse)
    if np.sum(pc1_loadings > 0) < np.sum(pc1_loadings < 0):
        pc1_loadings = -pc1_loadings

    # Normalize squared loadings to use as weights (sum to 1)
    squared_loadings = pc1_loadings ** 2
    normalized_weights = squared_loadings / squared_loadings.sum()

    component_names = standardized.columns.tolist()
    weights_map = {
        component_names[i]: round(float(normalized_weights[i]), 4)
        for i in range(len(component_names))
    }
    loadings_map = {
        component_names[i]: round(float(pc1_loadings[i]), 4)
        for i in range(len(component_names))
    }

    pc1_var_ratio = float(explained_ratio[0])
    console.print(
        f"  PCA: PC1 explains {pc1_var_ratio*100:.1f}% of component variance. "
        f"Weights: {weights_map}"
    )

    return {
        "pc1_explained_variance_ratio": round(pc1_var_ratio, 4),
        "pc1_loadings": loadings_map,
        "component_weights": weights_map,
        "note": (
            "PCA-derived weights from PC1 loadings on standardized component scores. "
            "High PC1 variance ratio indicates components co-vary strongly (good for composite). "
            "Caveat: PCA weights can introduce circularity if health outcomes are correlated "
            "with these components — use equal weights for confirmatory analysis."
        ),
    }


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
    # is_food_desert is binary — logistic regression is the correct model.
    # We report both the logistic coefficient (log-odds) and a linear probability
    # model (OLS) for comparison, with a warning that OLS on a binary DV can
    # produce predicted probabilities outside [0, 1].
    cols = ["is_food_desert", "poverty_rate"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        # ── Linear probability model (OLS on binary DV) ──
        m_ols = smf.ols("is_food_desert ~ poverty_rate", data=clean).fit()

        # ── Logistic regression (correct model for binary DV) ──
        logit_result = None
        if clean["is_food_desert"].nunique() == 2:
            try:
                m_logit = smf.logit("is_food_desert ~ poverty_rate", data=clean).fit(disp=False)
                logit_coef = m_logit.params.get("poverty_rate", np.nan)
                logit_or = np.exp(logit_coef)
                logit_p = m_logit.pvalues.get("poverty_rate", np.nan)
                logit_result = {
                    "log_odds_coef": round(float(logit_coef), 4),
                    "odds_ratio": round(float(logit_or), 4),
                    "p_value": float(f"{logit_p:.2e}"),
                    "pseudo_r_squared": round(float(m_logit.prsquared), 4),
                    "n_obs": int(m_logit.nobs),
                }
            except Exception as exc:
                console.print(f"  [yellow]Logistic (poverty→food_desert) failed: {exc}[/]")

        paths["poverty_to_food_desert"] = {
            "ols_coef": round(m_ols.params.get("poverty_rate", 0), 4),
            "ols_r_squared": round(m_ols.rsquared, 4),
            "ols_note": (
                "OLS on binary DV = linear probability model. "
                "Provided for comparability only; may predict probabilities outside [0, 1]."
            ),
            "logistic": logit_result,
            "preferred_model": "logistic",
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
        console.print("[cyan]Bivariate association diagram (not path coefficients — see docstring):[/]")
        for path, vals in paths.items():
            # poverty_to_food_desert uses different keys (logistic model)
            if "coef" in vals:
                console.print(f"  {path}: β={vals['coef']}, R²={vals['r_squared']}")
            elif "ols_coef" in vals:
                logit_or = (
                    vals.get("logistic", {}).get("odds_ratio", "n/a")
                    if vals.get("logistic") else "n/a"
                )
                console.print(
                    f"  {path}: OLS_β={vals['ols_coef']}, R²={vals['ols_r_squared']}, "
                    f"logistic_OR={logit_or}"
                )

    return paths
