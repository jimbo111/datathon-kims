"""Statistical analysis for the Food Desert + Chronic Disease project.

Phase 2: Food Access → Chronic Disease (OLS, odds ratios, partial correlation)
Phase 3: The Zip Code Effect (variance decomposition, life expectancy regression)
Phase 4: Race as a Residual Gap (interaction terms, residual analysis)

All functions take the master dataframe and return results + JSON-exportable dicts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from rich.console import Console

console = Console()
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def _export_json(data: dict, filename: str) -> Path:
    """Save results dict as JSON for Alice's frontend."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"[green]Exported → {path}[/]")
    return path


def _clean_for_regression(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop NaN rows and infinite values for the given columns."""
    subset = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return df.loc[subset.index]


# ─── Phase 2: Food Access → Chronic Disease ──────────────────────────────────


def run_phase2(master: pd.DataFrame) -> dict:
    """OLS regressions, odds ratios, and partial correlations for food access → disease.

    Returns a results dict with regression tables, odds ratios, and correlations.
    """
    console.rule("[bold]Phase 2: Food Access → Chronic Disease")
    results = {}

    # ── 2a. OLS: diabetes_pct ~ food access + income + uninsured ──
    reg_cols = ["diabetes_pct", "is_food_desert", "poverty_rate", "uninsured_pct"]
    df2 = _clean_for_regression(master, reg_cols)

    if len(df2) > 100:
        model_diabetes = smf.ols(
            "diabetes_pct ~ is_food_desert + poverty_rate + uninsured_pct", data=df2
        ).fit(cov_type="HC1")
        results["ols_diabetes"] = _ols_summary(model_diabetes, "diabetes_pct")
        console.print(f"  OLS diabetes: R²={model_diabetes.rsquared:.3f}, n={model_diabetes.nobs:.0f}")

        # VIF check for multicollinearity
        X_vif = df2[["is_food_desert", "poverty_rate", "uninsured_pct"]].dropna()
        X_vif = sm.add_constant(X_vif)
        vif_data = {}
        for i, col in enumerate(X_vif.columns[1:]):
            vif_val = variance_inflation_factor(X_vif.values, i + 1)
            vif_data[col] = round(vif_val, 2)
        results["vif"] = vif_data
        high_vif = {k: v for k, v in vif_data.items() if v > 5}
        if high_vif:
            console.print(f"  [yellow]VIF warning (>5): {high_vif}[/]")
        else:
            console.print(f"  VIF: {vif_data} (all <5, no multicollinearity concern)")
    else:
        console.print("[yellow]  Not enough data for diabetes OLS[/]")

    # ── 2b. OLS: obesity_pct ~ food access + income + uninsured ──
    reg_cols = ["obesity_pct", "is_food_desert", "poverty_rate", "uninsured_pct"]
    df2b = _clean_for_regression(master, reg_cols)

    if len(df2b) > 100:
        model_obesity = smf.ols(
            "obesity_pct ~ is_food_desert + poverty_rate + uninsured_pct", data=df2b
        ).fit(cov_type="HC1")
        results["ols_obesity"] = _ols_summary(model_obesity, "obesity_pct")
        console.print(f"  OLS obesity: R²={model_obesity.rsquared:.3f}, n={model_obesity.nobs:.0f}")
    else:
        console.print("[yellow]  Not enough data for obesity OLS[/]")

    # ── 2c. Odds ratio: food desert vs non-food-desert diabetes rates ──
    if "is_food_desert" in master.columns and "diabetes_pct" in master.columns:
        fd = master.dropna(subset=["is_food_desert", "diabetes_pct"])
        desert = fd[fd["is_food_desert"] == 1]["diabetes_pct"]
        non_desert = fd[fd["is_food_desert"] == 0]["diabetes_pct"]

        if len(desert) > 0 and len(non_desert) > 0:
            # Use median as threshold for "high diabetes" to compute odds ratio
            median_diabetes = fd["diabetes_pct"].median()
            a = (desert > median_diabetes).sum()  # desert + high diabetes
            b = (desert <= median_diabetes).sum()  # desert + low diabetes
            c = (non_desert > median_diabetes).sum()  # non-desert + high diabetes
            d = (non_desert <= median_diabetes).sum()  # non-desert + low diabetes

            if a > 0 and b > 0 and c > 0 and d > 0:
                odds_ratio = (a * d) / (b * c)
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_low = np.exp(np.log(odds_ratio) - 1.96 * se_log_or)
                ci_high = np.exp(np.log(odds_ratio) + 1.96 * se_log_or)

                results["odds_ratio_diabetes"] = {
                    "odds_ratio": round(odds_ratio, 3),
                    "ci_95_low": round(ci_low, 3),
                    "ci_95_high": round(ci_high, 3),
                    "desert_mean_diabetes": round(desert.mean(), 2),
                    "non_desert_mean_diabetes": round(non_desert.mean(), 2),
                    "n_desert": len(desert),
                    "n_non_desert": len(non_desert),
                }
                console.print(
                    f"  Odds ratio (diabetes): {odds_ratio:.2f} "
                    f"(95% CI: {ci_low:.2f}–{ci_high:.2f})"
                )

        # T-test: mean diabetes rate in food deserts vs not
        if len(desert) > 1 and len(non_desert) > 1:
            t_stat, t_p = stats.ttest_ind(desert, non_desert, equal_var=False)
            results["ttest_diabetes"] = {
                "t_statistic": round(t_stat, 3),
                "p_value": float(f"{t_p:.2e}"),
                "desert_mean": round(desert.mean(), 2),
                "non_desert_mean": round(non_desert.mean(), 2),
                "difference": round(desert.mean() - non_desert.mean(), 2),
            }

    # ── 2d. Partial correlation: food access → diabetes controlling for income ──
    partial_cols = ["diabetes_pct", "is_food_desert", "poverty_rate"]
    dfp = _clean_for_regression(master, partial_cols)
    if len(dfp) > 100:
        pc = _partial_correlation(dfp, "diabetes_pct", "is_food_desert", ["poverty_rate"])
        results["partial_corr_food_diabetes"] = pc
        console.print(f"  Partial corr (food→diabetes|income): r={pc['r']:.3f}, p={pc['p']:.2e}")

    _export_json(results, "phase2_food_access_disease.json")
    return results


# ─── Phase 3: The Zip Code Effect ────────────────────────────────────────────


def run_phase3(master: pd.DataFrame) -> dict:
    """Variance decomposition, life expectancy regression, zip code gaps.

    Returns results with standardized betas, variance stats, and gap metrics.
    """
    console.rule("[bold]Phase 3: The Zip Code Effect")
    results = {}

    # ── 3a. Variance decomposition: between-county vs within-county diabetes variance ──
    dfv = master.dropna(subset=["diabetes_pct"]).copy()
    dfv["county_fips"] = dfv["tract_fips"].str[:5]

    if len(dfv) > 100:
        grand_mean = dfv["diabetes_pct"].mean()
        n_total = len(dfv)
        total_var = dfv["diabetes_pct"].var(ddof=0)  # population variance for consistent decomposition

        # Weighted between-county variance (accounts for unequal group sizes)
        county_stats = dfv.groupby("county_fips")["diabetes_pct"].agg(["mean", "count"])
        between_var = ((county_stats["count"] * (county_stats["mean"] - grand_mean) ** 2).sum()
                       / n_total)
        within_var = total_var - between_var

        results["variance_decomposition"] = {
            "total_variance": round(total_var, 3),
            "between_county_variance": round(between_var, 3),
            "within_county_variance": round(within_var, 3),
            "pct_between": round(between_var / total_var * 100, 1),
            "pct_within": round(within_var / total_var * 100, 1),
        }
        console.print(
            f"  Variance: {between_var/total_var*100:.1f}% between-county, "
            f"{within_var/total_var*100:.1f}% within-county"
        )

    # ── 3b. Multivariate regression: life expectancy ──
    le_cols = [
        "life_expectancy", "median_household_income", "is_food_desert",
        "pct_black", "pct_bachelors_plus", "uninsured_pct",
    ]
    dfl = _clean_for_regression(master, le_cols)

    if len(dfl) > 100:
        # Standardize predictors for comparable betas
        X_cols = le_cols[1:]
        dfl_std = dfl.copy()
        for col in X_cols:
            dfl_std[col] = (dfl_std[col] - dfl_std[col].mean()) / dfl_std[col].std()

        formula = "life_expectancy ~ " + " + ".join(X_cols)
        model_le = smf.ols(formula, data=dfl_std).fit(cov_type="HC1")
        results["ols_life_expectancy"] = _ols_summary(model_le, "life_expectancy")

        # Standardized betas (already standardized predictors, so coefficients = std betas)
        std_betas = {
            var: round(model_le.params[var], 3)
            for var in X_cols if var in model_le.params
        }
        results["standardized_betas"] = std_betas
        dominant = max(std_betas.items(), key=lambda x: abs(x[1]))
        console.print(f"  Life exp R²={model_le.rsquared:.3f}, dominant predictor: {dominant[0]} (β={dominant[1]})")

    # ── 3c. Life expectancy gap by income quintile ──
    if "life_expectancy" in master.columns and "income_quintile" in master.columns:
        gap_df = master.dropna(subset=["life_expectancy", "income_quintile"])
        quintile_le = gap_df.groupby("income_quintile")["life_expectancy"].agg(["mean", "median", "count"])
        quintile_le = quintile_le.round(2)

        if len(quintile_le) >= 2:
            q1_le = quintile_le.iloc[0]["mean"]
            q5_le = quintile_le.iloc[-1]["mean"]
            gap = q5_le - q1_le

            results["life_expectancy_gap"] = {
                "poorest_quintile_mean_le": round(q1_le, 2),
                "richest_quintile_mean_le": round(q5_le, 2),
                "gap_years": round(gap, 2),
                "by_quintile": quintile_le.reset_index().to_dict(orient="records"),
            }
            console.print(f"  Life expectancy gap (Q5-Q1): {gap:.1f} years")

    _export_json(results, "phase3_zip_code_effect.json")
    return results


# ─── Phase 4: Race as a Residual Gap ─────────────────────────────────────────


def run_phase4(master: pd.DataFrame) -> dict:
    """Income × race → diabetes matrix, interaction regression, residual analysis.

    Returns results with the income-race matrix and regression findings.
    """
    console.rule("[bold]Phase 4: Race as a Residual Gap")
    results = {}

    # ── 4a. Income quintile × majority race → diabetes prevalence matrix ──
    matrix_cols = ["income_quintile", "majority_race", "diabetes_pct"]
    dfm = master.dropna(subset=matrix_cols)

    if len(dfm) > 100:
        matrix = dfm.pivot_table(
            index="income_quintile", columns="majority_race",
            values="diabetes_pct", aggfunc="mean",
        ).round(2)

        results["income_race_diabetes_matrix"] = {
            "matrix": matrix.reset_index().to_dict(orient="records"),
            "columns": matrix.columns.tolist(),
        }
        console.print("  Income × Race → Diabetes matrix:")
        console.print(str(matrix))

    # ── 4b. Regression with interaction: income × race (continuous) → diabetes ──
    int_cols = ["diabetes_pct", "pct_black", "median_household_income", "is_food_desert"]
    dfi = _clean_for_regression(master, int_cols)

    if len(dfi) > 100:
        dfi = dfi.copy()
        dfi["income_10k"] = dfi["median_household_income"] / 10_000
        # Use continuous pct_black (not median-split) to preserve information
        dfi["pct_black_std"] = (dfi["pct_black"] - dfi["pct_black"].mean()) / dfi["pct_black"].std()

        model_int = smf.ols(
            "diabetes_pct ~ income_10k * pct_black_std + is_food_desert",
            data=dfi,
        ).fit(cov_type="HC1")

        results["interaction_model"] = _ols_summary(model_int, "diabetes_pct (interaction)")
        console.print(f"  Interaction R²={model_int.rsquared:.3f}")

        # Extract interaction term significance
        int_term = "income_10k:pct_black_std"
        if int_term in model_int.params:
            int_coef = model_int.params[int_term]
            int_p = model_int.pvalues[int_term]
            results["interaction_term"] = {
                "coefficient": round(int_coef, 4),
                "p_value": float(f"{int_p:.2e}"),
                "significant": int_p < 0.05,
                "interpretation": (
                    "Income's protective effect on diabetes is weaker in higher-% Black tracts"
                    if int_coef > 0 else
                    "Income's protective effect on diabetes is stronger in higher-% Black tracts"
                ),
            }
            console.print(f"  Interaction coef={int_coef:.4f}, p={int_p:.2e}")

    # ── 4c. Residual analysis: is race significant after income + food access? ──
    res_cols = ["diabetes_pct", "pct_black", "poverty_rate", "is_food_desert", "uninsured_pct"]
    dfr = _clean_for_regression(master, res_cols)

    if len(dfr) > 100:
        # Model without race
        model_no_race = smf.ols(
            "diabetes_pct ~ poverty_rate + is_food_desert + uninsured_pct", data=dfr
        ).fit(cov_type="HC1")

        # Model with race
        model_with_race = smf.ols(
            "diabetes_pct ~ poverty_rate + is_food_desert + uninsured_pct + pct_black", data=dfr
        ).fit(cov_type="HC1")

        r2_change = model_with_race.rsquared - model_no_race.rsquared
        race_p = model_with_race.pvalues.get("pct_black", 1.0)

        results["residual_analysis"] = {
            "r2_without_race": round(model_no_race.rsquared, 4),
            "r2_with_race": round(model_with_race.rsquared, 4),
            "r2_change": round(r2_change, 4),
            "pct_black_coefficient": round(model_with_race.params.get("pct_black", 0), 4),
            "pct_black_p_value": float(f"{race_p:.2e}"),
            "race_still_significant": race_p < 0.05,
        }
        console.print(
            f"  Residual: R² without race={model_no_race.rsquared:.4f}, "
            f"with race={model_with_race.rsquared:.4f}, "
            f"ΔR²={r2_change:.4f}, p(race)={race_p:.2e}"
        )

    # ── 4d. Compare high-income Black tracts vs low-income White tracts ──
    comp_cols = ["diabetes_pct", "income_quintile", "majority_race"]
    dfc = master.dropna(subset=comp_cols)

    if len(dfc) > 50:
        high_inc_black = dfc[(dfc["income_quintile"] >= 4) & (dfc["majority_race"] == "Black")]["diabetes_pct"]
        low_inc_white = dfc[(dfc["income_quintile"] <= 2) & (dfc["majority_race"] == "White")]["diabetes_pct"]

        if len(high_inc_black) > 10 and len(low_inc_white) > 10:
            results["cross_comparison"] = {
                "high_income_black_mean_diabetes": round(high_inc_black.mean(), 2),
                "low_income_white_mean_diabetes": round(low_inc_white.mean(), 2),
                "gap": round(high_inc_black.mean() - low_inc_white.mean(), 2),
                "n_high_income_black": len(high_inc_black),
                "n_low_income_white": len(low_inc_white),
                "income_closes_gap": high_inc_black.mean() <= low_inc_white.mean(),
            }
            console.print(
                f"  High-income Black: {high_inc_black.mean():.1f}% diabetes, "
                f"Low-income White: {low_inc_white.mean():.1f}% diabetes"
            )

    _export_json(results, "phase4_race_residual_gap.json")
    return results


# ─── Helper functions ─────────────────────────────────────────────────────────


def _ols_summary(model, name: str) -> dict:
    """Convert a statsmodels OLS result to a JSON-serializable dict."""
    return {
        "name": name,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_statistic": round(model.fvalue, 2),
        "f_pvalue": float(f"{model.f_pvalue:.2e}"),
        "n_obs": int(model.nobs),
        "coefficients": {
            var: {
                "coef": round(model.params[var], 4),
                "std_err": round(model.bse[var], 4),
                "t_stat": round(model.tvalues[var], 3),
                "p_value": float(f"{model.pvalues[var]:.2e}"),
                "ci_low": round(model.conf_int().loc[var, 0], 4),
                "ci_high": round(model.conf_int().loc[var, 1], 4),
            }
            for var in model.params.index
        },
    }


def _partial_correlation(
    df: pd.DataFrame, y: str, x: str, controls: list[str]
) -> dict:
    """Compute partial correlation between x and y, controlling for other variables."""
    all_vars = [y, x] + controls
    data = df[all_vars].dropna()

    # Regress y on controls, get residuals
    X_ctrl = sm.add_constant(data[controls])
    resid_y = sm.OLS(data[y], X_ctrl).fit().resid

    # Regress x on controls, get residuals
    resid_x = sm.OLS(data[x], X_ctrl).fit().resid

    # Correlation of residuals = partial correlation
    r, p = stats.pearsonr(resid_x, resid_y)

    return {
        "r": round(r, 4),
        "p": float(f"{p:.2e}"),
        "n": len(data),
        "controlling_for": controls,
    }


# ─── Run all phases ──────────────────────────────────────────────────────────


def run_all_phases(master: pd.DataFrame) -> dict:
    """Run Phases 2-4 and return combined results."""
    return {
        "phase2": run_phase2(master),
        "phase3": run_phase3(master),
        "phase4": run_phase4(master),
    }
