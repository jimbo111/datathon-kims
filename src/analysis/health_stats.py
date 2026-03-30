"""Statistical analysis for the Food Desert + Chronic Disease project.

Phase 2: Food Access → Chronic Disease (OLS/WLS, odds ratios, partial correlation,
         logistic regression, Cohen's d, BH FDR correction)
Phase 3: The Zip Code Effect (variance decomposition, ICC, life expectancy regression
         with unstandardized + standardized betas, model diagnostics)
Phase 4: Race as a Residual Gap (interaction terms for Black + Hispanic, likelihood
         ratio F-test, Welch t-test with CI for cross-comparison)

All functions take the master dataframe and return results + JSON-exportable dicts.
BH FDR correction is applied globally across all p-values at the end of run_all_phases().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.regression.mixed_linear_model import MixedLM
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


def _bh_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values (q-values) in the same order as input.
    Implements the step-up procedure: q_i = p_(i) * m / i, enforcing monotonicity.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed)):
        i = m - rank  # rank from 1 in ascending order
        q = p * m / i
        running_min = min(running_min, q)
        adjusted[orig_idx] = running_min
    return adjusted


def _cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    """Pooled Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std(ddof=1) ** 2 + (n2 - 1) * group2.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return float("nan")
    return float((group1.mean() - group2.mean()) / pooled_std)


# ─── Phase 2: Food Access → Chronic Disease ──────────────────────────────────


def run_phase2(master: pd.DataFrame, use_wls: bool = False) -> dict:
    """OLS/WLS regressions, odds ratios, logistic regression, partial correlations.

    Parameters
    ----------
    master : pd.DataFrame
        The master merged dataframe.
    use_wls : bool
        If True and a ``population`` column is present, fits WLS using population
        as weights instead of unweighted OLS. Defaults to False (OLS).

    Returns a results dict with regression tables, odds ratios, t-tests (with
    Cohen's d), logistic regression, and partial correlations.
    All p-values are collected in ``results["_all_pvalues"]`` for BH correction
    by the caller (run_all_phases).
    """
    console.rule("[bold]Phase 2: Food Access → Chronic Disease")
    results = {}
    _pvals: dict[str, float] = {}  # label → raw p-value for BH correction

    has_population = "population" in master.columns

    def _fit_model(formula: str, df_clean: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Fit OLS or WLS depending on use_wls flag and data availability."""
        if use_wls and has_population:
            weights = df_clean["population"].clip(lower=1)
            return smf.wls(formula, data=df_clean, weights=weights).fit(cov_type="HC1")
        return smf.ols(formula, data=df_clean).fit(cov_type="HC1")

    # ── 2a. OLS/WLS: diabetes_pct ~ food access + income + uninsured ──
    reg_cols = ["diabetes_pct", "is_food_desert", "poverty_rate", "uninsured_pct"]
    if use_wls and has_population:
        reg_cols = reg_cols + ["population"]
    df2 = _clean_for_regression(master, reg_cols)

    model_label = "WLS" if (use_wls and has_population) else "OLS"

    if len(df2) > 100:
        model_diabetes = _fit_model(
            "diabetes_pct ~ is_food_desert + poverty_rate + uninsured_pct", df2
        )
        results["ols_diabetes"] = _ols_summary(model_diabetes, f"diabetes_pct ({model_label})")
        console.print(
            f"  {model_label} diabetes: R²={model_diabetes.rsquared:.3f}, "
            f"n={model_diabetes.nobs:.0f}"
        )
        # Collect p-values
        for var, pv in model_diabetes.pvalues.items():
            if var != "Intercept":
                _pvals[f"phase2_diabetes_{var}"] = float(pv)

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

        # ── Model diagnostics for Phase 2 diabetes OLS ──
        resid_diab = model_diabetes.resid
        # Jarque-Bera
        jb_d_stat, jb_d_p, jb_d_skew, jb_d_kurt = jarque_bera(resid_diab)
        # Breusch-Pagan
        X_bp_diab = sm.add_constant(df2[["is_food_desert", "poverty_rate", "uninsured_pct"]])
        bp_d_lm, bp_d_p, bp_d_f, bp_d_fp = het_breuschpagan(resid_diab, X_bp_diab)
        # Durbin-Watson
        dw_d = float(durbin_watson(resid_diab))
        # Cohen's f² = R² / (1 - R²)
        r2_diab = model_diabetes.rsquared
        f2_diab = r2_diab / (1 - r2_diab) if r2_diab < 1.0 else float("inf")

        results["diagnostics_diabetes_ols"] = {
            "jarque_bera": {
                "statistic": round(float(jb_d_stat), 4),
                "p_value": float(f"{jb_d_p:.2e}"),
                "skewness": round(float(jb_d_skew), 4),
                "kurtosis": round(float(jb_d_kurt), 4),
                "residuals_normal": jb_d_p > 0.05,
            },
            "breusch_pagan": {
                "lm_statistic": round(float(bp_d_lm), 4),
                "lm_p_value": float(f"{bp_d_p:.2e}"),
                "f_statistic": round(float(bp_d_f), 4),
                "f_p_value": float(f"{bp_d_fp:.2e}"),
                "homoscedastic": bp_d_p > 0.05,
            },
            "durbin_watson": {
                "statistic": round(dw_d, 4),
                "interpretation": (
                    "possible positive autocorrelation" if dw_d < 1.5
                    else "possible negative autocorrelation" if dw_d > 2.5
                    else "no autocorrelation concern"
                ),
            },
            "cohen_f_squared": round(f2_diab, 4),
            "effect_size_f2_interpretation": (
                "small" if f2_diab < 0.15
                else "medium" if f2_diab < 0.35
                else "large"
            ),
        }
        # NOTE: Diagnostic p-values (JB, BP) excluded from BH-FDR —
        # they test model assumptions, not inferential hypotheses.
        console.print(
            f"  Diagnostics (diabetes): JB p={jb_d_p:.2e}, BP p={bp_d_p:.2e}, "
            f"DW={dw_d:.3f}, f²={f2_diab:.3f}"
        )
    else:
        console.print("[yellow]  Not enough data for diabetes OLS[/]")

    # ── 2b. OLS/WLS: obesity_pct ~ food access + income + uninsured ──
    reg_cols_b = ["obesity_pct", "is_food_desert", "poverty_rate", "uninsured_pct"]
    if use_wls and has_population:
        reg_cols_b = reg_cols_b + ["population"]
    df2b = _clean_for_regression(master, reg_cols_b)

    if len(df2b) > 100:
        model_obesity = _fit_model(
            "obesity_pct ~ is_food_desert + poverty_rate + uninsured_pct", df2b
        )
        results["ols_obesity"] = _ols_summary(model_obesity, f"obesity_pct ({model_label})")
        console.print(f"  {model_label} obesity: R²={model_obesity.rsquared:.3f}, n={model_obesity.nobs:.0f}")
        for var, pv in model_obesity.pvalues.items():
            if var != "Intercept":
                _pvals[f"phase2_obesity_{var}"] = float(pv)
    else:
        console.print("[yellow]  Not enough data for obesity OLS[/]")

    # ── 2c. Odds ratio: food desert vs non-food-desert diabetes rates ──
    if "is_food_desert" in master.columns and "diabetes_pct" in master.columns:
        fd = master.dropna(subset=["is_food_desert", "diabetes_pct"])
        desert = fd[fd["is_food_desert"] == 1]["diabetes_pct"]
        non_desert = fd[fd["is_food_desert"] == 0]["diabetes_pct"]

        if len(desert) > 0 and len(non_desert) > 0:
            # Sensitivity check: compute OR at 25th, 50th, and 75th percentile thresholds
            or_by_threshold = {}
            for pct_label, pct_val in [("p25", 25), ("p50", 50), ("p75", 75)]:
                threshold = np.percentile(fd["diabetes_pct"], pct_val)
                a = (desert > threshold).sum()
                b = (desert <= threshold).sum()
                c = (non_desert > threshold).sum()
                d = (non_desert <= threshold).sum()
                if a > 0 and b > 0 and c > 0 and d > 0:
                    or_val = (a * d) / (b * c)
                    se_log = np.sqrt(1/a + 1/b + 1/c + 1/d)
                    or_by_threshold[pct_label] = {
                        "threshold": round(threshold, 2),
                        "odds_ratio": round(or_val, 3),
                        "ci_95_low": round(np.exp(np.log(or_val) - 1.96 * se_log), 3),
                        "ci_95_high": round(np.exp(np.log(or_val) + 1.96 * se_log), 3),
                    }

            # Primary OR uses median (p50) to preserve backward-compatible key
            if "p50" in or_by_threshold:
                p50 = or_by_threshold["p50"]
                results["odds_ratio_diabetes"] = {
                    "odds_ratio": p50["odds_ratio"],
                    "ci_95_low": p50["ci_95_low"],
                    "ci_95_high": p50["ci_95_high"],
                    "desert_mean_diabetes": round(desert.mean(), 2),
                    "non_desert_mean_diabetes": round(non_desert.mean(), 2),
                    "n_desert": len(desert),
                    "n_non_desert": len(non_desert),
                    "threshold_used": "median (p50)",
                    "sensitivity_by_threshold": or_by_threshold,
                }
                console.print(
                    f"  Odds ratio (diabetes, median split): {p50['odds_ratio']:.2f} "
                    f"(95% CI: {p50['ci_95_low']:.2f}–{p50['ci_95_high']:.2f})"
                )
                console.print(
                    f"  OR sensitivity: p25={or_by_threshold.get('p25', {}).get('odds_ratio', 'n/a')}, "
                    f"p50={p50['odds_ratio']}, "
                    f"p75={or_by_threshold.get('p75', {}).get('odds_ratio', 'n/a')}"
                )

        # ── Logistic regression for odds ratio (proper model-based approach) ──
        # Two versions: (a) above-median threshold (backward-compatible ecological OR)
        #               (b) CDC clinical benchmark >= 12.0% (proper logistic OR)
        if len(desert) > 0 and len(non_desert) > 0:
            df_logit = fd.copy()

            # (a) Median-based logistic OR (ecological, backward-compatible)
            median_diabetes = df_logit["diabetes_pct"].median()
            df_logit["high_diabetes_median"] = (df_logit["diabetes_pct"] > median_diabetes).astype(int)
            if df_logit["high_diabetes_median"].nunique() == 2:
                try:
                    logit_model = smf.logit(
                        "high_diabetes_median ~ is_food_desert", data=df_logit
                    ).fit(disp=False)
                    food_desert_coef = logit_model.params.get("is_food_desert", np.nan)
                    food_desert_p = logit_model.pvalues.get("is_food_desert", np.nan)
                    food_desert_ci = logit_model.conf_int().loc["is_food_desert"]
                    results["logistic_regression_diabetes"] = {
                        "log_odds_food_desert": round(food_desert_coef, 4),
                        "odds_ratio": round(np.exp(food_desert_coef), 3),
                        "ci_95_low": round(np.exp(food_desert_ci[0]), 3),
                        "ci_95_high": round(np.exp(food_desert_ci[1]), 3),
                        "p_value": float(f"{food_desert_p:.2e}"),
                        "pseudo_r_squared": round(logit_model.prsquared, 4),
                        "n_obs": int(logit_model.nobs),
                        "note": "Ecological logistic OR: outcome = above-median diabetes (binary); predictor = food desert flag",
                        "label": "ecological_logistic_or",
                    }
                    _pvals["phase2_logit_food_desert"] = float(food_desert_p)
                    console.print(
                        f"  Logistic OR ecological (food desert→high diabetes): "
                        f"{np.exp(food_desert_coef):.2f}, p={food_desert_p:.2e}"
                    )
                except Exception as exc:
                    console.print(f"  [yellow]Logistic regression (median) failed: {exc}[/]")

            # (b) CDC clinical benchmark logistic OR (diabetes_high = diabetes_pct > 12.0)
            # CDC defines diabetes prevalence >12% as elevated population burden
            CDC_DIABETES_THRESHOLD = 12.0
            df_logit["diabetes_high_cdc"] = (df_logit["diabetes_pct"] > CDC_DIABETES_THRESHOLD).astype(int)
            n_cdc_high = int(df_logit["diabetes_high_cdc"].sum())
            if df_logit["diabetes_high_cdc"].nunique() == 2 and n_cdc_high >= 30:
                try:
                    logit_cols_cdc = ["diabetes_high_cdc", "is_food_desert", "poverty_rate", "uninsured_pct"]
                    df_logit_cdc = df_logit.dropna(subset=logit_cols_cdc)
                    logit_cdc = smf.logit(
                        "diabetes_high_cdc ~ is_food_desert + poverty_rate + uninsured_pct",
                        data=df_logit_cdc,
                    ).fit(disp=False)
                    fd_coef_cdc = logit_cdc.params.get("is_food_desert", np.nan)
                    fd_p_cdc = logit_cdc.pvalues.get("is_food_desert", np.nan)
                    fd_ci_cdc = logit_cdc.conf_int().loc["is_food_desert"]
                    results["logistic_regression_diabetes_cdc"] = {
                        "threshold_pct": CDC_DIABETES_THRESHOLD,
                        "n_above_threshold": n_cdc_high,
                        "n_below_threshold": int(len(df_logit_cdc)) - n_cdc_high,
                        "food_desert_log_odds": round(fd_coef_cdc, 4),
                        "food_desert_odds_ratio": round(np.exp(fd_coef_cdc), 3),
                        "ci_95_low": round(np.exp(fd_ci_cdc[0]), 3),
                        "ci_95_high": round(np.exp(fd_ci_cdc[1]), 3),
                        "p_value": float(f"{fd_p_cdc:.2e}"),
                        "pseudo_r_squared": round(logit_cdc.prsquared, 4),
                        "n_obs": int(logit_cdc.nobs),
                        "controls": ["poverty_rate", "uninsured_pct"],
                        "note": (
                            "Proper logistic OR: outcome = diabetes_pct > 12% (CDC clinical benchmark). "
                            "Adjusted for poverty_rate and uninsured_pct. "
                            "Provides model-based OR comparable to, but distinct from, ecological median-split OR."
                        ),
                        "label": "cdc_benchmark_logistic_or",
                    }
                    _pvals["phase2_logit_cdc_food_desert"] = float(fd_p_cdc)
                    console.print(
                        f"  Logistic OR CDC (food desert→diabetes>12%): "
                        f"OR={np.exp(fd_coef_cdc):.2f}, p={fd_p_cdc:.2e} "
                        f"(n={logit_cdc.nobs:.0f}, adjusted for poverty+uninsured)"
                    )
                except Exception as exc:
                    console.print(f"  [yellow]Logistic regression (CDC benchmark) failed: {exc}[/]")
            else:
                console.print(
                    f"  [yellow]CDC logistic OR skipped: n_high={n_cdc_high} "
                    f"(need >=30 tracts with diabetes>12%)[/]"
                )

        # T-test: mean diabetes rate in food deserts vs not (with Cohen's d + partial eta²)
        if len(desert) > 1 and len(non_desert) > 1:
            t_stat, t_p = stats.ttest_ind(desert, non_desert, equal_var=False)
            d = _cohen_d(desert, non_desert)
            n1_t, n2_t = len(desert), len(non_desert)
            # Welch-Satterthwaite df for partial eta²
            s1_t, s2_t = desert.std(ddof=1), non_desert.std(ddof=1)
            df_t = (s1_t**2 / n1_t + s2_t**2 / n2_t) ** 2 / (
                (s1_t**2 / n1_t) ** 2 / (n1_t - 1) + (s2_t**2 / n2_t) ** 2 / (n2_t - 1)
            )
            # Partial eta² = t² / (t² + df) — proportion of variance explained by group
            partial_eta2 = t_stat**2 / (t_stat**2 + df_t) if df_t > 0 else float("nan")
            results["ttest_diabetes"] = {
                "t_statistic": round(t_stat, 3),
                "p_value": float(f"{t_p:.2e}"),
                "cohen_d": round(d, 4),
                "partial_eta_squared": round(float(partial_eta2), 4),
                "effect_size_interpretation": (
                    "small" if abs(d) < 0.5
                    else "medium" if abs(d) < 0.8
                    else "large"
                ),
                "desert_mean": round(desert.mean(), 2),
                "non_desert_mean": round(non_desert.mean(), 2),
                "difference": round(desert.mean() - non_desert.mean(), 2),
                "n_desert": n1_t,
                "n_non_desert": n2_t,
            }
            _pvals["phase2_ttest_diabetes"] = float(t_p)

    # ── 2d. Partial correlation: food access → diabetes controlling for income ──
    partial_cols = ["diabetes_pct", "is_food_desert", "poverty_rate"]
    dfp = _clean_for_regression(master, partial_cols)
    if len(dfp) > 100:
        pc = _partial_correlation(dfp, "diabetes_pct", "is_food_desert", ["poverty_rate"])
        results["partial_corr_food_diabetes"] = pc
        _pvals["phase2_partial_corr"] = float(pc["p"])
        console.print(f"  Partial corr (food→diabetes|income): r={pc['r']:.3f}, p={pc['p']:.2e}")

    results["_all_pvalues"] = _pvals
    _export_json(results, "phase2_food_access_disease.json")
    return results


# ─── Phase 3: The Zip Code Effect ────────────────────────────────────────────


def run_phase3(master: pd.DataFrame) -> dict:
    """Variance decomposition, ICC, life expectancy regression, zip code gaps.

    Returns results with standardized + unstandardized betas, variance stats,
    ICC, model diagnostics (Jarque-Bera, Breusch-Pagan), and gap metrics.
    All p-values are collected in ``results["_all_pvalues"]`` for BH correction.
    """
    console.rule("[bold]Phase 3: The Zip Code Effect")
    results = {}
    _pvals: dict[str, float] = {}

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

        # ── ICC (Intraclass Correlation Coefficient) ──
        # ICC = between_var / total_var — proportion of total variance due to county-level clustering
        # This is the ICC(1) formulation using the variance decomposition directly.
        # For a more formal multilevel model ICC, one would use a null random-intercept model.
        icc = between_var / total_var if total_var > 0 else 0.0

        results["variance_decomposition"] = {
            "total_variance": round(total_var, 3),
            "between_county_variance": round(between_var, 3),
            "within_county_variance": round(within_var, 3),
            "pct_between": round(between_var / total_var * 100, 1),
            "pct_within": round(within_var / total_var * 100, 1),
            "icc_anova": round(icc, 4),
            "icc_note": (
                "ICC(1) from ANOVA-based variance decomposition. "
                "Proportion of diabetes variance attributable to county-level clustering."
            ),
        }
        console.print(
            f"  Variance: {between_var/total_var*100:.1f}% between-county, "
            f"{within_var/total_var*100:.1f}% within-county | ICC(ANOVA)={icc:.4f}"
        )

        # ── ICC via null random-intercept MixedLM (model-based, more rigorous) ──
        # Requires at least 2 groups with multiple observations.
        n_counties = dfv["county_fips"].nunique()
        if n_counties >= 10:
            try:
                mlm_null = MixedLM(
                    endog=dfv["diabetes_pct"],
                    exog=np.ones((len(dfv), 1)),
                    groups=dfv["county_fips"],
                )
                mlm_fit = mlm_null.fit(reml=True, method="lbfgs", disp=False)
                # ICC = tau² / (tau² + sigma²)
                # mlm_fit.cov_re is the random-effects variance (tau²)
                # mlm_fit.scale is residual variance (sigma²)
                tau2 = float(mlm_fit.cov_re.iloc[0, 0])
                sigma2 = float(mlm_fit.scale)
                icc_mlm = tau2 / (tau2 + sigma2) if (tau2 + sigma2) > 0 else 0.0

                # ICC interpretation: <0.05 minimal, 0.05-0.10 small, 0.10-0.25 moderate, >0.25 large
                if icc_mlm < 0.05:
                    icc_interp = "minimal county-level clustering"
                elif icc_mlm < 0.10:
                    icc_interp = "small county-level clustering"
                elif icc_mlm < 0.25:
                    icc_interp = "moderate county-level clustering"
                else:
                    icc_interp = "large county-level clustering"

                results["variance_decomposition"]["icc_mlm"] = round(icc_mlm, 4)
                results["variance_decomposition"]["icc_mlm_tau2"] = round(tau2, 4)
                results["variance_decomposition"]["icc_mlm_sigma2"] = round(sigma2, 4)
                results["variance_decomposition"]["icc_mlm_interpretation"] = icc_interp
                results["variance_decomposition"]["icc_mlm_note"] = (
                    "Model-based ICC from null random-intercept MixedLM (REML). "
                    "tau² = between-county variance, sigma² = within-county residual variance. "
                    f"Interpretation: {icc_interp}."
                )
                console.print(
                    f"  ICC (MixedLM REML): {icc_mlm:.4f} — {icc_interp} "
                    f"(tau²={tau2:.3f}, sigma²={sigma2:.3f})"
                )
            except Exception as exc:
                console.print(f"  [yellow]MixedLM ICC failed: {exc}[/]")
                results["variance_decomposition"]["icc_mlm"] = None
                results["variance_decomposition"]["icc_mlm_note"] = f"MixedLM ICC computation failed: {exc}"

    # ── 3b. Multivariate regression: life expectancy ──
    le_cols = [
        "life_expectancy", "median_household_income", "is_food_desert",
        "pct_black", "pct_bachelors_plus", "uninsured_pct",
    ]
    dfl = _clean_for_regression(master, le_cols)

    if len(dfl) > 100:
        X_cols = le_cols[1:]

        # ── Unstandardized model (original units) ──
        formula_unstd = "life_expectancy ~ " + " + ".join(X_cols)
        model_le_unstd = smf.ols(formula_unstd, data=dfl).fit(cov_type="HC1")
        results["ols_life_expectancy_unstandardized"] = _ols_summary(
            model_le_unstd, "life_expectancy (unstandardized)"
        )

        # ── Standardized model (z-scored predictors) ──
        dfl_std = dfl.copy()
        for col in X_cols:
            dfl_std[col] = (dfl_std[col] - dfl_std[col].mean()) / dfl_std[col].std()

        formula_std = "life_expectancy ~ " + " + ".join(X_cols)
        model_le = smf.ols(formula_std, data=dfl_std).fit(cov_type="HC1")

        # Keep backward-compatible key pointing to standardized model
        results["ols_life_expectancy"] = _ols_summary(model_le, "life_expectancy (standardized predictors)")

        std_betas = {
            var: round(model_le.params[var], 3)
            for var in X_cols if var in model_le.params
        }
        results["standardized_betas"] = std_betas

        # Unstandardized betas for interpretability
        unstd_betas = {
            var: round(model_le_unstd.params[var], 4)
            for var in X_cols if var in model_le_unstd.params
        }
        results["unstandardized_betas"] = unstd_betas

        dominant = max(std_betas.items(), key=lambda x: abs(x[1]))
        console.print(
            f"  Life exp R²={model_le.rsquared:.3f}, dominant predictor: "
            f"{dominant[0]} (β_std={dominant[1]})"
        )

        # ── VIF check for Phase 3 life expectancy model ──
        X_vif_le = dfl[X_cols].dropna()
        X_vif_le_const = sm.add_constant(X_vif_le)
        vif_data_le = {}
        for i, col in enumerate(X_vif_le_const.columns[1:]):
            vif_val = variance_inflation_factor(X_vif_le_const.values, i + 1)
            vif_data_le[col] = round(vif_val, 2)
        results["vif_life_expectancy"] = vif_data_le
        high_vif_le = {k: v for k, v in vif_data_le.items() if v > 3}
        if high_vif_le:
            console.print(
                f"  [yellow]VIF warning (>3) in life exp model: {high_vif_le} "
                f"— education and income may be collinear[/]"
            )
        else:
            console.print(f"  VIF (life exp): {vif_data_le} (all <3, acceptable)")

        # ── Model diagnostics ──
        residuals = model_le_unstd.resid

        # Jarque-Bera test for residual normality
        jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)
        results["diagnostic_jarque_bera"] = {
            "jb_statistic": round(float(jb_stat), 4),
            "p_value": float(f"{jb_p:.2e}"),
            "skewness": round(float(jb_skew), 4),
            "kurtosis": round(float(jb_kurt), 4),
            "residuals_normal": jb_p > 0.05,
        }
        # Diagnostic p-value — excluded from BH-FDR (not inferential)
        console.print(
            f"  Jarque-Bera: stat={jb_stat:.2f}, p={jb_p:.2e} "
            f"({'normal' if jb_p > 0.05 else 'non-normal'} residuals)"
        )

        # Breusch-Pagan test for heteroscedasticity
        X_for_bp = sm.add_constant(dfl[X_cols])
        bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X_for_bp)
        results["diagnostic_breusch_pagan"] = {
            "lm_statistic": round(float(bp_lm), 4),
            "lm_p_value": float(f"{bp_p:.2e}"),
            "f_statistic": round(float(bp_f), 4),
            "f_p_value": float(f"{bp_fp:.2e}"),
            "homoscedastic": bp_p > 0.05,
            "note": "HC1 robust standard errors already applied to account for heteroscedasticity",
        }
        # Diagnostic p-value — excluded from BH-FDR (not inferential)
        console.print(
            f"  Breusch-Pagan: LM={bp_lm:.2f}, p={bp_p:.2e} "
            f"({'homoscedastic' if bp_p > 0.05 else 'heteroscedastic'})"
        )

        # Durbin-Watson test for residual autocorrelation
        # Note: spatial data may exhibit autocorrelation; DW tests for serial pattern in residual order.
        dw_stat = float(durbin_watson(residuals))
        # DW ≈ 2: no autocorrelation; <1.5: positive; >2.5: negative
        if dw_stat < 1.5:
            dw_interp = "possible positive autocorrelation"
        elif dw_stat > 2.5:
            dw_interp = "possible negative autocorrelation"
        else:
            dw_interp = "no autocorrelation concern"
        results["diagnostic_durbin_watson"] = {
            "dw_statistic": round(dw_stat, 4),
            "interpretation": dw_interp,
            "note": (
                "DW statistic for residual autocorrelation in sorted order. "
                "Spatial cross-sectional data may show geographic autocorrelation "
                "not captured by DW (which tests sequential order)."
            ),
        }
        console.print(f"  Durbin-Watson: {dw_stat:.4f} — {dw_interp}")

        for var, pv in model_le.pvalues.items():
            if var != "Intercept":
                _pvals[f"phase3_le_{var}"] = float(pv)

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

    results["_all_pvalues"] = _pvals
    _export_json(results, "phase3_zip_code_effect.json")
    return results


# ─── Phase 4: Race as a Residual Gap ─────────────────────────────────────────


def run_phase4(master: pd.DataFrame) -> dict:
    """Income × race → diabetes matrix, interaction regression, residual analysis.

    Adds pct_hispanic interaction term alongside pct_black, formal likelihood ratio
    F-test for R² improvement when adding race, and Welch t-test with CI for the
    high-income Black vs low-income White cross-comparison.

    Returns results with the income-race matrix and regression findings.
    All p-values are collected in ``results["_all_pvalues"]`` for BH correction.
    """
    console.rule("[bold]Phase 4: Race as a Residual Gap")
    results = {}
    _pvals: dict[str, float] = {}

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

    # ── 4b. Regression with interactions: income × pct_black + income × pct_hispanic ──
    int_cols = ["diabetes_pct", "pct_black", "pct_hispanic", "median_household_income", "is_food_desert"]
    dfi = _clean_for_regression(master, int_cols)

    if len(dfi) > 100:
        dfi = dfi.copy()
        dfi["income_10k"] = dfi["median_household_income"] / 10_000
        dfi["pct_black_std"] = (dfi["pct_black"] - dfi["pct_black"].mean()) / dfi["pct_black"].std()
        dfi["pct_hispanic_std"] = (
            (dfi["pct_hispanic"] - dfi["pct_hispanic"].mean()) / dfi["pct_hispanic"].std()
        )

        # Black interaction (original model, backward-compatible)
        model_int_black = smf.ols(
            "diabetes_pct ~ income_10k * pct_black_std + is_food_desert",
            data=dfi,
        ).fit(cov_type="HC1")

        results["interaction_model"] = _ols_summary(model_int_black, "diabetes_pct (income × Black interaction)")
        console.print(f"  Interaction (Black) R²={model_int_black.rsquared:.3f}")

        int_term_black = "income_10k:pct_black_std"
        if int_term_black in model_int_black.params:
            int_coef = model_int_black.params[int_term_black]
            int_p = model_int_black.pvalues[int_term_black]
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
            _pvals["phase4_interaction_black"] = float(int_p)
            console.print(f"  Interaction (Black) coef={int_coef:.4f}, p={int_p:.2e}")

        # Hispanic interaction (new model)
        model_int_hispanic = smf.ols(
            "diabetes_pct ~ income_10k * pct_hispanic_std + is_food_desert",
            data=dfi,
        ).fit(cov_type="HC1")

        results["interaction_model_hispanic"] = _ols_summary(
            model_int_hispanic, "diabetes_pct (income × Hispanic interaction)"
        )
        console.print(f"  Interaction (Hispanic) R²={model_int_hispanic.rsquared:.3f}")

        int_term_hispanic = "income_10k:pct_hispanic_std"
        if int_term_hispanic in model_int_hispanic.params:
            hisp_coef = model_int_hispanic.params[int_term_hispanic]
            hisp_p = model_int_hispanic.pvalues[int_term_hispanic]
            results["interaction_term_hispanic"] = {
                "coefficient": round(hisp_coef, 4),
                "p_value": float(f"{hisp_p:.2e}"),
                "significant": hisp_p < 0.05,
                "interpretation": (
                    "Income's protective effect on diabetes is weaker in higher-% Hispanic tracts"
                    if hisp_coef > 0 else
                    "Income's protective effect on diabetes is stronger in higher-% Hispanic tracts"
                ),
            }
            _pvals["phase4_interaction_hispanic"] = float(hisp_p)
            console.print(f"  Interaction (Hispanic) coef={hisp_coef:.4f}, p={hisp_p:.2e}")

    # ── 4c. Residual analysis: is race significant after income + food access? ──
    res_cols = ["diabetes_pct", "pct_black", "pct_hispanic", "poverty_rate",
                "is_food_desert", "uninsured_pct"]
    dfr = _clean_for_regression(master, res_cols)

    if len(dfr) > 100:
        # Model without race
        model_no_race = smf.ols(
            "diabetes_pct ~ poverty_rate + is_food_desert + uninsured_pct", data=dfr
        ).fit(cov_type="HC1")

        # Model with race (pct_black + pct_hispanic)
        model_with_race = smf.ols(
            "diabetes_pct ~ poverty_rate + is_food_desert + uninsured_pct + pct_black + pct_hispanic",
            data=dfr,
        ).fit(cov_type="HC1")

        r2_change = model_with_race.rsquared - model_no_race.rsquared
        race_p_black = model_with_race.pvalues.get("pct_black", 1.0)
        race_p_hispanic = model_with_race.pvalues.get("pct_hispanic", 1.0)

        # ── Formal F-test (likelihood ratio) for R² improvement ──
        # For OLS, the incremental F-test is exact: F = (ΔR² / q) / ((1-R²_full) / (n-k-1))
        # where q = number of added predictors, k = total predictors in full model.
        n_obs = model_with_race.nobs
        k_full = model_with_race.df_model       # excludes intercept
        k_restricted = model_no_race.df_model
        q_added = k_full - k_restricted          # number of added race vars (2)
        r2_full = model_with_race.rsquared
        r2_restricted = model_no_race.rsquared

        if q_added > 0 and r2_full < 1.0:
            f_increment = ((r2_full - r2_restricted) / q_added) / (
                (1 - r2_full) / (n_obs - k_full - 1)
            )
            f_increment_p = float(stats.f.sf(f_increment, q_added, n_obs - k_full - 1))
        else:
            f_increment = float("nan")
            f_increment_p = float("nan")

        results["residual_analysis"] = {
            "r2_without_race": round(model_no_race.rsquared, 4),
            "r2_with_race": round(model_with_race.rsquared, 4),
            "r2_change": round(r2_change, 4),
            "pct_black_coefficient": round(model_with_race.params.get("pct_black", 0), 4),
            "pct_black_p_value": float(f"{race_p_black:.2e}"),
            "pct_hispanic_coefficient": round(model_with_race.params.get("pct_hispanic", 0), 4),
            "pct_hispanic_p_value": float(f"{race_p_hispanic:.2e}"),
            "race_still_significant": race_p_black < 0.05 or race_p_hispanic < 0.05,
            "f_test_r2_improvement": {
                "f_statistic": round(f_increment, 4) if not np.isnan(f_increment) else None,
                "p_value": float(f"{f_increment_p:.2e}") if not np.isnan(f_increment_p) else None,
                "df_numerator": int(q_added),
                "df_denominator": int(n_obs - k_full - 1),
                "significant": f_increment_p < 0.05 if not np.isnan(f_increment_p) else False,
                "note": (
                    "Incremental F-test for whether adding pct_black + pct_hispanic "
                    "significantly improves model R²"
                ),
            },
        }
        _pvals["phase4_pct_black"] = float(race_p_black)
        _pvals["phase4_pct_hispanic"] = float(race_p_hispanic)
        if not np.isnan(f_increment_p):
            _pvals["phase4_f_test_race"] = f_increment_p
        console.print(
            f"  Residual: R² without race={model_no_race.rsquared:.4f}, "
            f"with race={model_with_race.rsquared:.4f}, ΔR²={r2_change:.4f}"
        )
        console.print(
            f"  F-test (race vars): F={f_increment:.2f}, p={f_increment_p:.2e}"
            if not np.isnan(f_increment)
            else "  F-test: could not compute"
        )

    # ── 4d. Compare high-income Black tracts vs low-income White tracts ──
    comp_cols = ["diabetes_pct", "income_quintile", "majority_race"]
    dfc = master.dropna(subset=comp_cols)

    if len(dfc) > 50:
        high_inc_black = dfc[(dfc["income_quintile"] >= 4) & (dfc["majority_race"] == "Black")]["diabetes_pct"]
        low_inc_white = dfc[(dfc["income_quintile"] <= 2) & (dfc["majority_race"] == "White")]["diabetes_pct"]

        n1, n2 = len(high_inc_black), len(low_inc_white)

        # Warn prominently if high-income Black sample is small
        small_n_warning = None
        if n1 < 50:
            small_n_warning = (
                f"CAUTION: n_high_income_black={n1} < 50. "
                "Results may be unstable; interpret with caution."
            )
            console.print(f"  [yellow bold]{small_n_warning}[/]")

        if n1 > 10 and n2 > 10:
            # Welch t-test (unequal variances) with 95% CI on the difference
            t_stat, t_p = stats.ttest_ind(high_inc_black, low_inc_white, equal_var=False)

            # 95% CI on difference of means (Welch-Satterthwaite df)
            s1, s2 = high_inc_black.std(ddof=1), low_inc_white.std(ddof=1)
            se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
            # Welch-Satterthwaite degrees of freedom
            df_ws = (s1**2 / n1 + s2**2 / n2) ** 2 / (
                (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
            )
            t_crit = stats.t.ppf(0.975, df_ws)
            mean_diff = high_inc_black.mean() - low_inc_white.mean()
            ci_low = mean_diff - t_crit * se_diff
            ci_high = mean_diff + t_crit * se_diff

            cohen_d_val = _cohen_d(high_inc_black, low_inc_white)

            results["cross_comparison"] = {
                "high_income_black_mean_diabetes": round(high_inc_black.mean(), 2),
                "low_income_white_mean_diabetes": round(low_inc_white.mean(), 2),
                "gap": round(mean_diff, 2),
                "n_high_income_black": n1,
                "n_low_income_white": n2,
                "income_closes_gap": high_inc_black.mean() <= low_inc_white.mean(),
                "small_n_warning": small_n_warning,
                "welch_ttest": {
                    "t_statistic": round(t_stat, 3),
                    "p_value": float(f"{t_p:.2e}"),
                    "df": round(df_ws, 1),
                    "ci_95_low": round(ci_low, 3),
                    "ci_95_high": round(ci_high, 3),
                    "cohen_d": round(cohen_d_val, 4),
                },
            }
            _pvals["phase4_cross_comparison"] = float(t_p)
            console.print(
                f"  High-income Black (n={n1}): {high_inc_black.mean():.1f}% diabetes, "
                f"Low-income White (n={n2}): {low_inc_white.mean():.1f}% diabetes"
            )
            console.print(
                f"  Welch t={t_stat:.3f}, p={t_p:.2e}, "
                f"95% CI=[{ci_low:.2f}, {ci_high:.2f}], d={cohen_d_val:.3f}"
            )

        # ── Within-quintile Black-White diabetes gap (Welch t-test per quintile) ──
        quintile_gap_results = []
        for q in sorted(dfc["income_quintile"].dropna().unique()):
            q_df = dfc[dfc["income_quintile"] == q]
            black_q = q_df[q_df["majority_race"] == "Black"]["diabetes_pct"].dropna()
            white_q = q_df[q_df["majority_race"] == "White"]["diabetes_pct"].dropna()
            n_b, n_w = len(black_q), len(white_q)

            if n_b < 2 or n_w < 2:
                quintile_gap_results.append({
                    "income_quintile": int(q),
                    "n_black": n_b,
                    "n_white": n_w,
                    "note": "insufficient data for comparison",
                })
                continue

            t_q, p_q = stats.ttest_ind(black_q, white_q, equal_var=False)
            s_b, s_w = black_q.std(ddof=1), white_q.std(ddof=1)
            se_q = np.sqrt(s_b**2 / n_b + s_w**2 / n_w)
            df_q = (s_b**2 / n_b + s_w**2 / n_w) ** 2 / (
                (s_b**2 / n_b) ** 2 / (n_b - 1) + (s_w**2 / n_w) ** 2 / (n_w - 1)
            )
            t_crit_q = stats.t.ppf(0.975, df_q)
            gap_q = black_q.mean() - white_q.mean()
            ci_q_low = gap_q - t_crit_q * se_q
            ci_q_high = gap_q + t_crit_q * se_q
            d_q = _cohen_d(black_q, white_q)

            quintile_gap_results.append({
                "income_quintile": int(q),
                "black_mean": round(float(black_q.mean()), 2),
                "white_mean": round(float(white_q.mean()), 2),
                "gap_black_minus_white": round(float(gap_q), 2),
                "n_black": n_b,
                "n_white": n_w,
                "welch_t": round(float(t_q), 3),
                "p_value": float(f"{p_q:.2e}"),
                "ci_95_low": round(float(ci_q_low), 3),
                "ci_95_high": round(float(ci_q_high), 3),
                "cohen_d": round(float(d_q), 4),
                "significant": p_q < 0.05,
            })
            _pvals[f"phase4_quintile_gap_q{int(q)}"] = float(p_q)

        if quintile_gap_results:
            results["within_quintile_black_white_gap"] = {
                "by_quintile": quintile_gap_results,
                "note": (
                    "Within each income quintile, Welch t-test comparing Black-majority vs "
                    "White-majority tract diabetes rates. Positive gap = Black tracts have "
                    "higher diabetes even within same income band."
                ),
            }
            console.print("  Within-quintile Black-White gaps:")
            for row in quintile_gap_results:
                if "gap_black_minus_white" in row:
                    console.print(
                        f"    Q{row['income_quintile']}: gap={row['gap_black_minus_white']:.2f}pp, "
                        f"p={row['p_value']}, n_B={row['n_black']}, n_W={row['n_white']}"
                    )

    results["_all_pvalues"] = _pvals
    _export_json(results, "phase4_race_residual_gap.json")
    return results


# ─── Helper functions ─────────────────────────────────────────────────────────


def _ols_summary(model, name: str) -> dict:
    """Convert a statsmodels OLS/WLS result to a JSON-serializable dict.

    Includes Cohen's f² = R² / (1 - R²) as a global effect size measure.
    Interpretation: f² < 0.02 negligible, 0.02–0.15 small, 0.15–0.35 medium, >0.35 large.
    """
    r2 = model.rsquared
    f2 = r2 / (1 - r2) if r2 < 1.0 else float("inf")
    f2_interp = (
        "negligible" if f2 < 0.02
        else "small" if f2 < 0.15
        else "medium" if f2 < 0.35
        else "large"
    )
    return {
        "name": name,
        "r_squared": round(r2, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_statistic": round(model.fvalue, 2),
        "f_pvalue": float(f"{model.f_pvalue:.2e}"),
        "n_obs": int(model.nobs),
        "cohen_f_squared": round(f2, 4),
        "effect_size_f2": f2_interp,
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


def run_all_phases(master: pd.DataFrame, use_wls: bool = False) -> dict:
    """Run Phases 2-4 and return combined results with global BH FDR correction.

    Parameters
    ----------
    master : pd.DataFrame
        The master merged dataframe.
    use_wls : bool
        Passed through to run_phase2. If True, uses WLS with population weights.

    Returns combined results including a ``fdr_correction`` key with all
    raw p-values, their BH-adjusted q-values, and significance at q<0.05.
    """
    phase2 = run_phase2(master, use_wls=use_wls)
    phase3 = run_phase3(master)
    phase4 = run_phase4(master)

    # ── Collect all p-values across phases for global BH FDR correction ──
    all_pvals_raw: dict[str, float] = {}
    for phase_key, phase_results in [("phase2", phase2), ("phase3", phase3), ("phase4", phase4)]:
        for label, pv in phase_results.get("_all_pvalues", {}).items():
            all_pvals_raw[label] = pv

    if all_pvals_raw:
        labels = list(all_pvals_raw.keys())
        raw_pvals = [all_pvals_raw[k] for k in labels]
        adjusted = _bh_correct(raw_pvals)

        fdr_table = [
            {
                "test": labels[i],
                "p_raw": float(f"{raw_pvals[i]:.2e}"),
                "q_bh": float(f"{adjusted[i]:.2e}"),
                "significant_bh_q05": adjusted[i] < 0.05,
            }
            for i in range(len(labels))
        ]
        fdr_table.sort(key=lambda x: x["p_raw"])

        fdr_results = {
            "method": "Benjamini-Hochberg FDR correction",
            "n_tests": len(labels),
            "alpha": 0.05,
            "tests": fdr_table,
        }
        _export_json(fdr_results, "fdr_correction_all_phases.json")
        console.print(
            f"\n[bold]BH FDR correction:[/] {len(labels)} tests, "
            f"{sum(1 for t in fdr_table if t['significant_bh_q05'])} survive q<0.05"
        )
    else:
        fdr_results = {"note": "No p-values collected for FDR correction"}

    return {
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
        "fdr_correction": fdr_results,
    }
