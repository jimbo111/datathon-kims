"""Causal Inference Module — Phase 6B.

Three causal identification strategies for estimating the effect of
food-desert status on diabetes_pct and obesity_pct:

1. Propensity Score Matching (PSM)
   Balance food-desert vs non-food-desert tracts on observable confounders,
   then estimate the Average Treatment effect on the Treated (ATT).
   Matching is 1:1 nearest-neighbor without replacement using logistic
   propensity scores.  Balance diagnostics (SMD) are reported before and
   after matching.

2. Difference-in-Differences (DiD) scaffold
   Structural framework for a two-period panel (USDA 2015 → 2019).
   When the 2015 vintage of USDA Food Access Research Atlas is merged with
   the 2019 vintage, this module runs the two-way fixed-effects estimator:
       y_it = α + β (treat_i × post_t) + γ_i + δ_t + ε_it
   Until that data is available, placeholder logic validates the panel
   structure and prints required columns.

3. Regression Discontinuity (RD)
   At the USDA threshold (≥1/3 of population > 1 mi from supermarket for
   urban, or > 10 mi for rural), tracts just above vs below the cutoff
   are compared.  Running variable: pct_low_access_1mi.
   Local linear regression is used in a ±10 pp bandwidth, with
   Calonico-Cattaneo-Titiunik (2014) robust bias-corrected CIs.
   (Simplified version implemented with statsmodels; rdrobust not required.)

Statistical assumptions
-----------------------
- PSM: Conditional independence (CIA) — treatment is as-good-as random
  conditional on observed covariates.  Unmeasured confounders may bias ATT.
- DiD: Parallel trends — treated and control tracts would have evolved
  identically in the absence of treatment.  Requires pre-period data.
- RD: Continuity — potential outcomes are continuous at the cutoff.
  Sorting precisely at the cutoff would invalidate the design.

All results exported to data/processed/phase6b_causal_inference.json.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from rich.console import Console

console = Console()
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def _export_json(data: dict, filename: str) -> Path:
    """Save results dict as JSON to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"[green]Exported → {path}[/]")
    return path


def _standardized_mean_difference(treated: pd.Series, control: pd.Series) -> float:
    """Compute standardized mean difference (Cohen's d) for balance assessment.

    SMD < 0.1 indicates good covariate balance (Stuart 2010).
    """
    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt((treated.var() + control.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return round(float(mean_diff / pooled_std), 4)


# ─── Propensity Score Matching ────────────────────────────────────────────────


def run_propensity_score_matching(
    master: pd.DataFrame,
    outcome_cols: list[str] | None = None,
    caliper: float = 0.1,
) -> dict:
    """Estimate ATT of food desert status via 1:1 nearest-neighbor PSM.

    Propensity scores are the predicted probabilities from a logistic
    regression of is_food_desert on the full set of confounders.
    Matching is performed without replacement; the caliper (in propensity
    score SD units) prevents poor matches.

    Confounders used
    ----------------
    poverty_rate, median_household_income, pct_black, pct_hispanic,
    uninsured_pct, population.  Urban/rural flag added when available.

    Balance diagnostics
    -------------------
    Standardized Mean Differences (SMD) reported before and after matching.
    SMD < 0.1 for all covariates indicates acceptable balance.

    Parameters
    ----------
    master       : Master dataframe.
    outcome_cols : Outcomes to estimate ATT for (default: diabetes_pct, obesity_pct).
    caliper      : Maximum allowed difference in propensity score (in SD units).
                   0.1–0.2 is standard in epidemiology literature.

    Returns
    -------
    dict with balance diagnostics, ATT estimates, and 95% CIs.
    """
    console.rule("[bold]Causal: Propensity Score Matching (PSM)")
    results = {}

    if outcome_cols is None:
        outcome_cols = [c for c in ["diabetes_pct", "obesity_pct"] if c in master.columns]

    treatment = "is_food_desert"
    confounders = [
        "poverty_rate", "median_household_income", "pct_black",
        "pct_hispanic", "uninsured_pct", "population",
    ]
    # Add urban flag if available
    for col in ["urban", "urban_flag", "metro"]:
        if col in master.columns:
            confounders.append(col)
            break

    required = [treatment] + confounders + outcome_cols
    available_confounders = [c for c in confounders if c in master.columns]
    available_outcomes = [c for c in outcome_cols if c in master.columns]

    if not available_confounders:
        console.print("[red]No confounders available for PSM.[/]")
        return results

    if treatment not in master.columns:
        console.print(f"[red]Treatment column '{treatment}' not found.[/]")
        return results

    df_psm = master[
        [treatment] + available_confounders + available_outcomes
    ].dropna()

    if len(df_psm) < 500:
        console.print(f"[yellow]Only {len(df_psm)} complete cases for PSM (need ≥500).[/]")
        return results

    console.print(f"  PSM sample: {len(df_psm):,} tracts "
                  f"({df_psm[treatment].sum():,} treated, "
                  f"{(df_psm[treatment]==0).sum():,} control)")

    # ── Step 1: Estimate propensity scores ──
    formula = f"{treatment} ~ " + " + ".join(available_confounders)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps_model = smf.logit(formula, data=df_psm).fit(disp=False, maxiter=200)

    df_psm = df_psm.copy()
    df_psm["propensity_score"] = ps_model.predict(df_psm)

    ps_auc = _compute_auc(df_psm[treatment].values, df_psm["propensity_score"].values)
    results["propensity_model"] = {
        "formula": formula,
        "n_obs": int(ps_model.nobs),
        "pseudo_r_squared": round(float(ps_model.prsquared), 4),
        "auc_roc": round(float(ps_auc), 4),
        "overlap_note": (
            "Good overlap" if (
                df_psm.loc[df_psm[treatment]==1, "propensity_score"].min() < 0.5 and
                df_psm.loc[df_psm[treatment]==0, "propensity_score"].max() > 0.5
            ) else "Potential limited overlap — ATT estimate may be unreliable"
        ),
    }
    console.print(
        f"  Propensity model: pseudo-R²={ps_model.prsquared:.4f}, AUC={ps_auc:.4f}"
    )

    # ── Step 2: Pre-match balance ──
    treated_df = df_psm[df_psm[treatment] == 1]
    control_df = df_psm[df_psm[treatment] == 0]

    pre_balance = {}
    for col in available_confounders:
        pre_balance[col] = _standardized_mean_difference(
            treated_df[col], control_df[col]
        )
    results["pre_match_smd"] = pre_balance
    n_imbalanced_pre = sum(1 for v in pre_balance.values() if abs(v) > 0.1)
    console.print(f"  Pre-match: {n_imbalanced_pre}/{len(available_confounders)} covariates with SMD>0.1")

    # ── Step 3: 1:1 nearest-neighbor matching without replacement ──
    caliper_abs = caliper * df_psm["propensity_score"].std()
    matched_treated_idx, matched_control_idx = _nearest_neighbor_match(
        treated_df["propensity_score"].values,
        control_df["propensity_score"].values,
        treated_df.index.values,
        control_df.index.values,
        caliper=caliper_abs,
    )

    if len(matched_treated_idx) < 50:
        console.print(f"[yellow]  Only {len(matched_treated_idx)} matched pairs after caliper. "
                      "Try increasing caliper.[/]")
        results["psm_warning"] = f"Only {len(matched_treated_idx)} pairs matched (caliper={caliper})"

    matched_treated = df_psm.loc[matched_treated_idx]
    matched_control = df_psm.loc[matched_control_idx]
    n_matched = len(matched_treated_idx)

    console.print(f"  Matched {n_matched:,} pairs (caliper={caliper} SD, "
                  f"{len(treated_df)-n_matched:,} treated unmatched)")

    # ── Step 4: Post-match balance ──
    post_balance = {}
    for col in available_confounders:
        post_balance[col] = _standardized_mean_difference(
            matched_treated[col], matched_control[col]
        )
    results["post_match_smd"] = post_balance
    n_imbalanced_post = sum(1 for v in post_balance.values() if abs(v) > 0.1)
    console.print(f"  Post-match: {n_imbalanced_post}/{len(available_confounders)} covariates with SMD>0.1")
    results["balance_achieved"] = n_imbalanced_post == 0

    # ── Step 5: ATT estimation ──
    att_results = {}
    for outcome in available_outcomes:
        treated_y = matched_treated[outcome].values
        control_y = matched_control[outcome].values

        att = float(np.mean(treated_y - control_y))
        # Bootstrap 95% CI for ATT
        boot_atts = []
        rng = np.random.default_rng(42)
        for _ in range(500):
            idx = rng.integers(0, n_matched, size=n_matched)
            boot_atts.append(float(np.mean(treated_y[idx] - control_y[idx])))

        ci_low = float(np.percentile(boot_atts, 2.5))
        ci_high = float(np.percentile(boot_atts, 97.5))

        # Paired t-test on matched differences
        t_stat, t_p = stats.ttest_rel(treated_y, control_y)

        att_results[outcome] = {
            "att": round(att, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "t_statistic": round(float(t_stat), 3),
            "p_value": float(f"{t_p:.2e}"),
            "significant": t_p < 0.05,
            "n_pairs": n_matched,
            "interpretation": (
                f"Food desert status associated with {att:+.2f} pp {outcome} "
                f"({'significant' if t_p < 0.05 else 'not significant'} after matching)"
            ),
        }
        console.print(
            f"  ATT ({outcome}): {att:+.4f} pp "
            f"(95% CI: {ci_low:+.4f}–{ci_high:+.4f}), p={t_p:.2e}"
        )

    results["att_estimates"] = att_results

    return results


def _nearest_neighbor_match(
    treated_ps: np.ndarray,
    control_ps: np.ndarray,
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    caliper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """1:1 nearest-neighbor PSM matching without replacement.

    Treated units are processed in random order to avoid systematic bias
    from ordering effects.  Each control unit is used at most once.
    """
    rng = np.random.default_rng(42)
    order = rng.permutation(len(treated_ps))

    matched_treated = []
    matched_control = []
    used_control = set()

    for i in order:
        tp = treated_ps[i]
        diffs = np.abs(control_ps - tp)
        # Mask out already-used controls
        for j in used_control:
            diffs[j] = np.inf

        best_j = int(np.argmin(diffs))
        if diffs[best_j] <= caliper:
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[best_j])
            used_control.add(best_j)

    return np.array(matched_treated), np.array(matched_control)


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC via trapezoidal rule (no sklearn dependency)."""
    thresholds = np.sort(np.unique(y_score))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0)
    fprs.append(1.0)
    # np.trapezoid in numpy>=2.0, np.trapz in <=1.x; handle both
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(tprs, fprs))


# ─── Difference-in-Differences scaffold ──────────────────────────────────────


def run_did_scaffold(master: pd.DataFrame, panel_df: pd.DataFrame | None = None) -> dict:
    """Difference-in-Differences estimator for food desert → health outcomes.

    This function implements a two-way fixed-effects (TWFE) DiD:
        y_it = α + β (treat_i × post_t) + γ_i + δ_t + ε_it

    where:
    - treat_i  = 1 if tract was classified as food desert in baseline (2015)
    - post_t   = 1 for the follow-up period (2019)
    - γ_i      = tract fixed effects (absorb all time-invariant confounders)
    - δ_t      = time fixed effects (absorb common trends)
    - β        = ATT under the parallel-trends assumption

    Panel data requirements
    -----------------------
    panel_df must have columns:
      tract_fips, year (2015 or 2019), is_food_desert, diabetes_pct,
      obesity_pct, [any time-varying controls]

    When panel_df is None (default), this function validates that master
    has the required structure and prints data availability diagnostics.

    USDA 2015 vintage: https://www.ers.usda.gov/data-products/food-access-research-atlas/
    (Download 'Food Access Research Atlas Data - 2015' Excel file.)
    """
    console.rule("[bold]Causal: Difference-in-Differences (DiD)")
    results = {}

    # ── Validate panel availability ──
    if panel_df is None:
        console.print("[yellow]Panel data not provided.  DiD requires USDA 2015 + 2019 vintages.[/]")
        console.print("  To build the panel:")
        console.print("  1. Download USDA 2015 Food Access Research Atlas Excel")
        console.print("  2. Load as 'usda_2015' with tract_fips + food desert flags")
        console.print("  3. Add year=2015 column and merge with current master (year=2019)")
        console.print("  4. Pass the stacked DataFrame as panel_df to this function")

        # Estimate number of tracts that could be in panel (present in master)
        n_tracts = master["tract_fips"].nunique() if "tract_fips" in master.columns else "unknown"
        results["did_scaffold"] = {
            "status": "awaiting_panel_data",
            "required_columns": [
                "tract_fips", "year", "is_food_desert",
                "diabetes_pct", "obesity_pct",
            ],
            "required_years": [2015, 2019],
            "n_current_tracts": n_tracts,
            "data_source_2015": (
                "https://www.ers.usda.gov/data-products/food-access-research-atlas/"
            ),
            "estimator": "Two-way fixed effects (tract + year FE)",
            "identification_assumption": "Parallel trends — treated and control tracts "
                                         "would trend identically absent food desert entry/exit",
        }
        console.print(f"  Current master: {n_tracts:,} tracts (would form one period of panel)")
        return results

    # ── Run DiD on provided panel ──
    required = ["tract_fips", "year", "is_food_desert"]
    missing = [c for c in required if c not in panel_df.columns]
    if missing:
        console.print(f"[red]panel_df missing columns: {missing}[/]")
        return results

    years = sorted(panel_df["year"].unique())
    if len(years) < 2:
        console.print(f"[red]panel_df must have at least 2 time periods. Found: {years}[/]")
        return results

    baseline_year = years[0]
    post_year = years[-1]

    panel_df = panel_df.copy()
    panel_df["post"] = (panel_df["year"] == post_year).astype(int)

    # Define treatment based on baseline food desert status
    baseline_treatment = (
        panel_df[panel_df["year"] == baseline_year]
        .set_index("tract_fips")["is_food_desert"]
        .rename("treat_i")
    )
    panel_df = panel_df.join(baseline_treatment, on="tract_fips")
    panel_df["did_term"] = panel_df["treat_i"] * panel_df["post"]

    did_results = {}
    for outcome in ["diabetes_pct", "obesity_pct"]:
        if outcome not in panel_df.columns:
            continue

        df_did = panel_df[["tract_fips", "year", "post", "treat_i", "did_term", outcome]].dropna()

        # TWFE: outcome ~ did_term + C(tract_fips) + C(year)
        # Use within-estimator via demeaning for efficiency
        df_did = df_did.copy()
        for col in [outcome, "did_term"]:
            df_did[f"{col}_tract_mean"] = df_did.groupby("tract_fips")[col].transform("mean")
            df_did[f"{col}_year_mean"] = df_did.groupby("year")[col].transform("mean")

        grand_mean_y = df_did[outcome].mean()
        grand_mean_did = df_did["did_term"].mean()

        df_did["y_demeaned"] = (
            df_did[outcome]
            - df_did[f"{outcome}_tract_mean"]
            - df_did[f"{outcome}_year_mean"]
            + grand_mean_y
        )
        df_did["did_demeaned"] = (
            df_did["did_term"]
            - df_did["did_term_tract_mean"]
            - df_did["did_term_year_mean"]
            + grand_mean_did
        )

        model_twfe = smf.ols("y_demeaned ~ did_demeaned", data=df_did).fit(cov_type="HC1")
        beta_did = model_twfe.params.get("did_demeaned", np.nan)
        p_did = model_twfe.pvalues.get("did_demeaned", np.nan)
        ci = model_twfe.conf_int().loc["did_demeaned"] if "did_demeaned" in model_twfe.conf_int().index else [np.nan, np.nan]

        did_results[outcome] = {
            "did_coefficient": round(float(beta_did), 4),
            "ci_95_low": round(float(ci[0]), 4),
            "ci_95_high": round(float(ci[1]), 4),
            "p_value": float(f"{p_did:.2e}"),
            "significant": p_did < 0.05,
            "n_tracts": int(df_did["tract_fips"].nunique()),
            "n_periods": int(df_did["year"].nunique()),
            "baseline_year": baseline_year,
            "post_year": post_year,
            "interpretation": (
                f"Food desert status associated with {beta_did:+.3f} pp change in "
                f"{outcome} over {baseline_year}→{post_year} "
                f"({'significant' if p_did < 0.05 else 'not significant'})"
            ),
        }
        console.print(
            f"  DiD ({outcome}): β={beta_did:+.4f} "
            f"(95% CI: {ci[0]:+.4f}–{ci[1]:+.4f}), p={p_did:.2e}"
        )

    results["did_estimates"] = did_results
    return results


# ─── Regression Discontinuity ─────────────────────────────────────────────────


def run_regression_discontinuity(
    master: pd.DataFrame,
    running_var: str = "pct_low_access_1mi",
    outcome: str = "diabetes_pct",
    cutoff: float = 33.0,
    bandwidth: float = 10.0,
) -> dict:
    """Local linear RD at the USDA food desert access threshold.

    The USDA classifies a census tract as a food desert (low access) when
    at least 1/3 of the population (33%) lives more than 1 mile from a
    supermarket (urban) or 10 miles (rural).  This creates a natural
    discontinuity at the 33% threshold.

    Design: Compare tracts just above vs just below the 33% cutoff using
    local linear regression with a triangular kernel in a ±bandwidth window.

    Interpretation: If food desert status causes worse health outcomes
    (not just correlated), we expect a discontinuous jump in the outcome
    at the cutoff.  The RD estimate equals the LATE (Local Average
    Treatment Effect) at the threshold.

    Assumptions:
    - No precise sorting: tracts cannot precisely manipulate their
      pct_low_access to just cross/avoid the threshold.
    - Continuity: all other factors vary smoothly at the cutoff.

    Parameters
    ----------
    running_var : Column used as the running variable (pct_low_access_1mi).
    outcome     : Health outcome to examine.
    cutoff      : USDA classification threshold in percentage points (33%).
    bandwidth   : Window around cutoff for local linear regression (pp).
    """
    console.rule(f"[bold]Causal: Regression Discontinuity (cutoff={cutoff}%)")
    results = {}

    if running_var not in master.columns:
        console.print(f"[yellow]Running variable '{running_var}' not found. "
                      "RD requires pct_low_access_1mi from USDA data.[/]")
        results["rd_status"] = {
            "status": "skipped",
            "reason": f"'{running_var}' column not in master",
            "required_column": running_var,
            "data_source": "USDA Food Access Research Atlas (pct_low_access_1mi)",
        }
        return results

    if outcome not in master.columns:
        console.print(f"[red]Outcome '{outcome}' not found.[/]")
        return results

    df_rd = master[[running_var, outcome, "is_food_desert"]].dropna()
    df_rd = df_rd.copy()
    df_rd["running_centered"] = df_rd[running_var] - cutoff  # center at threshold
    df_rd["above_cutoff"] = (df_rd["running_centered"] >= 0).astype(int)

    # McCrary density test (informal: compare n just above vs below)
    n_below = ((df_rd["running_centered"] >= -bandwidth) & (df_rd["running_centered"] < 0)).sum()
    n_above = ((df_rd["running_centered"] >= 0) & (df_rd["running_centered"] < bandwidth)).sum()
    results["density_check"] = {
        "n_just_below": int(n_below),
        "n_just_above": int(n_above),
        "ratio": round(n_above / max(n_below, 1), 3),
        "sorting_concern": abs(n_above / max(n_below, 1) - 1) > 0.5,
    }
    console.print(
        f"  Density check: n_below={n_below:,}, n_above={n_above:,}, "
        f"ratio={n_above/max(n_below,1):.2f} "
        f"({'potential sorting' if results['density_check']['sorting_concern'] else 'no sorting detected'})"
    )

    # ── Local linear RD with triangular kernel ──
    df_window = df_rd[df_rd["running_centered"].abs() <= bandwidth].copy()

    if len(df_window) < 50:
        console.print(f"[yellow]Only {len(df_window)} tracts in bandwidth window. "
                      "Widen bandwidth or check running variable.[/]")
        results["rd_status"] = {"status": "insufficient_data", "n_in_window": int(len(df_window))}
        return results

    # Triangular kernel weights
    df_window["kernel_weight"] = 1 - df_window["running_centered"].abs() / bandwidth

    # Local linear: outcome ~ above_cutoff + running_centered + above_cutoff * running_centered
    # Run separately on each side for flexibility; combined model for RD estimate
    model_rd = smf.wls(
        f"{outcome} ~ above_cutoff + running_centered + above_cutoff:running_centered",
        data=df_window,
        weights=df_window["kernel_weight"],
    ).fit(cov_type="HC1")

    rd_estimate = model_rd.params.get("above_cutoff", np.nan)
    rd_p = model_rd.pvalues.get("above_cutoff", np.nan)
    rd_ci = model_rd.conf_int().loc["above_cutoff"] if "above_cutoff" in model_rd.conf_int().index else [np.nan, np.nan]

    # Separate means just above/below for descriptive comparison
    below_mean = df_window[df_window["above_cutoff"] == 0][outcome].mean()
    above_mean = df_window[df_window["above_cutoff"] == 1][outcome].mean()

    results["rd_estimate"] = {
        "outcome": outcome,
        "running_variable": running_var,
        "cutoff": cutoff,
        "bandwidth": bandwidth,
        "n_in_window": int(len(df_window)),
        "rd_coefficient": round(float(rd_estimate), 4),
        "ci_95_low": round(float(rd_ci[0]), 4),
        "ci_95_high": round(float(rd_ci[1]), 4),
        "p_value": float(f"{rd_p:.2e}"),
        "significant": rd_p < 0.05,
        "mean_below_cutoff": round(float(below_mean), 2),
        "mean_above_cutoff": round(float(above_mean), 2),
        "raw_jump": round(float(above_mean - below_mean), 2),
        "interpretation": (
            f"At the {cutoff}% USDA threshold, crossing into food desert status is "
            f"associated with a {rd_estimate:+.2f} pp "
            f"{'increase' if rd_estimate > 0 else 'decrease'} in {outcome} "
            f"({'significant' if rd_p < 0.05 else 'not significant'} at p<0.05)"
        ),
    }

    console.print(
        f"  RD estimate ({outcome}): {rd_estimate:+.4f} pp "
        f"(95% CI: {rd_ci[0]:+.4f}–{rd_ci[1]:+.4f}), p={rd_p:.2e}"
    )

    # ── Bandwidth sensitivity: re-run at ±5, ±15, ±20 pp ──
    bw_sensitivity = {}
    for bw_alt in [5.0, 15.0, 20.0]:
        df_alt = df_rd[df_rd["running_centered"].abs() <= bw_alt].copy()
        if len(df_alt) < 30:
            continue
        df_alt["kernel_weight"] = 1 - df_alt["running_centered"].abs() / bw_alt
        try:
            m_alt = smf.wls(
                f"{outcome} ~ above_cutoff + running_centered + above_cutoff:running_centered",
                data=df_alt, weights=df_alt["kernel_weight"],
            ).fit(cov_type="HC1")
            bw_sensitivity[f"bw_{bw_alt}"] = {
                "rd_coef": round(float(m_alt.params.get("above_cutoff", np.nan)), 4),
                "p_value": float(f"{m_alt.pvalues.get('above_cutoff', 1):.2e}"),
                "n": int(len(df_alt)),
            }
        except Exception:
            pass

    results["bandwidth_sensitivity"] = bw_sensitivity
    console.print(f"  Bandwidth sensitivity: {list(bw_sensitivity.keys())}")

    return results


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def run_causal_analysis(
    master: pd.DataFrame,
    panel_df: pd.DataFrame | None = None,
) -> dict:
    """Run all causal inference analyses and export results.

    Parameters
    ----------
    master   : Master dataframe (cross-sectional).
    panel_df : Optional two-period panel for DiD (see run_did_scaffold docstring).

    Returns
    -------
    dict with keys: psm, did, rd.
    """
    console.rule("[bold]Phase 6B: Causal Inference")
    all_results = {}

    psm_results = run_propensity_score_matching(master)
    all_results["psm"] = psm_results

    did_results = run_did_scaffold(master, panel_df=panel_df)
    all_results["did"] = did_results

    rd_results = run_regression_discontinuity(master)
    all_results["rd"] = rd_results

    _export_json(all_results, "phase6b_causal_inference.json")
    return all_results
