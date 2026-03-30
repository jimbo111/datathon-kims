"""Robustness Checks Module — Phase 6E.

Validates that the core findings from Phases 2–4 are not artifacts of:
- Multiple testing (Benjamini-Hochberg FDR correction)
- Unweighted estimation ignoring tract population heterogeneity
- A specific food desert definition (USDA offers 4+ variants)
- Geographic clustering / influential states (leave-one-state-out CV)
- Sampling variability (bootstrap CIs on key OLS estimates)

All robustness checks target the primary finding:
    is_food_desert → diabetes_pct (β > 0, p < 0.05)

Checks run
----------
1. BH FDR correction: collect all p-values from OLS models, apply
   Benjamini-Hochberg (1995) correction at q=0.05.  Compare number
   of discoveries pre- and post-correction.

2. Population-weighted OLS: re-run Phase 2 OLS with WLS using the
   `population` column as analytic weights.  Downweights tiny tracts
   that may be noisy; upweights large urban tracts.

3. Food desert definition sensitivity: four USDA definitions:
   - food_desert_1_10   : ≥1 mi urban / ≥10 mi rural + low income (standard)
   - food_desert_half_10: ≥0.5 mi urban / ≥10 mi rural + low income (strict urban)
   - food_desert_1_20   : ≥1 mi urban / ≥20 mi rural + low income (strict rural)
   - food_desert_vehicle: vehicle access definition (poverty + no car)
   Coefficient and p-value for each definition are compared.

4. Leave-one-state-out (LOSO) CV: re-run OLS on all tracts EXCEPT
   each state in turn.  Checks whether any single state drives the
   finding (influential data concern).

5. Bootstrap CIs: 1000-iteration cluster bootstrap (clusters = states)
   for the food_desert coefficient in the primary OLS model.  Cluster
   bootstrap is appropriate because errors are likely correlated within
   states (due to shared state-level policy, climate, etc.).

All results exported to data/processed/phase6e_robustness.json.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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


def _ols_food_desert_coef(
    df: pd.DataFrame,
    fd_col: str = "is_food_desert",
    outcome: str = "diabetes_pct",
    controls: list[str] | None = None,
    weights: pd.Series | None = None,
) -> dict:
    """Fit OLS (or WLS) and return coefficient info for fd_col.

    Returns dict with coef, std_err, p_value, r_squared, n, or None on failure.
    """
    if controls is None:
        controls = ["poverty_rate", "uninsured_pct"]

    available = [c for c in controls if c in df.columns]
    formula = f"{outcome} ~ {fd_col} + " + " + ".join(available) if available else f"{outcome} ~ {fd_col}"

    cols_needed = [outcome, fd_col] + available
    df_fit = df[cols_needed].dropna()
    if len(df_fit) < 50:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if weights is not None:
                w = weights.loc[df_fit.index].fillna(weights.median())
                model = smf.wls(formula, data=df_fit, weights=w).fit(cov_type="HC1")
            else:
                model = smf.ols(formula, data=df_fit).fit(cov_type="HC1")
    except Exception as exc:
        return {"error": str(exc)}

    coef = model.params.get(fd_col, np.nan)
    p = model.pvalues.get(fd_col, np.nan)
    se = model.bse.get(fd_col, np.nan)
    ci = model.conf_int().loc[fd_col] if fd_col in model.conf_int().index else [np.nan, np.nan]

    return {
        "coef": round(float(coef), 4),
        "std_err": round(float(se), 4),
        "p_value": float(f"{p:.2e}"),
        "ci_95_low": round(float(ci[0]), 4),
        "ci_95_high": round(float(ci[1]), 4),
        "r_squared": round(float(model.rsquared), 4),
        "n": int(model.nobs),
        "significant": p < 0.05,
    }


# ─── 1. BH FDR correction ─────────────────────────────────────────────────────


def run_fdr_correction(
    phase_results: dict | None = None,
    master: pd.DataFrame | None = None,
    alpha_q: float = 0.05,
) -> dict:
    """Benjamini-Hochberg (1995) FDR correction across all OLS p-values.

    Collects all food-desert / key predictor p-values from Phase 2–4 models,
    applies BH correction at q=0.05, and reports how many remain significant.

    Under BH, the expected proportion of false discoveries among rejections
    is controlled at q=0.05 (less conservative than Bonferroni).

    Parameters
    ----------
    phase_results : Dict with keys 'phase2', 'phase3', 'phase4' and their
                    model outputs (from run_all_phases).  If None, p-values are
                    extracted by re-running key OLS models on master.
    master        : Required if phase_results is None.
    alpha_q       : FDR threshold (default 0.05 = 5% false discovery rate).
    """
    console.rule("[bold]Robustness: Benjamini-Hochberg FDR Correction")
    results = {}

    # Collect p-values from phase results or re-run models
    p_values = {}

    if phase_results is not None:
        # Extract from pre-computed results
        for phase_name, phase_dict in phase_results.items():
            if not isinstance(phase_dict, dict):
                continue
            for model_name, model_dict in phase_dict.items():
                if not isinstance(model_dict, dict):
                    continue
                # OLS model format (from _ols_summary)
                if "coefficients" in model_dict:
                    for var, var_dict in model_dict["coefficients"].items():
                        if var == "Intercept":
                            continue
                        key = f"{phase_name}.{model_name}.{var}"
                        p_values[key] = float(var_dict.get("p_value", 1.0))
                # Direct p-value fields
                for pkey in ["p_value", "pct_black_p_value"]:
                    if pkey in model_dict:
                        key = f"{phase_name}.{model_name}.{pkey}"
                        p_values[key] = float(model_dict[pkey])

    elif master is not None:
        # Re-run Phase 2 core models and extract p-values
        core_models = [
            ("diabetes_ols", "diabetes_pct", "is_food_desert"),
            ("obesity_ols", "obesity_pct", "is_food_desert"),
            ("le_ols", "life_expectancy", "is_food_desert"),
        ]
        for model_name, outcome, treatment in core_models:
            if outcome not in master.columns or treatment not in master.columns:
                continue
            res = _ols_food_desert_coef(master, fd_col=treatment, outcome=outcome)
            if res and "p_value" in res:
                p_values[f"{model_name}.{treatment}"] = res["p_value"]
                # Add control variable p-values from a fuller model
                controls = ["poverty_rate", "uninsured_pct", "pct_black", "pct_hispanic"]
                avail = [c for c in controls if c in master.columns]
                cols = [outcome, treatment] + avail
                df_fit = master[cols].dropna()
                if len(df_fit) > 100:
                    try:
                        formula = f"{outcome} ~ {treatment} + " + " + ".join(avail)
                        m = smf.ols(formula, data=df_fit).fit(cov_type="HC1")
                        for var in m.params.index:
                            if var != "Intercept":
                                p_values[f"{model_name}.{var}"] = float(m.pvalues[var])
                    except Exception:
                        pass
    else:
        console.print("[yellow]Provide phase_results or master to run FDR correction.[/]")
        return results

    if not p_values:
        console.print("[yellow]No p-values collected for FDR correction.[/]")
        return results

    # ── Apply BH correction ──
    tests = sorted(p_values.items(), key=lambda x: x[1])
    m_tests = len(tests)
    bh_threshold = []
    for i, (name, p) in enumerate(tests, 1):
        bh_threshold.append(p <= (i / m_tests) * alpha_q)

    # Find the largest k such that all tests 1…k are rejected
    rejected = []
    for i in range(m_tests - 1, -1, -1):
        if bh_threshold[i]:
            rejected = [tests[j][0] for j in range(i + 1)]
            break

    n_rejected = len(rejected)
    n_total = m_tests

    results["fdr_correction"] = {
        "method": "Benjamini-Hochberg (1995)",
        "fdr_threshold_q": alpha_q,
        "n_tests": n_total,
        "n_significant_uncorrected": sum(1 for _, p in tests if p < 0.05),
        "n_significant_bh": n_rejected,
        "pct_retained": round(n_rejected / max(n_total, 1) * 100, 1),
        "food_desert_survived_fdr": any("is_food_desert" in t for t in rejected),
        "rejected_tests": rejected[:50],  # cap for JSON size
        "all_p_values": {k: round(v, 4) for k, v in tests[:100]},
    }

    console.print(
        f"  BH FDR: {n_total} tests → {n_rejected} survive at q={alpha_q} "
        f"({'is_food_desert retained' if results['fdr_correction']['food_desert_survived_fdr'] else 'is_food_desert DROPPED'})"
    )

    return results


# ─── 2. Population-weighted OLS ───────────────────────────────────────────────


def run_population_weighted_ols(master: pd.DataFrame) -> dict:
    """WLS regression using population as analytic weights.

    Unweighted OLS treats every census tract equally regardless of population.
    Large urban tracts (pop > 10k) carry the same weight as rural tracts
    (pop < 500).  Population weighting lets outcomes from larger tracts
    dominate, which is appropriate when estimating the effect on the
    average American (rather than the average tract).

    Compares WLS vs OLS coefficients for is_food_desert.
    A large difference suggests the effect is concentrated in
    small/large tracts.
    """
    console.rule("[bold]Robustness: Population-Weighted OLS")
    results = {}

    if "population" not in master.columns:
        console.print("[yellow]'population' column not found. Skipping weighted OLS.[/]")
        return results

    outcomes = [c for c in ["diabetes_pct", "obesity_pct", "life_expectancy"] if c in master.columns]
    controls = ["poverty_rate", "uninsured_pct"]

    wls_results = {}
    ols_results_plain = {}

    for outcome in outcomes:
        # OLS (unweighted)
        ols_r = _ols_food_desert_coef(master, outcome=outcome, controls=controls)

        # WLS (population-weighted)
        wls_r = _ols_food_desert_coef(
            master, outcome=outcome, controls=controls,
            weights=master["population"].clip(lower=1),
        )

        if ols_r and wls_r:
            ols_results_plain[outcome] = ols_r
            wls_results[outcome] = wls_r

            coef_diff = abs(wls_r["coef"] - ols_r["coef"])
            console.print(
                f"  {outcome}: OLS β={ols_r['coef']:.4f} p={ols_r['p_value']}, "
                f"WLS β={wls_r['coef']:.4f} p={wls_r['p_value']} "
                f"(diff={coef_diff:.4f})"
            )

    results["population_weighted_ols"] = {
        "wls_results": wls_results,
        "ols_results": ols_results_plain,
        "interpretation": (
            "Large WLS–OLS differences would suggest the food desert effect "
            "is concentrated in unusually small or large tracts."
        ),
    }

    return results


# ─── 3. Food desert definition sensitivity ────────────────────────────────────


def run_definition_sensitivity(
    master: pd.DataFrame,
    outcome: str = "diabetes_pct",
) -> dict:
    """Compare food desert β across four USDA access definitions.

    USDA provides multiple definitions for 'low food access':
    - food_desert_1_10   : >1 mi (urban) or >10 mi (rural) + low-income tract [standard]
    - food_desert_half_10: >0.5 mi (urban) or >10 mi (rural) + low-income [stricter urban]
    - food_desert_1_20   : >1 mi (urban) or >20 mi (rural) + low-income [stricter rural]
    - food_desert_vehicle: low-income + low vehicle access (population-based)

    If the effect is robust, all four definitions should yield a
    positive, significant coefficient.  Unstable or sign-reversing
    coefficients would raise concern about construct validity.
    """
    console.rule("[bold]Robustness: Food Desert Definition Sensitivity")
    results = {}

    # Prefer the canonical column names from USDA Atlas
    fd_definitions = {
        "standard_1mi_10mi": "food_desert_1_10",
        "strict_urban_half_mi": "food_desert_half_10",
        "strict_rural_20mi": "food_desert_1_20",
        "vehicle_access": "food_desert_vehicle",
    }

    # Fallback: if columns are named differently (dataset-specific)
    fd_fallbacks = {
        "standard_1mi_10mi": ["LILATracts_1And10", "li_1_and_10", "food_desert"],
        "strict_urban_half_mi": ["LILATracts_halfAnd10", "li_half_and_10"],
        "strict_rural_20mi": ["LILATracts_1And20", "li_1_and_20"],
        "vehicle_access": ["LILATracts_Vehicle", "li_vehicle"],
    }

    # Resolve available column names
    resolved = {}
    for label, primary_col in fd_definitions.items():
        if primary_col in master.columns:
            resolved[label] = primary_col
        else:
            for fallback in fd_fallbacks.get(label, []):
                if fallback in master.columns:
                    resolved[label] = fallback
                    break

    if not resolved:
        console.print(
            "[yellow]No food desert variant columns found in master. "
            "USDA Atlas data may only include the standard definition.[/]"
        )
        results["definition_sensitivity"] = {
            "status": "skipped",
            "reason": "No alternative food desert definition columns available",
            "available_columns": [c for c in master.columns if "desert" in c.lower() or "lila" in c.lower()],
        }
        return results

    sensitivity_results = {}
    for label, col in resolved.items():
        df_def = master[[col, outcome, "poverty_rate", "uninsured_pct"]].copy()
        df_def[col] = pd.to_numeric(df_def[col], errors="coerce")
        res = _ols_food_desert_coef(df_def, fd_col=col, outcome=outcome)
        if res:
            sensitivity_results[label] = {**res, "column_used": col}
            console.print(
                f"  [{label}] ({col}): β={res['coef']:.4f}, "
                f"p={res['p_value']}, n={res['n']}"
            )

    # Consistency assessment
    significant_count = sum(1 for r in sensitivity_results.values() if r.get("significant"))
    coefs = [r["coef"] for r in sensitivity_results.values() if "coef" in r]
    results["definition_sensitivity"] = {
        "results_by_definition": sensitivity_results,
        "n_definitions_tested": len(sensitivity_results),
        "n_significant": significant_count,
        "coef_range": [round(min(coefs), 4), round(max(coefs), 4)] if coefs else [],
        "coef_mean": round(float(np.mean(coefs)), 4) if coefs else None,
        "finding_robust": significant_count == len(sensitivity_results),
        "interpretation": (
            "Finding is consistent across all tested USDA food desert definitions"
            if significant_count == len(sensitivity_results)
            else f"Only {significant_count}/{len(sensitivity_results)} definitions yield significant results"
        ),
    }

    return results


# ─── 4. Leave-one-state-out cross-validation ──────────────────────────────────


def run_loso_cv(
    master: pd.DataFrame,
    outcome: str = "diabetes_pct",
    fd_col: str = "is_food_desert",
) -> dict:
    """Leave-one-state-out cross-validation for OLS coefficient stability.

    For each of the 50 states + DC, re-run the primary OLS on all tracts
    EXCEPT that state.  If the coefficient is stable across all 51 leave-one-out
    models, the finding is not driven by any single state.

    Reports:
    - Distribution of leave-one-out coefficients
    - States whose exclusion produces the largest coefficient change
    - Whether is_food_desert is significant in all 51 leave-one-out models

    Requires a state identifier column.  Uses the first 2 characters of
    tract_fips (state FIPS code) if no dedicated state column exists.
    """
    console.rule("[bold]Robustness: Leave-One-State-Out CV")
    results = {}

    # Derive state column from FIPS
    df_loso = master.copy()
    if "state_fips" not in df_loso.columns:
        if "tract_fips" in df_loso.columns:
            df_loso["state_fips"] = df_loso["tract_fips"].astype(str).str[:2]
        elif "state" in df_loso.columns:
            df_loso["state_fips"] = df_loso["state"]
        else:
            console.print("[yellow]No state identifier available for LOSO CV.[/]")
            return results

    states = df_loso["state_fips"].dropna().unique()
    if len(states) < 5:
        console.print(f"[yellow]Only {len(states)} states found. LOSO requires multi-state data.[/]")
        return results

    console.print(f"  LOSO: running {len(states)} leave-one-out models…")

    loso_results = {}
    for state in sorted(states):
        df_leave = df_loso[df_loso["state_fips"] != state]
        res = _ols_food_desert_coef(df_leave, fd_col=fd_col, outcome=outcome)
        if res and "coef" in res:
            loso_results[state] = res

    if not loso_results:
        console.print("[yellow]No LOSO models completed.[/]")
        return results

    # Full-sample baseline
    full_res = _ols_food_desert_coef(master, fd_col=fd_col, outcome=outcome)

    coefs = [r["coef"] for r in loso_results.values()]
    sig_count = sum(1 for r in loso_results.values() if r.get("significant"))

    results["loso_cv"] = {
        "n_states": len(loso_results),
        "full_sample_coef": round(full_res["coef"], 4) if full_res else None,
        "loso_coef_mean": round(float(np.mean(coefs)), 4),
        "loso_coef_std": round(float(np.std(coefs)), 4),
        "loso_coef_min": round(float(np.min(coefs)), 4),
        "loso_coef_max": round(float(np.max(coefs)), 4),
        "pct_significant": round(sig_count / len(loso_results) * 100, 1),
        "all_significant": sig_count == len(loso_results),
        "coefficient_sign_stable": all(c > 0 for c in coefs) or all(c < 0 for c in coefs),
    }

    # Most influential states (largest deviation from full-sample coef)
    if full_res:
        deviations = {
            state: abs(res["coef"] - full_res["coef"])
            for state, res in loso_results.items()
        }
        top_influential = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:10]
        results["loso_cv"]["most_influential_states"] = [
            {
                "state_fips": state,
                "loso_coef": round(loso_results[state]["coef"], 4),
                "deviation_from_full": round(dev, 4),
            }
            for state, dev in top_influential
        ]

    console.print(
        f"  LOSO coef range: [{np.min(coefs):.4f}, {np.max(coefs):.4f}], "
        f"all significant: {results['loso_cv']['all_significant']}, "
        f"sign stable: {results['loso_cv']['coefficient_sign_stable']}"
    )

    return results


# ─── 5. Bootstrap confidence intervals (cluster bootstrap) ────────────────────


def run_bootstrap_ols(
    master: pd.DataFrame,
    outcome: str = "diabetes_pct",
    fd_col: str = "is_food_desert",
    n_bootstrap: int = 1000,
    cluster_col: str | None = None,
) -> dict:
    """Bootstrap CIs for the food desert OLS coefficient.

    Uses a cluster bootstrap (resampling entire states/counties) when
    cluster_col is specified.  Cluster bootstrapping accounts for
    within-cluster correlation in errors, which OLS HC1 standard errors
    may underestimate in geographically clustered data.

    Parameters
    ----------
    n_bootstrap : Number of bootstrap resamples.
    cluster_col : Column to use as the cluster unit.  If None, defaults to
                  first 5 digits of tract_fips (county FIPS) for county-level
                  clustering, which is a common choice for census-tract data.
    """
    console.rule(f"[bold]Robustness: Cluster Bootstrap CIs (n={n_bootstrap:,})")
    results = {}

    df_boot = master.copy()

    # Set cluster column
    if cluster_col is None:
        if "tract_fips" in df_boot.columns:
            df_boot["_cluster"] = df_boot["tract_fips"].astype(str).str[:5]  # county
        else:
            console.print("[yellow]No cluster column available. Falling back to IID bootstrap.[/]")
            df_boot["_cluster"] = np.arange(len(df_boot))
    else:
        df_boot["_cluster"] = df_boot[cluster_col]

    controls = ["poverty_rate", "uninsured_pct"]
    available = [c for c in controls if c in df_boot.columns]
    required_cols = [outcome, fd_col, "_cluster"] + available
    df_boot = df_boot[required_cols].dropna()

    clusters = df_boot["_cluster"].unique()
    n_clusters = len(clusters)

    if n_clusters < 20:
        console.print(f"[yellow]Only {n_clusters} clusters — bootstrap may be unstable.[/]")

    console.print(f"  Bootstrap: {len(df_boot):,} tracts, {n_clusters:,} clusters, {n_bootstrap:,} resamples…")

    # Point estimate
    full_res = _ols_food_desert_coef(
        df_boot.drop(columns=["_cluster"]), fd_col=fd_col, outcome=outcome, controls=available
    )
    point_coef = full_res["coef"] if full_res else np.nan

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_coefs = []

    for i in range(n_bootstrap):
        if (i + 1) % 250 == 0:
            console.print(f"  Bootstrap iteration {i+1}/{n_bootstrap}…", end="\r")

        boot_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        dfs = [df_boot[df_boot["_cluster"] == c] for c in boot_clusters]
        df_b = pd.concat(dfs, ignore_index=True)

        res = _ols_food_desert_coef(
            df_b.drop(columns=["_cluster"]),
            fd_col=fd_col, outcome=outcome, controls=available,
        )
        if res and "coef" in res:
            boot_coefs.append(res["coef"])

    console.print()  # newline after carriage-return

    if len(boot_coefs) < 100:
        console.print("[yellow]Fewer than 100 successful bootstrap iterations.[/]")
        return results

    boot_arr = np.array(boot_coefs)
    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))

    results["bootstrap_ols"] = {
        "outcome": outcome,
        "predictor": fd_col,
        "point_estimate": round(point_coef, 6),
        "ci_95_low": round(ci_low, 6),
        "ci_95_high": round(ci_high, 6),
        "significant": not (ci_low <= 0 <= ci_high),
        "n_bootstrap": n_bootstrap,
        "n_successful": len(boot_coefs),
        "n_clusters": int(n_clusters),
        "cluster_type": "county (first 5 FIPS digits)",
        "boot_coef_mean": round(float(boot_arr.mean()), 6),
        "boot_coef_std": round(float(boot_arr.std()), 6),
        "bias": round(float(boot_arr.mean() - point_coef), 6),
    }

    console.print(
        f"  Cluster bootstrap: β={point_coef:.4f}, "
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}] "
        f"({'significant' if not (ci_low <= 0 <= ci_high) else 'NOT significant'})"
    )

    return results


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def run_robustness_checks(
    master: pd.DataFrame,
    phase_results: dict | None = None,
) -> dict:
    """Run all robustness checks and export results.

    Parameters
    ----------
    master        : Master dataframe.
    phase_results : Optional dict from run_all_phases (for BH FDR correction).

    Returns
    -------
    dict with keys: fdr, weighted_ols, definition_sensitivity, loso, bootstrap.
    """
    console.rule("[bold]Phase 6E: Robustness Checks")
    all_results = {}

    fdr_results = run_fdr_correction(phase_results=phase_results, master=master)
    all_results["fdr"] = fdr_results

    weighted_results = run_population_weighted_ols(master)
    all_results["weighted_ols"] = weighted_results

    sensitivity_results = run_definition_sensitivity(master)
    all_results["definition_sensitivity"] = sensitivity_results

    loso_results = run_loso_cv(master)
    all_results["loso"] = loso_results

    bootstrap_results = run_bootstrap_ols(master, n_bootstrap=1000)
    all_results["bootstrap"] = bootstrap_results

    # ── Robustness summary ──
    summary = {
        "bh_fdr_survived": all_results["fdr"].get("fdr_correction", {}).get("food_desert_survived_fdr", None),
        "weighted_significant": (
            all_results["weighted_ols"]
            .get("population_weighted_ols", {})
            .get("wls_results", {})
            .get("diabetes_pct", {})
            .get("significant", None)
        ),
        "definition_robust": all_results["definition_sensitivity"].get("definition_sensitivity", {}).get("finding_robust", None),
        "loso_all_significant": all_results["loso"].get("loso_cv", {}).get("all_significant", None),
        "bootstrap_significant": all_results["bootstrap"].get("bootstrap_ols", {}).get("significant", None),
    }
    n_passed = sum(1 for v in summary.values() if v is True)
    n_total = sum(1 for v in summary.values() if v is not None)
    summary["robustness_score"] = f"{n_passed}/{n_total} checks passed"
    all_results["robustness_summary"] = summary

    console.print("[bold cyan]Robustness summary:[/]")
    for check, passed in summary.items():
        icon = "[green]PASS[/]" if passed is True else "[red]FAIL[/]" if passed is False else "[yellow]N/A[/]"
        console.print(f"  {check}: {icon}")

    _export_json(all_results, "phase6e_robustness.json")
    return all_results
