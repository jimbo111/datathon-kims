"""Formal Mediation Analysis — Phase 6C.

Implements Baron & Kenny (1986) four-step procedure plus bootstrap
confidence intervals for a single hypothesized causal pathway:

    poverty_rate → is_food_desert → obesity_pct → diabetes_pct

This represents the full socioeconomic deprivation cascade:
  Step A: poverty predicts food desert status (OLS on binary IV)
  Step B: poverty predicts diabetes (total effect, not via mediators)
  Step C: food desert + poverty predict obesity (mediator 1 model)
  Step D: obesity + food desert + poverty predict diabetes (outcome model)

Terminology
-----------
- Total effect   : effect of X on Y (ignoring mediators)
- Direct effect  : effect of X on Y after controlling for mediators
- Indirect effect: total - direct (= mediated portion)
- Proportion mediated: indirect / total × 100%

Sobel Test
----------
Tests H₀: indirect effect = 0.
    z = (a × b) / sqrt(b² × se_a² + a² × se_b²)
Conservative (underestimates indirect effect significance).

Bootstrap
---------
1000-iteration percentile bootstrap CI for indirect effects.
More accurate than Sobel for non-normal indirect effect distributions
(Preacher & Hayes 2008; MacKinnon et al. 2004).

Assumptions
-----------
1. No unmeasured confounders of X→M, X→Y, or M→Y relationships.
2. No measurement error in mediator.
3. No mediator-outcome confounders caused by the treatment (sequential ignorability).
4. Linear relationships (OLS used throughout).

All results exported to data/processed/phase6c_mediation.json.
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


def _ols_coef(formula: str, data: pd.DataFrame, var: str) -> tuple[float, float, float]:
    """Fit OLS and return (coefficient, std_error, p_value) for `var`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=data).fit(cov_type="HC1")
    coef = float(model.params.get(var, np.nan))
    se = float(model.bse.get(var, np.nan))
    p = float(model.pvalues.get(var, np.nan))
    return coef, se, p


# ─── Baron & Kenny four-step procedure ───────────────────────────────────────


def run_baron_kenny(
    master: pd.DataFrame,
    exposure: str = "poverty_rate",
    mediator1: str = "is_food_desert",
    mediator2: str = "obesity_pct",
    outcome: str = "diabetes_pct",
    covariates: list[str] | None = None,
) -> dict:
    """Baron & Kenny (1986) mediation analysis for the poverty → diabetes pathway.

    Tests four conditions for mediation (Baron & Kenny 1986):
    1. Exposure → outcome (total effect, path c): significant
    2. Exposure → mediator1 (path a1): significant
    3. Mediator1 + exposure → mediator2 (path a2): significant
    4. All predictors → outcome (path c'): direct effect of exposure attenuated

    For a two-mediator chain (X → M1 → M2 → Y), the analysis is conducted
    in stages following the sequential mediation framework
    (Taylor, MacKinnon & Tait 2008).

    Parameters
    ----------
    exposure   : Independent variable (X).
    mediator1  : First mediator (M1 in X → M1 → M2 → Y).
    mediator2  : Second mediator (M2).
    outcome    : Dependent variable (Y).
    covariates : Additional variables to include as covariates in all models.
                 Default: poverty_rate is already the exposure; no extra covariates.
    """
    console.rule("[bold]Mediation: Baron & Kenny Four-Step")
    results = {}

    if covariates is None:
        covariates = []

    required = [exposure, mediator1, mediator2, outcome]
    missing = [c for c in required if c not in master.columns]
    if missing:
        console.print(f"[red]Missing columns for mediation: {missing}[/]")
        return results

    all_cols = required + covariates
    df = master[all_cols].dropna()

    if len(df) < 200:
        console.print(f"[yellow]Only {len(df)} complete cases (need ≥200 for mediation).[/]")
        return results

    console.print(f"  Mediation sample: {len(df):,} tracts")
    console.print(f"  Pathway: {exposure} → {mediator1} → {mediator2} → {outcome}")

    cov_str = (" + " + " + ".join(covariates)) if covariates else ""

    # ── Step 1: Total effect X → Y (path c) ──
    c_total, c_se, c_p = _ols_coef(
        f"{outcome} ~ {exposure}{cov_str}", df, exposure
    )

    # ── Step 2: X → M1 (path a1) ──
    a1, a1_se, a1_p = _ols_coef(
        f"{mediator1} ~ {exposure}{cov_str}", df, exposure
    )

    # ── Step 3: M1 + X → M2 (path a2 = effect of M1 on M2 controlling for X) ──
    a2, a2_se, a2_p = _ols_coef(
        f"{mediator2} ~ {mediator1} + {exposure}{cov_str}", df, mediator1
    )

    # Also estimate direct X → M2 controlling for M1
    x_to_m2_controlling_m1, _, _ = _ols_coef(
        f"{mediator2} ~ {mediator1} + {exposure}{cov_str}", df, exposure
    )

    # ── Step 4: X + M1 + M2 → Y (paths c', b1, b2) ──
    c_prime, c_prime_se, c_prime_p = _ols_coef(
        f"{outcome} ~ {exposure} + {mediator1} + {mediator2}{cov_str}", df, exposure
    )
    b2, b2_se, b2_p = _ols_coef(
        f"{outcome} ~ {exposure} + {mediator1} + {mediator2}{cov_str}", df, mediator2
    )
    b1, b1_se, b1_p = _ols_coef(
        f"{outcome} ~ {exposure} + {mediator1} + {mediator2}{cov_str}", df, mediator1
    )

    # ── Baron & Kenny criteria ──
    bk_criteria = {
        "step1_c_significant": c_p < 0.05,
        "step2_a1_significant": a1_p < 0.05,
        "step3_a2_significant": a2_p < 0.05,
        "step4_c_prime_attenuated": abs(c_prime) < abs(c_total),
        "full_mediation": abs(c_prime) < abs(c_total) and c_prime_p >= 0.05,
        "partial_mediation": abs(c_prime) < abs(c_total) and c_prime_p < 0.05,
    }
    results["baron_kenny_steps"] = {
        "path_c_total": {"coef": round(c_total, 4), "se": round(c_se, 4), "p": float(f"{c_p:.2e}")},
        "path_a1_X_to_M1": {"coef": round(a1, 4), "se": round(a1_se, 4), "p": float(f"{a1_p:.2e}")},
        "path_a2_M1_to_M2": {"coef": round(a2, 4), "se": round(a2_se, 4), "p": float(f"{a2_p:.2e}")},
        "path_b2_M2_to_Y": {"coef": round(b2, 4), "se": round(b2_se, 4), "p": float(f"{b2_p:.2e}")},
        "path_b1_M1_to_Y": {"coef": round(b1, 4), "se": round(b1_se, 4), "p": float(f"{b1_p:.2e}")},
        "path_c_prime_direct": {"coef": round(c_prime, 4), "se": round(c_prime_se, 4), "p": float(f"{c_prime_p:.2e}")},
        "criteria": bk_criteria,
    }

    mediation_type = (
        "Full mediation" if bk_criteria["full_mediation"] else
        "Partial mediation" if bk_criteria["partial_mediation"] else
        "No mediation (direct effect persists, not attenuated)"
    )

    console.print(f"  Path c (total): β={c_total:.4f}, p={c_p:.2e}")
    console.print(f"  Path a1 (X→M1): β={a1:.4f}, p={a1_p:.2e}")
    console.print(f"  Path a2 (M1→M2): β={a2:.4f}, p={a2_p:.2e}")
    console.print(f"  Path b2 (M2→Y): β={b2:.4f}, p={b2_p:.2e}")
    console.print(f"  Path c' (direct): β={c_prime:.4f}, p={c_prime_p:.2e}")
    console.print(f"  Conclusion: [bold]{mediation_type}[/]")

    results["mediation_conclusion"] = mediation_type

    # ── Indirect effects (product-of-coefficients) ──
    # For X → M1 → M2 → Y:  indirect = a1 × a2 × b2
    # For X → M1 → Y:        indirect_m1_only = a1 × b1
    # For X → M2 → Y:        indirect_m2_only = x_to_m2 × b2
    indirect_full = a1 * a2 * b2
    indirect_m1_only = a1 * b1
    indirect_through_m2 = (c_total - c_prime)  # difference method

    results["indirect_effects"] = {
        "indirect_X_M1_M2_Y": round(indirect_full, 6),
        "indirect_X_M1_Y": round(indirect_m1_only, 6),
        "indirect_difference_method": round(indirect_through_m2, 6),
        "direct_effect": round(c_prime, 6),
        "total_effect": round(c_total, 6),
        "proportion_mediated_pct": round(
            abs(indirect_through_m2) / abs(c_total) * 100 if abs(c_total) > 1e-6 else 0, 2
        ),
    }

    return results


# ─── Sobel test ───────────────────────────────────────────────────────────────


def run_sobel_test(
    master: pd.DataFrame,
    exposure: str = "poverty_rate",
    mediator: str = "is_food_desert",
    outcome: str = "diabetes_pct",
    covariates: list[str] | None = None,
) -> dict:
    """Sobel (1982) test for the significance of a single mediated path.

    Tests H₀: a × b = 0 (indirect effect is zero).
    z = (a × b) / sqrt(b² × se_a² + a² × se_b²)

    Limitation: assumes normality of a × b sampling distribution,
    which is typically violated for small effects. Bootstrap CIs
    (run_bootstrap_mediation) are preferred but Sobel remains
    a useful supplementary check.

    Parameters
    ----------
    exposure  : Treatment variable X.
    mediator  : Single mediator M (test one path at a time).
    outcome   : Outcome Y.
    covariates: Additional covariates included in all models.
    """
    console.rule("[bold]Mediation: Sobel Test")
    results = {}

    if covariates is None:
        covariates = []

    required = [exposure, mediator, outcome]
    missing = [c for c in required if c not in master.columns]
    if missing:
        console.print(f"[red]Missing columns: {missing}[/]")
        return results

    df = master[required + covariates].dropna()
    cov_str = (" + " + " + ".join(covariates)) if covariates else ""

    # Path a: X → M
    a, a_se, a_p = _ols_coef(f"{mediator} ~ {exposure}{cov_str}", df, exposure)

    # Path b: M → Y (controlling for X)
    b, b_se, b_p = _ols_coef(
        f"{outcome} ~ {mediator} + {exposure}{cov_str}", df, mediator
    )

    # Sobel statistic
    indirect = a * b
    sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)

    if sobel_se == 0:
        console.print("[yellow]Sobel SE is zero — cannot compute test.[/]")
        return results

    z_sobel = indirect / sobel_se
    p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))

    results["sobel_test"] = {
        "pathway": f"{exposure} → {mediator} → {outcome}",
        "path_a_coef": round(a, 6),
        "path_a_se": round(a_se, 6),
        "path_b_coef": round(b, 6),
        "path_b_se": round(b_se, 6),
        "indirect_effect": round(indirect, 6),
        "sobel_se": round(sobel_se, 6),
        "z_statistic": round(z_sobel, 4),
        "p_value": float(f"{p_sobel:.2e}"),
        "significant": p_sobel < 0.05,
        "n": int(len(df)),
        "note": "Sobel test is conservative — use bootstrap CIs for inference",
    }

    console.print(
        f"  Sobel z={z_sobel:.4f}, p={p_sobel:.2e} — indirect={indirect:.6f} "
        f"({'significant' if p_sobel < 0.05 else 'not significant'})"
    )
    return results


# ─── Bootstrap mediation ──────────────────────────────────────────────────────


def run_bootstrap_mediation(
    master: pd.DataFrame,
    exposure: str = "poverty_rate",
    mediator1: str = "is_food_desert",
    mediator2: str = "obesity_pct",
    outcome: str = "diabetes_pct",
    covariates: list[str] | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for indirect effects (Preacher & Hayes 2008).

    Algorithm:
    1. Resample observations with replacement (n_bootstrap times).
    2. For each bootstrap sample, estimate all path coefficients via OLS.
    3. Compute indirect effects as products of path coefficients.
    4. Construct percentile CI from the bootstrap distribution.

    A CI that excludes zero indicates a significant indirect effect.
    This approach handles the skewed distribution of a×b products
    without relying on normality assumptions.

    Three indirect effects are estimated:
    - IE1: X → M1 → Y           (a1 × b1)
    - IE2: X → M2 → Y           (a2_direct × b2)
    - IE3: X → M1 → M2 → Y      (a1 × a2 × b2)  [sequential mediation]

    Parameters
    ----------
    n_bootstrap : Number of bootstrap resamples (1000 is standard; 5000 for publication).
    ci_level    : Confidence level for bootstrap percentile CIs (default: 0.95).
    """
    console.rule(f"[bold]Mediation: Bootstrap CIs (n={n_bootstrap:,})")
    results = {}

    if covariates is None:
        covariates = []

    required = [exposure, mediator1, mediator2, outcome]
    missing = [c for c in required if c not in master.columns]
    if missing:
        console.print(f"[red]Missing columns: {missing}[/]")
        return results

    all_cols = required + covariates
    df = master[all_cols].dropna().reset_index(drop=True)

    if len(df) < 200:
        console.print(f"[yellow]Only {len(df)} complete cases.[/]")
        return results

    cov_str = (" + " + " + ".join(covariates)) if covariates else ""

    console.print(f"  Bootstrap sample: {len(df):,} tracts, {n_bootstrap:,} resamples…")

    # Point estimates on full data
    a1, a1_se, _ = _ols_coef(f"{mediator1} ~ {exposure}{cov_str}", df, exposure)
    a2, a2_se, _ = _ols_coef(
        f"{mediator2} ~ {mediator1} + {exposure}{cov_str}", df, mediator1
    )
    a2x, _, _ = _ols_coef(
        f"{mediator2} ~ {mediator1} + {exposure}{cov_str}", df, exposure
    )
    b1, b1_se, _ = _ols_coef(
        f"{outcome} ~ {mediator1} + {exposure}{cov_str}", df, mediator1
    )
    b2, b2_se, _ = _ols_coef(
        f"{outcome} ~ {mediator1} + {mediator2} + {exposure}{cov_str}", df, mediator2
    )
    c_total, _, _ = _ols_coef(f"{outcome} ~ {exposure}{cov_str}", df, exposure)
    c_prime, _, _ = _ols_coef(
        f"{outcome} ~ {mediator1} + {mediator2} + {exposure}{cov_str}", df, exposure
    )

    ie1_point = a1 * b1
    ie2_point = a2x * b2
    ie3_point = a1 * a2 * b2

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_ie1, boot_ie2, boot_ie3 = [], [], []
    boot_total, boot_direct = [], []

    for i in range(n_bootstrap):
        if (i + 1) % 250 == 0:
            console.print(f"  Bootstrap iteration {i+1}/{n_bootstrap}…", end="\r")

        idx = rng.integers(0, len(df), size=len(df))
        df_b = df.iloc[idx].reset_index(drop=True)

        try:
            ba1, _, _ = _ols_coef(f"{mediator1} ~ {exposure}{cov_str}", df_b, exposure)
            ba2, _, _ = _ols_coef(
                f"{mediator2} ~ {mediator1} + {exposure}{cov_str}", df_b, mediator1
            )
            ba2x, _, _ = _ols_coef(
                f"{mediator2} ~ {mediator1} + {exposure}{cov_str}", df_b, exposure
            )
            bb1, _, _ = _ols_coef(
                f"{outcome} ~ {mediator1} + {exposure}{cov_str}", df_b, mediator1
            )
            bb2, _, _ = _ols_coef(
                f"{outcome} ~ {mediator1} + {mediator2} + {exposure}{cov_str}", df_b, mediator2
            )
            bc_total, _, _ = _ols_coef(f"{outcome} ~ {exposure}{cov_str}", df_b, exposure)
            bc_prime, _, _ = _ols_coef(
                f"{outcome} ~ {mediator1} + {mediator2} + {exposure}{cov_str}", df_b, exposure
            )

            boot_ie1.append(ba1 * bb1)
            boot_ie2.append(ba2x * bb2)
            boot_ie3.append(ba1 * ba2 * bb2)
            boot_total.append(bc_total)
            boot_direct.append(bc_prime)
        except Exception:
            continue

    console.print()  # newline after carriage-return progress

    alpha = 1 - ci_level
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    def _ci(boot_vals: list, point: float) -> dict:
        arr = np.array(boot_vals)
        return {
            "point_estimate": round(point, 6),
            "ci_low": round(float(np.percentile(arr, lo)), 6),
            "ci_high": round(float(np.percentile(arr, hi)), 6),
            "significant": not (np.percentile(arr, lo) <= 0 <= np.percentile(arr, hi)),
            "n_successful_boot": len(boot_vals),
        }

    results["bootstrap_indirect_effects"] = {
        "IE1_X_M1_Y": _ci(boot_ie1, ie1_point),
        "IE2_X_M2_Y": _ci(boot_ie2, ie2_point),
        "IE3_X_M1_M2_Y": _ci(boot_ie3, ie3_point),
        "total_effect": _ci(boot_total, c_total),
        "direct_effect": _ci(boot_direct, c_prime),
        "proportion_mediated_pct": round(
            abs(ie1_point + ie3_point) / abs(c_total) * 100
            if abs(c_total) > 1e-6 else 0, 2
        ),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "pathway": f"{exposure} → [{mediator1} → {mediator2}] → {outcome}",
    }

    for name, est in results["bootstrap_indirect_effects"].items():
        if isinstance(est, dict) and "point_estimate" in est:
            sig = "SIGNIFICANT" if est["significant"] else "not significant"
            console.print(
                f"  {name}: {est['point_estimate']:.6f} "
                f"({ci_level*100:.0f}% CI: {est['ci_low']:.6f}–{est['ci_high']:.6f}) [{sig}]"
            )

    return results


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def run_mediation_analysis(master: pd.DataFrame) -> dict:
    """Run full mediation analysis pipeline and export results.

    Pathway analyzed: poverty_rate → is_food_desert → obesity_pct → diabetes_pct

    Returns dict with Baron & Kenny steps, Sobel test, and bootstrap CIs.
    """
    console.rule("[bold]Phase 6C: Formal Mediation Analysis")
    all_results = {}

    bk_results = run_baron_kenny(master)
    all_results["baron_kenny"] = bk_results

    sobel_results = run_sobel_test(master)
    all_results["sobel"] = sobel_results

    boot_results = run_bootstrap_mediation(master, n_bootstrap=1000)
    all_results["bootstrap"] = boot_results

    # Summary table for reporting
    if "bootstrap" in all_results and "bootstrap_indirect_effects" in all_results["bootstrap"]:
        ies = all_results["bootstrap"]["bootstrap_indirect_effects"]
        summary_rows = []
        for name, vals in ies.items():
            if isinstance(vals, dict) and "point_estimate" in vals:
                summary_rows.append({
                    "effect": name,
                    "estimate": vals["point_estimate"],
                    "ci_low": vals["ci_low"],
                    "ci_high": vals["ci_high"],
                    "significant": vals["significant"],
                })
        all_results["summary_table"] = summary_rows

    _export_json(all_results, "phase6c_mediation.json")
    return all_results
