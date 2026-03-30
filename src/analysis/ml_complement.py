"""ML Complement Module — Phase 6D.

Complements the OLS/causal analysis with machine learning approaches:

1. XGBoost prediction of diabetes_pct
   - Captures nonlinear relationships and interactions that OLS misses
   - Tuned via 5-fold cross-validation with early stopping

2. SHAP (SHapley Additive exPlanations)
   - Feature importance that respects interactions (vs naive permutation importance)
   - SHAP summary data exported for visualization

3. ML vs OLS feature ranking comparison
   - Check whether ML corroborates OLS finding that food desert matters
     above and beyond income

4. K-means tract archetypes
   - Cluster 27k tracts into 3–5 profiles (e.g., "wealthy healthy", "poor rural desert",
     "urban minority", "suburban affluent")
   - Silhouette-based optimal k selection

5. Counterfactual simulation
   - For each food-desert tract: what would the XGBoost model predict
     if is_food_desert were flipped to 0?
   - Quantifies the "avoidable" diabetes burden attributable to food access

Packages required: xgboost, shap, sklearn
All results exported to data/processed/phase6d_ml_complement.json.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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


# ─── Feature set ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "is_food_desert",
    "poverty_rate",
    "median_household_income",
    "uninsured_pct",
    "pct_black",
    "pct_hispanic",
    "pct_white",
    "pct_bachelors_plus",
    "hpsa_shortage",
    "population",
    # NOTE: obesity_pct excluded — only ~15% of tracts have it, which would
    # reduce the training set from ~22K to ~4K tracts via dropna().
    # Use obesity_pct only when it is the target variable, not a feature.
]


# ─── XGBoost model ───────────────────────────────────────────────────────────


def run_xgboost_diabetes(
    master: pd.DataFrame,
    outcome: str = "diabetes_pct",
    n_cv_folds: int = 5,
) -> tuple[dict, object, list[str], pd.DataFrame]:
    """Train XGBoost regressor for diabetes_pct with 5-fold CV.

    Model details
    -------------
    - Objective: reg:squarederror
    - Hyperparameters: n_estimators=500 with early stopping (patience=30)
      on held-out fold validation set
    - Learning rate: 0.05 (conservative to avoid overfitting 27k rows)
    - max_depth=5, subsample=0.8, colsample_bytree=0.8

    Cross-validation
    ----------------
    5-fold CV reports mean and std of RMSE and R².
    Final model is trained on the full dataset for SHAP and counterfactuals.

    Returns
    -------
    (results_dict, fitted_model, feature_names, df_clean)
    """
    import xgboost as xgb

    console.rule(f"[bold]ML: XGBoost ({outcome})")
    results = {}

    available_features = [c for c in FEATURE_COLS if c in master.columns and c != outcome]
    if outcome not in master.columns:
        console.print(f"[red]Outcome '{outcome}' not found.[/]")
        return results, None, [], pd.DataFrame()

    if not available_features:
        console.print("[red]No feature columns available.[/]")
        return results, None, [], pd.DataFrame()

    df_clean = master[[outcome] + available_features].dropna().reset_index(drop=True)
    if len(df_clean) < 500:
        console.print(f"[yellow]Only {len(df_clean)} complete rows.[/]")
        return results, None, [], pd.DataFrame()

    X = df_clean[available_features].values.astype(np.float32)
    y = df_clean[outcome].values.astype(np.float32)

    console.print(f"  XGBoost dataset: {len(df_clean):,} tracts × {len(available_features)} features")

    # ── 5-fold cross-validation ──
    kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    cv_rmse, cv_r2 = [], []

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
        "verbosity": 0,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_cv = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=30,
            eval_metric="rmse",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_cv.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        preds = model_cv.predict(X_val)
        fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        fold_r2 = float(r2_score(y_val, preds))
        cv_rmse.append(fold_rmse)
        cv_r2.append(fold_r2)
        console.print(f"  Fold {fold}: RMSE={fold_rmse:.4f}, R²={fold_r2:.4f}")

    results["cv_performance"] = {
        "n_folds": n_cv_folds,
        "rmse_mean": round(float(np.mean(cv_rmse)), 4),
        "rmse_std": round(float(np.std(cv_rmse)), 4),
        "r2_mean": round(float(np.mean(cv_r2)), 4),
        "r2_std": round(float(np.std(cv_r2)), 4),
        "outcome": outcome,
        "n_features": len(available_features),
        "n_observations": len(df_clean),
    }
    console.print(
        f"  CV summary: RMSE={np.mean(cv_rmse):.4f}±{np.std(cv_rmse):.4f}, "
        f"R²={np.mean(cv_r2):.4f}±{np.std(cv_r2):.4f}"
    )

    # ── Final model on full data ──
    # Pass feature names so XGBoost uses them in importance dicts (avoids f0/f1 keys)
    df_X = pd.DataFrame(X, columns=available_features)
    console.print("  Fitting final model on full dataset…")
    final_model = xgb.XGBRegressor(**params, verbosity=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(df_X, y, verbose=False)

    # Built-in feature importances (gain) — keys are feature names when DataFrame used
    importances = final_model.get_booster().get_score(importance_type="gain")
    # Map from f0/f1… fallback to actual names if needed
    fi_named = {}
    for k, v in importances.items():
        if k in available_features:
            fi_named[k] = round(float(v), 2)
        elif k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(available_features):
            fi_named[available_features[int(k[1:])]] = round(float(v), 2)
        else:
            fi_named[k] = round(float(v), 2)

    results["xgb_feature_importance_gain"] = dict(
        sorted(fi_named.items(), key=lambda x: x[1], reverse=True)
    )

    console.print("  Top features (gain):")
    for feat, imp in list(results["xgb_feature_importance_gain"].items())[:6]:
        console.print(f"    {feat}: {imp:.2f}")

    return results, final_model, available_features, df_clean


# ─── SHAP analysis ────────────────────────────────────────────────────────────


def run_shap_analysis(
    model,
    X: np.ndarray,
    feature_names: list[str],
    df_clean: pd.DataFrame,
    outcome: str = "diabetes_pct",
    n_shap_sample: int = 3000,
) -> dict:
    """Compute SHAP values and export summary data.

    SHAP values represent each feature's marginal contribution to a
    prediction, averaged over all possible feature coalitions (Shapley 1953).
    Unlike permutation importance, SHAP correctly handles correlated features.

    SHAP summary exported:
    - Mean absolute SHAP per feature (global importance)
    - SHAP values for a sample of 3000 tracts (for scatter plots)
    - Interaction: is_food_desert SHAP values stratified by income quintile

    Parameters
    ----------
    n_shap_sample: Number of tracts to compute exact SHAP for (full dataset
                   is slow; 3000 is sufficient for visualization).
    """
    import shap

    console.rule("[bold]ML: SHAP Analysis")
    results = {}

    if model is None or len(feature_names) == 0:
        console.print("[yellow]No model provided for SHAP.[/]")
        return results

    n_sample = min(n_shap_sample, len(X))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=n_sample, replace=False)
    X_sample = X[sample_idx].astype(np.float32)

    console.print(f"  Computing SHAP values for {n_sample:,} tracts…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP (global importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = {
        feature_names[i]: round(float(mean_abs_shap[i]), 6)
        for i in range(len(feature_names))
    }
    shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
    results["shap_importance"] = shap_importance

    # Food desert SHAP rank
    shap_rank = list(shap_importance.keys()).index("is_food_desert") + 1 if "is_food_desert" in shap_importance else "N/A"
    results["food_desert_shap_rank"] = shap_rank

    console.print(f"  SHAP importance rank of is_food_desert: #{shap_rank}")
    console.print("  Top 6 features by mean |SHAP|:")
    for feat, val in list(shap_importance.items())[:6]:
        console.print(f"    {feat}: {val:.6f}")

    # SHAP scatter data (for frontend charts) — top 5 features
    top5_features = list(shap_importance.keys())[:5]
    scatter_data = {}
    for feat in top5_features:
        if feat not in feature_names:
            continue
        feat_idx = feature_names.index(feat)
        feat_vals = X_sample[:, feat_idx].tolist()
        shap_vals = shap_values[:, feat_idx].tolist()
        scatter_data[feat] = {
            "feature_values": [round(v, 4) for v in feat_vals[:500]],  # 500 pts for JSON size
            "shap_values": [round(v, 6) for v in shap_vals[:500]],
        }
    results["shap_scatter_data"] = scatter_data

    # Food desert SHAP distribution
    if "is_food_desert" in feature_names:
        fd_idx = feature_names.index("is_food_desert")
        fd_shap = shap_values[:, fd_idx]
        results["food_desert_shap_distribution"] = {
            "mean": round(float(fd_shap.mean()), 6),
            "std": round(float(fd_shap.std()), 6),
            "min": round(float(fd_shap.min()), 6),
            "max": round(float(fd_shap.max()), 6),
            "pct_positive": round(float((fd_shap > 0).mean() * 100), 1),
            "interpretation": (
                "Food desert positively contributes to higher diabetes predictions "
                f"in {((fd_shap > 0).mean()*100):.1f}% of sampled tracts"
            ),
        }

    return results


# ─── ML vs OLS comparison ─────────────────────────────────────────────────────


def compare_ml_ols(
    shap_importance: dict,
    ols_results: dict | None = None,
) -> dict:
    """Compare ML (SHAP) and OLS feature importance rankings.

    Higher correlation between rankings suggests both methods agree on
    what drives diabetes.  Divergences may indicate nonlinear effects
    captured by XGBoost but not OLS.

    Parameters
    ----------
    shap_importance: Ordered dict of {feature: mean_abs_shap} from SHAP.
    ols_results    : Phase 2 OLS results dict (optional); used to extract
                     OLS coefficient magnitudes.
    """
    console.rule("[bold]ML: ML vs OLS Feature Ranking Comparison")
    results = {}

    if not shap_importance:
        console.print("[yellow]No SHAP importance to compare.[/]")
        return results

    ml_ranks = {feat: i + 1 for i, feat in enumerate(shap_importance.keys())}
    results["ml_shap_ranks"] = ml_ranks

    if ols_results and "ols_diabetes" in ols_results:
        ols_coefs = ols_results["ols_diabetes"].get("coefficients", {})
        # Rank by absolute coefficient
        ols_abs = {k: abs(v["coef"]) for k, v in ols_coefs.items() if k != "Intercept"}
        ols_sorted = sorted(ols_abs.items(), key=lambda x: x[1], reverse=True)
        ols_ranks = {feat: i + 1 for i, (feat, _) in enumerate(ols_sorted)}
        results["ols_coef_ranks"] = ols_ranks

        # Spearman rank correlation on common features
        common = [f for f in ml_ranks if f in ols_ranks]
        if len(common) >= 3:
            from scipy.stats import spearmanr
            ml_r = [ml_ranks[f] for f in common]
            ols_r = [ols_ranks[f] for f in common]
            rho, p = spearmanr(ml_r, ols_r)
            results["spearman_rank_correlation"] = {
                "rho": round(float(rho), 4),
                "p_value": float(f"{p:.2e}"),
                "n_common_features": len(common),
                "agreement": "Strong" if abs(rho) > 0.7 else "Moderate" if abs(rho) > 0.4 else "Weak",
            }
            console.print(
                f"  Spearman rank correlation (ML vs OLS): ρ={rho:.4f}, p={p:.2e} "
                f"[{results['spearman_rank_correlation']['agreement']} agreement]"
            )

    # Key finding: does food desert rank consistently?
    if "is_food_desert" in ml_ranks:
        console.print(f"  is_food_desert ML rank: #{ml_ranks['is_food_desert']}")

    return results


# ─── K-means tract archetypes ─────────────────────────────────────────────────


def run_tract_archetypes(
    master: pd.DataFrame,
    k_range: tuple[int, int] = (3, 6),
    clustering_features: list[str] | None = None,
) -> dict:
    """K-means clustering to identify distinct census tract archetypes.

    Feature set for clustering (standardized before fitting):
    poverty_rate, median_household_income, pct_black, pct_hispanic,
    is_food_desert, uninsured_pct, diabetes_pct, obesity_pct, pct_bachelors_plus

    Optimal k is selected by silhouette score (Rousseeuw 1987).
    Silhouette ∈ [-1, 1]; higher is better; >0.5 indicates well-separated clusters.

    Archetype profiles report mean of each variable per cluster, allowing
    substantive interpretation (e.g., "Cluster 3: high-poverty, food desert, minority").

    Parameters
    ----------
    k_range           : (min_k, max_k) range of k values to test.
    clustering_features: Features to use for clustering.  Defaults to a curated set.
    """
    console.rule("[bold]ML: K-Means Tract Archetypes")
    results = {}

    if clustering_features is None:
        clustering_features = [
            "poverty_rate", "median_household_income", "pct_black",
            "pct_hispanic", "pct_white", "is_food_desert", "uninsured_pct",
            "diabetes_pct", "obesity_pct", "pct_bachelors_plus",
        ]

    available = [c for c in clustering_features if c in master.columns]
    if len(available) < 3:
        console.print("[red]Insufficient features for clustering.[/]")
        return results

    df_cl = master[available].dropna().reset_index(drop=True)
    if len(df_cl) < 300:
        console.print(f"[yellow]Only {len(df_cl)} rows for clustering.[/]")
        return results

    console.print(f"  Clustering: {len(df_cl):,} tracts × {len(available)} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cl[available])

    # ── Select optimal k ──
    k_min, k_max = k_range
    silhouette_scores = {}

    console.print(f"  Testing k={k_min}–{k_max-1}…")
    for k in range(k_min, k_max):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=42)
        silhouette_scores[k] = round(float(sil), 4)
        console.print(f"    k={k}: silhouette={sil:.4f}")

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    results["silhouette_scores"] = silhouette_scores
    results["optimal_k"] = optimal_k
    console.print(f"  Optimal k={optimal_k} (silhouette={silhouette_scores[optimal_k]:.4f})")

    # ── Final clustering with optimal k ──
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    df_cl["cluster"] = km_final.fit_predict(X_scaled)

    # Cluster profiles
    profiles = (
        df_cl.groupby("cluster")[available]
        .mean()
        .round(4)
        .reset_index()
    )
    cluster_sizes = df_cl["cluster"].value_counts().sort_index().to_dict()

    # Auto-label clusters by dominant characteristics
    cluster_labels = {}
    for _, row in profiles.iterrows():
        k_id = int(row["cluster"])
        size = cluster_sizes.get(k_id, 0)
        label_parts = [f"Cluster {k_id} (n={size:,})"]

        if "poverty_rate" in available:
            pov = row["poverty_rate"]
            label_parts.append("high-poverty" if pov > profiles["poverty_rate"].median() else "low-poverty")
        if "is_food_desert" in available:
            fd = row["is_food_desert"]
            if fd > 0.4:
                label_parts.append("food desert")
        if "diabetes_pct" in available:
            dm = row["diabetes_pct"]
            label_parts.append(f"diabetes={dm:.1f}%")

        cluster_labels[k_id] = " | ".join(label_parts)

    results["cluster_profiles"] = profiles.to_dict(orient="records")
    results["cluster_sizes"] = {int(k): int(v) for k, v in cluster_sizes.items()}
    results["cluster_labels"] = cluster_labels

    console.print("  Cluster profiles (means):")
    for _, row in profiles.iterrows():
        k_id = int(row["cluster"])
        dm = row.get("diabetes_pct", "N/A")
        pov = row.get("poverty_rate", "N/A")
        fd = row.get("is_food_desert", "N/A")
        console.print(f"    {cluster_labels[k_id]}: diabetes={dm}, poverty={pov}, food_desert={fd}")

    # Merge cluster labels back for export
    # df_cl was reset_index(drop=True), so its index is 0…n-1.
    # Re-derive tract_fips by aligning with the same dropna operation.
    if "tract_fips" in master.columns:
        tract_fips_aligned = (
            master[["tract_fips"] + available]
            .dropna(subset=available)
            .reset_index(drop=True)["tract_fips"]
        )
        cluster_assignments = pd.DataFrame({
            "tract_fips": tract_fips_aligned.values,
            "cluster": df_cl["cluster"].values,
        })
        top_per_cluster = []
        for c_id in range(optimal_k):
            mask = cluster_assignments["cluster"] == c_id
            top_per_cluster.append({
                "cluster": int(c_id),
                "label": cluster_labels[c_id],
                "sample_tracts": cluster_assignments.loc[mask, "tract_fips"].head(5).tolist(),
            })
        results["cluster_sample_tracts"] = top_per_cluster

    return results


# ─── Counterfactual simulation ────────────────────────────────────────────────


def run_counterfactual_simulation(
    model,
    df_clean: pd.DataFrame,
    feature_names: list[str],
    outcome: str = "diabetes_pct",
) -> dict:
    """Predict diabetes rates if all food-desert tracts had is_food_desert=0.

    For each food-desert tract, the XGBoost model predicts what diabetes_pct
    would be if the tract magically gained food access (is_food_desert → 0),
    holding all other features constant.

    This is NOT a causal estimate — it reflects what the model predicts
    under feature manipulation, which is only causal if the model has
    learned the true causal data-generating process.  Use alongside PSM
    and RD results as a complementary signal.

    Metrics reported:
    - Average reduction in predicted diabetes for food-desert tracts
    - Total "avoidable" cases (reduction × population) across all treated tracts
    - Distribution of counterfactual reductions

    Parameters
    ----------
    model        : Fitted XGBoost model.
    df_clean     : DataFrame with features and tract_fips used for predictions.
    feature_names: List matching the columns used during model training.
    """
    console.rule("[bold]ML: Counterfactual Simulation (food desert → 0)")
    results = {}

    if model is None or df_clean is None or len(df_clean) == 0:
        console.print("[yellow]No fitted model for counterfactual simulation.[/]")
        return results

    if "is_food_desert" not in feature_names:
        console.print("[yellow]is_food_desert not in feature set — cannot simulate.[/]")
        return results

    if "is_food_desert" not in df_clean.columns:
        console.print("[yellow]is_food_desert column missing from df_clean.[/]")
        return results

    fd_mask = df_clean["is_food_desert"] == 1
    df_fd = df_clean[fd_mask].copy()

    if len(df_fd) < 10:
        console.print("[yellow]Fewer than 10 food-desert tracts — skipping counterfactual.[/]")
        return results

    console.print(f"  Simulating {len(df_fd):,} food-desert tracts with is_food_desert=0…")

    X_fd = df_fd[feature_names].values.astype(np.float32)
    fd_col_idx = feature_names.index("is_food_desert")

    # Factual predictions
    y_factual = model.predict(X_fd)

    # Counterfactual: flip is_food_desert to 0
    X_cf = X_fd.copy()
    X_cf[:, fd_col_idx] = 0.0
    y_counterfactual = model.predict(X_cf)

    reduction = y_factual - y_counterfactual  # positive = food desert increases diabetes
    results["counterfactual_simulation"] = {
        "n_food_desert_tracts": int(len(df_fd)),
        "mean_factual_diabetes": round(float(y_factual.mean()), 4),
        "mean_counterfactual_diabetes": round(float(y_counterfactual.mean()), 4),
        "mean_reduction_pp": round(float(reduction.mean()), 4),
        "median_reduction_pp": round(float(np.median(reduction)), 4),
        "std_reduction_pp": round(float(reduction.std()), 4),
        "pct_tracts_with_reduction": round(float((reduction > 0).mean() * 100), 1),
        "max_reduction_pp": round(float(reduction.max()), 4),
        "min_reduction_pp": round(float(reduction.min()), 4),
    }

    # Population-weighted total avoidable cases
    if "population" in df_fd.columns:
        pop = df_fd["population"].fillna(df_fd["population"].median()).values
        avoidable_cases = (reduction / 100 * pop).sum()
        results["counterfactual_simulation"]["population_weighted_avoidable_diabetes_cases"] = round(
            float(avoidable_cases), 0
        )
        console.print(
            f"  Population-weighted avoidable cases: "
            f"{avoidable_cases:,.0f} diabetes cases"
        )

    # Distribution buckets
    reduction_buckets = {
        "reduced_gt_2pp": int((reduction > 2).sum()),
        "reduced_1_to_2pp": int(((reduction >= 1) & (reduction <= 2)).sum()),
        "reduced_lt_1pp": int(((reduction > 0) & (reduction < 1)).sum()),
        "no_change_or_increase": int((reduction <= 0).sum()),
    }
    results["counterfactual_simulation"]["reduction_distribution"] = reduction_buckets

    console.print(
        f"  Mean reduction: {reduction.mean():.4f} pp "
        f"(SD={reduction.std():.4f}, range={reduction.min():.4f}–{reduction.max():.4f})"
    )

    # Top 25 tracts with largest potential reduction
    if "tract_fips" in df_fd.columns:
        cf_df = df_fd[["tract_fips"]].copy()
        cf_df["factual_diabetes"] = y_factual
        cf_df["counterfactual_diabetes"] = y_counterfactual
        cf_df["reduction_pp"] = reduction
        top25 = cf_df.sort_values("reduction_pp", ascending=False).head(25)
        results["top_25_reduction_tracts"] = top25.round(4).to_dict(orient="records")

    return results


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def run_ml_analysis(
    master: pd.DataFrame,
    ols_results: dict | None = None,
) -> dict:
    """Run full ML complement pipeline and export results.

    Parameters
    ----------
    master     : Master dataframe.
    ols_results: Phase 2 OLS results dict for ML vs OLS comparison (optional).

    Returns
    -------
    dict with keys: xgboost, shap, ml_ols_comparison, archetypes, counterfactual.
    """
    import shap as shap_lib  # import only to verify availability
    import xgboost  # import only to verify availability

    console.rule("[bold]Phase 6D: ML Complement")
    all_results = {}

    # XGBoost
    xgb_results, model, feature_names, df_clean = run_xgboost_diabetes(master)
    all_results["xgboost"] = xgb_results

    # SHAP
    if model is not None and len(feature_names) > 0:
        X_arr = df_clean[feature_names].values.astype(np.float32)
        shap_results = run_shap_analysis(model, X_arr, feature_names, df_clean)
        all_results["shap"] = shap_results

        # ML vs OLS comparison
        shap_importance = shap_results.get("shap_importance", {})
        comparison = compare_ml_ols(shap_importance, ols_results)
        all_results["ml_ols_comparison"] = comparison

        # Counterfactual
        cf_results = run_counterfactual_simulation(model, df_clean, feature_names)
        all_results["counterfactual"] = cf_results
    else:
        console.print("[yellow]Skipping SHAP/counterfactual (no model fitted).[/]")

    # K-means archetypes
    archetypes = run_tract_archetypes(master)
    all_results["archetypes"] = archetypes

    _export_json(all_results, "phase6d_ml_complement.json")
    return all_results
