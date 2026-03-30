"""Spatial Statistics Module — Phase 6A.

Implements spatial autocorrelation, spatial regression models, and
Geographically Weighted Regression (GWR) to test whether food-desert /
diabetes patterns cluster in space above and beyond what aspatial OLS
would predict.

Key methods
-----------
- Global Moran's I  : overall spatial autocorrelation of diabetes_pct
- Local LISA        : identify hot-spots, cold-spots, and spatial outliers
- Spatial Lag Model : spatially-lagged dependent variable (SLM / SAR)
- Spatial Error Model: spatially-correlated residuals (SEM)
- GWR               : locally-varying coefficients across the US

Assumptions & limitations
--------------------------
- Queen contiguity weights assume adjacent tracts are spatially related.
  This is conservative; row-standardization ensures mean neighbor influence = 1.
- GWR bandwidth is selected via AICc minimization (golden section search).
- Census-tract shapefiles are required.  Use `download_tiger_shapefiles()`
  to fetch them from the Census TIGER/Line server if not present locally.
- Results are approximate when run on a subset of states; full US requires
  ~27 k tracts and ~10–30 min for GWR depending on hardware.

Packages required: libpysal, esda, spreg, mgwr, geopandas, mapclassify
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _export_json(data: dict, filename: str) -> Path:
    """Save results dict as JSON to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"[green]Exported → {path}[/]")
    return path


# ─── Shapefile downloader ─────────────────────────────────────────────────────


def download_tiger_shapefiles(year: int = 2020, overwrite: bool = False) -> Path:
    """Download all US census-tract shapefiles from TIGER/Line.

    The Census publishes one shapefile per state.  This function downloads
    all 50 states + DC (51 files) and concatenates them into a single GeoParquet
    at data/raw/census_tracts.parquet.

    Parameters
    ----------
    year     : TIGER/Line vintage (2020 matches CDC PLACES tract IDs).
    overwrite: Re-download even if the output file already exists.

    Returns
    -------
    Path to the merged GeoParquet file.
    """
    import geopandas as gpd
    import requests

    out_path = DATA_RAW / "census_tracts.parquet"
    if out_path.exists() and not overwrite:
        console.print(f"[cyan]Shapefiles already cached → {out_path}[/]")
        return out_path

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # FIPS codes for 50 states + DC
    state_fips = [
        "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18",
        "19","20","21","22","23","24","25","26","27","28","29","30","31","32","33",
        "34","35","36","37","38","39","40","41","42","44","45","46","47","48","49",
        "50","51","53","54","55","56",
    ]

    base = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT"
    gdfs = []

    console.print(f"[bold]Downloading {len(state_fips)} state tract shapefiles (TIGER {year})…[/]")
    for fips in state_fips:
        url = f"{base}/tl_{year}_{fips}_tract.zip"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            import io, zipfile
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                shp_name = [n for n in zf.namelist() if n.endswith(".shp")][0]
                with zf.open(shp_name) as shp:
                    gdf = gpd.read_file(shp)
            gdfs.append(gdf)
            console.print(f"  [{fips}] {len(gdf):,} tracts", end="\r")
        except Exception as exc:
            console.print(f"  [yellow]Warning: failed to download state {fips}: {exc}[/]")

    if not gdfs:
        raise RuntimeError("No shapefiles downloaded.  Check network access.")

    merged = pd.concat(gdfs, ignore_index=True)
    merged = merged.to_crs("EPSG:4326")

    # Standardize FIPS column
    if "GEOID" in merged.columns:
        merged = merged.rename(columns={"GEOID": "tract_fips"})

    merged.to_parquet(out_path, index=False)
    console.print(f"\n[green]Saved {len(merged):,} tracts → {out_path}[/]")
    return out_path


# ─── Build spatial weights ────────────────────────────────────────────────────


def build_spatial_weights(
    master: pd.DataFrame,
    shapefile_path: Optional[Path] = None,
) -> tuple:
    """Build queen-contiguity spatial weights matrix from census tract geometries.

    Parameters
    ----------
    master         : Master dataframe with tract_fips column.
    shapefile_path : Path to a GeoParquet / shapefile with tract geometries.
                     Defaults to data/raw/census_tracts.parquet.

    Returns
    -------
    (gdf, W) where gdf is the GeoDataFrame aligned to master and W is the
    row-standardized libpysal Queen weights object.
    """
    import geopandas as gpd
    from libpysal.weights import Queen

    if shapefile_path is None:
        shapefile_path = DATA_RAW / "census_tracts.parquet"

    if not shapefile_path.exists():
        console.print("[yellow]Shapefile not found. Attempting auto-download…[/]")
        shapefile_path = download_tiger_shapefiles()

    console.print(f"[cyan]Loading tract geometries from {shapefile_path.name}…[/]")
    gdf = gpd.read_parquet(shapefile_path) if shapefile_path.suffix == ".parquet" else gpd.read_file(shapefile_path)

    # Align to master tracts
    tracts_in_master = set(master["tract_fips"].dropna().astype(str))
    gdf["tract_fips"] = gdf["tract_fips"].astype(str).str.zfill(11)
    gdf = gdf[gdf["tract_fips"].isin(tracts_in_master)].copy()

    if len(gdf) < 100:
        raise ValueError(f"Only {len(gdf)} tracts matched between master and shapefile. Check FIPS alignment.")

    console.print(f"  Matched {len(gdf):,} tracts to geometries")

    # Build Queen contiguity weights (shared edge or vertex = neighbor)
    console.print("  Building Queen contiguity weights (this may take 1–3 min for full US)…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W = Queen.from_dataframe(gdf, idVariable="tract_fips")

    W.transform = "r"  # row-standardize: each row sums to 1
    console.print(f"  Weights: {W.n} tracts, mean neighbors={W.mean_neighbors:.1f}, "
                  f"islands={len(W.islands)}")

    return gdf, W


# ─── Global Moran's I ─────────────────────────────────────────────────────────


def run_global_morans_i(
    master: pd.DataFrame,
    W=None,
    gdf=None,
    variable: str = "diabetes_pct",
    shapefile_path: Optional[Path] = None,
) -> dict:
    """Compute global Moran's I for spatial autocorrelation.

    Moran's I tests H₀: values are randomly distributed in space.
    I > 0  → similar values cluster (positive autocorrelation).
    I ≈ 0  → random spatial pattern.
    I < 0  → dissimilar values cluster (negative autocorrelation).

    Significance is assessed via a permutation test (999 simulations)
    rather than the asymptotic normal approximation to avoid the
    assumption of normality in tract-level health data.

    Parameters
    ----------
    master  : Master dataframe.
    W       : Pre-built libpysal weights (built if None).
    variable: Column to test for spatial autocorrelation.
    """
    from esda.moran import Moran

    console.rule(f"[bold]Spatial: Global Moran's I ({variable})")
    results = {}

    if variable not in master.columns:
        console.print(f"[red]Column '{variable}' not found in master.[/]")
        return results

    if W is None:
        gdf, W = build_spatial_weights(master, shapefile_path)

    # Align values to weights index order
    df_clean = master.dropna(subset=["tract_fips", variable]).copy()
    df_clean["tract_fips"] = df_clean["tract_fips"].astype(str).str.zfill(11)
    df_indexed = df_clean.set_index("tract_fips")

    common = [t for t in W.id_order if t in df_indexed.index]
    if len(common) < 100:
        console.print(f"[red]Only {len(common)} tracts overlap with weights. Aborting.[/]")
        return results

    y = df_indexed.loc[common, variable].values

    # Subset weights matrix to only tracts with valid outcome data
    from libpysal.weights import w_subset
    W_sub = w_subset(W, common)
    W_sub.transform = "r"

    moran = Moran(y, W_sub, permutations=999)

    results["global_morans_i"] = {
        "variable": variable,
        "moran_i": round(float(moran.I), 4),
        "expected_i": round(float(moran.EI), 4),
        "z_score": round(float(moran.z_norm), 4),
        "p_value_norm": round(float(moran.p_norm), 4),
        "p_value_sim": round(float(moran.p_sim), 4),
        "n_permutations": 999,
        "n_tracts": len(common),
        "interpretation": (
            "Strong positive spatial autocorrelation — diabetes rates cluster geographically"
            if moran.I > 0.3 else
            "Moderate positive spatial autocorrelation" if moran.I > 0.1 else
            "Weak or no spatial autocorrelation"
        ),
    }

    console.print(
        f"  Moran's I={moran.I:.4f}, z={moran.z_norm:.2f}, "
        f"p(sim)={moran.p_sim:.4f}  [{'SIGNIFICANT' if moran.p_sim < 0.05 else 'not significant'}]"
    )
    return results


# ─── Local LISA ───────────────────────────────────────────────────────────────


def run_local_lisa(
    master: pd.DataFrame,
    W=None,
    gdf=None,
    variable: str = "diabetes_pct",
    significance: float = 0.05,
    shapefile_path: Optional[Path] = None,
) -> dict:
    """Compute Local Indicators of Spatial Association (LISA / Local Moran's I).

    Each tract receives a cluster label:
    - HH (High-High): high value surrounded by high neighbors → hot-spot
    - LL (Low-Low)  : low value surrounded by low neighbors → cold-spot
    - HL (High-Low) : high value surrounded by low neighbors → spatial outlier
    - LH (Low-High) : low value surrounded by high neighbors → spatial outlier
    - ns            : not statistically significant

    p-values are based on 999 conditional permutations (Anselin 1995).

    Returns cluster label counts and the top 50 hot-spot tracts.
    """
    from esda.moran import Moran_Local

    console.rule(f"[bold]Spatial: Local LISA ({variable})")
    results = {}

    if variable not in master.columns:
        console.print(f"[red]Column '{variable}' not found in master.[/]")
        return results

    if W is None:
        gdf, W = build_spatial_weights(master, shapefile_path)

    df_clean = master.dropna(subset=["tract_fips", variable]).copy()
    df_clean["tract_fips"] = df_clean["tract_fips"].astype(str).str.zfill(11)
    df_indexed = df_clean.set_index("tract_fips")

    common = [t for t in W.id_order if t in df_indexed.index]
    if len(common) < 100:
        console.print(f"[red]Only {len(common)} tracts overlap with weights.[/]")
        return results

    y = df_indexed.loc[common, variable].values

    # Subset weights matrix to only tracts with valid outcome data
    from libpysal.weights import w_subset
    W_sub = w_subset(W, common)
    W_sub.transform = "r"

    lisa = Moran_Local(y, W_sub, permutations=999)

    # Cluster labels following Anselin convention
    # lisa.q: 1=HH, 2=LH, 3=LL, 4=HL
    q_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    labels = []
    for i in range(len(common)):
        if lisa.p_sim[i] < significance:
            labels.append(q_map.get(lisa.q[i], "ns"))
        else:
            labels.append("ns")

    label_series = pd.Series(labels, index=common)
    counts = label_series.value_counts().to_dict()

    results["lisa"] = {
        "variable": variable,
        "significance_threshold": significance,
        "n_tracts": len(common),
        "cluster_counts": counts,
        "pct_hotspot_HH": round(counts.get("HH", 0) / len(common) * 100, 2),
        "pct_coldspot_LL": round(counts.get("LL", 0) / len(common) * 100, 2),
    }

    # Top 50 hot-spot tracts (HH, highest local I)
    # Use positional boolean mask (same order as lisa arrays and y)
    pos_mask = np.array(labels) == "HH"
    if pos_mask.sum() > 0:
        hot_tracts = pd.DataFrame({
            "tract_fips": np.array(common)[pos_mask],
            "local_moran_i": lisa.Is[pos_mask],
            "p_sim": lisa.p_sim[pos_mask],
            variable: y[pos_mask],
        }).sort_values("local_moran_i", ascending=False).head(50)

        results["top_hotspot_tracts"] = hot_tracts.to_dict(orient="records")

    console.print(
        f"  LISA clusters: HH={counts.get('HH',0):,} hot-spots, "
        f"LL={counts.get('LL',0):,} cold-spots, "
        f"HL={counts.get('HL',0):,} / LH={counts.get('LH',0):,} outliers, "
        f"ns={counts.get('ns',0):,}"
    )
    return results


# ─── Spatial regression models ────────────────────────────────────────────────


def run_spatial_regression(
    master: pd.DataFrame,
    W=None,
    gdf=None,
    outcome: str = "diabetes_pct",
    shapefile_path: Optional[Path] = None,
) -> dict:
    """Fit Spatial Lag Model (SLM) and Spatial Error Model (SEM) for outcome.

    Spatial Lag Model (SAR / SLM)
        y = ρ W y + X β + ε
        Captures spatial spillovers: a tract's outcome partly depends on
        neighbor outcomes (e.g., regional food environment).

    Spatial Error Model (SEM)
        y = X β + λ W u + ε
        Captures spatially correlated residuals due to omitted spatial variables
        (e.g., climate, regional policy) without postulating outcome spillovers.

    Model selection: Lagrange Multiplier tests (Anselin 1988) guide which model
    is more appropriate.  Both are estimated via ML using spreg.

    Diagnostics reported: log-likelihood, AIC, BIC, Moran's I on residuals,
    LM tests, and coefficient tables.
    """
    import spreg

    console.rule(f"[bold]Spatial: Spatial Regression ({outcome})")
    results = {}

    reg_cols = [outcome, "is_food_desert", "poverty_rate", "uninsured_pct",
                "pct_black", "pct_hispanic", "median_household_income"]
    reg_cols = [c for c in reg_cols if c in master.columns]

    if outcome not in reg_cols:
        console.print(f"[red]Outcome '{outcome}' not available.[/]")
        return results

    df_clean = master[["tract_fips"] + reg_cols].dropna()
    df_clean = df_clean.copy()
    df_clean["tract_fips"] = df_clean["tract_fips"].astype(str).str.zfill(11)

    if W is None:
        gdf, W = build_spatial_weights(master, shapefile_path)

    # Align to weights order, then remove island tracts (no neighbors)
    from libpysal.weights import w_subset
    common = [t for t in W.id_order if t in df_clean["tract_fips"].values]
    W_sub = w_subset(W, common)
    W_sub.transform = "r"

    # Remove islands (tracts with zero neighbors) — required for ML_Lag/ML_Error
    islands = set(W_sub.islands)
    if islands:
        console.print(f"  [yellow]Removing {len(islands)} island tracts (no neighbors)[/]")
        common = [t for t in common if t not in islands]
        W_sub = w_subset(W_sub, common)
        W_sub.transform = "r"

    df_aligned = df_clean.set_index("tract_fips").loc[common]

    if len(df_aligned) < 200:
        console.print(f"[red]Insufficient aligned tracts ({len(df_aligned)}) for spatial regression.[/]")
        return results

    y_vals = df_aligned[[outcome]].values
    feature_cols = [c for c in reg_cols if c != outcome]
    X_vals = df_aligned[feature_cols].values
    X_names = feature_cols

    console.print(f"  Fitting OLS baseline (n={len(df_aligned):,})…")
    # OLS baseline with spatial diagnostics
    ols = spreg.OLS(
        y_vals, X_vals,
        w=W_sub,
        name_y=outcome,
        name_x=X_names,
        spat_diag=True,
    )
    results["ols_baseline"] = {
        "r_squared": round(float(ols.r2), 4),
        "adj_r_squared": round(float(ols.ar2), 4),
        "aic": round(float(ols.aic), 2),
        "lm_lag_p": round(float(ols.lm_lag[1]), 4),
        "lm_error_p": round(float(ols.lm_error[1]), 4),
        "rlm_lag_p": round(float(ols.rlm_lag[1]), 4),
        "rlm_error_p": round(float(ols.rlm_error[1]), 4),
        "moran_i_residuals": round(float(ols.moran_res[0]), 4),
        "model_preferred": (
            "Spatial Lag" if ols.lm_lag[1] < ols.lm_error[1] else "Spatial Error"
        ),
    }
    console.print(
        f"  OLS R²={ols.r2:.4f}, LM-lag p={ols.lm_lag[1]:.4f}, "
        f"LM-error p={ols.lm_error[1]:.4f} → prefer {results['ols_baseline']['model_preferred']}"
    )

    console.print("  Fitting Spatial Lag Model (ML)…")
    slm = spreg.ML_Lag(
        y_vals, X_vals, w=W_sub,
        name_y=outcome,
        name_x=X_names,
    )
    results["spatial_lag_model"] = {
        "rho": round(float(slm.rho), 4),
        "rho_z": round(float(slm.z_stat[-1][0]), 3),
        "rho_p": round(float(slm.z_stat[-1][1]), 4),
        "pseudo_r_squared": round(float(slm.pr2), 4),
        "log_likelihood": round(float(slm.logll), 2),
        "aic": round(float(slm.aic), 2),
        "coefficients": {
            X_names[i]: {
                "coef": round(float(slm.betas[i + 1][0]), 4),
                "z_stat": round(float(slm.z_stat[i][0]), 3),
                "p_value": round(float(slm.z_stat[i][1]), 4),
            }
            for i in range(len(X_names))
        },
        "interpretation": (
            f"ρ={slm.rho:.4f} — "
            + ("significant spatial spillovers" if slm.z_stat[-1][1] < 0.05
               else "no significant spatial lag")
        ),
    }
    console.print(
        f"  SLM: ρ={slm.rho:.4f}, p={slm.z_stat[-1][1]:.4f}, "
        f"pseudo-R²={slm.pr2:.4f}, AIC={slm.aic:.1f}"
    )

    console.print("  Fitting Spatial Error Model (ML)…")
    sem = spreg.ML_Error(
        y_vals, X_vals, w=W_sub,
        name_y=outcome,
        name_x=X_names,
    )
    results["spatial_error_model"] = {
        "lambda": round(float(sem.lam), 4),
        "lambda_z": round(float(sem.z_stat[-1][0]), 3),
        "lambda_p": round(float(sem.z_stat[-1][1]), 4),
        "pseudo_r_squared": round(float(sem.pr2), 4),
        "log_likelihood": round(float(sem.logll), 2),
        "aic": round(float(sem.aic), 2),
        "coefficients": {
            X_names[i]: {
                "coef": round(float(sem.betas[i + 1][0]), 4),
                "z_stat": round(float(sem.z_stat[i][0]), 3),
                "p_value": round(float(sem.z_stat[i][1]), 4),
            }
            for i in range(len(X_names))
        },
        "interpretation": (
            f"λ={sem.lam:.4f} — "
            + ("significant spatial error correlation" if sem.z_stat[-1][1] < 0.05
               else "no significant spatial error")
        ),
    }
    console.print(
        f"  SEM: λ={sem.lam:.4f}, p={sem.z_stat[-1][1]:.4f}, "
        f"pseudo-R²={sem.pr2:.4f}, AIC={sem.aic:.1f}"
    )

    # Model comparison
    results["model_comparison"] = {
        "ols_aic": results["ols_baseline"]["aic"],
        "slm_aic": results["spatial_lag_model"]["aic"],
        "sem_aic": results["spatial_error_model"]["aic"],
        "best_model": min(
            [("OLS", results["ols_baseline"]["aic"]),
             ("SLM", results["spatial_lag_model"]["aic"]),
             ("SEM", results["spatial_error_model"]["aic"])],
            key=lambda x: x[1]
        )[0],
    }

    return results


# ─── Geographically Weighted Regression ──────────────────────────────────────


def run_gwr(
    master: pd.DataFrame,
    gdf=None,
    outcome: str = "diabetes_pct",
    max_tracts: int = 5000,
    shapefile_path: Optional[Path] = None,
) -> dict:
    """Fit Geographically Weighted Regression (GWR) showing regional coefficient variation.

    GWR fits a separate weighted regression at each observation location,
    using a Gaussian kernel to weight nearby observations more heavily.
    Bandwidth is selected by minimizing AICc (Fotheringham et al. 2002).

    Key outputs:
    - Local R² per tract (where does the model fit well?)
    - Local coefficient for is_food_desert (where is the effect strongest?)
    - Global vs local comparison

    GWR is computationally intensive.  To keep runtime < 5 min,
    `max_tracts` caps the sample size (default 5,000 tracts).
    Set max_tracts=0 to run on the full dataset.

    Packages required: mgwr, geopandas
    """
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    import geopandas as gpd

    console.rule(f"[bold]Spatial: Geographically Weighted Regression ({outcome})")
    results = {}

    if shapefile_path is None:
        shapefile_path = DATA_RAW / "census_tracts.parquet"

    reg_cols = [outcome, "is_food_desert", "poverty_rate", "uninsured_pct", "pct_black"]
    reg_cols = [c for c in reg_cols if c in master.columns]

    if outcome not in reg_cols:
        console.print(f"[red]Outcome '{outcome}' not available.[/]")
        return results

    df_clean = master[["tract_fips"] + reg_cols].dropna()

    # Subsample if needed
    if max_tracts and len(df_clean) > max_tracts:
        console.print(f"  [yellow]Subsampling to {max_tracts:,} tracts for GWR (set max_tracts=0 for full run)[/]")
        df_clean = df_clean.sample(n=max_tracts, random_state=42)

    df_clean["tract_fips"] = df_clean["tract_fips"].astype(str).str.zfill(11)

    # Load geometries for centroids
    if gdf is None:
        if not shapefile_path.exists():
            console.print("[yellow]Shapefile not found. Attempting auto-download…[/]")
            shapefile_path = download_tiger_shapefiles()
        gdf = gpd.read_parquet(shapefile_path) if shapefile_path.suffix == ".parquet" else gpd.read_file(shapefile_path)
        gdf["tract_fips"] = gdf["tract_fips"].astype(str).str.zfill(11)

    # Merge centroids
    gdf_proj = gdf.to_crs("EPSG:5070")  # Albers Equal Area for US
    gdf_proj["centroid_x"] = gdf_proj.geometry.centroid.x
    gdf_proj["centroid_y"] = gdf_proj.geometry.centroid.y

    df_merged = df_clean.merge(
        gdf_proj[["tract_fips", "centroid_x", "centroid_y"]],
        on="tract_fips", how="inner"
    )

    if len(df_merged) < 100:
        console.print(f"[red]Only {len(df_merged)} tracts after geometry merge.[/]")
        return results

    console.print(f"  GWR sample: {len(df_merged):,} tracts")

    coords = df_merged[["centroid_x", "centroid_y"]].values
    y_vals = df_merged[[outcome]].values
    feature_cols = [c for c in reg_cols if c != outcome]
    X_vals = df_merged[feature_cols].values

    console.print("  Selecting GWR bandwidth via AICc (golden section search)…")
    selector = Sel_BW(coords, y_vals, X_vals, kernel="gaussian", fixed=False)
    bw = selector.search(criterion="AICc")
    console.print(f"  Optimal bandwidth: {bw:.1f} (adaptive nearest neighbors)")

    console.print("  Fitting GWR…")
    gwr_model = GWR(coords, y_vals, X_vals, bw=bw, kernel="gaussian", fixed=False)
    gwr_results = gwr_model.fit()

    # Summary of local coefficients
    local_r2 = gwr_results.localR2.flatten()
    local_params = gwr_results.params  # shape: (n, k+1)

    food_desert_idx = feature_cols.index("is_food_desert") + 1  # +1 for intercept
    local_food_desert_coef = local_params[:, food_desert_idx]

    results["gwr_summary"] = {
        "outcome": outcome,
        "n_tracts": len(df_merged),
        "bandwidth": round(float(bw), 1),
        "kernel": "gaussian (adaptive)",
        "aic": round(float(gwr_results.aic), 2),
        "aicc": round(float(gwr_results.aicc), 2),
        "bic": round(float(gwr_results.bic), 2),
        "r_squared_local_mean": round(float(local_r2.mean()), 4),
        "r_squared_local_std": round(float(local_r2.std()), 4),
        "food_desert_coef_mean": round(float(local_food_desert_coef.mean()), 4),
        "food_desert_coef_std": round(float(local_food_desert_coef.std()), 4),
        "food_desert_coef_min": round(float(local_food_desert_coef.min()), 4),
        "food_desert_coef_max": round(float(local_food_desert_coef.max()), 4),
        "spatial_variability_detected": float(local_food_desert_coef.std()) > 0.5,
    }

    # Top 50 tracts by local food desert effect
    local_df = df_merged[["tract_fips"]].copy()
    local_df["local_r2"] = local_r2
    local_df["local_food_desert_coef"] = local_food_desert_coef
    for i, col in enumerate(feature_cols):
        local_df[f"local_coef_{col}"] = local_params[:, i + 1]

    results["top_50_high_effect_tracts"] = (
        local_df.sort_values("local_food_desert_coef", ascending=False)
        .head(50)
        .to_dict(orient="records")
    )

    # Local R² distribution quantiles
    r2_quantiles = np.quantile(local_r2, [0.1, 0.25, 0.5, 0.75, 0.9])
    results["local_r2_quantiles"] = {
        "p10": round(float(r2_quantiles[0]), 4),
        "p25": round(float(r2_quantiles[1]), 4),
        "median": round(float(r2_quantiles[2]), 4),
        "p75": round(float(r2_quantiles[3]), 4),
        "p90": round(float(r2_quantiles[4]), 4),
    }

    console.print(
        f"  GWR mean local R²={local_r2.mean():.4f}, "
        f"food desert coef range=[{local_food_desert_coef.min():.3f}, {local_food_desert_coef.max():.3f}]"
    )

    return results


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def run_spatial_analysis(
    master: pd.DataFrame,
    shapefile_path: Optional[Path] = None,
    run_gwr_flag: bool = True,
) -> dict:
    """Run all spatial analyses and export results.

    Parameters
    ----------
    master         : Master dataframe with tract_fips and health columns.
    shapefile_path : Path to census tract geometries (downloaded if absent).
    run_gwr_flag   : Set False to skip GWR (saves ~5 min compute time).

    Returns
    -------
    dict with keys: global_morans_i, lisa, spatial_regression, gwr (optional).
    """
    console.rule("[bold]Phase 6A: Spatial Statistics")
    all_results = {}

    # Build weights once and reuse
    try:
        gdf, W = build_spatial_weights(master, shapefile_path)
    except Exception as exc:
        console.print(f"[red]Failed to build spatial weights: {exc}[/]")
        console.print("[yellow]Spatial analysis skipped. Provide census tract shapefiles.[/]")
        return all_results

    moran_results = run_global_morans_i(master, W=W, gdf=gdf)
    all_results.update(moran_results)

    lisa_results = run_local_lisa(master, W=W, gdf=gdf)
    all_results.update(lisa_results)

    spatial_reg_results = run_spatial_regression(master, W=W, gdf=gdf)
    all_results.update(spatial_reg_results)

    if run_gwr_flag:
        gwr_results = run_gwr(master, gdf=gdf, shapefile_path=shapefile_path)
        all_results.update(gwr_results)
    else:
        console.print("[yellow]GWR skipped (run_gwr_flag=False).[/]")

    _export_json(all_results, "phase6a_spatial_analysis.json")
    return all_results
