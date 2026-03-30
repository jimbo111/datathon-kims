"""Merge all health datasets into a single master dataframe on census tract FIPS.

Handles the 2010/2020 census tract mismatch:
- CDC PLACES + ACS use 2020 census tracts
- USDA + USALEEP use 2010 census tracts
- HRSA is at county level (stable across censuses)

Most tract FIPS codes are unchanged between censuses, so a direct join
captures ~85-95% of tracts. The merge quality report documents coverage.

Additions:
- Population-weighted merge quality check (% of US population covered)
- Per-state warnings when >15% of tracts are lost due to census-year mismatch
- validate_master() function for impossible-value detection
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from src.loaders.health_data import (
    FIPS_COL,
    load_acs,
    load_hrsa,
    load_life_expectancy,
    load_places,
    load_usda,
)

console = Console()
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_master(
    census_api_key: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Merge all 5 health datasets on census tract FIPS → master dataframe.

    Returns a single DataFrame with one row per tract and columns from all sources.
    Runs _report_quality() (tract counts + population coverage + per-state tract loss)
    and validate_master() after building.
    """
    console.rule("[bold]Building master dataframe")

    usda = load_usda()
    places = load_places()
    acs = load_acs(census_api_key)
    life_exp = load_life_expectancy()
    hrsa = load_hrsa()

    # ── Start with USDA as the base (food desert flags are the core IV) ──
    master = usda.copy()
    n_base = len(master)
    console.print(f"[cyan]Base (USDA): {n_base:,} tracts[/]")

    # ── Validate food_desert_1_10 presence (required for core analysis) ──
    if "food_desert_1_10" not in master.columns:
        raise ValueError(
            "USDA data is missing the 'food_desert_1_10' column. "
            "This column is required to derive 'is_food_desert' for all downstream analyses. "
            "Check the USDA loader and raw data file."
        )

    # ── Merge CDC PLACES (health outcomes) ──
    before = master[FIPS_COL].nunique()
    # validate="m:1" ensures PLACES has one row per FIPS (no duplicate joins)
    master = master.merge(places, on=FIPS_COL, how="left", validate="m:1")
    assert len(master) == n_base, (
        f"Row count changed after PLACES merge: {n_base} → {len(master)}. "
        "Possible duplicate FIPS in PLACES dataset."
    )
    matched = master["obesity_pct"].notna().sum()
    console.print(f"  + PLACES: {matched:,}/{before:,} matched ({matched/before*100:.1f}%)")

    # ── Merge ACS (demographics) ──
    master = master.merge(acs, on=FIPS_COL, how="left", validate="m:1")
    assert len(master) == n_base, (
        f"Row count changed after ACS merge: {n_base} → {len(master)}. "
        "Possible duplicate FIPS in ACS dataset."
    )
    matched = master["median_household_income"].notna().sum()
    console.print(f"  + ACS: {matched:,}/{before:,} matched ({matched/before*100:.1f}%)")

    # ── Merge Life Expectancy ──
    master = master.merge(life_exp, on=FIPS_COL, how="left", validate="m:1")
    assert len(master) == n_base, (
        f"Row count changed after Life Expectancy merge: {n_base} → {len(master)}. "
        "Possible duplicate FIPS in life expectancy dataset."
    )
    matched = master["life_expectancy"].notna().sum() if "life_expectancy" in master.columns else 0
    console.print(f"  + Life Exp: {matched:,}/{before:,} matched ({matched/before*100:.1f}%)")

    # ── Merge HRSA (county level → first 5 digits of tract FIPS) ──
    master["county_fips"] = master[FIPS_COL].str[:5]
    # HRSA is county-level so many tracts map to one county — "m:1" is correct here
    master = master.merge(hrsa, on="county_fips", how="left", validate="m:1")
    assert len(master) == n_base, (
        f"Row count changed after HRSA merge: {n_base} → {len(master)}. "
        "Possible duplicate county_fips in HRSA dataset."
    )
    matched = master["hpsa_score"].notna().sum() if "hpsa_score" in master.columns else 0
    console.print(f"  + HRSA: {matched:,}/{before:,} matched ({matched/before*100:.1f}%)")

    # ── Derived columns ──
    if "food_desert_1_10" in master.columns:
        master["is_food_desert"] = master["food_desert_1_10"].fillna(0).astype(int)

    race_cols = {"pct_white": "White", "pct_black": "Black", "pct_hispanic": "Hispanic"}
    avail = {k: v for k, v in race_cols.items() if k in master.columns}
    if avail:
        race_df = master[list(avail.keys())].fillna(0)
        max_pct = race_df.max(axis=1)
        top_race = race_df.idxmax(axis=1).map(avail)
        has_any = master[list(avail.keys())].notna().any(axis=1)
        master["majority_race"] = np.where(
            ~has_any, "Unknown",
            np.where(max_pct >= 40, top_race, "Other")
        )

    if "median_household_income" in master.columns:
        master["income_quintile"] = pd.qcut(
            master["median_household_income"].rank(method="first"),
            5, labels=[1, 2, 3, 4, 5],
        ).astype("Int64")

    # ── Drop helper column ──
    master = master.drop(columns=["county_fips"], errors="ignore")

    # ── Merge quality report ──
    _report_quality(master)

    # ── Validate for impossible values ──
    validate_master(master)

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out = DATA_PROCESSED / "master.parquet"
        master.to_parquet(out, index=False)
        console.print(f"\n[green bold]Master saved → {out}[/]")
        console.print(f"  {master.shape[0]:,} rows × {master.shape[1]} cols")

        # Also save schema for Alice
        schema = pd.DataFrame({
            "column": master.columns,
            "dtype": master.dtypes.astype(str).values,
            "non_null_pct": ((1 - master.isnull().mean()) * 100).round(1).values,
            "sample": [str(master[c].dropna().iloc[0])[:60] if master[c].notna().any() else "" for c in master.columns],
        })
        schema.to_csv(DATA_PROCESSED / "master_schema.csv", index=False)
        console.print("[green]Schema → data/processed/master_schema.csv[/]")

    return master


def _report_quality(df: pd.DataFrame) -> None:
    """Print merge quality metrics including population-weighted coverage and per-state tract loss."""
    console.rule("[bold]Merge Quality Report")
    console.print(f"Total tracts: {len(df):,}")

    key_cols = [
        "obesity_pct", "diabetes_pct", "life_expectancy",
        "median_household_income", "poverty_rate", "pct_black",
        "hpsa_score", "is_food_desert",
    ]
    for col in key_cols:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            console.print(f"  {col}: {100-null_pct:.1f}% coverage ({null_pct:.1f}% missing)")

    # ── Population-weighted coverage ──
    # Determine the best available population column
    pop_col = None
    for candidate in ["population", "acs_total_pop", "places_population"]:
        if candidate in df.columns and df[candidate].notna().sum() > 100:
            pop_col = candidate
            break

    if pop_col is not None:
        total_pop = df[pop_col].sum()
        if total_pop > 0:
            console.print(f"\n[bold]Population-weighted coverage (using {pop_col}):[/]")
            for col in key_cols:
                if col in df.columns:
                    covered_pop = df.loc[df[col].notna(), pop_col].sum()
                    pct_pop = covered_pop / total_pop * 100
                    console.print(f"  {col}: {pct_pop:.1f}% of US population covered")
    else:
        console.print("\n[yellow]  No population column found — skipping population-weighted coverage[/]")

    # ── Per-state tract loss warning (>15% missing a key outcome) ──
    # Extract state FIPS from tract_fips (first 2 digits)
    if FIPS_COL in df.columns and "diabetes_pct" in df.columns:
        state_col = df[FIPS_COL].str[:2]
        state_df = df.copy()
        state_df["_state"] = state_col
        state_coverage = (
            state_df.groupby("_state")["diabetes_pct"]
            .apply(lambda s: s.isna().mean() * 100)
            .rename("missing_pct")
        )
        high_loss_states = state_coverage[state_coverage > 15].sort_values(ascending=False)
        if not high_loss_states.empty:
            console.print("\n[yellow bold]States with >15% tract loss (diabetes_pct missing):[/]")
            for state_fips, missing_pct in high_loss_states.items():
                n_tracts = (state_df["_state"] == state_fips).sum()
                console.print(
                    f"  [yellow]State {state_fips}: {missing_pct:.1f}% missing "
                    f"({n_tracts:,} tracts total) — likely 2010/2020 census mismatch[/]"
                )
        else:
            console.print("\n  [green]No states with >15% tract loss[/]")

    # Usable rows (have food desert flag + at least one health outcome)
    health_cols = [c for c in ["obesity_pct", "diabetes_pct"] if c in df.columns]
    if health_cols and "is_food_desert" in df.columns:
        usable = df.dropna(subset=health_cols + ["is_food_desert"])
        console.print(f"\n  [bold]Usable for analysis: {len(usable):,} tracts[/]")


def validate_master(df: pd.DataFrame) -> dict:
    """Check for impossible values in the master dataframe.

    Validates:
    - Percentage columns (poverty_rate, pct_black, pct_white, pct_hispanic,
      pct_bachelors_plus, uninsured_pct, obesity_pct, diabetes_pct,
      physical_inactivity_pct, high_bp_pct, depression_pct) are in [0, 100].
    - life_expectancy is in [50, 100] years (plausible US census-tract range).
    - median_household_income is positive (> 0).
    - population is positive (> 0).

    Returns a dict with validation results including count and sample FIPS of
    offending rows. Logs warnings to console for any violations found.
    """
    console.rule("[bold]Master Validation")
    issues: dict[str, dict] = {}

    # ── Percentage columns: must be in [0, 100] ──
    pct_cols = [
        "poverty_rate", "pct_black", "pct_white", "pct_hispanic",
        "pct_bachelors_plus", "uninsured_pct", "obesity_pct", "diabetes_pct",
        "physical_inactivity_pct", "high_bp_pct", "depression_pct",
        "pct_low_access_1mi", "pct_low_access_10mi", "pct_lowinclow_access_1mi",
    ]
    for col in pct_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bad_mask = (series < 0) | (series > 100)
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            sample_fips = (
                df.loc[bad_mask, FIPS_COL].head(5).tolist()
                if FIPS_COL in df.columns else []
            )
            issues[col] = {
                "rule": "percentage must be in [0, 100]",
                "n_violations": bad_count,
                "sample_fips": sample_fips,
            }
            console.print(
                f"  [red]INVALID {col}: {bad_count} rows outside [0, 100] "
                f"— sample FIPS: {sample_fips}[/]"
            )

    # ── Life expectancy: plausible US census-tract range [50, 100] ──
    if "life_expectancy" in df.columns:
        le = pd.to_numeric(df["life_expectancy"], errors="coerce")
        bad_mask = le.notna() & ((le < 50) | (le > 100))
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            sample_fips = (
                df.loc[bad_mask, FIPS_COL].head(5).tolist()
                if FIPS_COL in df.columns else []
            )
            issues["life_expectancy"] = {
                "rule": "life_expectancy must be in [50, 100]",
                "n_violations": bad_count,
                "sample_fips": sample_fips,
            }
            console.print(
                f"  [red]INVALID life_expectancy: {bad_count} rows outside [50, 100] "
                f"— sample FIPS: {sample_fips}[/]"
            )

    # ── Median household income: must be positive ──
    if "median_household_income" in df.columns:
        inc = pd.to_numeric(df["median_household_income"], errors="coerce")
        bad_mask = inc.notna() & (inc <= 0)
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            sample_fips = (
                df.loc[bad_mask, FIPS_COL].head(5).tolist()
                if FIPS_COL in df.columns else []
            )
            issues["median_household_income"] = {
                "rule": "median_household_income must be > 0",
                "n_violations": bad_count,
                "sample_fips": sample_fips,
            }
            console.print(
                f"  [red]INVALID median_household_income: {bad_count} rows <= 0 "
                f"— sample FIPS: {sample_fips}[/]"
            )

    # ── Population: must be positive (> 0) ──
    pop_col = next((c for c in ["population", "acs_total_pop"] if c in df.columns), None)
    if pop_col is not None:
        pop = pd.to_numeric(df[pop_col], errors="coerce")
        bad_mask = pop.notna() & (pop <= 0)
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            sample_fips = (
                df.loc[bad_mask, FIPS_COL].head(5).tolist()
                if FIPS_COL in df.columns else []
            )
            issues[pop_col] = {
                "rule": "population must be > 0",
                "n_violations": bad_count,
                "sample_fips": sample_fips,
            }
            console.print(
                f"  [red]INVALID {pop_col}: {bad_count} rows <= 0 "
                f"— sample FIPS: {sample_fips}[/]"
            )

    if not issues:
        console.print("  [green]All validation checks passed — no impossible values detected[/]")
    else:
        console.print(
            f"  [yellow bold]{len(issues)} column(s) with validation issues. "
            "Review before analysis.[/]"
        )

    return issues
