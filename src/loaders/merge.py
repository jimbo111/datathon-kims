"""Merge all health datasets into a single master dataframe on census tract FIPS.

Handles the 2010/2020 census tract mismatch:
- CDC PLACES + ACS use 2020 census tracts
- USDA + USALEEP use 2010 census tracts
- HRSA is at county level (stable across censuses)

Most tract FIPS codes are unchanged between censuses, so a direct join
captures ~85-95% of tracts. The merge quality report documents coverage.
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
    """
    console.rule("[bold]Building master dataframe")

    usda = load_usda()
    places = load_places()
    acs = load_acs(census_api_key)
    life_exp = load_life_expectancy()
    hrsa = load_hrsa()

    # ── Start with USDA as the base (food desert flags are the core IV) ──
    master = usda.copy()
    console.print(f"[cyan]Base (USDA): {len(master):,} tracts[/]")

    # ── Merge CDC PLACES (health outcomes) ──
    before = master[FIPS_COL].nunique()
    master = master.merge(places, on=FIPS_COL, how="left")
    matched = master["obesity_pct"].notna().sum()
    console.print(f"  + PLACES: {matched:,}/{before:,} matched ({matched/before*100:.1f}%)")

    # ── Merge ACS (demographics) ──
    master = master.merge(acs, on=FIPS_COL, how="left")
    matched = master["median_household_income"].notna().sum()
    console.print(f"  + ACS: {matched:,}/{before:,} matched ({matched/before*100:.1f}%)")

    # ── Merge Life Expectancy ──
    master = master.merge(life_exp, on=FIPS_COL, how="left")
    matched = master["life_expectancy"].notna().sum() if "life_expectancy" in master.columns else 0
    console.print(f"  + Life Exp: {matched:,}/{before:,} matched ({matched/before*100:.1f}%)")

    # ── Merge HRSA (county level → first 5 digits of tract FIPS) ──
    master["county_fips"] = master[FIPS_COL].str[:5]
    master = master.merge(hrsa, on="county_fips", how="left")
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
    """Print merge quality metrics."""
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

    # Usable rows (have food desert flag + at least one health outcome)
    health_cols = [c for c in ["obesity_pct", "diabetes_pct"] if c in df.columns]
    if health_cols and "is_food_desert" in df.columns:
        usable = df.dropna(subset=health_cols + ["is_food_desert"])
        console.print(f"\n  [bold]Usable for analysis: {len(usable):,} tracts[/]")
