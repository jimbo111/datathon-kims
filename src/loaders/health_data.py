"""Health dataset downloaders and loaders.

Downloads, cleans, and standardizes five federal datasets:
1. USDA Food Access Research Atlas (food desert classification by tract)
2. CDC PLACES (chronic disease prevalence by tract)
3. ACS 5-Year Estimates (demographics by tract via Census API)
4. CDC Life Expectancy at Birth (by tract, USALEEP)
5. HRSA HPSA (primary care shortage areas)

All datasets are cached locally after first download.
"""

import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from tqdm import tqdm

console = Console()

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
FIPS_COL = "tract_fips"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Datathon Research)"})

# ─── USDA Food Access Research Atlas ──────────────────────────────────────────

USDA_URL = (
    "https://www.ers.usda.gov/media/5627/"
    "food-access-research-atlas-data-download-2019.zip?v=47524"
)
USDA_DIR = DATA_RAW / "food_atlas"

USDA_KEEP = {
    "CensusTract": FIPS_COL,
    "State": "state",
    "County": "county",
    "Urban": "urban",
    "Pop2010": "population",
    "PovertyRate": "usda_poverty_rate",
    "MedianFamilyIncome": "usda_median_family_income",
    "LILATracts_1And10": "food_desert_1_10",
    "LILATracts_halfAnd10": "food_desert_half_10",
    "LILATracts_1And20": "food_desert_1_20",
    "LILATracts_Vehicle": "food_desert_vehicle",
    "lapop1share": "pct_low_access_1mi",
    "lapop10share": "pct_low_access_10mi",
    "lalowi1share": "pct_lowinclow_access_1mi",
    "TractLOWI": "low_income_tract",
    "TractSNAP": "snap_tract",
    "NUMGQTRS": "group_quarters_flag",
}


def download_usda() -> Path:
    """Download and extract the USDA Food Access Atlas ZIP."""
    USDA_DIR.mkdir(parents=True, exist_ok=True)
    if any(USDA_DIR.glob("*.xlsx")) or any(USDA_DIR.glob("*.csv")):
        console.print("[yellow]USDA: already downloaded, skipping.[/]")
        return USDA_DIR

    console.print("[cyan]Downloading USDA Food Access Research Atlas (~9 MB)...[/]")
    resp = _SESSION.get(USDA_URL, timeout=180)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(USDA_DIR)
    console.print(f"[green]USDA extracted → {USDA_DIR}[/]")
    return USDA_DIR


def load_usda() -> pd.DataFrame:
    """Load and clean USDA Food Access Atlas → one row per tract."""
    download_usda()

    xlsx = list(USDA_DIR.rglob("*.xlsx"))
    csv = sorted(USDA_DIR.rglob("*.csv"), key=lambda f: f.stat().st_size, reverse=True)
    if xlsx:
        console.print(f"[cyan]Reading {xlsx[0].name} (may take ~30s)...[/]")
        df = pd.read_excel(xlsx[0], sheet_name=0, engine="openpyxl")
    elif csv:
        # Pick the largest CSV (skip ReadMe.csv, VariableLookup.csv, etc.)
        console.print(f"[cyan]Reading {csv[0].name}...[/]")
        df = pd.read_csv(csv[0])
    else:
        raise FileNotFoundError(f"No data files in {USDA_DIR}")

    available = {k: v for k, v in USDA_KEEP.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    df[FIPS_COL] = df[FIPS_COL].astype(str).str.zfill(11)

    if "group_quarters_flag" in df.columns:
        df = df[df["group_quarters_flag"] == 0].drop(columns=["group_quarters_flag"])

    console.print(f"[green]USDA: {df.shape[0]:,} tracts × {df.shape[1]} cols[/]")
    return df


# ─── CDC PLACES (Census Tract Level) ─────────────────────────────────────────

PLACES_API = "https://data.cdc.gov/resource/cwsq-ngmh.csv"
PLACES_MEASURES = ["OBESITY", "DIABETES", "LPA", "BPHIGH", "ACCESS2", "DEPRESSION", "CHD"]
PLACES_CACHE = DATA_RAW / "cdc_places_tract.parquet"


def download_places() -> pd.DataFrame:
    """Download CDC PLACES census-tract data for selected health measures."""
    if PLACES_CACHE.exists():
        console.print("[yellow]CDC PLACES: loading from cache.[/]")
        return pd.read_parquet(PLACES_CACHE)

    console.print("[cyan]Downloading CDC PLACES (census tract, 7 measures)...[/]")
    measures_str = "','".join(PLACES_MEASURES)
    where = (
        f"measureid IN ('{measures_str}') "
        f"AND data_value_type='Crude prevalence'"
    )

    frames = []
    offset = 0
    limit = 50_000
    while True:
        params = {"$where": where, "$limit": limit, "$offset": offset}
        resp = _SESSION.get(PLACES_API, params=params, timeout=180)
        resp.raise_for_status()
        batch = pd.read_csv(io.StringIO(resp.text))
        if batch.empty:
            break
        frames.append(batch)
        console.print(f"  fetched {offset + len(batch):,} rows...")
        if len(batch) < limit:
            break
        offset += limit

    if not frames:
        raise RuntimeError("CDC PLACES download returned no data. Check network connection.")
    df = pd.concat(frames, ignore_index=True)
    PLACES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PLACES_CACHE, index=False)
    console.print(f"[green]CDC PLACES: {df.shape[0]:,} rows cached.[/]")
    return df


def load_places() -> pd.DataFrame:
    """Load CDC PLACES and pivot to one row per tract."""
    raw = download_places()

    pivot = raw.pivot_table(
        index="locationid", columns="measureid",
        values="data_value", aggfunc="first",
    ).reset_index()

    rename = {
        "locationid": FIPS_COL,
        "OBESITY": "obesity_pct",
        "DIABETES": "diabetes_pct",
        "LPA": "physical_inactivity_pct",
        "BPHIGH": "high_bp_pct",
        "ACCESS2": "uninsured_pct",
        "DEPRESSION": "depression_pct",
        "CHD": "chd_pct",
    }
    pivot = pivot.rename(columns={k: v for k, v in rename.items() if k in pivot.columns})

    pop = raw.groupby("locationid")["totalpopulation"].first().reset_index()
    pop.columns = [FIPS_COL, "places_population"]
    pivot = pivot.merge(pop, on=FIPS_COL, how="left")

    pivot[FIPS_COL] = pivot[FIPS_COL].astype(str).str.zfill(11)

    console.print(f"[green]PLACES pivoted: {pivot.shape[0]:,} tracts × {pivot.shape[1]} cols[/]")
    return pivot


# ─── ACS 5-Year Estimates (Census API) ───────────────────────────────────────

CENSUS_API = "https://api.census.gov/data/2022/acs/acs5"
ACS_VARS = {
    "B19013_001E": "median_household_income",
    "B17001_001E": "_poverty_universe",
    "B17001_002E": "_below_poverty",
    "B02001_001E": "_total_pop",
    "B02001_002E": "_pop_white",
    "B02001_003E": "_pop_black",
    "B03003_003E": "_pop_hispanic",
    "B15003_001E": "_edu_universe",
    "B15003_022E": "_edu_bach",
    "B15003_023E": "_edu_mast",
    "B15003_024E": "_edu_prof",
    "B15003_025E": "_edu_doct",
}
ACS_CACHE = DATA_RAW / "acs_tract.parquet"

STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56",
]


def download_acs(api_key: Optional[str] = None) -> pd.DataFrame:
    """Download ACS 5-Year tract-level demographics via Census API."""
    if ACS_CACHE.exists():
        console.print("[yellow]ACS: loading from cache.[/]")
        return pd.read_parquet(ACS_CACHE)

    console.print("[cyan]Downloading ACS 5-Year (tract level, 52 states)...[/]")
    var_list = ",".join(ACS_VARS.keys())
    frames = []

    for st in tqdm(STATE_FIPS, desc="ACS by state"):
        params = {"get": f"NAME,{var_list}", "for": "tract:*", "in": f"state:{st}"}
        if api_key:
            params["key"] = api_key
        try:
            resp = _SESSION.get(CENSUS_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            frames.append(pd.DataFrame(data[1:], columns=data[0]))
        except Exception as e:
            console.print(f"[red]  state {st} failed: {e}[/]")

    if not frames:
        raise RuntimeError("ACS download returned no data. Check network/Census API.")
    df = pd.concat(frames, ignore_index=True)
    ACS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ACS_CACHE, index=False)
    console.print(f"[green]ACS: {df.shape[0]:,} tracts cached.[/]")
    return df


def load_acs(api_key: Optional[str] = None) -> pd.DataFrame:
    """Load ACS data and compute derived percentages."""
    raw = download_acs(api_key)

    raw[FIPS_COL] = raw["state"] + raw["county"] + raw["tract"]
    raw = raw.rename(columns={k: v for k, v in ACS_VARS.items() if k in raw.columns})

    for col in ACS_VARS.values():
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = pd.DataFrame({FIPS_COL: raw[FIPS_COL]})

    # Census API uses negative sentinel values for suppressed/missing data
    income = raw["median_household_income"].copy()
    income = income.where(income > 0, np.nan)
    df["median_household_income"] = income

    pop = raw["_total_pop"].copy()
    pop = pop.where(pop >= 0, np.nan)
    pov = raw["_poverty_universe"].copy()
    pov = pov.where(pov >= 0, np.nan)

    df["poverty_rate"] = np.where(pov > 0, raw["_below_poverty"] / pov * 100, np.nan)
    df["pct_white"] = np.where(pop > 0, raw["_pop_white"] / pop * 100, np.nan)
    df["pct_black"] = np.where(pop > 0, raw["_pop_black"] / pop * 100, np.nan)
    df["pct_hispanic"] = np.where(pop > 0, raw["_pop_hispanic"] / pop * 100, np.nan)

    edu = raw["_edu_universe"]
    bach_plus = (
        raw["_edu_bach"].fillna(0)
        + raw["_edu_mast"].fillna(0)
        + raw["_edu_prof"].fillna(0)
        + raw["_edu_doct"].fillna(0)
    )
    df["pct_bachelors_plus"] = np.where(edu > 0, bach_plus / edu * 100, np.nan)
    df["acs_total_pop"] = pop

    console.print(f"[green]ACS cleaned: {df.shape[0]:,} tracts × {df.shape[1]} cols[/]")
    return df


# ─── CDC Life Expectancy (USALEEP) ───────────────────────────────────────────

LIFE_EXP_FTP = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NVSS/USALEEP/CSV/US_B.CSV"
LIFE_EXP_CACHE = DATA_RAW / "life_expectancy_tract.parquet"


def download_life_expectancy() -> pd.DataFrame:
    """Download CDC Life Expectancy at Birth by census tract (from CDC FTP)."""
    if LIFE_EXP_CACHE.exists():
        console.print("[yellow]Life expectancy: loading from cache.[/]")
        return pd.read_parquet(LIFE_EXP_CACHE)

    console.print("[cyan]Downloading CDC Life Expectancy (USALEEP, ~25 MB)...[/]")
    resp = _SESSION.get(LIFE_EXP_FTP, timeout=180)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.content.decode("latin-1")))
    LIFE_EXP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LIFE_EXP_CACHE, index=False)
    console.print(f"[green]Life expectancy: {df.shape[0]:,} rows cached.[/]")
    return df


def load_life_expectancy() -> pd.DataFrame:
    """Load and clean life expectancy data — extract e(0) for each tract.

    The CDC FTP life table has columns: Tract ID, STATE2KX, CNTY2KX, TRACT2KX,
    Age Group, nq(x), l(x), nd(x), nL(x), T(x), e(x), se(nq(x)), se(e(x)).
    We filter for 'Under 1' age group to get life expectancy at birth e(0).
    """
    raw = download_life_expectancy()

    # FTP format: "Tract ID" is the 11-digit FIPS, "e(x)" is life expectancy,
    # "Age Group" contains "Under 1" for birth life expectancy
    fips_col_name = next((c for c in raw.columns if "tract" in c.lower() and "id" in c.lower()), raw.columns[0])
    le_col_name = next((c for c in raw.columns if c.strip() == "e(x)"), None)
    se_col_name = next((c for c in raw.columns if c.strip() == "se(e(x))"), None)
    age_col_name = next((c for c in raw.columns if "age" in c.lower()), None)

    # Filter for life expectancy at birth (Under 1 age group)
    if age_col_name:
        birth = raw[raw[age_col_name].str.strip().str.lower() == "under 1"].copy()
    else:
        birth = raw.copy()

    df = pd.DataFrame()
    df[FIPS_COL] = birth[fips_col_name].astype(str).str.strip().str.zfill(11)

    if le_col_name:
        df["life_expectancy"] = pd.to_numeric(birth[le_col_name], errors="coerce")
    if se_col_name:
        df["life_expectancy_se"] = pd.to_numeric(birth[se_col_name], errors="coerce")

    df = df.dropna(subset=["life_expectancy"])
    df = df.drop_duplicates(subset=[FIPS_COL])

    console.print(f"[green]Life expectancy: {df.shape[0]:,} tracts[/]")
    return df


# ─── HRSA HPSA (Primary Care Shortage Areas) ─────────────────────────────────

HRSA_URL = "https://data.hrsa.gov/DataDownload/DD_Files/BCD_HPSA_FCT_DET_PC.csv"
HRSA_CACHE = DATA_RAW / "hpsa_primary_care.csv"


def download_hrsa() -> Path:
    """Download HRSA HPSA primary care data."""
    if HRSA_CACHE.exists():
        console.print("[yellow]HRSA: already downloaded, skipping.[/]")
        return HRSA_CACHE

    console.print("[cyan]Downloading HRSA HPSA Primary Care...[/]")
    HRSA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    resp = _SESSION.get(HRSA_URL, timeout=180)
    resp.raise_for_status()
    HRSA_CACHE.write_bytes(resp.content)
    console.print(f"[green]HRSA saved → {HRSA_CACHE}[/]")
    return HRSA_CACHE


def load_hrsa() -> pd.DataFrame:
    """Load HRSA HPSA and aggregate to county FIPS level."""
    download_hrsa()
    df = pd.read_csv(HRSA_CACHE, low_memory=False, encoding="latin-1")

    # Find key columns by partial name match
    def _find_col(df, patterns):
        for p in patterns:
            matches = [c for c in df.columns if p.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    fips_col = _find_col(df, ["Common County FIPS", "county fips", "FIPS"])
    score_col = _find_col(df, ["HPSA Score", "hpsa_score"])
    status_col = _find_col(df, ["HPSA Status", "hpsa_status"])

    if fips_col is None:
        console.print(f"[red]HRSA: no FIPS column found. Cols: {df.columns.tolist()[:10]}[/]")
        return pd.DataFrame(columns=["county_fips", "hpsa_score", "hpsa_shortage"])

    result = pd.DataFrame()
    result["county_fips"] = df[fips_col].astype(str).str.zfill(5)

    if score_col:
        result["hpsa_score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        result["hpsa_score"] = np.nan

    if status_col:
        result["hpsa_shortage"] = df[status_col].str.lower().str.contains("designated", na=False).astype(int)
    else:
        result["hpsa_shortage"] = 1  # if in HPSA file at all, it's a shortage area

    agg = result.groupby("county_fips").agg(
        hpsa_score=("hpsa_score", "max"),
        hpsa_shortage=("hpsa_shortage", "max"),
    ).reset_index()

    console.print(f"[green]HRSA: {agg.shape[0]:,} counties[/]")
    return agg


# ─── Download All ─────────────────────────────────────────────────────────────

def download_all(census_api_key: Optional[str] = None) -> None:
    """Download all five datasets (idempotent — skips cached)."""
    console.rule("[bold]Downloading all health datasets")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    download_usda()
    download_places()
    download_acs(census_api_key)
    download_life_expectancy()
    download_hrsa()
    console.rule("[bold green]All downloads complete")
