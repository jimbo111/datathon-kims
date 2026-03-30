"""Data endpoints — load, query, and inspect health datasets."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.services.data_service import get_data_service

router = APIRouter()
data_svc = get_data_service()

# ── Security: resolve DATA_DIR for path traversal validation ──
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _validate_filename(filename: str, subdir: str) -> None:
    """Reject filenames containing path traversal sequences.

    Raises HTTPException 400 if:
    - filename contains '..' or '/'
    - resolved path escapes DATA_DIR/subdir
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename '{filename}': path separators and '..' are not allowed.",
        )
    resolved = (DATA_DIR / subdir / filename).resolve()
    allowed_root = (DATA_DIR / subdir).resolve()
    if not str(resolved).startswith(str(allowed_root)):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename '{filename}': resolved path escapes the data directory.",
        )


@router.get("/datasets")
async def list_datasets():
    """List available datasets in data/raw and data/processed."""
    return data_svc.list_datasets()


@router.post("/load/{filename}")
async def load_dataset(filename: str, subdir: str = Query("raw")):
    """Load a dataset into memory."""
    _validate_filename(filename, subdir)
    data_svc.load(filename, subdir=subdir)
    return {"loaded": filename, "shape": list(data_svc.current_shape())}


@router.post("/load-master")
async def load_master():
    """Load the master health dataset (data/processed/master.parquet)."""
    data_svc.load_master()
    return {"loaded": "master.parquet", "shape": list(data_svc.current_shape())}


@router.get("/info")
async def dataset_info():
    """Return column info for the loaded dataset."""
    return JSONResponse(content=data_svc.info())


@router.get("/head")
async def head(n: int = Query(20, ge=1, le=500)):
    """Return first n rows as JSON records."""
    return JSONResponse(content=data_svc.head(n))


@router.get("/columns")
async def columns():
    """Return column names and dtypes."""
    return data_svc.columns()


@router.get("/stats")
async def summary_stats():
    """Return extended summary statistics."""
    return JSONResponse(content=data_svc.stats())


# ── Health-specific endpoints ──


@router.get("/food-desert-comparison")
async def food_desert_comparison():
    """Compare health outcomes in food deserts vs non-food-deserts."""
    return JSONResponse(content=data_svc.food_desert_comparison())


@router.get("/income-quintile-stats")
async def income_quintile_stats():
    """Health metrics by income quintile."""
    return JSONResponse(content=data_svc.income_quintile_stats())


@router.get("/race-diabetes-matrix")
async def race_diabetes_matrix():
    """Income × race → diabetes prevalence matrix for heatmap."""
    return JSONResponse(content=data_svc.race_diabetes_matrix())


@router.get("/analysis/{phase}")
async def get_analysis(phase: str):
    """Load pre-computed analysis JSON (phase2, phase3, phase4, phase5)."""
    return JSONResponse(content=data_svc.load_analysis_json(phase))


@router.get("/hdi-ranked")
async def hdi_ranked(top_n: int = Query(50, ge=1, le=500)):
    """Top N most disadvantaged tracts by Health Disadvantage Index."""
    return JSONResponse(content=data_svc.hdi_ranked_tracts(top_n))
