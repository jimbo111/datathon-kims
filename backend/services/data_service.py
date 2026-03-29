"""Data service — manages health datasets and analysis results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.loaders.loader import load_data

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DataService:
    def __init__(self):
        self._df: pd.DataFrame | None = None
        self._name: str | None = None
        self._analysis_cache: dict = {}

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No dataset loaded. POST /api/data/load/{filename} first.")
        return self._df

    def load(self, filename: str, subdir: str = "raw") -> None:
        self._df = load_data(filename, subdir=subdir)
        self._name = filename

    def load_master(self) -> None:
        """Load the pre-built master.parquet health dataset."""
        path = DATA_DIR / "processed" / "master.parquet"
        if not path.exists():
            raise FileNotFoundError(
                "master.parquet not found. Run the data pipeline first: "
                "from src.loaders import build_master; build_master()"
            )
        self._df = pd.read_parquet(path)
        self._name = "master.parquet"

    def current_shape(self) -> tuple[int, int]:
        return self.df.shape

    def list_datasets(self) -> dict:
        result = {}
        for sub in ["raw", "processed", "external"]:
            d = DATA_DIR / sub
            if d.exists():
                files = [f.name for f in d.iterdir() if f.is_file() and f.name != ".gitkeep"]
                if files:
                    result[sub] = files
        return result

    def info(self) -> dict:
        df = self.df
        info = []
        for col in df.columns:
            s = df[col]
            info.append({
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "null": int(s.isna().sum()),
                "unique": int(s.nunique()),
                "sample": _safe_sample(s),
            })
        return {"name": self._name, "shape": list(df.shape), "columns": info}

    def head(self, n: int = 20) -> list[dict]:
        return self.df.head(n).replace({np.nan: None}).to_dict(orient="records")

    def columns(self) -> dict:
        return {
            "numeric": self.df.select_dtypes(include="number").columns.tolist(),
            "categorical": self.df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime": self.df.select_dtypes(include="datetime").columns.tolist(),
        }

    def stats(self) -> list[dict]:
        desc = self.df.describe(include="all").T
        desc = desc.where(desc.notna(), None)
        return desc.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")

    # ── Health-specific queries ──

    def food_desert_comparison(self) -> dict:
        """Compare health outcomes in food deserts vs non-food-deserts."""
        df = self.df
        if "is_food_desert" not in df.columns:
            return {"error": "is_food_desert column not found"}

        metrics = ["diabetes_pct", "obesity_pct", "life_expectancy", "high_bp_pct"]
        available = [m for m in metrics if m in df.columns]

        result = {}
        for col in available:
            desert = df[df["is_food_desert"] == 1][col].dropna()
            non_desert = df[df["is_food_desert"] == 0][col].dropna()
            result[col] = {
                "food_desert_mean": round(desert.mean(), 2) if len(desert) > 0 else None,
                "non_food_desert_mean": round(non_desert.mean(), 2) if len(non_desert) > 0 else None,
                "difference": round(desert.mean() - non_desert.mean(), 2) if len(desert) > 0 and len(non_desert) > 0 else None,
            }
        return result

    def income_quintile_stats(self) -> list[dict]:
        """Health metrics broken down by income quintile."""
        df = self.df
        if "income_quintile" not in df.columns:
            return []

        metrics = ["diabetes_pct", "obesity_pct", "life_expectancy", "physical_inactivity_pct"]
        available = [m for m in metrics if m in df.columns]

        groups = df.groupby("income_quintile")[available].mean().round(2)
        return groups.reset_index().to_dict(orient="records")

    def race_diabetes_matrix(self) -> dict:
        """Income quintile × majority race → diabetes prevalence matrix."""
        df = self.df
        required = ["income_quintile", "majority_race", "diabetes_pct"]
        if not all(c in df.columns for c in required):
            return {"error": "Required columns not found"}

        matrix = df.pivot_table(
            index="income_quintile", columns="majority_race",
            values="diabetes_pct", aggfunc="mean",
        ).round(2)
        return {
            "matrix": matrix.reset_index().replace({np.nan: None}).to_dict(orient="records"),
            "races": matrix.columns.tolist(),
        }

    def load_analysis_json(self, phase: str) -> dict:
        """Load pre-computed analysis results from JSON."""
        filenames = {
            "phase2": "phase2_food_access_disease.json",
            "phase3": "phase3_zip_code_effect.json",
            "phase4": "phase4_race_residual_gap.json",
            "phase5": "phase5_health_index.json",
        }
        fname = filenames.get(phase)
        if not fname:
            return {"error": f"Unknown phase: {phase}"}

        path = DATA_DIR / "processed" / fname
        if not path.exists():
            return {"error": f"{fname} not found. Run analysis first."}

        with open(path) as f:
            return json.load(f)

    def hdi_ranked_tracts(self, top_n: int = 50) -> list[dict]:
        """Return top N most disadvantaged tracts by HDI score."""
        path = DATA_DIR / "processed" / "health_disadvantage_index.parquet"
        if not path.exists():
            return []
        hdi = pd.read_parquet(path).sort_values("hdi_score", ascending=False)
        return hdi.head(top_n).replace({np.nan: None}).to_dict(orient="records")


def _safe_sample(s: pd.Series) -> str | None:
    if s.notna().any():
        val = s.dropna().iloc[0]
        text = str(val)
        return text[:80] if len(text) > 80 else text
    return None


# ── Singleton ──

_instance: DataService | None = None


def get_data_service() -> DataService:
    """Return a shared DataService instance, auto-loading master.parquet if available."""
    global _instance
    if _instance is None:
        _instance = DataService()
        try:
            _instance.load_master()
        except FileNotFoundError:
            pass  # master.parquet not built yet — will use sample data fallback
    return _instance
