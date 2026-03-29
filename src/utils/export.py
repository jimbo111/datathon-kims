"""Submission and export helpers."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()

SUBMISSIONS_DIR = Path(__file__).resolve().parents[2] / "submissions"


def save_submission(
    df: pd.DataFrame,
    filename: str | None = None,
    fmt: str = "csv",
    **kwargs,
) -> Path:
    """Save a DataFrame as a submission file with timestamp."""
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"submission_{ts}"
    if not filename.endswith(f".{fmt}"):
        filename = f"{filename}.{fmt}"

    path = SUBMISSIONS_DIR / filename

    if fmt == "csv":
        df.to_csv(path, index=False, **kwargs)
    elif fmt == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif fmt == "excel":
        df.to_excel(path, index=False, **kwargs)
    elif fmt == "json":
        df.to_json(path, orient="records", **kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    console.print(f"[green]Saved submission:[/] {path} ({df.shape[0]:,} rows)")
    return path


def save_figure(fig, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure to submissions/."""
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SUBMISSIONS_DIR / f"{name}_{ts}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    console.print(f"[green]Saved figure:[/] {path}")
    return path
