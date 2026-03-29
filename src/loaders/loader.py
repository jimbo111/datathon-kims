"""Universal data loader — handles CSV, Excel, Parquet, JSON, SQL, and APIs."""

from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_data(
    source: str,
    subdir: str = "raw",
    *,
    sheet_name: str | int = 0,
    sql_con: Any = None,
    **kwargs,
) -> pd.DataFrame:
    """Load data from any common format. Auto-detects by extension.

    Args:
        source: filename inside data/{subdir}, full path, URL, or SQL query.
        subdir: subdirectory under data/ ("raw", "processed", "external").
        sheet_name: for Excel files.
        sql_con: SQLAlchemy connection for SQL queries.
        **kwargs: passed through to the pandas reader.
    """
    # SQL query
    if sql_con is not None:
        console.print(f"[cyan]Loading SQL query...[/]")
        return pd.read_sql(source, sql_con, **kwargs)

    # URL
    if source.startswith(("http://", "https://")):
        console.print(f"[cyan]Fetching from URL: {source}[/]")
        if source.endswith(".parquet"):
            return pd.read_parquet(source, **kwargs)
        if source.endswith(".json"):
            return pd.read_json(source, **kwargs)
        return pd.read_csv(source, **kwargs)

    # Local file
    path = Path(source)
    if not path.is_absolute():
        path = DATA_DIR / subdir / source

    if not path.exists():
        raise FileNotFoundError(f"No file at {path}")

    suffix = path.suffix.lower()
    console.print(f"[cyan]Loading {path.name}[/] ({suffix})")

    readers = {
        ".csv": lambda: pd.read_csv(path, **kwargs),
        ".tsv": lambda: pd.read_csv(path, sep="\t", **kwargs),
        ".xlsx": lambda: pd.read_excel(path, sheet_name=sheet_name, **kwargs),
        ".xls": lambda: pd.read_excel(path, sheet_name=sheet_name, **kwargs),
        ".parquet": lambda: pd.read_parquet(path, **kwargs),
        ".json": lambda: pd.read_json(path, **kwargs),
        ".jsonl": lambda: pd.read_json(path, lines=True, **kwargs),
        ".feather": lambda: pd.read_feather(path, **kwargs),
    }

    reader = readers.get(suffix)
    if reader is None:
        raise ValueError(f"Unsupported format: {suffix}")

    return reader()


def describe_data(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print a rich summary of a DataFrame."""
    console.rule(f"[bold]{name}")
    console.print(f"Shape: [green]{df.shape[0]:,}[/] rows x [green]{df.shape[1]}[/] cols")

    # Types
    table = Table(title="Column Info", show_lines=False)
    table.add_column("Column", style="bold")
    table.add_column("Type")
    table.add_column("Non-Null")
    table.add_column("Unique")
    table.add_column("Sample")

    for col in df.columns:
        series = df[col]
        sample = str(series.dropna().iloc[0]) if series.notna().any() else "N/A"
        if len(sample) > 40:
            sample = sample[:37] + "..."
        table.add_row(
            col,
            str(series.dtype),
            f"{series.notna().sum():,}",
            f"{series.nunique():,}",
            sample,
        )
    console.print(table)

    # Missing
    missing = df.isnull().sum()
    if missing.any():
        console.print("\n[yellow]Missing values:[/]")
        for col, count in missing[missing > 0].items():
            pct = count / len(df) * 100
            console.print(f"  {col}: {count:,} ({pct:.1f}%)")

    console.rule()
