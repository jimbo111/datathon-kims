"""Reusable plotting functions — quick EDA charts."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_style():
    """Apply consistent plot styling."""
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


set_style()


def distribution_grid(df: pd.DataFrame, cols: list[str] | None = None, bins: int = 30):
    """Histogram grid for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(cols):
        df[col].hist(ax=axes[i], bins=bins, edgecolor="white")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


def correlation_matrix(df: pd.DataFrame, method: str = "pearson", annot: bool = True):
    """Heatmap of correlations for numeric columns."""
    corr = df.select_dtypes(include="number").corr(method=method)
    fig, ax = plt.subplots(figsize=(max(8, len(corr)), max(6, len(corr) * 0.8)))
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title(f"Correlation Matrix ({method})")
    plt.tight_layout()
    return fig


def category_counts(df: pd.DataFrame, col: str, top_n: int = 20, horizontal: bool = True):
    """Bar chart of value counts for a categorical column."""
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.4)) if horizontal else (10, 6))
    if horizontal:
        counts.plot.barh(ax=ax, edgecolor="white")
        ax.invert_yaxis()
    else:
        counts.plot.bar(ax=ax, edgecolor="white")
    ax.set_title(f"{col} — Top {top_n} Values")
    plt.tight_layout()
    return fig


def scatter_pair(df: pd.DataFrame, x: str, y: str, hue: str | None = None):
    """Scatter plot with optional hue."""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, ax=ax)
    ax.set_title(f"{y} vs {x}")
    plt.tight_layout()
    return fig


def time_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str | None = None):
    """Line plot for time series data."""
    ts = df.set_index(date_col)[value_col]
    if freq:
        ts = ts.resample(freq).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ts.plot(ax=ax)
    ax.set_title(f"{value_col} over time")
    plt.tight_layout()
    return fig
