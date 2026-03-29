"""Chart service — generates Plotly JSON figures for the health dashboard.

Uses real tract-level data from master.parquet (aggregated to state level for
choropleths). Falls back to sample data if the pipeline hasn't been run yet.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from backend.services.data_service import get_data_service

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# State FIPS → 2-letter postal code mapping
STATE_FIPS_TO_POSTAL = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR",
}

POSTAL_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}

DARK_LAYOUT = dict(
    geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#e1e4e8",
    margin=dict(t=50, b=10, l=10, r=10),
)

DARK_XY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#e1e4e8",
    margin=dict(t=50, b=50, l=60, r=20),
    xaxis=dict(gridcolor="#21262d"),
    yaxis=dict(gridcolor="#21262d"),
)

# Cached state-level aggregation
_state_cache: pd.DataFrame | None = None


def _build_state_summary() -> pd.DataFrame:
    """Aggregate tract-level master data to state level for choropleths."""
    svc = get_data_service()
    df = svc.df

    df = df.copy()
    df["state_fips"] = df["tract_fips"].str[:2]
    df["state_postal"] = df["state_fips"].map(STATE_FIPS_TO_POSTAL)

    agg = df.groupby("state_postal").agg(
        food_access_score=("pct_low_access_1mi", lambda x: round((1 - x.mean() / 100) * 100, 1) if x.notna().any() else np.nan),
        pct_food_desert=("is_food_desert", lambda x: round(x.mean() * 100, 1)),
        diabetes_rate=("diabetes_pct", "mean"),
        obesity_rate=("obesity_pct", "mean"),
        median_income=("median_household_income", "median"),
        pct_uninsured=("uninsured_pct", "mean"),
        pct_minority=("pct_white", lambda x: round(100 - x.mean(), 1) if x.notna().any() else np.nan),
        life_expectancy=("life_expectancy", "mean"),
    ).reset_index()

    agg = agg.rename(columns={"state_postal": "state"})
    agg["state_name"] = agg["state"].map(POSTAL_TO_NAME)

    # Round numeric columns
    for col in ["diabetes_rate", "obesity_rate", "life_expectancy", "pct_uninsured"]:
        if col in agg.columns:
            agg[col] = agg[col].round(1)
    if "median_income" in agg.columns:
        agg["median_income"] = agg["median_income"].astype("Int64")

    return agg.dropna(subset=["state"])


def _get_health_df() -> pd.DataFrame:
    """Get state-level health data, with fallback to sample data."""
    global _state_cache
    if _state_cache is not None:
        return _state_cache

    try:
        _state_cache = _build_state_summary()
    except (ValueError, KeyError, FileNotFoundError):
        from backend.services.sample_data import generate_health_data
        _state_cache = generate_health_data()

    return _state_cache


def _get_df():
    """Get the raw loaded dataframe for generic chart methods."""
    return get_data_service().df


def _decode_bdata(obj):
    """Recursively decode Plotly 6.x binary-encoded arrays to plain lists."""
    import base64
    import struct

    DTYPE_MAP = {
        "i1": ("b", 1), "u1": ("B", 1),
        "i2": ("<h", 2), "u2": ("<H", 2),
        "i4": ("<i", 4), "u4": ("<I", 4),
        "f4": ("<f", 4), "f8": ("<d", 8),
    }

    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            raw = base64.b64decode(obj["bdata"])
            fmt, size = DTYPE_MAP.get(obj["dtype"], ("<d", 8))
            count = len(raw) // size
            values = list(struct.unpack(f"<{count}{fmt[-1]}", raw))
            return values
        return {k: _decode_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bdata(item) for item in obj]
    return obj


def _fig_to_dict(fig: go.Figure) -> dict:
    raw = json.loads(pio.to_json(fig))
    return _decode_bdata(raw)


class ChartService:

    # ── Generic charts (Data Explorer) ──

    def distribution(self, column: str, bins: int = 30) -> dict:
        fig = px.histogram(_get_df(), x=column, nbins=bins, title=f"Distribution of {column}")
        fig.update_layout(bargap=0.05)
        return _fig_to_dict(fig)

    def correlation_matrix(self, method: str = "pearson") -> dict:
        corr = _get_df().select_dtypes(include="number").corr(method=method)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title=f"Correlation Matrix ({method})")
        return _fig_to_dict(fig)

    def bar(self, column: str, top_n: int = 20) -> dict:
        counts = _get_df()[column].value_counts().head(top_n).reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(counts, x="count", y=column, orientation="h", title=f"{column} — Top {top_n}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return _fig_to_dict(fig)

    def scatter(self, x: str, y: str, hue: str | None = None) -> dict:
        fig = px.scatter(_get_df(), x=x, y=y, color=hue, opacity=0.6, title=f"{y} vs {x}")
        return _fig_to_dict(fig)

    def time_series(self, date_col: str, value_col: str, freq: str | None = None) -> dict:
        ts = _get_df().copy()
        ts[date_col] = ts[date_col].astype("datetime64[ns]")
        if freq:
            ts = ts.set_index(date_col).resample(freq)[[value_col]].mean().reset_index()
        fig = px.line(ts, x=date_col, y=value_col, title=f"{value_col} over time")
        return _fig_to_dict(fig)

    # ── Health Choropleths (state-level aggregation) ──

    def choropleth_food_access(self) -> dict:
        df = _get_health_df()
        fig = px.choropleth(
            df, locations="state", locationmode="USA-states",
            color="food_access_score", color_continuous_scale="RdYlGn",
            range_color=[20, 95], scope="usa", hover_name="state_name",
            hover_data=["food_access_score", "pct_food_desert", "median_income"],
            title="Food Access Score by State",
        )
        fig.update_layout(**DARK_LAYOUT, coloraxis_colorbar_title="Score")
        return _fig_to_dict(fig)

    def choropleth_diabetes(self) -> dict:
        df = _get_health_df()
        fig = px.choropleth(
            df, locations="state", locationmode="USA-states",
            color="diabetes_rate", color_continuous_scale="YlOrRd",
            range_color=[5, 18], scope="usa", hover_name="state_name",
            hover_data=["diabetes_rate", "obesity_rate", "life_expectancy"],
            title="Diabetes Rate (%) by State",
        )
        fig.update_layout(**DARK_LAYOUT, coloraxis_colorbar_title="%")
        return _fig_to_dict(fig)

    def choropleth_obesity(self) -> dict:
        df = _get_health_df()
        fig = px.choropleth(
            df, locations="state", locationmode="USA-states",
            color="obesity_rate", color_continuous_scale="YlOrRd",
            range_color=[20, 45], scope="usa", hover_name="state_name",
            hover_data=["obesity_rate", "diabetes_rate", "food_access_score"],
            title="Obesity Rate (%) by State",
        )
        fig.update_layout(**DARK_LAYOUT, coloraxis_colorbar_title="%")
        return _fig_to_dict(fig)

    def choropleth_life_expectancy(self) -> dict:
        df = _get_health_df()
        fig = px.choropleth(
            df, locations="state", locationmode="USA-states",
            color="life_expectancy", color_continuous_scale="RdYlGn",
            range_color=[72, 83], scope="usa", hover_name="state_name",
            hover_data=["life_expectancy", "median_income", "pct_uninsured"],
            title="Life Expectancy at Birth by State",
        )
        fig.update_layout(**DARK_LAYOUT, coloraxis_colorbar_title="Years")
        return _fig_to_dict(fig)

    def scatter_food_vs_diabetes(self) -> dict:
        df = _get_health_df()
        fig = px.scatter(
            df, x="food_access_score", y="diabetes_rate",
            color="pct_minority", size="median_income", size_max=20,
            color_continuous_scale="Viridis", hover_name="state_name",
            hover_data=["obesity_rate", "life_expectancy", "pct_food_desert"],
            title="Food Access vs Diabetes Rate (size = income, color = % minority)",
            trendline="ols",
        )
        fig.update_layout(**DARK_XY_LAYOUT, xaxis_title="Food Access Score", yaxis_title="Diabetes Rate (%)")
        return _fig_to_dict(fig)

    def scatter_food_vs_obesity(self) -> dict:
        df = _get_health_df()
        fig = px.scatter(
            df, x="food_access_score", y="obesity_rate",
            color="pct_minority", size="median_income", size_max=20,
            color_continuous_scale="Viridis", hover_name="state_name",
            hover_data=["diabetes_rate", "life_expectancy"],
            title="Food Access vs Obesity Rate (size = income, color = % minority)",
            trendline="ols",
        )
        fig.update_layout(**DARK_XY_LAYOUT, xaxis_title="Food Access Score", yaxis_title="Obesity Rate (%)")
        return _fig_to_dict(fig)

    def bar_life_expectancy_by_income(self) -> dict:
        df = _get_health_df().copy()
        df["income_quintile"] = pd.qcut(df["median_income"], 5, labels=[
            "Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"
        ])
        grouped = df.groupby("income_quintile", observed=True)["life_expectancy"].mean().reset_index()
        fig = px.bar(
            grouped, x="income_quintile", y="life_expectancy",
            color="life_expectancy", color_continuous_scale="RdYlGn",
            title="Life Expectancy by Income Quintile",
        )
        fig.update_layout(**DARK_XY_LAYOUT, xaxis_title="Income Quintile",
                          yaxis_title="Life Expectancy (years)", yaxis_range=[70, 85], showlegend=False)
        return _fig_to_dict(fig)

    def health_summary_table(self) -> list[dict]:
        return _get_health_df().sort_values("food_access_score").head(10).to_dict(orient="records")

    # ── Phase 4: Income × Race Heatmap ──

    def heatmap_income_race_diabetes(self) -> dict:
        """Income quintile × majority race → diabetes prevalence heatmap."""
        svc = get_data_service()
        data = svc.race_diabetes_matrix()

        if "error" in data:
            # Fallback: return empty figure
            fig = go.Figure()
            fig.add_annotation(text="Data not available — run pipeline first",
                               showarrow=False, font=dict(size=16, color="#e1e4e8"))
            fig.update_layout(**DARK_XY_LAYOUT, title="Income × Race → Diabetes (no data)")
            return _fig_to_dict(fig)

        matrix_records = data["matrix"]
        races = [r for r in data["races"] if r != "Other"]
        df_matrix = pd.DataFrame(matrix_records)
        df_matrix = df_matrix.set_index("income_quintile")
        available_races = [r for r in races if r in df_matrix.columns]

        fig = px.imshow(
            df_matrix[available_races].values,
            x=available_races,
            y=[f"Q{int(q)}" for q in df_matrix.index],
            color_continuous_scale="YlOrRd",
            text_auto=".1f",
            title="Diabetes Prevalence: Income Quintile × Majority Race",
            labels=dict(x="Majority Race", y="Income Quintile", color="Diabetes %"),
        )

        # Annotate the key comparison
        if "Black" in available_races and "White" in available_races:
            bi = available_races.index("Black")
            wi = available_races.index("White")
            n_rows = len(df_matrix)
            if n_rows >= 2:
                hi_black = df_matrix[available_races].iloc[-1, bi]
                lo_white = df_matrix[available_races].iloc[0, wi]
                if pd.notna(hi_black) and pd.notna(lo_white):
                    fig.add_annotation(
                        x=bi, y=n_rows - 1,
                        text=f"Richest Black: {hi_black:.1f}%",
                        showarrow=True, arrowhead=2, font=dict(size=10, color="white"),
                        bgcolor="rgba(0,0,0,0.7)", bordercolor="white",
                    )
                    fig.add_annotation(
                        x=wi, y=0,
                        text=f"Poorest White: {lo_white:.1f}%",
                        showarrow=True, arrowhead=2, font=dict(size=10, color="white"),
                        bgcolor="rgba(0,0,0,0.7)", bordercolor="white",
                    )

        fig.update_layout(**DARK_XY_LAYOUT)
        return _fig_to_dict(fig)

    # ── Phase 5A: HDI Choropleth ──

    def choropleth_hdi(self) -> dict:
        """Health Disadvantage Index choropleth (aggregated to state level)."""
        hdi_path = DATA_DIR / "processed" / "health_disadvantage_index.parquet"
        if not hdi_path.exists():
            fig = go.Figure()
            fig.add_annotation(text="HDI data not available — run pipeline first",
                               showarrow=False, font=dict(size=16, color="#e1e4e8"))
            fig.update_layout(**DARK_XY_LAYOUT, title="Health Disadvantage Index (no data)")
            return _fig_to_dict(fig)

        hdi = pd.read_parquet(hdi_path)
        hdi["state_fips"] = hdi["tract_fips"].str[:2]
        hdi["state_postal"] = hdi["state_fips"].map(STATE_FIPS_TO_POSTAL)

        state_hdi = hdi.groupby("state_postal").agg(
            hdi_score=("hdi_score", "mean"),
            diabetes_pct=("diabetes_pct", "mean"),
            life_expectancy=("life_expectancy", "mean"),
        ).reset_index().rename(columns={"state_postal": "state"})
        state_hdi["state_name"] = state_hdi["state"].map(POSTAL_TO_NAME)
        state_hdi = state_hdi.round(2)

        fig = px.choropleth(
            state_hdi, locations="state", locationmode="USA-states",
            color="hdi_score", color_continuous_scale="RdYlGn_r",
            scope="usa", hover_name="state_name",
            hover_data=["hdi_score", "diabetes_pct", "life_expectancy"],
            title="Health Disadvantage Index by State (higher = more disadvantaged)",
        )
        fig.update_layout(**DARK_LAYOUT, coloraxis_colorbar_title="HDI Score")
        return _fig_to_dict(fig)

    # ── Phase 5A: Path Diagram ──

    def path_diagram(self) -> dict:
        """Bivariate associations: poverty → food desert → obesity → diabetes."""
        svc = get_data_service()
        p5 = svc.load_analysis_json("phase5")

        paths = p5.get("path_diagram", {})
        if not paths:
            fig = go.Figure()
            fig.add_annotation(text="Path data not available — run analysis first",
                               showarrow=False, font=dict(size=16, color="#e1e4e8"))
            fig.update_layout(**DARK_XY_LAYOUT, title="Path Diagram (no data)")
            return _fig_to_dict(fig)

        # Node positions
        nodes = [
            {"label": "Poverty\nRate", "x": 0.1, "y": 0.5},
            {"label": "Food\nDesert", "x": 0.37, "y": 0.5},
            {"label": "Obesity", "x": 0.63, "y": 0.5},
            {"label": "Diabetes", "x": 0.9, "y": 0.5},
        ]

        fig = go.Figure()

        # Draw boxes for each node
        for node in nodes:
            fig.add_shape(
                type="rect",
                x0=node["x"] - 0.08, x1=node["x"] + 0.08,
                y0=node["y"] - 0.15, y1=node["y"] + 0.15,
                fillcolor="#1f6feb", line=dict(color="#58a6ff", width=2),
                xref="paper", yref="paper",
            )
            fig.add_annotation(
                x=node["x"], y=node["y"], text=f"<b>{node['label']}</b>",
                showarrow=False, font=dict(size=13, color="white"),
                xref="paper", yref="paper",
            )

        # Draw arrows between nodes with coefficients
        arrows = [
            ("poverty_to_food_desert", 0, 1),
            ("food_desert_to_obesity", 1, 2),
            ("obesity_to_diabetes", 2, 3),
        ]

        for path_key, from_idx, to_idx in arrows:
            if path_key in paths:
                coef = paths[path_key]["coef"]
                r2 = paths[path_key]["r_squared"]
                x0 = nodes[from_idx]["x"] + 0.08
                x1 = nodes[to_idx]["x"] - 0.08
                mid_x = (x0 + x1) / 2

                fig.add_annotation(
                    x=x1, y=0.5, ax=x0, ay=0.5,
                    xref="paper", yref="paper", axref="paper", ayref="paper",
                    showarrow=True, arrowhead=3, arrowsize=1.5,
                    arrowcolor="#58a6ff", arrowwidth=2,
                )
                fig.add_annotation(
                    x=mid_x, y=0.68,
                    text=f"β={coef}<br>R²={r2}",
                    showarrow=False, font=dict(size=11, color="#8b949e"),
                    xref="paper", yref="paper",
                )

        # Direct path (curved arrow on top)
        if "poverty_to_diabetes_direct" in paths:
            d = paths["poverty_to_diabetes_direct"]
            fig.add_annotation(
                x=0.5, y=0.92,
                text=f"Direct: β={d['coef']}, R²={d['r_squared']}",
                showarrow=False, font=dict(size=11, color="#f0883e"),
                xref="paper", yref="paper",
                bgcolor="rgba(0,0,0,0.6)", bordercolor="#f0883e",
            )
            fig.add_shape(
                type="path",
                path="M 0.18,0.65 C 0.35,0.95 0.65,0.95 0.82,0.65",
                line=dict(color="#f0883e", width=2, dash="dash"),
                xref="paper", yref="paper",
            )

        fig.update_layout(
            title="Bivariate Associations Along Hypothesized Pathway",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e1e4e8",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(t=60, b=20, l=20, r=20),
            height=350,
        )
        return _fig_to_dict(fig)
