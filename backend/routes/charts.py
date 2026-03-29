"""Chart endpoints — return Plotly JSON for frontend rendering."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.services.chart_service import ChartService

router = APIRouter()
charts = ChartService()


@router.get("/")
async def list_charts():
    """List all available chart endpoints."""
    return {
        "available": [
            {"name": "distribution", "params": ["column", "bins"]},
            {"name": "correlation", "params": ["method"]},
            {"name": "bar", "params": ["column", "top_n"]},
            {"name": "scatter", "params": ["x", "y", "hue"]},
            {"name": "timeseries", "params": ["date_col", "value_col", "freq"]},
            {"name": "health/choropleth/food-access", "params": []},
            {"name": "health/choropleth/diabetes", "params": []},
            {"name": "health/choropleth/obesity", "params": []},
            {"name": "health/choropleth/life-expectancy", "params": []},
            {"name": "health/scatter/food-vs-diabetes", "params": []},
            {"name": "health/scatter/food-vs-obesity", "params": []},
            {"name": "health/bar/life-expectancy-income", "params": []},
            {"name": "health/summary", "params": []},
        ]
    }


@router.get("/distribution")
async def distribution(column: str, bins: int = Query(30, ge=5, le=200)):
    fig_json = charts.distribution(column, bins=bins)
    return JSONResponse(content=fig_json)


@router.get("/correlation")
async def correlation(method: str = Query("pearson", pattern="^(pearson|spearman|kendall)$")):
    fig_json = charts.correlation_matrix(method=method)
    return JSONResponse(content=fig_json)


@router.get("/bar")
async def bar_chart(column: str, top_n: int = Query(20, ge=1, le=100)):
    fig_json = charts.bar(column, top_n=top_n)
    return JSONResponse(content=fig_json)


@router.get("/scatter")
async def scatter(x: str, y: str, hue: str | None = None):
    fig_json = charts.scatter(x, y, hue=hue)
    return JSONResponse(content=fig_json)


@router.get("/timeseries")
async def timeseries(date_col: str, value_col: str, freq: str | None = None):
    fig_json = charts.time_series(date_col, value_col, freq=freq)
    return JSONResponse(content=fig_json)


# --- Health Analysis Map Endpoints ---

@router.get("/health/choropleth/food-access")
async def choropleth_food_access():
    return JSONResponse(content=charts.choropleth_food_access())


@router.get("/health/choropleth/diabetes")
async def choropleth_diabetes():
    return JSONResponse(content=charts.choropleth_diabetes())


@router.get("/health/choropleth/obesity")
async def choropleth_obesity():
    return JSONResponse(content=charts.choropleth_obesity())


@router.get("/health/choropleth/life-expectancy")
async def choropleth_life_expectancy():
    return JSONResponse(content=charts.choropleth_life_expectancy())


@router.get("/health/scatter/food-vs-diabetes")
async def scatter_food_vs_diabetes():
    return JSONResponse(content=charts.scatter_food_vs_diabetes())


@router.get("/health/scatter/food-vs-obesity")
async def scatter_food_vs_obesity():
    return JSONResponse(content=charts.scatter_food_vs_obesity())


@router.get("/health/bar/life-expectancy-income")
async def bar_life_expectancy_income():
    return JSONResponse(content=charts.bar_life_expectancy_by_income())


@router.get("/health/summary")
async def health_summary():
    return JSONResponse(content=charts.health_summary_table())


@router.get("/health/heatmap/income-race-diabetes")
async def heatmap_income_race_diabetes():
    return JSONResponse(content=charts.heatmap_income_race_diabetes())


@router.get("/health/choropleth/hdi")
async def choropleth_hdi():
    return JSONResponse(content=charts.choropleth_hdi())


@router.get("/health/path-diagram")
async def path_diagram():
    return JSONResponse(content=charts.path_diagram())
