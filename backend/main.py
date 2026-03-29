"""FastAPI entry point — datathon demo server."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import charts, data


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=30)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Datathon 2026 — Team 2Kim", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(data.router, prefix="/api/data", tags=["data"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
