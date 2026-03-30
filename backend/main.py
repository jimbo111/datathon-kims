"""FastAPI entry point — datathon demo server."""

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import charts, data

# ── CORS origins: read from env variable ALLOWED_ORIGINS (comma-separated)
# or fall back to localhost defaults for local development.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Create React App / Next.js dev server
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _DEFAULT_ORIGINS
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=30)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Datathon 2026 — Team 2Kim", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(data.router, prefix="/api/data", tags=["data"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
