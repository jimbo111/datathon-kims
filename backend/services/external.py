"""External API client — async HTTP calls to third-party APIs."""

from typing import Any

import httpx


class ExternalAPIClient:
    """Reusable async client for calling external APIs.

    Usage in routes:
        from fastapi import Request

        @router.get("/external-stuff")
        async def get_stuff(request: Request):
            client = ExternalAPIClient(request.app.state.http_client)
            data = await client.get("https://api.example.com/data", params={"q": "test"})
            return data
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self._client = http_client

    async def get(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> Any:
        resp = await self._client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def post(self, url: str, *, json: dict | None = None, headers: dict | None = None) -> Any:
        resp = await self._client.post(url, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def fetch_raw(self, url: str, **kwargs) -> httpx.Response:
        """Return the raw response for non-JSON APIs."""
        resp = await self._client.get(url, **kwargs)
        resp.raise_for_status()
        return resp
