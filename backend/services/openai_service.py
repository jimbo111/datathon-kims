"""OpenAI integration service."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    return AsyncOpenAI(api_key=api_key)


async def chat_completion(
    prompt: str,
    *,
    system: str = "You are a helpful data analysis assistant.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Quick helper for single-turn completions."""
    client = get_openai_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
