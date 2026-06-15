"""
Mneme Client
============

Async client for Mneme memory system — drop-in replacement for Mem0Client
in the LoCoMo benchmark pipeline.

Instead of Mem0's auto-extraction API, Mneme stores raw conversation turns
as structured memories and uses semantic search for retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class MnemeClient:
    """Async client for Mneme memory system.

    Implements the same async interface as Mem0Client so it can be used
    as a drop-in in benchmark pipelines.

    Args:
        host: Mneme server URL. Defaults to MNEME_HOST env or http://localhost:8989.
        max_retries: Maximum retry attempts for API calls.
        retry_delay: Base delay in seconds between retries.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        host: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
    ):
        self.host = (host or os.getenv("MNEME_HOST", "http://localhost:8989")).rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> MnemeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # =========================================================================
    # Add — store conversation messages as memories
    # =========================================================================

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Store a conversation chunk into Mneme.

        Mneme doesn't do auto-extraction like Mem0. Instead, we:
        - Concatenate messages into a single memory text
        - Store with metadata (user_id, timestamp)
        - Tag as 'conversation' type
        """
        if not messages:
            return {"results": []}

        session = await self._get_session()

        # Concatenate messages into a single memory
        text_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            if content:
                text_parts.append(f"[{role}] {content}")

        if not text_parts:
            return {"results": []}

        memory_text = " | ".join(text_parts)

        payload: dict[str, Any] = {
            "content": memory_text,
            "type": "conversation",
            "tags": [f"user:{user_id}"],
        }

        if timestamp is not None:
            payload["metadata"] = {"timestamp": timestamp, "user_id": user_id}
        elif observation_date:
            payload["metadata"] = {"observation_date": observation_date, "user_id": user_id}
        else:
            payload["metadata"] = {"user_id": user_id}

        for attempt in range(self.max_retries):
            try:
                async with session.post(f"{self.host}/v1/memories", json=payload) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    resp.raise_for_status()
                    data = await resp.json()

                # Normalize to match Mem0 response format
                memory_id = data.get("id", str(time.time()))
                return {
                    "results": [{
                        "id": memory_id,
                        "memory": memory_text,
                        "event": "ADD",
                    }]
                }

            except Exception as exc:
                logger.warning(
                    "Mneme ADD attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Mneme ADD failed after %d attempts", self.max_retries)
                    return None

        return None

    # =========================================================================
    # Search — semantic search over memories
    # =========================================================================

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Search memories via Mneme's semantic search.

        Returns list of results in Mem0-compatible format.
        """
        session = await self._get_session()

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.host}/v1/memories/search",
                    json={"query": query, "limit": top_k},
                ) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    resp.raise_for_status()
                    data = await resp.json()

                # Normalize to Mem0 format
                results = data.get("results", data) if isinstance(data, dict) else data
                if not isinstance(results, list):
                    results = []

                normalised = []
                for r in results:
                    entry: dict[str, Any] = {
                        "memory": r.get("content", r.get("memory", "")),
                        "score": r.get("score", r.get("similarity", 0)),
                        "id": r.get("id", ""),
                    }
                    if r.get("created_at"):
                        entry["created_at"] = r["created_at"]
                    if r.get("updated_at"):
                        entry["updated_at"] = r["updated_at"]
                    normalised.append(entry)

                normalised.sort(key=lambda x: x.get("score", 0), reverse=True)
                return normalised

            except Exception as exc:
                logger.warning(
                    "Mneme SEARCH attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Mneme SEARCH failed after %d attempts", self.max_retries)
                    return []

        return []

    # =========================================================================
    # Delete — clear memories
    # =========================================================================

    async def delete_user(self, user_id: str) -> bool:
        """Delete all memories. Mneme doesn't scope by user_id, so just clear all."""
        session = await self._get_session()
        try:
            async with session.delete(f"{self.host}/v1/memories") as resp:
                resp.raise_for_status()
            logger.info("Cleared all Mneme memories")
            return True
        except Exception as exc:
            logger.warning("Failed to clear Mneme memories: %s", exc)
            return False
