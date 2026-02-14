# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Automatic batching service for /v1/audio/speech requests.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backends import get_backend, initialize_backend
from ..config import (
    SPEECH_BATCH_DEBOUNCE_MS,
    SPEECH_BATCH_ENABLED,
    SPEECH_BATCH_MAX_QUEUE,
    SPEECH_BATCH_MAX_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class _QueuedSpeechRequest:
    payload: Dict[str, Any]
    future: asyncio.Future
    queued_at: float


class SpeechRequestBatcher:
    """Collects concurrent speech requests and executes them as micro-batches."""

    def __init__(
        self,
        debounce_ms: int = 200,
        max_batch_size: int = 4,
        max_queue_size: int = 256,
    ):
        self.debounce_s = max(0.0, debounce_ms / 1000.0)
        self.max_batch_size = max(1, max_batch_size)
        self._queue: asyncio.Queue[_QueuedSpeechRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._worker_lock = asyncio.Lock()
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    async def submit(
        self,
        *,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        if self._closed:
            raise RuntimeError("Speech batcher is closed")

        await self._ensure_worker()

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        item = _QueuedSpeechRequest(
            payload={
                "text": text,
                "voice": voice,
                "language": language,
                "instruct": instruct,
                "speed": speed,
            },
            future=fut,
            queued_at=time.perf_counter(),
        )

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            raise RuntimeError("Speech batching queue is full")

        return await fut

    async def shutdown(self) -> None:
        self._closed = True

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        self._fail_pending(RuntimeError("Speech batcher shut down"))

    async def _ensure_worker(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return

        async with self._worker_lock:
            if self._worker_task and not self._worker_task.done():
                return
            self._worker_task = asyncio.create_task(self._worker_loop(), name="speech-batch-worker")

    async def _worker_loop(self) -> None:
        try:
            while True:
                first_item = await self._queue.get()
                batch = [first_item]

                if self._queue.qsize() > 0:
                    self._drain_nowait(batch)
                elif self.debounce_s > 0 and len(batch) < self.max_batch_size:
                    deadline = time.perf_counter() + self.debounce_s
                    while len(batch) < self.max_batch_size:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        try:
                            item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        except asyncio.TimeoutError:
                            break
                        batch.append(item)
                        self._drain_nowait(batch)

                await self._process_batch(batch)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Speech batch worker crashed: %s", e, exc_info=True)
            self._fail_pending(RuntimeError(f"Speech batch worker crashed: {e}"))
            raise

    def _drain_nowait(self, batch: List[_QueuedSpeechRequest]) -> None:
        while len(batch) < self.max_batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            batch.append(item)

    async def _process_batch(self, batch: List[_QueuedSpeechRequest]) -> None:
        try:
            backend = get_backend()
            if not backend.is_ready():
                await initialize_backend()

            payloads = [item.payload for item in batch]
            started = time.perf_counter()
            results = await backend.generate_speech_batch(payloads)
            elapsed_ms = (time.perf_counter() - started) * 1000

            if len(results) != len(batch):
                raise RuntimeError(
                    f"Batched backend returned {len(results)} results for {len(batch)} requests"
                )

            oldest_wait_ms = (time.perf_counter() - min(i.queued_at for i in batch)) * 1000
            logger.info(
                "Speech batch | size=%d infer_ms=%.1f oldest_wait_ms=%.1f",
                len(batch),
                elapsed_ms,
                oldest_wait_ms,
            )

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

    def _fail_pending(self, exc: Exception) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not item.future.done():
                item.future.set_exception(exc)


_speech_batcher: Optional[SpeechRequestBatcher] = None
_speech_batcher_loop: Optional[asyncio.AbstractEventLoop] = None


def get_speech_batcher() -> SpeechRequestBatcher:
    """Get loop-bound singleton speech batcher."""
    global _speech_batcher, _speech_batcher_loop

    loop = asyncio.get_running_loop()
    if (
        _speech_batcher is None
        or _speech_batcher.is_closed()
        or _speech_batcher_loop is None
        or _speech_batcher_loop.is_closed()
        or _speech_batcher_loop is not loop
    ):
        _speech_batcher = SpeechRequestBatcher(
            debounce_ms=SPEECH_BATCH_DEBOUNCE_MS,
            max_batch_size=SPEECH_BATCH_MAX_SIZE,
            max_queue_size=SPEECH_BATCH_MAX_QUEUE,
        )
        _speech_batcher_loop = loop

    return _speech_batcher


async def submit_speech_request(
    *,
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Submit a request to the speech batcher or direct backend fallback."""
    if not SPEECH_BATCH_ENABLED:
        backend = get_backend()
        if not backend.is_ready():
            await initialize_backend()
        return await backend.generate_speech(
            text=text,
            voice=voice,
            language=language,
            instruct=instruct,
            speed=speed,
        )

    batcher = get_speech_batcher()
    return await batcher.submit(
        text=text,
        voice=voice,
        language=language,
        instruct=instruct,
        speed=speed,
    )


async def shutdown_speech_batcher() -> None:
    """Shutdown global speech batcher if running."""
    global _speech_batcher, _speech_batcher_loop

    if _speech_batcher is not None:
        await _speech_batcher.shutdown()

    _speech_batcher = None
    _speech_batcher_loop = None
