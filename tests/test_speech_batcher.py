# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Tests for automatic speech batching service."""

import asyncio

import numpy as np
import pytest

from api.backends import factory
from api.backends.factory import reset_backend
from api.services.speech_batcher import SpeechRequestBatcher


class _FakeBatchBackend:
    def __init__(self):
        self.batch_sizes = []

    def is_ready(self):
        return True

    async def generate_speech_batch(self, requests):
        self.batch_sizes.append(len(requests))
        await asyncio.sleep(0.01)
        return [
            (np.array([float(len(req["text"]))], dtype=np.float32), 24000)
            for req in requests
        ]


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    yield
    reset_backend()


@pytest.mark.asyncio
async def test_batcher_collects_parallel_requests_within_debounce_window():
    backend = _FakeBatchBackend()
    factory._backend_instance = backend

    batcher = SpeechRequestBatcher(debounce_ms=120, max_batch_size=4, max_queue_size=64)
    try:
        results = await asyncio.gather(
            batcher.submit(text="hello one", voice="Vivian"),
            batcher.submit(text="hello two", voice="Vivian"),
            batcher.submit(text="hello three", voice="Vivian"),
        )

        assert len(results) == 3
        assert backend.batch_sizes == [3]
    finally:
        await batcher.shutdown()


@pytest.mark.asyncio
async def test_batcher_splits_large_parallel_load_into_multiple_batches():
    backend = _FakeBatchBackend()
    factory._backend_instance = backend

    batcher = SpeechRequestBatcher(debounce_ms=120, max_batch_size=4, max_queue_size=64)
    try:
        tasks = [
            asyncio.create_task(batcher.submit(text=f"request-{i}", voice="Vivian"))
            for i in range(6)
        ]
        await asyncio.gather(*tasks)

        assert backend.batch_sizes == [4, 2]
    finally:
        await batcher.shutdown()
