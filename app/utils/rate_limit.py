"""A simple per-client sliding-window rate limiter.

Because the demo runs on a free LLM quota, a public URL could otherwise be
drained by anyone. This keeps one visitor from burning the whole quota.
"""

import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0, clock=time.monotonic):
        self.max_requests = max_requests
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> float:
        """Record a hit. Returns 0 if allowed, otherwise seconds to wait before retrying."""
        if self.max_requests <= 0:
            return 0.0  # limiter disabled
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return max(0.1, self.window - (now - hits[0]))
        hits.append(now)
        return 0.0
