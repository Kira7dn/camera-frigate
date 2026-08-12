"""In-process notification delivery metrics."""

from collections import Counter
from threading import Lock

_lock = Lock()
_counts: Counter[tuple[str, str]] = Counter()
_latency_total: Counter[str] = Counter()
_latency_count: Counter[str] = Counter()
_queue_depth: dict[str, int] = {}


def increment(provider: str, outcome: str) -> None:
    with _lock:
        _counts[(provider, outcome)] += 1


def observe_latency(provider: str, seconds: float) -> None:
    with _lock:
        _latency_total[provider] += seconds
        _latency_count[provider] += 1


def set_queue_depth(provider: str, depth: int) -> None:
    with _lock:
        _queue_depth[provider] = depth


def snapshot() -> dict[str, dict[str, float | int]]:
    with _lock:
        providers = {provider for provider, _ in _counts}
        providers.update(_latency_count)
        providers.update(_queue_depth)
        return {
            provider: {
                **{
                    outcome: count
                    for (metric_provider, outcome), count in _counts.items()
                    if metric_provider == provider
                },
                "latency_seconds_total": float(_latency_total[provider]),
                "latency_count": _latency_count[provider],
                "queue_depth": _queue_depth.get(provider, 0),
            }
            for provider in providers
        }
