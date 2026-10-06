from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Sequence


def merged_params(defaults: Dict[str, Any] | None, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    out = deepcopy(defaults or {})
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged_params(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def p(params: Dict[str, Any] | None, key: str, default: Any) -> Any:
    return default if not params or key not in params else params[key]


def as_pair(value: Any, default: Sequence[float]) -> tuple[float, float]:
    if value is None:
        value = default
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if len(value) != 2:
        raise ValueError(f"Expected pair, got {value!r}")
    a, b = float(value[0]), float(value[1])
    return (a, b) if a <= b else (b, a)


def irange(rng, value: Any, default: Sequence[int]) -> int:
    a, b = as_pair(value, default)
    ia, ib = int(round(a)), int(round(b))
    return ia if ia == ib else int(rng.integers(ia, ib + 1))


def frange(rng, value: Any, default: Sequence[float]) -> float:
    a, b = as_pair(value, default)
    return a if a == b else float(rng.uniform(a, b))


def color3(value: Any, default=(128, 128, 128)) -> tuple[int, int, int]:
    value = default if value is None else value
    if isinstance(value, (int, float)):
        v = int(round(value))
        return (v, v, v)
    if len(value) != 3:
        raise ValueError(f"Expected BGR triplet, got {value!r}")
    return tuple(int(round(x)) for x in value)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
