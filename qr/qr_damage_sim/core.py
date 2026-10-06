from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DamageLayer:
    name: str
    category: str
    mask: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DamageResult:
    image: np.ndarray
    union_mask: np.ndarray
    layers: List[DamageLayer]
    metadata: Dict[str, Any]


def ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.copy()


def ensure_mask(mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != shape_hw:
        import cv2
        m = cv2.resize(m.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8) * 255


def mask_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def overlap_ratio(mask: np.ndarray, region: Optional[np.ndarray]) -> float:
    if region is None:
        return 0.0
    region_b = region > 0
    denom = int(region_b.sum())
    if denom == 0:
        return 0.0
    return float(((mask > 0) & region_b).sum() / denom)
