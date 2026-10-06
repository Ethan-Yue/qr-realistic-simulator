"""Optional QR scene-layout sampling from an audited localization manifest.

This samples observed bounding-box geometry only. It does not validate that
the source image was a photograph or that a code was physically damaged.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LayoutBox:
    image_id: str
    box_index: int
    area_ratio: float
    center_xy_fraction: tuple[float, float]


def load_layout_boxes(path: str | Path) -> tuple[list[LayoutBox], str]:
    source = Path(path)
    raw = source.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "box_index", "width", "height", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Layout manifest needs columns: {sorted(required)}")
        boxes = []
        for line_number, row in enumerate(reader, start=2):
            try:
                width, height = float(row["width"]), float(row["height"])
                x1, y1, x2, y2 = (float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                index = int(row["box_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric box at line {line_number}") from exc
            values = np.array([width, height, x1, y1, x2, y2], dtype=float)
            if not np.isfinite(values).all() or not (width > 0 and height > 0 and index > 0):
                raise ValueError(f"Non-finite or non-positive geometry at line {line_number}")
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Out-of-image box at line {line_number}")
            area = (x2 - x1) * (y2 - y1) / (width * height)
            boxes.append(LayoutBox(
                image_id=row["image_id"],
                box_index=index,
                area_ratio=area,
                center_xy_fraction=((x1 + x2) / (2 * width), (y1 + y2) / (2 * height)),
            ))
    if not boxes:
        raise ValueError("Layout manifest contains no valid boxes")
    return boxes, source_sha256


def sample_layout_box(boxes: list[LayoutBox], rng: np.random.Generator) -> LayoutBox:
    if not boxes:
        raise ValueError("Cannot sample an empty layout profile")
    return boxes[int(rng.integers(0, len(boxes)))]
