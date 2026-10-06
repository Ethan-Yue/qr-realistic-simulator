"""Framework-neutral loader for paired synthetic QR restoration data.

The returned arrays are float32 CHW RGB images in [0, 1] and binary masks.
PyTorch's default DataLoader can collate these NumPy arrays without a hard
PyTorch dependency in this project.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path, mode: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def _qr_crop(image: np.ndarray, quad_xy: list[list[float]], side_px: int,
             context_ratio: float, is_mask: bool) -> np.ndarray:
    points = np.asarray(quad_xy, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("metadata.quad_xy must be four finite xy points")
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    span = max(float(right - left), float(bottom - top)) * (1 + 2 * context_ratio)
    if span <= 0:
        raise ValueError("QR quadrilateral has zero area")
    center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
    scale = side_px / span
    matrix = np.array([[scale, 0, (side_px - 1) / 2 - center_x * scale],
                       [0, scale, (side_px - 1) / 2 - center_y * scale]], dtype=np.float32)
    return cv2.warpAffine(
        image, matrix, (side_px, side_px),
        flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101,
        borderValue=0,
    )


class RestorationPairDataset:
    """Load damaged input, aligned clean target, and optional binary masks."""

    def __init__(self, root: str | Path, manifest: str | Path | None = None,
                 split: str | None = "train", crop_qr: bool = True,
                 output_size: int = 256, context_ratio: float = .25):
        self.root = Path(root).resolve()
        manifest_path = Path(manifest).resolve() if manifest else self.root / "restoration_pairs.csv"
        if split not in {"train", "val", "test", None}:
            raise ValueError("split must be train, val, test, or None")
        if output_size < 32 or context_ratio < 0:
            raise ValueError("output_size must be >=32 and context_ratio non-negative")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if split is None or row["split"] == split]
        self.crop_qr = crop_qr
        self.output_size = output_size
        self.context_ratio = context_ratio

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        meta = json.loads((self.root / row["metadata"]).read_text(encoding="utf-8"))
        damaged = _read(self.root / row["damaged_image"], cv2.IMREAD_COLOR)
        clean = _read(self.root / row["clean_target"], cv2.IMREAD_COLOR)
        masks = {name: _read(self.root / row[name], cv2.IMREAD_GRAYSCALE)
                 for name in ("qr_mask", "damage_mask", "visible_change_mask")}
        if self.crop_qr:
            quad = meta["quad_xy"]
            damaged = _qr_crop(damaged, quad, self.output_size, self.context_ratio, False)
            clean = _qr_crop(clean, quad, self.output_size, self.context_ratio, False)
            masks = {name: _qr_crop(mask, quad, self.output_size, self.context_ratio, True)
                     for name, mask in masks.items()}
        to_chw_rgb = lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        return {
            "input": to_chw_rgb(damaged),
            "target": to_chw_rgb(clean),
            "qr_mask": (masks["qr_mask"] > 0)[None].astype(np.float32),
            "damage_mask": (masks["damage_mask"] > 0)[None].astype(np.float32),
            "visible_change_mask": (masks["visible_change_mask"] > 0)[None].astype(np.float32),
            "sample_id": row["sample_id"],
            "target_payload": meta.get("qr_payload") if row["payload_ground_truth_available"] == "1" else "",
            "split": row["split"],
        }
