"""Subtle, reproducible coupling between a QR mark and its local substrate.

The input layer is BGRA, with a binary QR-module mask.  This module only
changes the appearance of existing marks: the output dimensions and module
mask are unchanged.  It does not simulate missing ink or later damage.
"""

from __future__ import annotations

import cv2
import numpy as np


# (family, substrate colour fraction, texture gain, grain gain,
#  edge Gaussian sigma in pixels, minimum edge coverage)
_PROFILES: dict[str, tuple[str, float, float, float, float, float]] = {
    "laser_print": ("toner", 0.012, 2.0, 0.8, 0.36, 0.94),
    "inkjet_print": ("porous_ink", 0.040, 6.0, 2.2, 0.68, 0.82),
    "offset_print": ("offset_ink", 0.026, 4.0, 1.2, 0.50, 0.88),
    "thermal_print": ("thermal", 0.025, 3.0, 1.2, 0.54, 0.88),
    "thermal_transfer": ("ribbon", 0.014, 2.0, 0.9, 0.42, 0.92),
    "digital_inkjet": ("porous_ink", 0.040, 6.0, 2.0, 0.66, 0.83),
    "flexo_print": ("flexographic", 0.035, 5.0, 1.8, 0.63, 0.85),
    "industrial_inkjet": ("porous_ink", 0.040, 6.0, 2.0, 0.69, 0.82),
    "sticker": ("label", 0.007, 1.0, 0.6, 0.38, 0.94),
    "gravure_print": ("gravure", 0.022, 3.5, 1.1, 0.48, 0.90),
    "uv_inkjet": ("uv_ink", 0.012, 2.0, 1.0, 0.40, 0.93),
    "digital_print": ("digital_ink", 0.018, 2.5, 1.0, 0.44, 0.91),
    "laser_mark": ("laser_surface", 0.075, 5.0, 0.8, 0.47, 0.91),
    "laser_engrave": ("laser_surface", 0.090, 6.0, 1.0, 0.54, 0.88),
    "screen_print": ("screen_ink", 0.020, 3.0, 1.0, 0.46, 0.91),
    "uv_print": ("uv_ink", 0.012, 2.0, 0.8, 0.39, 0.94),
    "pad_print": ("pad_ink", 0.025, 3.0, 1.1, 0.53, 0.89),
    "laser_etch": ("laser_surface", 0.090, 5.0, 0.8, 0.48, 0.90),
    "laser_burn": ("burnt_wood", 0.070, 7.0, 1.2, 0.62, 0.86),
    "heat_transfer": ("transfer_film", 0.013, 2.0, 0.8, 0.42, 0.92),
    "direct_inkjet": ("porous_ink", 0.050, 7.0, 2.4, 0.70, 0.81),
    "jacquard": ("woven", 0.065, 6.0, 1.2, 0.50, 0.90),
    "embroidery": ("stitched", 0.055, 6.0, 1.4, 0.58, 0.87),
    "underglaze": ("fired_pigment", 0.030, 3.0, 0.8, 0.43, 0.92),
    "spray_print": ("porous_ink", 0.038, 5.0, 1.8, 0.63, 0.85),
    "photo_print": ("photo_ink", 0.014, 2.0, 0.7, 0.38, 0.94),
    "uv_roll_print": ("uv_ink", 0.013, 2.0, 0.8, 0.42, 0.93),
    "lcd_oled": ("display", 0.0, 0.0, 0.0, 0.0, 1.0),
}

_FIBROUS = {"paper", "cardboard", "paper_box", "wood", "textile", "poster"}
_SMOOTH = {"metal", "glass", "plastic_shell", "plastic_pack", "acrylic_pvc", "ceramic"}


def couple_ink(
    rgba: np.ndarray,
    local_mask: np.ndarray,
    background_bgr: np.ndarray,
    carrier: str,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Return a lightly surface-coupled BGRA layer and JSON-safe parameters.

    ``background_bgr`` is the QR-sized local background, ideally after a label
    plate has been composited.  The binary ``local_mask`` remains the ground
    truth; edge alpha only models subpixel coverage *inside* marked modules.
    All stochastic variation is drawn from ``rng``.
    """
    rgba = np.asarray(rgba)
    local_mask = np.asarray(local_mask)
    background_bgr = np.asarray(background_bgr)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba must have shape (H, W, 4)")
    if local_mask.shape != rgba.shape[:2]:
        raise ValueError("local_mask must match rgba height and width")
    if background_bgr.shape != (*rgba.shape[:2], 3):
        raise ValueError("background_bgr must have shape (H, W, 3)")

    profile = _PROFILES.get(method, ("generic_ink", 0.020, 3.0, 1.0, 0.45, 0.91))
    family, substrate_fraction, texture_gain, grain_gain, edge_sigma, edge_floor = profile
    marked = (local_mask > 0) & (rgba[..., 3] > 0)
    result = np.clip(rgba, 0, 255).astype(np.uint8, copy=True)

    if method == "lcd_oled" or not np.any(marked):
        return result, {
            "surface_family": family,
            "surface_applied": False,
            "substrate_color_fraction": 0.0,
            "texture_gain": 0.0,
            "grain_gain": 0.0,
            "edge_sigma_px": 0.0,
            "edge_min_coverage": 1.0,
        }

    # A physical label carries its own print surface; its ink should not be
    # tinted by the cardboard, glass or plastic beneath it.
    if carrier == "adhesive_label" or method == "sticker":
        substrate_fraction = min(substrate_fraction, 0.008)
        texture_gain = min(texture_gain, 1.5)
    elif carrier in _FIBROUS:
        texture_gain *= 1.20
    elif carrier in _SMOOTH:
        texture_gain *= 0.80

    # Retain measured surface contrast.  Fixed denominators avoid magnifying
    # nearly flat generated backgrounds or JPEG artifacts into strong noise.
    bg = np.clip(background_bgr, 0, 255).astype(np.float32)
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0, sigmaY=3.0)
    texture = np.clip((gray - low) / 24.0, -1.0, 1.0)

    # A weak correlated adhesion field makes repeated samples less uniform.
    # It does not remove pixels or alter the binary QR ground-truth mask.
    height, width = marked.shape
    coarse = rng.normal(0.0, 1.0, size=(max(2, (height + 5) // 6), max(2, (width + 5) // 6)))
    grain = cv2.resize(coarse.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    grain = np.clip(grain, -2.0, 2.0)
    strength = float(rng.uniform(0.85, 1.15))

    base = result[..., :3].astype(np.float32)
    ink = base * (1.0 - substrate_fraction) + bg * substrate_fraction
    brightness_shift = strength * (texture_gain * texture + grain_gain * grain)
    ink += brightness_shift[..., None]
    ink = np.clip(ink, 0, 255).astype(np.uint8)
    result[..., :3][marked] = ink[marked]

    # Blur the *coverage* only within already marked pixels.  There is no
    # expansion outside local_mask, so QR size, placement and mask stay exact.
    coverage = cv2.GaussianBlur(marked.astype(np.float32), (0, 0), sigmaX=edge_sigma, sigmaY=edge_sigma)
    coverage = np.maximum(coverage, edge_floor)
    alpha = result[..., 3].astype(np.float32)
    alpha[marked] *= coverage[marked]
    result[..., 3] = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)

    return result, {
        "surface_family": family,
        "surface_applied": True,
        "substrate_color_fraction": round(float(substrate_fraction), 4),
        "texture_gain": round(float(texture_gain * strength), 4),
        "grain_gain": round(float(grain_gain * strength), 4),
        "edge_sigma_px": float(edge_sigma),
        "edge_min_coverage": float(edge_floor),
    }
