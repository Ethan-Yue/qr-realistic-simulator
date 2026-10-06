"""Paired camera simulation for clean and damaged QR samples.

The same sampled camera state is applied to both images. Masks describe the
geometric footprint of the QR ink and damage before optical blur/JPEG spread.
"""

from __future__ import annotations

import cv2
import numpy as np


_REFLECTIVE_CARRIERS = {
    "metal": (0.10, 0.25),
    "glass": (0.08, 0.22),
    "acrylic_pvc": (0.07, 0.19),
    "screen": (0.07, 0.20),
    "plastic_pack": (0.03, 0.12),
    "plastic_shell": (0.02, 0.10),
    "ceramic": (0.02, 0.10),
}


def _motion_kernel(length_px: int, angle_degrees: float) -> np.ndarray:
    """Normalized straight-line exposure PSF in image coordinates."""
    length = max(3, int(length_px) | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    radius = (length - 1) / 2.0
    angle = np.deg2rad(angle_degrees)
    dx, dy = radius * np.cos(angle), radius * np.sin(angle)
    center = length // 2
    cv2.line(kernel,
             (int(round(center - dx)), int(round(center - dy))),
             (int(round(center + dx)), int(round(center + dy))),
             1.0, 1, cv2.LINE_AA)
    total = float(kernel.sum())
    if total <= 0:
        kernel[center, center] = 1.0
        total = 1.0
    return kernel / total


def _strong_glare(height: int, width: int, rng, strength_override: float | None = None) -> tuple[np.ndarray, dict]:
    """Broad reflected-light lobe plus a narrow saturated streak."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = float(rng.uniform(0.18, 0.82))
    cy = float(rng.uniform(0.18, 0.82))
    angle = float(rng.uniform(-75.0, 75.0))
    theta = np.deg2rad(angle)
    dx, dy = xx / width - cx, yy / height - cy
    along = np.cos(theta) * dx + np.sin(theta) * dy
    across = -np.sin(theta) * dx + np.cos(theta) * dy
    strength = float(rng.uniform(0.75, 1.6) if strength_override is None else strength_override)
    broad = np.exp(-0.5 * ((along / .22) ** 2 + (across / .09) ** 2))
    streak = np.exp(-0.5 * ((along / .38) ** 2 + (across / .014) ** 2))
    field = (strength * (.72 * broad + .28 * streak)).astype(np.float32)
    return field, {
        "enabled": True, "strength_linear": strength,
        "center_xy_fraction": [cx, cy], "angle_degrees": angle,
        "model": "elliptical lobe plus narrow streak",
    }


def _image_u8(image: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 BGR image")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.copy()


def _mask_u8(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return (arr > 0).astype(np.uint8) * 255


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_u8(image: np.ndarray) -> np.ndarray:
    x = np.clip(image, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def _lighting_fields(height: int, width: int, carrier: str, mode: str, rng):
    if mode == "mild":
        ambient = float(rng.uniform(0.91, 1.08))
        gradient_x = float(rng.uniform(-0.07, 0.07))
        gradient_y = float(rng.uniform(-0.07, 0.07))
        vignette = float(rng.uniform(0.0, 0.07))
        blue_gain = float(rng.uniform(0.98, 1.02))
        highlight_scale = 0.58
    else:
        ambient = float(rng.uniform(0.78, 1.15))
        gradient_x = float(rng.uniform(-0.16, 0.16))
        gradient_y = float(rng.uniform(-0.16, 0.16))
        vignette = float(rng.uniform(0.02, 0.17))
        blue_gain = float(rng.uniform(0.94, 1.06))
        highlight_scale = 1.0

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx + 0.5) / width * 2.0 - 1.0
    ny = (yy + 0.5) / height * 2.0 - 1.0
    radius_sq = np.minimum((nx * nx + ny * ny) / 2.0, 1.0)
    illumination = np.clip(
        ambient + gradient_x * nx + gradient_y * ny - vignette * radius_sq,
        0.45,
        1.35,
    ).astype(np.float32)
    white_balance_bgr = np.array([blue_gain, 1.0, 1.0 / blue_gain], dtype=np.float32)

    highlight = np.zeros((height, width), dtype=np.float32)
    highlight_info = {"enabled": False, "strength": 0.0}
    if carrier in _REFLECTIVE_CARRIERS:
        low, high = _REFLECTIVE_CARRIERS[carrier]
        strength = float(rng.uniform(low, high) * highlight_scale)
        center_x = float(rng.uniform(0.16, 0.84))
        center_y = float(rng.uniform(0.16, 0.84))
        radius_x = float(rng.uniform(0.12, 0.34))
        radius_y = float(rng.uniform(0.25, 0.68))
        angle = float(rng.uniform(-75.0, 75.0))
        theta = np.deg2rad(angle)
        dx = xx / width - center_x
        dy = yy / height - center_y
        rx = np.cos(theta) * dx + np.sin(theta) * dy
        ry = -np.sin(theta) * dx + np.cos(theta) * dy
        highlight = (strength * np.exp(-0.5 * ((rx / radius_x) ** 2 + (ry / radius_y) ** 2))).astype(np.float32)
        highlight_info = {
            "enabled": True,
            "strength": strength,
            "center_xy_fraction": [center_x, center_y],
            "radii_fraction": [radius_x, radius_y],
            "angle_degrees": angle,
        }

    lighting_info = {
        "ambient": ambient,
        "gradient_xy": [gradient_x, gradient_y],
        "vignette": vignette,
        "white_balance_bgr": white_balance_bgr.astype(float).tolist(),
        "highlight": highlight_info,
    }
    return illumination, white_balance_bgr, highlight, lighting_info


def _apply_optics(
    image: np.ndarray,
    illumination: np.ndarray,
    white_balance_bgr: np.ndarray,
    highlight: np.ndarray,
    blur_sigma: float,
    shared_noise: np.ndarray,
    jpeg_quality: int | None,
    motion_kernel: np.ndarray | None = None,
) -> np.ndarray:
    linear = _srgb_to_linear(image)
    linear = linear * illumination[..., None] * white_balance_bgr[None, None, :]
    linear += highlight[..., None]
    out = _linear_to_u8(linear)
    if blur_sigma > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    if motion_kernel is not None:
        out = cv2.filter2D(out, -1, motion_kernel, borderType=cv2.BORDER_REFLECT_101)
    if shared_noise is not None:
        out = np.clip(np.rint(out.astype(np.float32) + shared_noise), 0, 255).astype(np.uint8)
    if jpeg_quality is not None:
        ok, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG camera simulation failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG camera simulation produced an unreadable image")
        out = decoded
    return out


def capture_pair(
    clean_bgr: np.ndarray,
    damaged_bgr: np.ndarray,
    qr_mask: np.ndarray,
    damage_mask: np.ndarray | None,
    quad_xy: np.ndarray,
    carrier: str,
    rng,
    mode: str = "mild",
    effect_overrides: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply one sampled capture to a paired clean/damaged QR image.

    Returns ``(clean_capture, damaged_capture, qr_mask_capture,
    damage_mask_capture, quad_capture, metadata)``. All image and mask outputs
    retain the input spatial size and use ``uint8``. ``quad_capture`` is float32.
    ``mode`` is one of ``none``, ``mild``, ``handheld`` or ``challenging``.
    """
    if mode not in {"none", "mild", "handheld", "challenging"}:
        raise ValueError(f"Unknown capture mode: {mode!r}")
    overrides = dict(effect_overrides or {})
    if overrides and mode != "challenging":
        raise ValueError("effect_overrides require capture mode 'challenging'")
    if set(overrides) - {"motion_length_px", "exposure_ev", "glare_strength"}:
        raise ValueError("Unknown acquisition effect override")
    if "motion_length_px" in overrides:
        length = overrides["motion_length_px"]
        if not isinstance(length, int) or (length != 0 and not (3 <= length <= 31 and length % 2 == 1)):
            raise ValueError("motion_length_px must be 0 or an odd integer from 3 to 31")
    if "exposure_ev" in overrides and not (-2 <= float(overrides["exposure_ev"]) <= 3):
        raise ValueError("exposure_ev must be between -2 and 3")
    if "glare_strength" in overrides and not (0 <= float(overrides["glare_strength"]) <= 2.5):
        raise ValueError("glare_strength must be between 0 and 2.5")
    clean = _image_u8(clean_bgr, "clean_bgr")
    damaged = _image_u8(damaged_bgr, "damaged_bgr")
    if clean.shape != damaged.shape:
        raise ValueError("clean_bgr and damaged_bgr must have identical shapes")
    height, width = clean.shape[:2]
    qrm = _mask_u8(qr_mask, (height, width), "qr_mask")
    dm = _mask_u8(damage_mask, (height, width), "damage_mask")
    quad = np.asarray(quad_xy, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("quad_xy must contain four finite xy points")

    identity = np.eye(3, dtype=np.float32)
    if mode == "none":
        return clean, damaged, qrm, dm, quad.copy(), {
            "mode": "none",
            "carrier": carrier,
            "image_size_hw": [height, width],
            "perspective_matrix": identity.astype(float).tolist(),
            "quad_xy": quad.astype(float).tolist(),
            "mask_scope": "geometric footprint before optical spread",
        }

    corner_limit = 0.013 if mode == "mild" else 0.045
    translation_limit = 0.008 if mode == "mild" else 0.025
    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    scale_xy = np.array([width, height], dtype=np.float32)
    jitter = rng.uniform(-corner_limit, corner_limit, size=(4, 2)).astype(np.float32) * scale_xy
    translation = rng.uniform(-translation_limit, translation_limit, size=2).astype(np.float32) * scale_xy
    dst = src + jitter + translation
    homography = cv2.getPerspectiveTransform(src, dst).astype(np.float32)
    output_size = (width, height)

    warped_clean = cv2.warpPerspective(
        clean, homography, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    warped_damaged = cv2.warpPerspective(
        damaged, homography, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    warped_qrm = cv2.warpPerspective(
        qrm, homography, output_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    warped_dm = cv2.warpPerspective(
        dm, homography, output_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    quad_capture = cv2.perspectiveTransform(quad.reshape(1, 4, 2), homography).reshape(4, 2)

    illumination, wb, highlight, lighting_info = _lighting_fields(height, width, carrier, mode, rng)
    exposure_ev = 0.0
    glare_info = {"enabled": False, "strength_linear": 0.0}
    motion_info = {"enabled": False, "length_px": 0, "angle_degrees": None}
    motion_kernel = None
    if mode == "challenging":
        if "exposure_ev" in overrides:
            exposure_ev = float(overrides["exposure_ev"])
        elif rng.random() < .72:
            exposure_ev = float(rng.uniform(.35, 1.25))
        use_glare = (float(overrides["glare_strength"]) > 0 if "glare_strength" in overrides
                     else rng.random() < (.72 if carrier in _REFLECTIVE_CARRIERS else .15))
        if use_glare:
            glare, glare_info = _strong_glare(height, width, rng, overrides.get("glare_strength"))
            highlight = highlight + glare
        if "motion_length_px" in overrides:
            length = overrides["motion_length_px"]
        else:
            length = int(rng.choice([3, 5, 7, 9, 11])) if rng.random() < .72 else 0
        if length > 0:
            angle = float(rng.uniform(-90.0, 90.0))
            motion_kernel = _motion_kernel(length, angle)
            motion_info = {"enabled": True, "length_px": length, "angle_degrees": angle}
    illumination = illumination * (2.0 ** exposure_ev)
    blur_sigma = float(rng.uniform(0.18, 0.65) if mode == "mild" else rng.uniform(0.65, 1.55))
    noise_sigma = float(rng.uniform(0.35, 1.35) if mode == "mild" else rng.uniform(1.25, 3.2))
    shared_noise = rng.normal(0.0, noise_sigma, size=(height, width, 3)).astype(np.float32)
    jpeg_probability = 0.25 if mode == "mild" else 0.75
    jpeg_quality = None
    if rng.random() < jpeg_probability:
        jpeg_quality = int(rng.integers(92, 99) if mode == "mild" else rng.integers(77, 95))

    clean_capture = _apply_optics(
        warped_clean, illumination, wb, highlight, blur_sigma, shared_noise, jpeg_quality, motion_kernel
    )
    damaged_capture = _apply_optics(
        warped_damaged, illumination, wb, highlight, blur_sigma, shared_noise, jpeg_quality, motion_kernel
    )
    metadata = {
        "mode": mode,
        "carrier": carrier,
        "image_size_hw": [height, width],
        "perspective_matrix": homography.astype(float).tolist(),
        "source_corners_xy": src.astype(float).tolist(),
        "destination_corners_xy": dst.astype(float).tolist(),
        "quad_xy": quad_capture.astype(float).tolist(),
        "lighting": lighting_info,
        "exposure_ev": exposure_ev,
        "strong_glare": glare_info,
        "motion_blur": motion_info,
        "near_white_fraction_clean": float((np.min(clean_capture, axis=2) >= 250).mean()),
        "effect_overrides_requested": overrides or None,
        "blur_sigma_px": blur_sigma,
        "gaussian_noise_sigma_srgb": noise_sigma,
        "shared_noise_realization": True,
        "jpeg_quality": jpeg_quality,
        "mask_scope": "geometric footprint before optical spread",
    }
    return (
        clean_capture,
        damaged_capture,
        (warped_qrm > 0).astype(np.uint8) * 255,
        (warped_dm > 0).astype(np.uint8) * 255,
        quad_capture.astype(np.float32),
        metadata,
    )
