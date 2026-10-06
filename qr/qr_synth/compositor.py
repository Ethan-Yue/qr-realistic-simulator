from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

from .materials import generate_material
from .printing import render_qr
from .contrast import select_qr_color, contrast_ratio, PLATE_ALLOWED_METHODS
from .surface import couple_ink


def _alpha_blend(bg, fg):
    a = fg[..., 3:4].astype(np.float32) / 255.0
    out = bg.astype(np.float32) * (1.0 - a) + fg[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _resize_rgba(rgba, side):
    return cv2.resize(rgba, (side, side), interpolation=cv2.INTER_NEAREST)


def _resize_mask(mask, side):
    return cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)


def _make_clean_label_plate(side: int, carrier: str):
    """浅色底板属于载体/标签结构，不属于污损增强。"""
    plate = np.zeros((side, side, 4), np.uint8)
    margin = max(1, int(side * 0.025))
    alpha = 210 if carrier == "glass" else 255
    cv2.rectangle(plate, (margin, margin), (side-margin-1, side-margin-1), (244,244,244,alpha), -1)
    return plate


def _paste_rgba(canvas_bgr, rgba, x0, y0):
    h, w = rgba.shape[:2]
    roi = canvas_bgr[y0:y0+h, x0:x0+w]
    canvas_bgr[y0:y0+h, x0:x0+w] = _alpha_blend(roi, rgba)
    return canvas_bgr


def _sample_placement(size, side, rng, position_jitter, center_fraction=None):
    jitter_px = int(round(size * position_jitter))
    dx = int(rng.integers(-jitter_px, jitter_px+1)) if jitter_px > 0 else 0
    dy = int(rng.integers(-jitter_px, jitter_px+1)) if jitter_px > 0 else 0
    if center_fraction is None:
        x0_base = y0_base = (size-side)//2
    else:
        x_center, y_center = (float(value) * size for value in center_fraction)
        if not np.isfinite([x_center, y_center]).all() or not (0 <= x_center <= size and 0 <= y_center <= size):
            raise ValueError("center_fraction must contain two values in [0, 1]")
        x0_base, y0_base = round(x_center - side / 2), round(y_center - side / 2)
    x0 = int(np.clip(x0_base + dx, 0, size-side))
    y0 = int(np.clip(y0_base + dy, 0, size-side))
    return x0, y0


def _decide_white_mode(carrier, method, requested_mode, rng):
    """
    决定白模块/静区是否透明。

    transparent:
        只有黑模块落到背景上，白模块和静区完全透出载体背景。
    opaque:
        在 QR 外接正方形下方增加浅色底板，然后再放黑模块。

    规则：
    1) structural label / sticker 默认为 opaque；
    2) 其它直接印刷/打标/雕刻类默认为 transparent；
    3) 若 requested_mode=random，则在“可双态”的方法上随机；
    4) 若用户强制 transparent / opaque，则尽量按用户要求执行。
    """
    structural_plate = carrier == "adhesive_label" or method == "sticker"

    # 这些方法现实里既可能直接让背景充当白模块，也可能带浅色底板
    dual_mode_methods = {"thermal_print", "thermal_transfer", "heat_transfer", "screen_print", "uv_print", "uv_inkjet"}

    if requested_mode == "transparent":
        if structural_plate:
            return False, "structural_label_forced_opaque"
        return True, "forced_transparent"

    if requested_mode == "opaque":
        return False, "forced_opaque"

    # random / auto
    if structural_plate:
        return False, "structural_label"

    if requested_mode == "random" and method in dual_mode_methods:
        is_transparent = bool(rng.random() < 0.5)
        return is_transparent, "random_dual_mode"

    # 其余直接成码默认透明白模块
    return True, "default_transparent"


def compose_one(qr_mask, carrier, method, rng, size=768, background_root: Path | None=None,
                qr_scale_range=(0.68,0.76), position_jitter=0.015,
                min_contrast=3.0, max_background_resamples=12,
                white_module_mode="random", render_profile="ideal",
                module_count_with_border: int | None = None,
                center_fraction: tuple[float, float] | None = None):
    """
    干净基线 + 自动颜色对比度保护 + 白模块透明/不透明控制。

    white_module_mode:
      - random       : 默认；部分可双态工艺随机透明/不透明，结构标签保持不透明
      - transparent  : 尽量仅保留黑模块，白模块与静区透明
      - opaque       : 在 QR 外接正方形下增加浅色底板

    不包含光照、噪声、模糊、透视、曲面、污损、缺墨等增强。
    """
    if render_profile not in {"ideal", "material"}:
        raise ValueError("render_profile must be 'ideal' or 'material'")
    if size < 64 or max_background_resamples < 1:
        raise ValueError("size must be >=64 and max_background_resamples >=1")
    scale = float(rng.uniform(*qr_scale_range))
    side = max(32, int(round(size * scale)))
    if module_count_with_border is not None:
        modules = int(module_count_with_border)
        if modules < 1 or modules > size:
            raise ValueError("module_count_with_border must be between 1 and image size")
        pitch = max(1, min(int(round(side / modules)), size // modules))
        side = modules * pitch

    white_transparent, white_mode_reason = _decide_white_mode(
        carrier, method, white_module_mode, rng
    )

    best = None
    for attempt in range(max_background_resamples):
        bg, bg_info = generate_material(
            carrier, size, rng, background_root=background_root, return_info=True
        )
        x0, y0 = _sample_placement(size, side, rng, position_jitter, center_fraction)
        roi = bg[y0:y0+side, x0:x0+side]

        # 评估二维码对比度时：
        # - opaque 模式按浅色底板评估；
        # - transparent 模式按真实背景评估。
        plate = _make_clean_label_plate(side, carrier)
        eval_roi = _alpha_blend(roi, plate) if not white_transparent else roi
        decision = select_qr_color(eval_roi, carrier, method, min_contrast=min_contrast)

        use_plate = not white_transparent
        candidate_reason = white_mode_reason

        # random/transparent 模式下，若选择透明白模块但对比度不足，
        # 对允许加底板的工艺可退化为浅色底板，以保证随机模式里仍有一部分可读样本。
        if white_transparent and not decision["passes"] and method in PLATE_ALLOWED_METHODS and white_module_mode == "random":
            use_plate = True
            eval_roi = _alpha_blend(roi, plate)
            decision = select_qr_color(eval_roi, carrier, method, min_contrast=min_contrast)
            candidate_reason = "random_fallback_plate"

        candidate = (decision["contrast_ratio"], bg, bg_info, x0, y0, decision, use_plate, attempt, candidate_reason)
        if best is None or candidate[0] > best[0]:
            best = candidate
        if decision["passes"]:
            break

    ratio, bg, bg_info, x0, y0, decision, use_plate, attempt, white_mode_reason = best

    background_mean = float(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).mean())
    rgba, local_mask = render_qr(
        qr_mask, carrier, method,
        background_mean=background_mean,
        color_override=decision["color_bgr"],
    )
    rgba = _resize_rgba(rgba, side)
    local_mask = _resize_mask(local_mask, side)

    surface_info = {"applied": False, "profile": "ideal"}
    if render_profile == "material":
        surface_bg = bg[y0:y0+side, x0:x0+side].copy()
        if use_plate:
            surface_bg = _alpha_blend(surface_bg, _make_clean_label_plate(side, carrier))
        rgba, surface_info = couple_ink(
            rgba, local_mask, surface_bg, carrier, method, rng
        )

    if use_plate:
        bg = _paste_rgba(bg, _make_clean_label_plate(side, carrier), x0, y0)

    out = _paste_rgba(bg, rgba, x0, y0)
    mask = np.zeros((size,size), np.uint8)
    mask[y0:y0+side, x0:x0+side] = local_mask

    quad = np.array([
        [x0,y0], [x0+side-1,y0], [x0+side-1,y0+side-1], [x0,y0+side-1]
    ], dtype=np.float32)

    effective_transparent = not use_plate
    rendered_roi = out[y0:y0+side, x0:x0+side]
    ink_pixels = rendered_roi[local_mask > 0]
    other_pixels = rendered_roi[local_mask == 0]
    rendered_ratio = contrast_ratio(
        np.median(ink_pixels, axis=0), np.median(other_pixels, axis=0)
    ) if len(ink_pixels) and len(other_pixels) else 0.0
    info = {
        "qr_color_bgr": [int(v) for v in decision["color_bgr"]],
        "background_median_bgr": [round(float(v),2) for v in decision["background_bgr"]],
        "contrast_ratio": round(float(decision["contrast_ratio"]),4),
        "min_contrast": float(min_contrast),
        "contrast_passes": bool(decision["contrast_ratio"] >= min_contrast),
        "rendered_contrast_ratio": round(float(rendered_ratio), 4),
        "rendered_contrast_passes": bool(rendered_ratio >= min_contrast),
        "used_light_plate": bool(use_plate),
        "background_resample_attempts": int(attempt+1),
        "white_module_mode_requested": str(white_module_mode),
        "white_modules_transparent": bool(effective_transparent),
        "white_underlay_type": "light_plate" if use_plate else "none",
        "white_module_mode_reason": str(white_mode_reason),
        "render_profile": render_profile,
        "surface_coupling": surface_info,
        "background_source": bg_info,
    }
    return out, mask, quad, info
