from __future__ import annotations
import cv2
import numpy as np

# 仅这些工艺允许在低对比度时增加浅色底板。
# 直接印刷、雕刻、蚀刻等工艺不会凭空增加白底，而是重新选择背景/位置。
PLATE_ALLOWED_METHODS = {
    "sticker", "thermal_print", "thermal_transfer", "heat_transfer",
}

# 优先使用符合常见真实工艺的深色候选；只有工艺本身允许浅色标记时才使用浅色。
DARK_CANDIDATES = [
    (8, 8, 8),        # 黑
    (28, 28, 28),     # 深灰
    (55, 25, 18),     # 深蓝黑（BGR）
    (25, 45, 70),     # 深棕/深红黑（BGR）
    (28, 55, 25),     # 深绿黑（BGR）
]
LIGHT_CANDIDATES = [
    (245, 245, 245),
    (225, 225, 225),
]

# Constrain processes with characteristic mark colours before contrast selection.
# These are plausible appearance ranges, not calibrated device measurements.
METHOD_DARK_CANDIDATES = {
    "thermal_print": [(38, 38, 38), (52, 52, 52)],
    "laser_burn": [(27, 43, 61), (35, 54, 73)],
    "underglaze": [(145, 66, 27), (125, 55, 22)],
    "lcd_oled": [(3, 3, 3), (12, 12, 12)],
}
METAL_LASER_CANDIDATES = {
    "laser_mark": [(62, 62, 66), (84, 84, 88)],
    "laser_engrave": [(38, 38, 40), (62, 62, 66)],
}
# Etched glass is a translucent, frosted mark. Very dark ink-like colours make
# it indistinguishable from screen printing on a pale glass background.
GLASS_ETCH_CANDIDATES = [(105, 108, 111), (123, 126, 129)]


def relative_luminance_bgr(color):
    """sRGB BGR -> WCAG relative luminance [0,1]."""
    b, g, r = np.asarray(color, dtype=np.float32) / 255.0
    rgb = np.array([r, g, b], dtype=np.float32)
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def contrast_ratio(color_a, color_b):
    la = relative_luminance_bgr(color_a)
    lb = relative_luminance_bgr(color_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def robust_background_color(roi):
    """用中位数代表局部背景色，避免少量纹理像素主导判断。"""
    return np.median(roi.reshape(-1, 3), axis=0).astype(np.float32)


def choose_best_color(roi, candidates):
    bg = robust_background_color(roi)
    scored = [(contrast_ratio(c, bg), tuple(int(v) for v in c)) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], float(scored[0][0]), tuple(float(v) for v in bg)


def method_allows_light_modules(carrier, method):
    """浅色模块只用于真实工艺中合理的场景，例如深色塑料激光标记/刻蚀。"""
    if method in {"laser_mark", "laser_engrave", "laser_etch"}:
        return carrier in {"plastic_shell", "metal", "glass", "acrylic_pvc"}
    return False


def select_qr_color(roi, carrier, method, min_contrast=3.0):
    """
    颜色策略：
      1. 默认优先深色 QR；
      2. 工艺允许时，深色/浅色候选一起比较；
      3. 返回局部对比度，供 compositor 决定重采样或加底板。
    """
    if carrier == "glass" and method == "laser_etch":
        candidates = list(GLASS_ETCH_CANDIDATES)
    elif carrier == "metal" and method in {"laser_mark", "laser_engrave"}:
        candidates = list(METAL_LASER_CANDIDATES[method])
    else:
        candidates = list(METHOD_DARK_CANDIDATES.get(method, DARK_CANDIDATES))
    if method_allows_light_modules(carrier, method):
        candidates += LIGHT_CANDIDATES
    color, ratio, bg = choose_best_color(roi, candidates)
    return {
        "color_bgr": color,
        "contrast_ratio": ratio,
        "background_bgr": bg,
        "passes": ratio >= float(min_contrast),
    }
