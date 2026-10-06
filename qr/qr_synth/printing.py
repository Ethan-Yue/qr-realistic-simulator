from __future__ import annotations
import cv2
import numpy as np


def _make_ink(mask, color=(20, 20, 20), alpha=255):
    """理想、无缺陷的成码层。mask=255 表示 QR 黑模块。"""
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[..., :3] = np.array(color, np.uint8)
    rgba[..., 3] = np.where(mask > 0, alpha, 0).astype(np.uint8)
    return rgba


def _regular_dot_fill(mask, period=4, dot=3, color=(15, 15, 15)):
    """
    规则网点/喷墨颗粒，仅用于表示工艺的理想结构，不做随机缺墨。
    注意：外轮廓仍保持完整，避免把“工艺纹理”变成污损。
    """
    h, w = mask.shape
    rgba = _make_ink(mask, color=color, alpha=255)

    # 在黑模块内部轻微调制亮度形成规则点阵，不改变 mask 几何形状
    yy, xx = np.mgrid[0:h, 0:w]
    dotmask = ((xx % period) < dot) & ((yy % period) < dot) & (mask > 0)
    rgba[dotmask, :3] = np.clip(np.array(color) + 12, 0, 255)
    return rgba


def _thread_fill(mask, color=(18, 18, 24), step=5, mode="woven"):
    """理想织造/刺绣纹理：只改变模块内部纹理，不引入断线、变形或缺失。"""
    rgba = _make_ink(mask, color=color, alpha=255)
    if mode == "woven":
        for x in range(0, mask.shape[1], step):
            idx = (mask[:, x:x + 1] > 0)
            rgba[:, x:x + 1, :3][idx] = np.clip(np.array(color) + 20, 0, 255)
        for y in range(0, mask.shape[0], step):
            idx = (mask[y:y + 1, :] > 0)
            rgba[y:y + 1, :, :3][idx] = np.clip(np.array(color) + 8, 0, 255)
    else:  # embroidery
        for y in range(0, mask.shape[0], step):
            idx = (mask[y:y + 1, :] > 0)
            rgba[y:y + 1, :, :3][idx] = np.clip(np.array(color) + 24, 0, 255)
    return rgba


def render_qr(qr_mask, carrier, method, background_mean=230.0, color_override=None):
    """
    只模拟“理想工艺外观”，不模拟打印缺陷/污损。

    不包含：
    - 缺墨、断点、条带
    - 边缘毛刺/随机扩散
    - 模糊、形变
    - 随机噪声
    - 光照高光

    返回：
      rgba   : BGRA 成码层
      fgmask : 二值模块 mask，几何上与原 QR 一致
    """
    m = (qr_mask > 0).astype(np.uint8) * 255

    # 由局部背景对比度模块选出的颜色优先。工艺内部纹理仍保持不变。
    if color_override is not None:
        base_override = tuple(int(v) for v in color_override)
    else:
        base_override = None

    # ---------- 纸张/通用印刷：理想、完整、清晰 ----------
    if method in {"laser_print", "digital_print", "photo_print"}:
        return _make_ink(m, base_override or (10, 10, 10)), m

    if method in {"inkjet_print", "digital_inkjet", "direct_inkjet", "spray_print"}:
        # 保留规则墨滴观感，但不做随机缺墨
        return _regular_dot_fill(m, period=4, dot=3, color=base_override or (14, 14, 14)), m

    if method in {"offset_print", "gravure_print", "flexo_print", "uv_roll_print"}:
        # 规则网点只是理想印刷结构，不改变二维码轮廓
        return _regular_dot_fill(m, period=5, dot=4, color=base_override or (12, 12, 12)), m

    if method == "thermal_print":
        # 热敏纸典型灰黑，但保持理想完整
        return _make_ink(m, base_override or (42, 42, 42)), m

    if method in {"thermal_transfer", "heat_transfer"}:
        return _make_ink(m, base_override or (8, 8, 8)), m

    if method == "industrial_inkjet":
        # 工业喷码可呈规则点阵，但不删除点、不破坏结构
        return _regular_dot_fill(m, period=5, dot=3, color=base_override or (26, 26, 26)), m

    # ---------- 丝印 / 移印 / UV ----------
    if method == "screen_print":
        return _make_ink(m, base_override or (6, 6, 6)), m

    if method == "pad_print":
        return _make_ink(m, base_override or (18, 18, 18)), m

    if method in {"uv_print", "uv_inkjet"}:
        return _make_ink(m, base_override or (4, 4, 4)), m

    # ---------- 激光/蚀刻/烧灼：只表现材料颜色差异 ----------
    if method == "laser_mark":
        # 深色塑料上通常采用浅色激光标记；浅色材质则用深色
        if carrier == "plastic_shell" and background_mean < 128:
            col = (226, 226, 226)
        elif carrier == "metal":
            col = (65, 65, 68)
        else:
            col = (35, 35, 35)
        return _make_ink(m, base_override or col), m

    if method == "laser_burn":
        # 木材烧灼的理想棕黑色，不加烧焦毛边
        return _make_ink(m, base_override or (28, 48, 68)), m

    if method == "laser_engrave":
        # 用固定灰阶表达刻蚀/凹刻，不做阴影/高光
        col = (92, 92, 92) if background_mean > 150 else (205, 205, 205)
        return _make_ink(m, base_override or col), m

    if method == "laser_etch":
        col = (150, 150, 150) if background_mean > 160 else (220, 220, 220)
        return _make_ink(m, base_override or col), m

    # ---------- 纺织 ----------
    if method == "jacquard":
        return _thread_fill(m, color=base_override or (20, 20, 28), step=5, mode="woven"), m

    if method == "embroidery":
        return _thread_fill(m, color=base_override or (15, 15, 22), step=6, mode="embroidery"), m

    # ---------- 陶瓷 ----------
    if method == "underglaze":
        # 典型釉下蓝；仍然是完整理想模块
        return _make_ink(m, base_override or (150, 70, 30)), m

    # ---------- 标签/贴纸 ----------
    if method == "sticker":
        return _make_ink(m, base_override or (8, 8, 8)), m

    # ---------- 屏幕 ----------
    if method == "lcd_oled":
        # 屏幕自身 RGB 栅格由背景材质层负责；二维码保持干净黑色
        return _make_ink(m, base_override or (0, 0, 0)), m

    return _make_ink(m, base_override or (12, 12, 12)), m
