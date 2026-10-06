from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import hashlib
import cv2
import numpy as np

from .utils import read_cv_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    den = x.max() - x.min()
    if den < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return x / den


def _lowfreq_texture(rng, h, w, sigma=10.0):
    """仅用于生成材质本身的纹理，不是相机噪声或污损。"""
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return _norm01(n)


def _paper(size, rng):
    h = w = size
    base_val = int(rng.integers(238, 250))
    base = np.full((h, w, 3), base_val, np.float32)

    # 细纸纤维：属于材质纹理，而非污损
    for _ in range(int(rng.integers(45, 85))):
        y = int(rng.integers(0, h))
        x = int(rng.integers(0, w))
        ln = int(rng.integers(8, 48))
        c = float(rng.integers(base_val - 16, base_val - 4))
        cv2.line(base, (x, y), (min(w - 1, x + ln), y), (c, c, c), 1)

    tex = _lowfreq_texture(rng, h, w, sigma=3.5)
    base += (tex[..., None] - 0.5) * 4.0
    return np.clip(base, 0, 255).astype(np.uint8)


def _adhesive_label(size, rng):
    h = w = size
    base_val = int(rng.integers(238, 252))
    base = np.full((h, w, 3), base_val, np.float32)
    tex = _lowfreq_texture(rng, h, w, sigma=4.0)
    base += (tex[..., None] - 0.5) * 3.5
    return np.clip(base, 0, 255).astype(np.uint8)


def _cardboard(size, rng):
    h = w = size
    # BGR，随机牛皮纸色阶
    color = np.array([
        int(rng.integers(145, 174)),
        int(rng.integers(170, 198)),
        int(rng.integers(188, 218)),
    ], dtype=np.float32)
    base = np.tile(color, (h, w, 1))

    # 纸浆/纤维纹理
    tex = _lowfreq_texture(rng, h, w, sigma=2.0)
    base += (tex[..., None] - 0.5) * 10.0

    # 极弱的瓦楞方向纹，不做起伏/变形
    step = int(rng.integers(10, 16))
    for y in range(0, h, step):
        c = tuple(np.clip(color - rng.uniform(4, 10), 0, 255).tolist())
        cv2.line(base, (0, y), (w - 1, y), c, 1)
    return np.clip(base, 0, 255).astype(np.uint8)


def _plastic_pack(size, rng):
    h = w = size
    # 平整塑料膜，仅保留平面色与微观纹理；不生成褶皱/反光
    palette = [
        (240, 240, 240), (226, 236, 246), (236, 244, 230),
        (246, 236, 226), (230, 230, 242)
    ]
    color = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    base = np.tile(color, (h, w, 1))
    tex = _lowfreq_texture(rng, h, w, sigma=5.0)
    base += (tex[..., None] - 0.5) * 3.0
    return np.clip(base, 0, 255).astype(np.uint8)


def _paper_box(size, rng):
    h = w = size
    palette = [
        (246, 246, 246), (240, 244, 248), (244, 242, 236),
        (236, 246, 240), (246, 238, 242)
    ]
    color = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    base = np.tile(color, (h, w, 1))
    tex = _lowfreq_texture(rng, h, w, sigma=4.0)
    base += (tex[..., None] - 0.5) * 3.0
    return np.clip(base, 0, 255).astype(np.uint8)


def _metal(size, rng):
    h = w = size
    # 中性拉丝金属，仅生成方向纹，不加高光、阴影、曝光变化
    mean = float(rng.integers(175, 215))
    base = np.full((h, w), mean, np.float32)
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=7.0, sigmaY=0.25)
    n = _norm01(n) - 0.5
    base += n * rng.uniform(10, 18)
    return np.dstack([base, base, base]).clip(0, 255).astype(np.uint8)


def _plastic_shell(size, rng):
    h = w = size
    palette = [
        (232, 232, 232), (215, 215, 215), (58, 58, 58), (38, 38, 38),
        (222, 228, 236)
    ]
    color = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    base = np.tile(color, (h, w, 1))
    tex = _lowfreq_texture(rng, h, w, sigma=2.2)
    base += (tex[..., None] - 0.5) * 4.0
    return np.clip(base, 0, 255).astype(np.uint8)


def _glass(size, rng):
    h = w = size
    # 平整、无额外反光的玻璃底；仅用轻微冷色和微纹理提示材质
    color = np.array([
        int(rng.integers(232, 244)),
        int(rng.integers(238, 249)),
        int(rng.integers(240, 251)),
    ], dtype=np.float32)
    base = np.tile(color, (h, w, 1))
    tex = _lowfreq_texture(rng, h, w, sigma=7.0)
    base += (tex[..., None] - 0.5) * 2.5
    return np.clip(base, 0, 255).astype(np.uint8)


def _wood(size, rng):
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    freq = float(rng.uniform(0.018, 0.035))
    phase = float(rng.uniform(0, np.pi * 2))
    wav = np.sin(xx * freq + 0.7 * np.sin(yy * 0.010) + phase)
    fine = np.sin(xx * freq * 4.2 + phase * 0.3) * 0.18
    grain = wav * 0.72 + fine

    base = np.zeros((h, w, 3), np.float32)
    # 随机木色，但不引入光照梯度
    b0 = float(rng.integers(110, 145))
    g0 = float(rng.integers(160, 190))
    r0 = float(rng.integers(188, 220))
    base[..., 0] = b0 + grain * 16
    base[..., 1] = g0 + grain * 20
    base[..., 2] = r0 + grain * 24
    return np.clip(base, 0, 255).astype(np.uint8)


def _textile(size, rng):
    h = w = size
    palette = [
        (236, 236, 236), (215, 220, 225), (205, 212, 220),
        (228, 220, 210), (210, 225, 214)
    ]
    color = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    base = np.tile(color, (h, w, 1))

    step_x = int(rng.integers(3, 6))
    step_y = int(rng.integers(3, 6))
    darker = tuple(np.clip(color - 16, 0, 255).tolist())
    lighter = tuple(np.clip(color + 8, 0, 255).tolist())
    for x in range(0, w, step_x):
        cv2.line(base, (x, 0), (x, h - 1), darker, 1)
    for y in range(0, h, step_y):
        cv2.line(base, (0, y), (w - 1, y), lighter, 1)
    return np.clip(base, 0, 255).astype(np.uint8)


def _ceramic(size, rng):
    h = w = size
    palette = [(244, 244, 244), (238, 242, 246), (246, 242, 236), (235, 244, 240)]
    color = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    base = np.tile(color, (h, w, 1))

    # 釉面细小天然斑点，不模拟污渍
    for _ in range(int(rng.integers(40, 90))):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        c = tuple(np.clip(color - rng.uniform(6, 14), 0, 255).tolist())
        cv2.circle(base, (x, y), 1, c, -1)
    return np.clip(base, 0, 255).astype(np.uint8)


def _acrylic(size, rng):
    h = w = size
    palette = [(242, 246, 248), (236, 242, 246), (246, 246, 246)]
    color = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    base = np.tile(color, (h, w, 1))
    tex = _lowfreq_texture(rng, h, w, sigma=9.0)
    base += (tex[..., None] - 0.5) * 2.0
    return np.clip(base, 0, 255).astype(np.uint8)


def _poster(size, rng):
    h = w = size
    palette = [
        (246, 246, 246), (232, 242, 248), (244, 236, 232),
        (235, 245, 238), (242, 236, 246)
    ]
    color = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    base = np.tile(color, (h, w, 1))
    tex = _lowfreq_texture(rng, h, w, sigma=4.0)
    base += (tex[..., None] - 0.5) * 3.0
    return np.clip(base, 0, 255).astype(np.uint8)


def _ticket(size, rng):
    h = w = size
    base = np.full((h, w, 3), int(rng.integers(240, 250)), np.float32)
    # 干净、规则的防伪细纹；不是污损
    line_color = tuple(int(x) for x in rng.choice([
        [228, 222, 238], [224, 234, 238], [236, 226, 222]
    ]))
    step = int(rng.integers(14, 22))
    for k in range(-h, w, step):
        cv2.line(base, (k, 0), (k + h, h - 1), line_color, 1)
    return np.clip(base, 0, 255).astype(np.uint8)


def _screen(size, rng):
    h = w = size
    base_val = int(rng.integers(242, 251))
    base = np.full((h, w, 3), base_val, np.uint8)
    # 固定规则子像素栅格；属于屏幕介质本身
    for x in range(0, w, 3):
        base[:, x:x + 1, 2] = np.clip(base_val + 4, 0, 255)
        if x + 1 < w:
            base[:, x + 1:x + 2, 1] = np.clip(base_val + 4, 0, 255)
        if x + 2 < w:
            base[:, x + 2:x + 3, 0] = np.clip(base_val + 4, 0, 255)
    return base


GENERATORS = {
    "paper": _paper,
    "adhesive_label": _adhesive_label,
    "cardboard": _cardboard,
    "plastic_pack": _plastic_pack,
    "paper_box": _paper_box,
    "metal": _metal,
    "plastic_shell": _plastic_shell,
    "glass": _glass,
    "wood": _wood,
    "textile": _textile,
    "ceramic": _ceramic,
    "acrylic_pvc": _acrylic,
    "poster": _poster,
    "ticket_card": _ticket,
    "screen": _screen,
}


@lru_cache(maxsize=4096)
def _file_sha256(path_text: str) -> str:
    digest = hashlib.sha256()
    with open(path_text, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def random_real_background(root: Path, carrier: str, size: int, rng,
                           return_info: bool = False):
    """
    如果 assets/backgrounds/<carrier>/ 中有干净的实拍/扫描材质背景，优先随机抽取。
    注意：这里不做亮度、噪声、形变等增强，只做随机选图 + 随机方形裁剪 + resize。
    """
    folder = root / carrier
    if not folder.exists():
        return None

    files = sorted(
        (p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.relative_to(root).as_posix().lower(),
    )
    if not files:
        return None

    p = files[int(rng.integers(0, len(files)))]
    try:
        im = read_cv_image(p, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None

    h, w = im.shape[:2]
    s = min(h, w)
    if s < 8:
        return None

    x0 = int(rng.integers(0, max(1, w - s + 1)))
    y0 = int(rng.integers(0, max(1, h - s + 1)))
    crop = im[y0:y0 + s, x0:x0 + s]
    output = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    if not return_info:
        return output
    return output, {
        "source_kind": "background_library_image",
        "source_path": p.relative_to(root).as_posix(),
        "source_sha256": _file_sha256(str(p.resolve())),
        "source_size_hw": [int(h), int(w)],
        "crop_xywh": [int(x0), int(y0), int(s), int(s)],
        "output_size_hw": [int(size), int(size)],
    }


def generate_material(carrier: str, size: int, rng, background_root: Path | None = None,
                      real_bg_prob: float = 1.00, return_info: bool = False):
    """
    只负责“材质背景随机性”。
    不做：污损、形变、光照、相机噪声、模糊、JPEG 压缩。
    """
    if background_root is not None and rng.random() < real_bg_prob:
        x = random_real_background(background_root, carrier, size, rng, return_info=return_info)
        if x is not None:
            return x

    image = GENERATORS[carrier](size, rng)
    if not return_info:
        return image
    return image, {
        "source_kind": "procedural_material",
        "generator": carrier,
        "output_size_hw": [int(size), int(size)],
    }
