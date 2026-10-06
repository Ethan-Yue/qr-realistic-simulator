from __future__ import annotations

from typing import Dict, Optional, Tuple
import math
import cv2
import numpy as np


def _rng(rng=None):
    return rng if rng is not None else np.random.default_rng()


def severity_scale(severity: str) -> float:
    return {"mild": 0.65, "medium": 1.0, "severe": 1.55}.get(str(severity).lower(), 1.0)


def qr_bbox(qr_mask: Optional[np.ndarray], h: int, w: int) -> Tuple[int, int, int, int]:
    if qr_mask is None or not np.any(qr_mask > 0):
        return 0, 0, w, h
    ys, xs = np.where(qr_mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(4, int((x1 - x0) * 0.08))
    pad_y = max(4, int((y1 - y0) * 0.08))
    return max(0, x0-pad_x), max(0, y0-pad_y), min(w, x1+pad_x), min(h, y1+pad_y)


def choose_center(h: int, w: int, rng, qr_mask: Optional[np.ndarray] = None, qr_bias: float = 0.65):
    if qr_mask is not None and rng.random() < qr_bias and np.any(qr_mask > 0):
        ys, xs = np.where(qr_mask > 0)
        k = int(rng.integers(0, len(xs)))
        return int(xs[k]), int(ys[k])
    return int(rng.integers(0, w)), int(rng.integers(0, h))


def feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.float32) / 255.0
    k = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), sigma)


def blend_color(image: np.ndarray, mask: np.ndarray, color_bgr, opacity: float = 1.0, feather_sigma: float = 0.0):
    alpha = feather(mask, feather_sigma) * float(opacity)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    color = np.empty_like(image, dtype=np.float32)
    color[:] = np.array(color_bgr, dtype=np.float32)
    out = image.astype(np.float32) * (1-alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_toward(image: np.ndarray, mask: np.ndarray, target: float, opacity: float = 1.0, feather_sigma: float = 0.0):
    t = (target, target, target)
    return blend_color(image, mask, t, opacity, feather_sigma)


def irregular_blob_mask(shape, rng, center=None, radius_range=(15, 80), n_blobs=(1, 4), smooth=9, qr_mask=None, qr_bias=0.65):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    count = int(rng.integers(n_blobs[0], n_blobs[1] + 1))
    for _ in range(count):
        cx, cy = center if center is not None else choose_center(h, w, rng, qr_mask, qr_bias)
        r = int(rng.integers(max(2, radius_range[0]), max(radius_range[0]+1, radius_range[1]+1)))
        n = int(rng.integers(7, 15))
        pts=[]
        for i in range(n):
            a = 2*math.pi*i/n + rng.normal(0, 0.08)
            rr = r * rng.uniform(0.55, 1.15)
            x = int(np.clip(cx + math.cos(a)*rr, 0, w-1))
            y = int(np.clip(cy + math.sin(a)*rr, 0, h-1))
            pts.append([x,y])
        cv2.fillPoly(m, [np.asarray(pts,np.int32)], 255)
    if smooth > 0:
        k = int(smooth) | 1
        m = cv2.GaussianBlur(m, (k,k), 0)
        m = (m > 35).astype(np.uint8)*255
    return m


def speckle_mask(shape, rng, count, radius_range=(1,5), qr_mask=None, qr_bias=0.5):
    h,w=shape
    # 高清图按分辨率同比放大微小污点，避免 2K/4K 图上的污损变成亚像素级。
    rs=max(1.0, min(h,w)/1024.0)
    r0=max(1,int(round(radius_range[0]*rs))); r1=max(r0,int(round(radius_range[1]*rs)))
    m=np.zeros((h,w),np.uint8)
    for _ in range(max(1,int(count))):
        cx,cy=choose_center(h,w,rng,qr_mask,qr_bias)
        rx=int(rng.integers(r0, r1+1))
        ry=max(1,int(rx*rng.uniform(0.5,1.6)))
        cv2.ellipse(m,(cx,cy),(rx,ry),float(rng.uniform(0,180)),0,360,255,-1)
    return m


def scratch_mask(shape, rng, count=5, width_range=(1,4), length_ratio=(0.08,0.35), qr_mask=None, qr_bias=0.75):
    h,w=shape
    m=np.zeros((h,w),np.uint8)
    base=min(h,w)
    rs=max(1.0, base/1024.0)
    wr0=max(1,int(round(width_range[0]*rs))); wr1=max(wr0,int(round(width_range[1]*rs)))
    for _ in range(max(1,int(count))):
        cx,cy=choose_center(h,w,rng,qr_mask,qr_bias)
        length=int(base*rng.uniform(*length_ratio))
        angle=float(rng.uniform(0,2*math.pi))
        x0=int(cx-math.cos(angle)*length/2); y0=int(cy-math.sin(angle)*length/2)
        x1=int(cx+math.cos(angle)*length/2); y1=int(cy+math.sin(angle)*length/2)
        thickness=int(rng.integers(wr0, wr1+1))
        cv2.line(m,(x0,y0),(x1,y1),255,thickness,cv2.LINE_AA)
    return m


def random_rect_mask(shape, rng, size_ratio=(0.08,0.28), qr_mask=None, qr_bias=0.7, angle=True):
    h,w=shape
    cx,cy=choose_center(h,w,rng,qr_mask,qr_bias)
    rw=max(2,int(w*rng.uniform(*size_ratio)))
    rh=max(2,int(h*rng.uniform(size_ratio[0]*0.35,size_ratio[1]*0.65)))
    a=float(rng.uniform(-25,25) if angle else 0)
    rect=((cx,cy),(rw,rh),a)
    box=cv2.boxPoints(rect).astype(np.int32)
    m=np.zeros((h,w),np.uint8); cv2.fillPoly(m,[box],255)
    return m


def erode_mask_random(mask: np.ndarray, rng, frac=0.15):
    ys,xs=np.where(mask>0)
    if len(xs)==0:
        return np.zeros_like(mask)
    out=np.zeros_like(mask)
    n=max(1,int(len(xs)*frac))
    idx=rng.choice(len(xs),size=min(n,len(xs)),replace=False)
    out[ys[idx],xs[idx]]=255
    k=int(rng.integers(2,6))*2+1
    out=cv2.dilate(out,np.ones((k,k),np.uint8),iterations=1)
    out=cv2.bitwise_and(out,mask)
    return out


def module_patch_mask(qr_mask: np.ndarray, rng, patch_count=6, patch_ratio=(0.02,0.08)):
    h,w=qr_mask.shape
    x0,y0,x1,y1=qr_bbox(qr_mask,h,w)
    m=np.zeros_like(qr_mask)
    bw=max(1,x1-x0); bh=max(1,y1-y0)
    for _ in range(max(1,int(patch_count))):
        rw=max(2,int(bw*rng.uniform(*patch_ratio)))
        rh=max(2,int(bh*rng.uniform(*patch_ratio)))
        cx=int(rng.integers(x0,x1)); cy=int(rng.integers(y0,y1))
        cv2.rectangle(m,(max(0,cx-rw//2),max(0,cy-rh//2)),(min(w-1,cx+rw//2),min(h-1,cy+rh//2)),255,-1)
    return cv2.bitwise_and(m,qr_mask)
