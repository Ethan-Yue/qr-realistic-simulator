from __future__ import annotations
from pathlib import Path
import hashlib
import cv2
import numpy as np
import qrcode


def read_cv_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image through Python's Unicode-aware file API on Windows."""
    path = Path(path)
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_cv_image(path: Path, image: np.ndarray) -> None:
    """Encode and write an image, failing instead of ignoring an OpenCV write error."""
    path = Path(path)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())


def derive_seed(base_seed: int, combo_id: int, sample_index: int, stage: int = 0) -> int:
    """Stable per-sample/stage seed, independent of combo selection or loop order."""
    parts = (int(base_seed), int(combo_id), int(sample_index), int(stage))
    if any(value < 0 for value in parts):
        raise ValueError("Seed coordinates must be non-negative integers")
    return int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint32)[0])

ECC_LEVELS = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}


def make_payload(combo_id: int, sample_index: int, sample_seed: int, rng,
                 mode: str = "id", length_range: tuple[int, int] = (36, 96),
                 alphabet_mode: str = "alphanumeric") -> str:
    """Create a unique, reproducible synthetic payload with optional length variation."""
    if mode == "id":
        return f"QRDATASET://combo/{combo_id}/sample/{sample_index}/seed/{sample_seed}"
    if mode != "variable":
        raise ValueError(f"Unknown payload mode: {mode}")
    low, high = (int(length_range[0]), int(length_range[1]))
    if low < 1 or high < low:
        raise ValueError("length_range must contain positive ascending integers")
    # The payload must not contain the carrier/marking label. Otherwise a
    # restoration model can exploit a content-to-class shortcut.
    prefix = "QRDATASET://"
    length = max(len(prefix), int(rng.integers(low, high + 1)))
    alphabets = {
        "alphanumeric": b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-./:",
        "byte": b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    }
    if alphabet_mode not in alphabets:
        raise ValueError(f"Unknown payload alphabet: {alphabet_mode}")
    alphabet = np.frombuffer(alphabets[alphabet_mode], dtype="S1")
    tail = b"".join(rng.choice(alphabet, size=length - len(prefix)).tolist()).decode("ascii")
    return prefix + tail


def make_qr_mask(data: str, pixels=420, border_modules=4,
                 error_correction="M", version=None, return_info=False):
    """
    生成原始 QR。
    返回:
      original_bgr: 原始黑白 QR PNG
      ink_mask:     255=黑模块, 0=背景/静区
    """
    ecc = str(error_correction).upper()
    if ecc not in ECC_LEVELS:
        raise ValueError(f"Unknown QR error correction level: {error_correction}")
    qr=qrcode.QRCode(
        version=version,
        error_correction=ECC_LEVELS[ecc],
        box_size=1,
        border=border_modules,
    )
    qr.add_data(data)
    qr.make(fit=version is None)
    matrix=np.asarray(qr.get_matrix(), dtype=np.uint8)
    module_count=int(matrix.shape[0])
    # Every module occupies the same integer number of source pixels. Directly
    # resizing a QR to an arbitrary side can create uneven modules and an
    # undecodable clean target (especially at high versions).
    module_pitch=max(1, int(np.ceil(int(pixels) / module_count)))
    arr=np.repeat(np.repeat(np.where(matrix > 0, 0, 255).astype(np.uint8),
                            module_pitch, axis=0), module_pitch, axis=1)
    mask=(arr<128).astype(np.uint8)*255
    original=cv2.cvtColor(arr,cv2.COLOR_GRAY2BGR)
    if return_info:
        return original, mask, {
            "version": int(qr.version),
            "error_correction": ecc,
            "border_modules": int(border_modules),
            "payload_length": len(data),
            "module_count_with_border": module_count,
            "source_module_pitch_px": module_pitch,
            "source_side_px": int(arr.shape[0]),
        }
    return original,mask

def load_qr_image(path: Path, pixels=420, return_info=False, require_decodable=False):
    """Normalize a user QR input and optionally validate its decoded payload."""
    im=read_cv_image(path,cv2.IMREAD_GRAYSCALE)
    input_size_hw=[int(im.shape[0]), int(im.shape[1])]
    im=cv2.resize(im,(pixels,pixels),interpolation=cv2.INTER_AREA)
    _,bw=cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # 保证背景白、码黑
    if bw.mean()<127:
        bw=255-bw
    mask=(bw<128).astype(np.uint8)*255
    original=cv2.cvtColor(bw,cv2.COLOR_GRAY2BGR)
    if return_info or require_decodable:
        detector=cv2.QRCodeDetector()
        payload=""
        for candidate in (original, cv2.bitwise_not(original)):
            try:
                payload,_,_=detector.detectAndDecode(candidate)
            except cv2.error:
                payload=""
            if payload:
                break
        if require_decodable and not payload:
            raise ValueError(f"QR input is not decodable after normalization: {path}")
        sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        info={
            "source_path": str(path),
            "source_sha256": sha,
            "source_size_hw": input_size_hw,
            "normalized_size_hw": [int(bw.shape[0]), int(bw.shape[1])],
            "decoded_payload": payload or None,
            "decoder": "OpenCV QRCodeDetector normal-or-inverted",
        }
        if return_info:
            return original,mask,info
    return original,mask

def list_qr_inputs(folder: Path):
    exts={".png",".jpg",".jpeg",".bmp",".webp"}
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])

def slugify_combo(c):
    return f"{int(c['id']):02d}_{c['carrier']}_{c['method']}"
