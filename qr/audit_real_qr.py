"""Audit a QR detection dataset without treating decoded strings as ground truth.

The source images are read only. VOC boxes are preferred; YOLO labels are used
only when a matching VOC XML is absent. Crops preserve native pixel resolution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELDS = [
    "image_id", "source_image", "source_sha256", "width", "height", "split",
    "has_voc", "has_yolo", "annotation_source", "annotation_count", "box_index",
    "class_name", "xmin", "ymin", "xmax", "ymax", "box_width", "box_height",
    "box_area_ratio", "crop_path", "crop_width", "crop_height",
    "opencv_full_returned", "zxing_full_returned", "opencv_crop_returned",
    "zxing_crop_returned", "full_decoders_agree", "crop_decoders_agree",
]
IMAGE_FIELDS = [
    "image_id", "source_image", "source_sha256", "width", "height", "split",
    "has_voc", "has_yolo", "annotation_source", "annotation_count",
    "opencv_full_returned", "zxing_full_returned", "full_decoders_agree",
]


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError):
        return None


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode crop: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def read_splits(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for folder in (root / "ImageSets" / "Main", root / "ImageSets"):
            path = folder / f"{name}.txt"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                stem = line.strip().split(" ")[0]
                if stem and stem not in result:
                    result[stem] = name
            break
    return result


def parse_voc(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            box = tuple(float(bnd.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        boxes.append((obj.findtext("name", "unknown"), box))
    return boxes


def parse_yolo(path: Path, width: int, height: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    boxes = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id, xc, yc, bw, bh = (float(part) for part in parts[:5])
        except ValueError:
            continue
        boxes.append((f"yolo_{int(class_id)}", (
            width * (xc - bw / 2), height * (yc - bh / 2),
            width * (xc + bw / 2), height * (yc + bh / 2),
        )))
    return boxes


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = box
    coordinates = (
        max(0, min(width, int(np.floor(x1)))),
        max(0, min(height, int(np.floor(y1)))),
        max(0, min(width, int(np.ceil(x2)))),
        max(0, min(height, int(np.ceil(y2)))),
    )
    return coordinates if coordinates[2] > coordinates[0] and coordinates[3] > coordinates[1] else None


def crop_with_context(image: np.ndarray, box: tuple[int, int, int, int], pad_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    dx = max(4, int(round((x2 - x1) * pad_ratio)))
    dy = max(4, int(round((y2 - y1) * pad_ratio)))
    return image[max(0, y1 - dy):min(height, y2 + dy), max(0, x1 - dx):min(width, x2 + dx)].copy()


def opencv_decode(detector: cv2.QRCodeDetector, image: np.ndarray) -> str:
    for candidate in (image, cv2.bitwise_not(image)):
        try:
            text, _, _ = detector.detectAndDecode(candidate)
        except cv2.error:
            text = ""
        if text:
            return text
    return ""


def zxing_decode(zxingcpp, image: np.ndarray) -> str:
    if zxingcpp is None:
        return ""
    for candidate in (image, cv2.bitwise_not(image)):
        try:
            result = zxingcpp.read_barcode(candidate)
            text = result.text if result is not None else ""
        except (RuntimeError, ValueError):
            text = ""
        if text:
            return text
    return ""


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p10", "p25", "median", "p75", "p90", "max")}
    arr = np.asarray(values, dtype=np.float64)
    return {key: float(np.quantile(arr, q)) for key, q in (
        ("min", 0), ("p10", .1), ("p25", .25), ("median", .5),
        ("p75", .75), ("p90", .9), ("max", 1),
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Folder containing images/ and Annotations/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad-ratio", type=float, default=.2)
    parser.add_argument("--export-crops", action="store_true")
    parser.add_argument("--decode", action="store_true", help="Try OpenCV and optional ZXing-C++ on whole images and crops")
    args = parser.parse_args()
    if args.pad_ratio < 0:
        parser.error("--pad-ratio must be non-negative")
    root = args.dataset_root.resolve()
    image_dir = root / "images"
    if not image_dir.is_dir():
        parser.error(f"Image directory does not exist: {image_dir}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    splits = read_splits(root)
    detector = cv2.QRCodeDetector() if args.decode else None
    zxingcpp = None
    if args.decode:
        try:
            import zxingcpp as zxingcpp_module
            zxingcpp = zxingcpp_module
        except ImportError:
            pass

    counts = Counter()
    dimensions = Counter()
    source_hashes = defaultdict(list)
    area_ratios: list[float] = []
    short_sides: list[float] = []
    rows = []
    image_rows = []
    for index, path in enumerate(files, start=1):
        image = read_image(path)
        if image is None:
            counts["unreadable_images"] += 1
            continue
        height, width = image.shape[:2]
        dimensions[f"{width}x{height}"] += 1
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes[sha].append(path.name)
        voc = root / "Annotations" / f"{path.stem}.xml"
        yolo = root / "labels" / f"{path.stem}.txt"
        counts["images"] += 1
        counts["voc_files"] += voc.is_file()
        counts["yolo_files"] += yolo.is_file()
        boxes = []
        annotation_source = "none"
        if voc.is_file():
            try:
                boxes = parse_voc(voc)
                annotation_source = "voc"
            except ET.ParseError:
                counts["invalid_voc_files"] += 1
        if not boxes and yolo.is_file():
            boxes = parse_yolo(yolo, width, height)
            annotation_source = "yolo"
            counts["yolo_fallback_images"] += 1
        valid_boxes = []
        for class_name, raw_box in boxes:
            box = clamp_box(raw_box, width, height)
            if box is None:
                counts["invalid_boxes"] += 1
            else:
                valid_boxes.append((class_name, box))
        if not valid_boxes:
            counts["unlabeled_images"] += 1
        if len(valid_boxes) > 1:
            counts["multi_box_images"] += 1
        split = splits.get(path.stem, "unassigned")
        counts[f"split_{split}_images"] += 1
        whole_cv = opencv_decode(detector, image) if detector is not None else ""
        whole_zx = zxing_decode(zxingcpp, image) if detector is not None else ""
        if args.decode:
            counts["opencv_full_returned_images"] += bool(whole_cv)
            counts["zxing_full_returned_images"] += bool(whole_zx)
            counts["full_decoders_agree_images"] += bool(whole_cv and whole_zx and whole_cv == whole_zx)
        image_rows.append({
            "image_id": path.stem, "source_image": f"images/{path.name}", "source_sha256": sha,
            "width": width, "height": height, "split": split, "has_voc": int(voc.is_file()),
            "has_yolo": int(yolo.is_file()), "annotation_source": annotation_source,
            "annotation_count": len(valid_boxes),
            "opencv_full_returned": int(bool(whole_cv)) if args.decode else "",
            "zxing_full_returned": int(bool(whole_zx)) if args.decode and zxingcpp is not None else "",
            "full_decoders_agree": int(bool(whole_cv and whole_zx and whole_cv == whole_zx)) if args.decode and zxingcpp is not None else "",
        })
        for box_index, (class_name, box) in enumerate(valid_boxes, start=1):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1) / (width * height)
            area_ratios.append(area)
            short_sides.append(float(min(x2 - x1, y2 - y1)))
            crop = crop_with_context(image, box, args.pad_ratio)
            crop_rel = Path("crops") / f"{path.stem}__{box_index:02d}.png"
            if args.export_crops:
                write_png(output / crop_rel, crop)
            crop_cv = opencv_decode(detector, crop) if detector is not None else ""
            crop_zx = zxing_decode(zxingcpp, crop) if detector is not None else ""
            if args.decode:
                counts["opencv_crop_returned_boxes"] += bool(crop_cv)
                counts["zxing_crop_returned_boxes"] += bool(crop_zx)
                counts["crop_decoders_agree_boxes"] += bool(crop_cv and crop_zx and crop_cv == crop_zx)
            rows.append({
                "image_id": path.stem, "source_image": f"images/{path.name}", "source_sha256": sha,
                "width": width, "height": height, "split": split, "has_voc": int(voc.is_file()),
                "has_yolo": int(yolo.is_file()), "annotation_source": annotation_source,
                "annotation_count": len(valid_boxes), "box_index": box_index, "class_name": class_name,
                "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                "box_width": x2 - x1, "box_height": y2 - y1, "box_area_ratio": round(area, 8),
                "crop_path": crop_rel.as_posix() if args.export_crops else "",
                "crop_width": crop.shape[1], "crop_height": crop.shape[0],
                "opencv_full_returned": int(bool(whole_cv)) if args.decode else "",
                "zxing_full_returned": int(bool(whole_zx)) if args.decode and zxingcpp is not None else "",
                "opencv_crop_returned": int(bool(crop_cv)) if args.decode else "",
                "zxing_crop_returned": int(bool(crop_zx)) if args.decode and zxingcpp is not None else "",
                "full_decoders_agree": int(bool(whole_cv and whole_zx and whole_cv == whole_zx)) if args.decode and zxingcpp is not None else "",
                "crop_decoders_agree": int(bool(crop_cv and crop_zx and crop_cv == crop_zx)) if args.decode and zxingcpp is not None else "",
            })
        if index % 100 == 0 or index == len(files):
            print(f"Audited {index}/{len(files)} images", flush=True)

    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "images.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(image_rows)
    duplicate_groups = [names for names in source_hashes.values() if len(names) > 1]
    summary = {
        "source_root": str(root),
        "source_repository": "https://gitcode.com/open-source-toolkit/24565/blob/main/README.md",
        "source_images_are_modified": False,
        "readme_description": "1085-image YOLOv5 QR recognition dataset; not a physical-damage ground-truth dataset",
        "decode_interpretation": "Non-empty decoder returns only; payload truth and photo authenticity are not established",
        "settings": {"pad_ratio": args.pad_ratio, "export_crops": args.export_crops, "decode": args.decode,
                     "zxingcpp_available": zxingcpp is not None},
        "counts": dict(counts),
        "annotated_boxes": len(rows),
        "box_area_fraction_quantiles": quantiles(area_ratios),
        "box_short_side_px_quantiles": quantiles(short_sides),
        "common_dimensions": dimensions.most_common(20),
        "exact_duplicate_groups": duplicate_groups,
        "limitations": [
            "VOC/YOLO boxes identify QR locations, not damage masks or clean targets.",
            "Decoded strings have not been checked against independently recorded payloads.",
            "Images may be advertisements, illustrations, screenshots, photographs, or composites; source type requires manual curation.",
            "Repository README claims MIT, but LICENSE contains unfilled copyright placeholders and image provenance is not documented.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'summary.json'}, {output / 'images.csv'}, and {output / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
