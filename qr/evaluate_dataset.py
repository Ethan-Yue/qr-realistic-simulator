"""Engineering checks for generated QR pairs; this is not a realism benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from qr_synth.utils import read_cv_image


def read_image(path: Path, grayscale: bool = False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return read_cv_image(path, flag)


def decode(detector, image, expected):
    try:
        payload, _, _ = detector.detectAndDecode(image)
    except cv2.error:
        payload = ""
    inverted = False
    if not payload:
        try:
            payload, _, _ = detector.detectAndDecode(cv2.bitwise_not(image))
            inverted = bool(payload)
        except cv2.error:
            payload = ""
    detected = bool(payload)
    return detected, (payload == expected if expected else None), inverted


def decode_zxing(zxingcpp, image, expected):
    """Optional independent decoder for an engineering cross-check."""
    payload = ""
    inverted = False
    for candidate, is_inverted in ((image, False), (cv2.bitwise_not(image), True)):
        try:
            result = zxingcpp.read_barcode(candidate)
            payload = result.text if result is not None else ""
        except (RuntimeError, ValueError):
            payload = ""
        if payload:
            inverted = is_inverted
            break
    return bool(payload), (payload == expected if expected else None), inverted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="outputs_damaged or outputs directory")
    parser.add_argument("--out", type=Path, default=Path("evaluation_summary.json"))
    parser.add_argument("--limit-per-combo", type=int, default=0)
    parser.add_argument("--zxingcpp", action="store_true", help="Also evaluate with ZXing-C++ (optional zxing-cpp package)")
    args = parser.parse_args()

    detector = cv2.QRCodeDetector()
    zxingcpp = None
    if args.zxingcpp:
        try:
            import zxingcpp
        except ImportError as exc:
            raise RuntimeError("Install the optional zxing-cpp package for --zxingcpp") from exc
    grouped = defaultdict(list)
    run_manifest_path = args.root / "run_manifest.json"
    run_manifest = None
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("status") != "complete":
            raise RuntimeError(f"Generation has not completed: {run_manifest_path}")
        combo_dirs = [args.root / name for name in run_manifest["combo_directories"]]
    else:
        combo_dirs = sorted(args.root.iterdir())
    for combo in combo_dirs:
        metadata_dir = combo / "metadata"
        if not metadata_dir.is_dir():
            if run_manifest is not None:
                raise FileNotFoundError(metadata_dir)
            continue
        if run_manifest is not None:
            count = int(run_manifest["samples_per_combo"])
            files = [metadata_dir / f"{i:06d}.json" for i in range(count)]
            for path in files:
                if not path.exists():
                    raise FileNotFoundError(path)
        else:
            files = sorted(metadata_dir.glob("*.json"))
        if args.limit_per_combo > 0:
            files = files[:args.limit_per_combo]
        for meta_path in files:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stem = meta_path.stem
            clean_path = combo / "clean_images" / f"{stem}.png"
            if not clean_path.exists():
                clean_path = combo / "images" / f"{stem}.png"
            damaged_path = combo / "damaged_images" / f"{stem}.png"
            if not damaged_path.exists():
                damaged_path = clean_path
            clean = read_image(clean_path)
            damaged = read_image(damaged_path)
            qrm_path = combo / "qr_masks" / f"{stem}.png"
            if not qrm_path.exists():
                qrm_path = combo / "masks" / f"{stem}.png"
            qrm = read_image(qrm_path, grayscale=True)
            dmask_path = combo / "damage_masks" / f"{stem}.png"
            dmask = read_image(dmask_path, grayscale=True) if dmask_path.exists() else np.zeros_like(qrm)
            visible_path = combo / "visible_change_masks" / f"{stem}.png"
            visible = read_image(visible_path, grayscale=True) if visible_path.exists() else None
            shape_ok = clean.shape == damaged.shape and qrm.shape == dmask.shape == clean.shape[:2]
            if visible is not None:
                shape_ok = shape_ok and visible.shape == clean.shape[:2]
            mask_values_ok = set(np.unique(qrm)).issubset({0, 255}) and set(np.unique(dmask)).issubset({0, 255})
            if visible is not None:
                mask_values_ok = mask_values_ok and set(np.unique(visible)).issubset({0, 255})
            # An externally supplied QR may have a string inferred by OpenCV,
            # but that is not independent payload ground truth.
            expected = "" if meta.get("qr_input_info") else str(meta.get("qr_payload") or "")
            if not expected and str(meta.get("qr_source", "")).startswith("QRDATASET://"):
                expected = str(meta["qr_source"])
            original = read_image(combo / "qr_original" / f"{stem}.png")
            original_decoded, original_correct, _ = decode(detector, original, expected)
            clean_decoded, clean_correct, clean_inverted = decode(detector, clean, expected)
            damaged_decoded, damaged_correct, damaged_inverted = decode(detector, damaged, expected)
            group = str(meta.get("combo_id", combo.name))
            record = {
                "shape_ok": shape_ok,
                "mask_values_ok": mask_values_ok,
                "original_decoded": original_decoded,
                "original_correct": original_correct,
                "clean_decoded": clean_decoded,
                "clean_correct": clean_correct,
                "clean_inverted": clean_inverted,
                "damaged_decoded": damaged_decoded,
                "damaged_correct": damaged_correct,
                "damaged_inverted": damaged_inverted,
                "qr_damage_ratio": float(((qrm > 0) & (dmask > 0)).sum() / max(1, (qrm > 0).sum())),
                "rendered_contrast_ratio": meta.get("contrast_control", {}).get("rendered_contrast_ratio"),
            }
            if zxingcpp is not None:
                for label, image in (("original", original), ("clean", clean), ("damaged", damaged)):
                    found, correct, inverted = decode_zxing(zxingcpp, image, expected)
                    record[f"zxing_{label}_decoded"] = found
                    record[f"zxing_{label}_correct"] = correct
                    record[f"zxing_{label}_inverted"] = inverted
            grouped[group].append(record)

    rows = []
    for group, records in sorted(grouped.items(), key=lambda item: int(item[0])):
        n = len(records)
        known_clean = [x["clean_correct"] for x in records if x["clean_correct"] is not None]
        known_damaged = [x["damaged_correct"] for x in records if x["damaged_correct"] is not None]
        known_original = [x["original_correct"] for x in records if x["original_correct"] is not None]
        contrasts = [x["rendered_contrast_ratio"] for x in records if x["rendered_contrast_ratio"] is not None]
        row = {
            "combo_id": int(group),
            "samples": n,
            "shape_and_mask_valid": sum(x["shape_ok"] and x["mask_values_ok"] for x in records),
            "original_decode_rate": sum(x["original_decoded"] for x in records) / n,
            "original_payload_accuracy": sum(known_original) / len(known_original) if known_original else None,
            "clean_decode_rate": sum(x["clean_decoded"] for x in records) / n,
            "damaged_decode_rate": sum(x["damaged_decoded"] for x in records) / n,
            "clean_inverted_decode_rate": sum(x["clean_inverted"] for x in records) / n,
            "damaged_inverted_decode_rate": sum(x["damaged_inverted"] for x in records) / n,
            "clean_payload_accuracy": sum(known_clean) / len(known_clean) if known_clean else None,
            "damaged_payload_accuracy": sum(known_damaged) / len(known_damaged) if known_damaged else None,
            "mean_qr_damage_ratio": float(np.mean([x["qr_damage_ratio"] for x in records])),
            "median_rendered_contrast_ratio": float(np.median(contrasts)) if contrasts else None,
        }
        if zxingcpp is not None:
            for label in ("original", "clean", "damaged"):
                known = [x[f"zxing_{label}_correct"] for x in records if x[f"zxing_{label}_correct"] is not None]
                row[f"zxing_{label}_decode_rate"] = sum(x[f"zxing_{label}_decoded"] for x in records) / n
                row[f"zxing_{label}_payload_accuracy"] = sum(known) / len(known) if known else None
                row[f"zxing_{label}_inverted_decode_rate"] = sum(x[f"zxing_{label}_inverted"] for x in records) / n
        rows.append(row)

    all_records = [record for records in grouped.values() for record in records]
    totals = {
        "shape_and_mask_valid": sum(x["shape_ok"] and x["mask_values_ok"] for x in all_records),
    }
    for label in ("original", "clean", "damaged"):
        totals[f"opencv_{label}_decoded"] = sum(x[f"{label}_decoded"] for x in all_records)
        totals[f"opencv_{label}_correct_payload"] = sum(x[f"{label}_correct"] is True for x in all_records)
        if zxingcpp is not None:
            totals[f"zxing_{label}_decoded"] = sum(x[f"zxing_{label}_decoded"] for x in all_records)
            totals[f"zxing_{label}_correct_payload"] = sum(x[f"zxing_{label}_correct"] is True for x in all_records)

    summary = {
        "scope": "synthetic engineering check only; no claim of photographic realism or real-domain generalization",
        "root": str(args.root.resolve()),
        "run_manifest": str(run_manifest_path.resolve()) if run_manifest is not None else None,
        "combos": len(rows),
        "samples": sum(row["samples"] for row in rows),
        "opencv_version": cv2.__version__,
        "secondary_decoder": "ZXing-C++" if zxingcpp is not None else None,
        "totals": totals,
        "per_combo": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Checked {summary['samples']} samples across {summary['combos']} combos -> {args.out}")


if __name__ == "__main__":
    main()
