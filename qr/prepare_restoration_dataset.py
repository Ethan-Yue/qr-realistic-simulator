"""Build a leakage-checked paired manifest for synthetic QR restoration training.

The manifest references existing generated files; it does not copy images or
claim that synthetic validation/test scores measure physical-code transfer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


FIELDS = [
    "sample_id", "split", "group_id", "combo_id", "carrier", "marking_process",
    "damaged_image", "clean_target", "qr_original", "qr_mask", "damage_mask",
    "visible_change_mask", "metadata", "payload_sha256", "payload_ground_truth_available",
    "qr_version", "error_correction", "background_source_sha256", "background_source_kind",
    "layout_profile", "damage_spatial_mode", "damage_types", "damage_qr_overlap_ratio",
    "visible_change_area_ratio",
]


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)


def read_png(path: Path, mode: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, mode)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def safe_relative(root: Path, combo_dir: Path, name: str) -> str:
    path = (combo_dir / name).resolve()
    if not path.is_relative_to(combo_dir.resolve()) or not path.is_relative_to(root):
        raise ValueError(f"Output path escapes generator root: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.relative_to(root).as_posix()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Completed synthesize_with_damage.py output directory")
    parser.add_argument("--out", type=Path, default=None, help="CSV path; default ROOT/restoration_pairs.csv")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--train", type=float, default=.8)
    parser.add_argument("--val", type=float, default=.1)
    parser.add_argument("--include-undamaged", action="store_true", help="Keep identity pairs with no damage")
    args = parser.parse_args()
    if not (0 < args.train < 1 and 0 < args.val < 1 and args.train + args.val < 1):
        parser.error("Require positive train/val fractions whose sum is below 1")
    root = args.root.resolve()
    run_manifest_path = root / "run_manifest.json"
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("entry_point") != "synthesize_with_damage.py":
        raise ValueError("A complete synthesize_with_damage.py run_manifest.json is required")
    output = (args.out or root / "restoration_pairs.csv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = Counter()
    for combo_name in run["combo_directories"]:
        combo_dir = (root / combo_name).resolve()
        if not combo_dir.is_relative_to(root):
            raise ValueError(f"Invalid combination directory: {combo_name}")
        for index in range(int(run["samples_per_combo"])):
            stem = f"{index:06d}"
            meta_path = combo_dir / "metadata" / f"{stem}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("damage_enabled", False) or not meta.get("damage", {}).get("damage_layer_count", 0):
                if not args.include_undamaged:
                    skipped["no_damage_layer"] += 1
                    continue
            if float(meta.get("visible_change_area_ratio", 0)) <= 0 and not args.include_undamaged:
                skipped["no_visible_change"] += 1
                continue

            outputs = meta["outputs"]
            paths = {field: safe_relative(root, combo_dir, outputs[key]) for field, key in (
                ("damaged_image", "damaged_image"), ("clean_target", "clean_image"),
                ("qr_original", "qr_original"), ("qr_mask", "qr_mask"),
                ("damage_mask", "damage_mask"), ("visible_change_mask", "visible_change_mask"),
            )}
            clean = read_png(root / paths["clean_target"], cv2.IMREAD_COLOR)
            damaged = read_png(root / paths["damaged_image"], cv2.IMREAD_COLOR)
            if clean.shape != damaged.shape:
                raise ValueError(f"Paired image shapes differ: {combo_name}/{stem}")
            for field in ("qr_mask", "damage_mask", "visible_change_mask"):
                mask = read_png(root / paths[field], cv2.IMREAD_GRAYSCALE)
                if mask.shape != clean.shape[:2] or not np.isin(np.unique(mask), [0, 255]).all():
                    raise ValueError(f"Invalid binary mask {field}: {combo_name}/{stem}")

            structure = meta.get("qr_structure") or {}
            payload = meta.get("qr_payload")
            payload_gt = bool(structure and isinstance(payload, str) and payload)
            qr_input = meta.get("qr_input_info") or {}
            qr_fingerprint = sha_text(payload) if payload_gt else (
                qr_input.get("source_sha256") or sha_text(str(meta.get("qr_source", "")))
            )
            background = (meta.get("contrast_control") or {}).get("background_source") or {}
            background_sha = background.get("source_sha256") or ""
            rows.append({
                "sample_id": f"{combo_name}/{stem}", "combo_id": int(meta["combo_id"]),
                "carrier": meta["carrier"], "marking_process": meta["method"],
                **paths, "metadata": meta_path.relative_to(root).as_posix(),
                "payload_sha256": sha_text(payload) if payload_gt else "",
                "payload_ground_truth_available": int(payload_gt),
                "qr_version": structure.get("version", ""),
                "error_correction": structure.get("error_correction", ""),
                "background_source_sha256": background_sha,
                "background_source_kind": background.get("source_kind", "unknown"),
                "layout_profile": meta.get("layout_profile", "closeup"),
                "damage_spatial_mode": meta.get("damage_spatial_mode", "full_canvas"),
                "damage_types": "|".join(meta.get("damage", {}).get("damage_types", [])),
                "damage_qr_overlap_ratio": meta.get("damage", {}).get("qr_damage_ratio", ""),
                "visible_change_area_ratio": meta.get("visible_change_area_ratio", ""),
                "_group_keys": [key for key in (
                    f"background:{background_sha}" if background_sha else "",
                    f"symbol:{qr_fingerprint}" if qr_fingerprint else "",
                ) if key],
            })

    if not rows:
        raise ValueError("No eligible damaged image pairs found")
    dsu = DisjointSet(len(rows))
    first_for_key = {}
    for i, row in enumerate(rows):
        for key in row["_group_keys"]:
            if key in first_for_key:
                dsu.union(i, first_for_key[key])
            else:
                first_for_key[key] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[dsu.find(i)].append(i)
    groups = list(components.values())
    random.Random(args.seed).shuffle(groups)
    targets = {"train": args.train * len(rows), "val": args.val * len(rows),
               "test": (1 - args.train - args.val) * len(rows)}
    assigned = Counter()
    for group in groups:
        label = max(("train", "val", "test"), key=lambda name: (targets[name] - assigned[name]) / targets[name])
        group_id = sha_text("|".join(sorted(rows[i]["sample_id"] for i in group)))[:16]
        for i in group:
            rows[i]["split"] = label
            rows[i]["group_id"] = group_id
        assigned[label] += len(group)

    key_splits = {}
    for row in rows:
        for key in row["_group_keys"]:
            if key in key_splits and key_splits[key] != row["split"]:
                raise AssertionError(f"Grouping leakage for {key}")
            key_splits[key] = row["split"]
    rows.sort(key=lambda row: row["sample_id"])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in FIELDS} for row in rows)
    report = {
        "scope": "Synthetic paired restoration training manifest; synthetic split is not a physical-code test",
        "generator_root": str(root), "generator_run_manifest": str(run_manifest_path),
        "pairs": len(rows), "split_counts": dict(assigned), "group_count": len(groups),
        "excluded": dict(skipped), "group_keys": "transitive components of exact background source SHA-256 and QR payload/source SHA-256",
        "group_leakage": 0, "known_generated_payload_pairs": sum(int(row["payload_ground_truth_available"]) for row in rows),
        "limitations": [
            "Validation and test partitions are synthetic and cannot measure transfer to physical QR codes.",
            "Exact hashes do not identify near-duplicate generated backgrounds or equivalent physical specimens.",
            "clean_target is the paired undamaged rendered image; qr_original is an ideal symbol in a different coordinate frame.",
        ],
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} paired rows to {output}; report: {report_path}")


if __name__ == "__main__":
    main()
