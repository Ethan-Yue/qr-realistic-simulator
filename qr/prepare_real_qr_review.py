"""Prepare duplicate-safe splits and manual-review sheets from audit_real_qr.py output."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_FIELDS = ["image_id", "source_image", "source_sha256", "original_split", "grouped_split"]
CURATION_FIELDS = [
    "image_id", "box_index", "source_image", "crop_path", "source_sha256",
    "visual_source_type", "physical_carrier", "marking_method", "visible_damage",
    "damage_type", "physical_specimen_id", "payload_ground_truth_available",
    "image_rights_verified", "reviewer", "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--train", type=float, default=.7)
    parser.add_argument("--val", type=float, default=.2)
    parser.add_argument("--overwrite-curation", action="store_true",
                        help="Replace an existing manual curation sheet after verifying the audit rows")
    args = parser.parse_args()
    if not (0 < args.train < 1 and 0 < args.val < 1 and args.train + args.val < 1):
        parser.error("Require 0 < train, val and train + val < 1")
    root = args.audit_dir.resolve()
    images = read_csv(root / "images.csv")
    boxes = read_csv(root / "manifest.csv")
    if len({row["image_id"] for row in images}) != len(images):
        raise ValueError("Duplicate image_id values in images.csv")

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in images:
        by_hash[row["source_sha256"]].append(row)
    duplicate_groups = [group for group in by_hash.values() if len(group) > 1]
    cross_split = [group for group in duplicate_groups if len({row["split"] for row in group}) > 1]
    cross_named = [group for group in cross_split if len({row["split"] for row in group if row["split"] in {"train", "val", "test"}}) > 1]

    # Assign each exact-hash group to one new split, preserving approximate
    # image counts and never using the source split as an assignment rule.
    rng = random.Random(args.seed)
    groups = list(by_hash.items())
    rng.shuffle(groups)
    targets = {"train": args.train * len(images), "val": args.val * len(images),
               "test": (1 - args.train - args.val) * len(images)}
    assigned = Counter()
    group_split = {}
    for sha, group in groups:
        label = max(("train", "val", "test"), key=lambda name: (targets[name] - assigned[name]) / targets[name])
        group_split[sha] = label
        assigned[label] += len(group)

    split_rows = [{
        "image_id": row["image_id"], "source_image": row["source_image"],
        "source_sha256": row["source_sha256"], "original_split": row["split"],
        "grouped_split": group_split[row["source_sha256"]],
    } for row in images]
    write_csv(root / "split_grouped.csv", SPLIT_FIELDS, split_rows)

    review_rows = [{
        "image_id": row["image_id"], "box_index": row["box_index"],
        "source_image": row["source_image"], "crop_path": row["crop_path"],
        "source_sha256": row["source_sha256"], "visual_source_type": "unknown",
        "physical_carrier": "unknown", "marking_method": "unknown",
        "visible_damage": "uncertain", "damage_type": "unknown",
        "physical_specimen_id": "unknown", "payload_ground_truth_available": "no",
        "image_rights_verified": "no", "reviewer": "", "notes": "",
    } for row in boxes]
    curation_path = root / "curation_template.csv"
    curation_action = "created"
    if curation_path.exists():
        existing = read_csv(curation_path)
        key = lambda row: (row["image_id"], row["box_index"], row["source_sha256"])
        if {key(row) for row in existing} != {key(row) for row in review_rows}:
            raise ValueError("Existing curation sheet refers to different audit rows; preserve it and resolve the mismatch before continuing")
        if args.overwrite_curation:
            write_csv(curation_path, CURATION_FIELDS, review_rows)
            curation_action = "overwritten"
        else:
            curation_action = "preserved"
    else:
        write_csv(curation_path, CURATION_FIELDS, review_rows)

    per_image_boxes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in boxes:
        per_image_boxes[row["image_id"]].append(row)
    single_box = [records[0] for records in per_image_boxes.values() if len(records) == 1]
    strata = []
    for lo, hi, label in ((0, .01, "<1%"), (.01, .04, "1–4%"), (.04, .12, "4–12%"), (.12, 1.01, "≥12%")):
        sample = [row for row in single_box if lo <= float(row["box_area_ratio"]) < hi]
        strata.append({
            "box_area_stratum": label, "single_box_images": len(sample),
            "opencv_full_returned": sum(row["opencv_full_returned"] == "1" for row in sample),
            "opencv_crop_returned": sum(row["opencv_crop_returned"] == "1" for row in sample),
            "zxing_full_returned": sum(row["zxing_full_returned"] == "1" for row in sample),
            "zxing_crop_returned": sum(row["zxing_crop_returned"] == "1" for row in sample),
        })
    any_crop = {}
    for decoder in ("opencv", "zxing"):
        full = {row["image_id"] for row in images if row[f"{decoder}_full_returned"] == "1"}
        crop = {name for name, records in per_image_boxes.items()
                if any(row[f"{decoder}_crop_returned"] == "1" for row in records)}
        any_crop[decoder] = {"full_images": len(full), "any_crop_images": len(crop),
                             "crop_only": len(crop - full), "full_only": len(full - crop),
                             "both": len(full & crop)}

    report = {
        "scope": "Engineering audit of QR localization and decoder returns; no photo-realism or payload-correctness inference",
        "image_count": len(images), "box_count": len(boxes),
        "exact_duplicate_groups": len(duplicate_groups),
        "extra_exact_duplicate_images": sum(len(group) - 1 for group in duplicate_groups),
        "original_cross_split_duplicate_groups_including_unassigned": len(cross_split),
        "original_cross_train_val_test_duplicate_groups": len(cross_named),
        "cross_split_examples": [
            [{"image_id": row["image_id"], "original_split": row["split"]} for row in group]
            for group in cross_split
        ],
        "new_split_seed": args.seed, "new_split_counts": dict(assigned),
        "new_split_exact_hash_leakage": 0,
        "new_split_limitation": "Exact SHA-256 grouping does not remove resized, cropped, or near-duplicate images, nor group by original physical specimen.",
        "decoder_returns_by_image": any_crop,
        "single_box_area_strata": strata,
        "curation_guidance": "Manually identify photo/graphic/screenshot/composite/unknown and physical damage. Do not treat unknown as real photographs.",
    }
    (root / "review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {root / 'split_grouped.csv'} and {root / 'review_report.json'}; "
          f"{curation_action} {curation_path}")


if __name__ == "__main__":
    main()
