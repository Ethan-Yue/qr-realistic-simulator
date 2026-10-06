"""Fetch a small, provenance-recorded library of sourced material images.

Poly Haven diffuse maps follow its photo-based texture standard under CC0. The
paper image is an author-supplied CC0 texture hosted by Wikimedia Commons. The files are
kept in a separate background root so the existing AI-texture library stays
available for controlled comparisons.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from urllib.parse import quote
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "assets" / "backgrounds_real"
USER_AGENT = "QRMaterialResearch/0.1 (academic QR simulation; provenance recorded)"
POLY_HAVEN = {
    "metal": "metal_plate_02",
    "wood": "ash_veneer",
    "textile": "cotton_jersey",
    "ceramic": "anti_skid_tiles",
}
PAPER = {
    "carrier": "paper",
    "filename": "wikimedia_old_paper_texture.jpg",
    "url": "https://upload.wikimedia.org/wikipedia/commons/8/83/" + quote("Текстура_бумаги.jpg"),
    "source_page": "https://commons.wikimedia.org/wiki/File:" + quote("Текстура_бумаги.jpg"),
    "creator": "Iroi su",
    "license": "CC0 1.0",
    "origin": "Old-paper texture described as creator's own work on the source page",
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def save_image(carrier: str, filename: str, data: bytes) -> dict:
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or min(decoded.shape[:2]) < 512:
        raise ValueError(f"Invalid or too-small material image: {carrier}/{filename}")
    path = OUTPUT / carrier / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "relative_path": path.relative_to(OUTPUT).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
        "bytes": len(data),
    }


def main() -> None:
    records = []
    asset_index = json.loads(fetch("https://api.polyhaven.com/assets?t=textures"))
    for carrier, slug in POLY_HAVEN.items():
        metadata = asset_index[slug]
        files = json.loads(fetch(f"https://api.polyhaven.com/files/{slug}"))
        diffuse = files["Diffuse"]["1k"]["jpg"]
        data = fetch(diffuse["url"])
        actual_md5 = hashlib.md5(data).hexdigest()
        if actual_md5 != diffuse["md5"]:
            raise ValueError(f"Poly Haven MD5 mismatch: {slug}")
        record = save_image(carrier, f"polyhaven_{slug}_diff_1k.jpg", data)
        record.update({
            "carrier": carrier,
            "source": "Poly Haven",
            "asset_id": slug,
            "asset_title": metadata.get("name"),
            "source_page": f"https://polyhaven.com/a/{slug}",
            "download_url": diffuse["url"],
            "creator": metadata.get("authors", {}),
            "license": "CC0 1.0",
            "capture_provenance": "Poly Haven photo-based texture standard; individual camera capture not independently audited",
            "source_md5": actual_md5,
        })
        records.append(record)
        print(f"[OK] {carrier}: {slug} {record['width']}x{record['height']}")

    paper_data = fetch(PAPER["url"])
    paper = save_image(PAPER["carrier"], PAPER["filename"], paper_data)
    paper.update(PAPER)
    paper["source"] = "Wikimedia Commons"
    records.append(paper)
    print(f"[OK] paper: {paper['width']}x{paper['height']}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "provenance.json").write_text(json.dumps({
        "schema_version": 1,
        "purpose": "Small traceable material-image subset, not a full 15-carrier library or physical QR-photo collection",
        "license_review_date": "2026-10-07",
        "poly_haven_license": "https://polyhaven.com/license",
        "poly_haven_texture_standard": "https://docs.polyhaven.com/en/technical-standards/textures",
        "assets": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
