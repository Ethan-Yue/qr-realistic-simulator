from __future__ import annotations

import csv, json, shutil
from pathlib import Path
import cv2

from qr_synth.utils import read_cv_image, write_cv_image

from .engine import DamageEngine


def _read_mask(path):
    if not path:
        return None
    p=Path(path)
    if not p.exists():
        return None
    return read_cv_image(p,cv2.IMREAD_GRAYSCALE)


def process_manifest(
    manifest_csv,
    output_dir,
    seed=42,
    severity="medium",
    variants_per_image=1,
    common_count=(1,2),
    carrier_count=(0,1),
    method_count=(0,1),
    damage_config=None,
):
    """批处理入口。

    manifest 列：image, carrier, method, qr_mask(可选), source_qr(可选)
    damage_config：可传 damage_plan_example.yaml / random_controls_example.yaml。
    输出：damaged / masks / clean / source_qr / metadata / layer_masks
    """
    manifest_csv=Path(manifest_csv); out=Path(output_dir)
    for d in ["damaged","masks","clean","source_qr","metadata","layer_masks"]:
        (out/d).mkdir(parents=True,exist_ok=True)
    engine=DamageEngine(seed=seed); records=[]
    with manifest_csv.open("r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))

    for row_idx,row in enumerate(rows):
        img_path=Path(row["image"]); image=read_cv_image(img_path,cv2.IMREAD_COLOR)
        qrm=_read_mask(row.get("qr_mask","")); carrier=row["carrier"].strip(); method=row["method"].strip()
        for v in range(int(variants_per_image)):
            if damage_config:
                res=engine.apply_from_config(image,carrier,method,damage_config,qr_mask=qrm)
            else:
                res=engine.apply(image,carrier,method,qrm,severity,common_count,carrier_count,method_count)
            stem=f"{img_path.stem}__dmg{v:02d}"
            write_cv_image(out/"damaged"/f"{stem}.png",res.image)
            write_cv_image(out/"masks"/f"{stem}.png",res.union_mask)
            shutil.copy2(img_path,out/"clean"/f"{stem}{img_path.suffix.lower()}")
            source_qr=row.get("source_qr","").strip()
            if source_qr and Path(source_qr).exists():
                shutil.copy2(source_qr,out/"source_qr"/f"{stem}{Path(source_qr).suffix.lower()}")
            for li,layer in enumerate(res.layers):
                d=out/"layer_masks"/layer.name; d.mkdir(parents=True,exist_ok=True)
                write_cv_image(d/f"{stem}__layer{li:02d}.png",layer.mask)
            meta={**res.metadata,"source_image":str(img_path),"qr_mask":row.get("qr_mask","") or None,"source_qr":source_qr or None,"output_stem":stem}
            with (out/"metadata"/f"{stem}.json").open("w",encoding="utf-8") as jf:
                json.dump(meta,jf,ensure_ascii=False,indent=2)
            records.append(meta)
    with (out/"metadata.jsonl").open("w",encoding="utf-8") as jf:
        for r in records:
            jf.write(json.dumps(r,ensure_ascii=False)+"\n")
    return records
