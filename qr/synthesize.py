from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import cv2
import numpy as np

from qr_synth.utils import make_qr_mask, make_payload, load_qr_image, list_qr_inputs, slugify_combo, derive_seed, write_cv_image
from qr_synth.compositor import compose_one
from qr_synth.layout import load_layout_boxes, sample_layout_box


def parse_args():
    p=argparse.ArgumentParser(description="QR 载体 × 成码方式 仿真数据集生成器")
    p.add_argument("--config", default="configs/combinations.yaml")
    p.add_argument("--output", default="outputs")
    p.add_argument("--background-root", default="assets/backgrounds",
                   help="可选：真实空载体背景图库根目录。目录名与 carrier 一致。")
    p.add_argument("--qr-input", default="assets/qr_input",
                   help="可选：自己的原始二维码目录；为空时自动生成二维码。")
    p.add_argument("--per-combo", type=int, default=100, help="每个组合生成多少张")
    p.add_argument("--size", type=int, default=768, help="输出方形分辨率")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--combo-ids", default="all",
                   help="例如 1,2,17,18；默认 all")
    p.add_argument("--min-contrast", type=float, default=3.0,
                   help="QR模块与局部背景的最小WCAG对比度，默认3.0")
    p.add_argument("--white-module-mode", default="random",
                   choices=["random", "transparent", "opaque"],
                   help="白模块/静区渲染模式：random(默认，部分场景随机透明/不透明)、transparent(透明)、opaque(浅色底板)")
    p.add_argument("--render-profile", default="ideal", choices=["ideal", "material"],
                   help="ideal=原始干净印刷层；material=轻微材质与印刷层耦合")
    p.add_argument("--qr-ecc", default="M", choices=["L", "M", "Q", "H", "random"])
    p.add_argument("--payload-mode", default="id", choices=["id", "variable"])
    p.add_argument("--payload-alphabet", default="alphanumeric", choices=["alphanumeric", "byte"])
    p.add_argument("--payload-length-range", default="36,96", help="variable 模式的目标字符数下限,上限")
    p.add_argument("--qr-source-size", type=int, default=420, help="生成或规范化 QR 原图的最小/目标像素边长")
    p.add_argument("--layout-profile", default="closeup", choices=["closeup", "external_boxes"],
                   help="closeup=原近景布局；external_boxes=按定位框面积和中心位置采样，仅供场景尺度压力测试")
    p.add_argument("--layout-manifest", default=None,
                   help="external_boxes 模式的 audit_real_qr.py 定位框 manifest.csv")
    p.add_argument("--allow-undecodable-qr-input", action="store_true",
                   help="允许外部 QR 原图经 OpenCV 规范化后仍无法解码；此时不能计算载荷正确率")
    return p.parse_args()


def main():
    args=parse_args()
    if args.per_combo < 1:
        raise ValueError('--per-combo must be positive')
    with open(args.config,"r",encoding="utf-8") as f:
        cfg=yaml.safe_load(f)
    combos=cfg["combinations"]
    if args.combo_ids!="all":
        ids={int(x) for x in args.combo_ids.split(",")}
        combos=[c for c in combos if int(c["id"]) in ids]
    if not combos:
        raise ValueError('No configured combinations match --combo-ids')
    if args.layout_profile == "external_boxes":
        if not args.layout_manifest:
            raise ValueError("--layout-manifest is required for external_boxes")
        layout_boxes, layout_manifest_sha256 = load_layout_boxes(args.layout_manifest)
    else:
        layout_boxes, layout_manifest_sha256 = [], None

    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_manifest_path=output/'run_manifest.json'
    run_manifest={
        'schema_version': 1,
        'status': 'running',
        'entry_point': 'synthesize.py',
        'arguments': vars(args),
        'combo_directories': [slugify_combo(c) for c in combos],
        'samples_per_combo': args.per_combo,
        'layout_manifest_sha256': layout_manifest_sha256,
    }
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    bgroot=Path(args.background_root)
    qrroot=Path(args.qr_input)
    qrfiles=list_qr_inputs(qrroot) if qrroot.exists() else []
    if args.qr_source_size < 64:
        raise ValueError("--qr-source-size must be >=64")
    for qpath in qrfiles:
        load_qr_image(qpath, pixels=args.qr_source_size, return_info=True,
                      require_decodable=not args.allow_undecodable_qr_input)
    length_range=tuple(int(x) for x in args.payload_length_range.split(','))
    if len(length_range) != 2 or length_range[0] < 1 or length_range[1] < length_range[0]:
        raise ValueError("--payload-length-range must be two ascending positive integers")

    for combo in combos:
        combo_dir=output/slugify_combo(combo)
        img_dir=combo_dir/"images"
        mask_dir=combo_dir/"masks"
        qr_dir=combo_dir/"qr_original"
        meta_dir=combo_dir/"metadata"
        for d in [img_dir,mask_dir,qr_dir,meta_dir]:
            d.mkdir(parents=True,exist_ok=True)

        for i in range(args.per_combo):
            sample_seed=derive_seed(args.seed, int(combo["id"]), i, stage=0)
            rng=np.random.default_rng(sample_seed)
            qr_seed=derive_seed(args.seed, int(combo["id"]), i, stage=3)
            layout_seed=derive_seed(args.seed, int(combo["id"]), i, stage=4) if layout_boxes else None
            qr_rng=np.random.default_rng(qr_seed)
            layout_box = sample_layout_box(
                layout_boxes, np.random.default_rng(layout_seed)
            ) if layout_boxes else None
            layout_kwargs = {
                "qr_scale_range": (layout_box.area_ratio ** .5, layout_box.area_ratio ** .5),
                "position_jitter": 0.0,
                "center_fraction": layout_box.center_xy_fraction,
            } if layout_box else {}

            if qrfiles:
                qpath=qrfiles[int(rng.integers(0,len(qrfiles)))]
                qr_original, qr_mask, qr_input_info=load_qr_image(
                    qpath,pixels=args.qr_source_size,return_info=True,
                    require_decodable=not args.allow_undecodable_qr_input)
                qr_source=str(qpath)
                qr_payload=qr_input_info["decoded_payload"]
                qr_structure=None
            else:
                payload=make_payload(int(combo['id']), i, sample_seed, qr_rng,
                                     mode=args.payload_mode, length_range=length_range,
                                     alphabet_mode=args.payload_alphabet)
                ecc=args.qr_ecc if args.qr_ecc != "random" else str(qr_rng.choice(["L", "M", "Q", "H"]))
                qr_original, qr_mask, qr_structure=make_qr_mask(
                    payload,pixels=args.qr_source_size,border_modules=4,error_correction=ecc,return_info=True)
                qr_source=payload
                qr_payload=payload
                qr_input_info=None

            final, mask, quad, contrast_info=compose_one(
                qr_mask=qr_mask,
                carrier=combo["carrier"],
                method=combo["method"],
                rng=rng,
                size=args.size,
                background_root=bgroot if bgroot.exists() else None,
                min_contrast=args.min_contrast,
                white_module_mode=args.white_module_mode,
                render_profile=args.render_profile,
                module_count_with_border=(qr_structure or {}).get("module_count_with_border"),
                **layout_kwargs,
            )

            stem=f"{i:06d}"
            write_cv_image(img_dir/f"{stem}.png",final)
            write_cv_image(mask_dir/f"{stem}.png",mask)
            write_cv_image(qr_dir/f"{stem}.png",qr_original)

            meta={
                "combo_id": int(combo["id"]),
                "carrier": combo["carrier"],
                "carrier_cn": combo["carrier_cn"],
                "method": combo["method"],
                "method_cn": combo["method_cn"],
                "sample_index": i,
                "seed": sample_seed,
                "seed_scheme": "SeedSequence(base_seed, combo_id, sample_index, stage=0)",
                "qr_seed": qr_seed,
                "layout_seed": layout_seed,
                "qr_structure": qr_structure,
                "payload_mode": args.payload_mode if qr_structure is not None else "user_image",
                "payload_alphabet": args.payload_alphabet if qr_structure is not None else None,
                "qr_source": qr_source,
                "qr_payload": qr_payload,
                "qr_input_info": qr_input_info,
                "quad_xy": np.asarray(quad).round(2).tolist(),
                "layout_profile": args.layout_profile,
                "layout_reference": {
                    "image_id": layout_box.image_id,
                    "box_index": layout_box.box_index,
                    "requested_box_area_ratio": layout_box.area_ratio,
                    "requested_center_xy_fraction": list(layout_box.center_xy_fraction),
                    "reference_manifest_sha256": layout_manifest_sha256,
                    "rendered_box_area_ratio": float(abs(cv2.contourArea(np.asarray(quad, dtype=np.float32))) / (args.size * args.size)),
                    "area_equivalent_square": True,
                } if layout_box else None,
                "mask_definition": "255=QR成码模块的几何范围（可为浅色）, 0=其他",
                "clean_baseline": args.render_profile == "ideal",
                "augmentations_applied": [],
                "randomness": "仅材质背景随机抽样/随机裁剪/程序化纹理 + 极小QR尺度与位置变化 + 白模块透明/不透明模式控制",
                "image_size": [args.size,args.size],
                "white_module_mode": args.white_module_mode,
                "render_profile": args.render_profile,
                "contrast_control": contrast_info,
            }
            with open(meta_dir/f"{stem}.json","w",encoding="utf-8") as f:
                json.dump(meta,f,ensure_ascii=False,indent=2)

        print(f"[OK] {combo['id']:02d} {combo['carrier_cn']} + {combo['method_cn']} -> {combo_dir}")

    run_manifest['status']='complete'
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__=="__main__":
    main()
