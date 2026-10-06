from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import cv2
import numpy as np

from qr_synth.utils import make_qr_mask, make_payload, load_qr_image, list_qr_inputs, slugify_combo, derive_seed, write_cv_image
from qr_synth.compositor import compose_one
from qr_synth.capture import capture_pair
from qr_synth.layout import load_layout_boxes, sample_layout_box
from qr_damage_sim import DamageEngine
from qr_damage_sim.core import overlap_ratio


def parse_pair(text: str):
    parts = [x.strip() for x in str(text).split(',') if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected form a,b')
    return (int(parts[0]), int(parts[1]))


def parse_ratio_pair(text: str):
    parts = [x.strip() for x in str(text).split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected QR damage fraction range min,max')
    try:
        low, high = (float(x) for x in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('QR damage fractions must be numeric') from exc
    if not (0 <= low <= high <= 1):
        raise argparse.ArgumentTypeError('QR damage fractions must satisfy 0 <= min <= max <= 1')
    return low, high


def parse_args():
    p=argparse.ArgumentParser(description='QR 载体 × 成码方式 + 污损仿真数据集生成器')
    # clean generation args
    p.add_argument('--config', default='configs/combinations.yaml')
    p.add_argument('--output', default='outputs_damaged')
    p.add_argument('--background-root', default='assets/backgrounds')
    p.add_argument('--require-background-library', action='store_true',
                   help='仅允许从所设 background-root 的载体目录取图；缺图或读图失败时直接报错')
    p.add_argument('--qr-input', default='assets/qr_input')
    p.add_argument('--per-combo', type=int, default=100)
    p.add_argument('--size', type=int, default=768)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--combo-ids', default='all')
    p.add_argument('--min-contrast', type=float, default=3.0)
    p.add_argument('--white-module-mode', default='random', choices=['random','transparent','opaque'])
    p.add_argument('--render-profile', default='material', choices=['ideal', 'material'],
                   help='ideal=原始干净印刷层；material=轻微材质与印刷层耦合')
    p.add_argument('--capture-mode', default='mild', choices=['none', 'mild', 'handheld', 'challenging'],
                   help='none=平面数字图；mild/handheld=普通采集；challenging=可出现运动模糊、强反光与过曝')
    p.add_argument('--capture-motion-px', type=int, default=None,
                   help='challenging 模式下指定运动核长度；0 关闭，正数为 3–31 的奇数')
    p.add_argument('--capture-exposure-ev', type=float, default=None,
                   help='challenging 模式下指定曝光增益 EV，允许 -2 至 3')
    p.add_argument('--capture-glare-strength', type=float, default=None,
                   help='challenging 模式下指定线性强反光强度；0 关闭，范围 0–2.5')
    p.add_argument('--qr-ecc', default='M', choices=['L', 'M', 'Q', 'H', 'random'])
    p.add_argument('--payload-mode', default='variable', choices=['id', 'variable'])
    p.add_argument('--payload-alphabet', default='alphanumeric', choices=['alphanumeric', 'byte'])
    p.add_argument('--payload-length-range', default='36,96', help='variable 模式的目标字符数下限,上限')
    p.add_argument('--qr-source-size', type=int, default=420, help='生成或规范化 QR 原图的最小/目标像素边长')
    p.add_argument('--layout-profile', default='closeup', choices=['closeup', 'external_boxes'],
                   help='closeup=原近景布局；external_boxes=按定位框面积和中心位置采样，仅供场景尺度压力测试')
    p.add_argument('--layout-manifest', default=None,
                   help='external_boxes 模式的 audit_real_qr.py 定位框 manifest.csv')
    p.add_argument('--allow-undecodable-qr-input', action='store_true',
                   help='允许外部 QR 原图经 OpenCV 规范化后仍无法解码；此时不能计算载荷正确率')
    # damage args
    p.add_argument('--damage-mode', default='random', choices=['random','config','none'],
                   help='random=按通用/载体/成码规则随机叠加；config=按YAML精确叠加；none=不叠加，仅导出clean')
    p.add_argument('--damage-config', default='damage_configs/damage_plan_example.yaml',
                   help='damage-mode=config 时使用的YAML配置')
    p.add_argument('--damage-severity', default='medium', choices=['mild','medium','severe'])
    p.add_argument('--target-qr-damage-ratio', type=parse_ratio_pair, default=None,
                   help='目标 QR 黑色标记像素污损率下限,上限，例如 0.05,0.10；按输出几何掩码筛选，非模块个数比例')
    p.add_argument('--max-damage-attempts', type=int, default=32,
                   help='设置目标污损率时每个样本的最多候选次数；若无候选落入区间则报错')
    p.add_argument('--damage-spatial-mode', default='auto', choices=['auto', 'full_canvas', 'qr_context'],
                   help='auto=近景按整图、小码区按 QR 邻域生成污损；可显式指定供消融')
    p.add_argument('--damage-common', type=parse_pair, default=(1,2), help='随机模式下通用污损数量范围，例如 1,2')
    p.add_argument('--damage-carrier', type=parse_pair, default=(0,1), help='随机模式下载体专属污损数量范围，例如 0,1')
    p.add_argument('--damage-method', type=parse_pair, default=(0,1), help='随机模式下成码方式污损数量范围，例如 0,1')
    p.add_argument('--damage-rules-dir', default=None, help='可选：自定义污损规则目录，默认使用内置 damage_configs')
    p.add_argument('--allow-incompatible', action='store_true', help='允许使用与当前组合不兼容的污损名（一般不建议）')
    return p.parse_args()


def main():
    args=parse_args()
    if args.per_combo < 1:
        raise ValueError('--per-combo must be positive')
    capture_overrides = {
        key: value for key, value in {
            'motion_length_px': args.capture_motion_px,
            'exposure_ev': args.capture_exposure_ev,
            'glare_strength': args.capture_glare_strength,
        }.items() if value is not None
    }
    if capture_overrides and args.capture_mode != 'challenging':
        raise ValueError('Capture effect overrides require --capture-mode challenging')
    if args.capture_motion_px is not None and args.capture_motion_px != 0 and not (3 <= args.capture_motion_px <= 31 and args.capture_motion_px % 2 == 1):
        raise ValueError('--capture-motion-px must be 0 or an odd number from 3 to 31')
    if args.capture_exposure_ev is not None and not (-2 <= args.capture_exposure_ev <= 3):
        raise ValueError('--capture-exposure-ev must be between -2 and 3')
    if args.capture_glare_strength is not None and not (0 <= args.capture_glare_strength <= 2.5):
        raise ValueError('--capture-glare-strength must be between 0 and 2.5')
    if args.max_damage_attempts < 1:
        raise ValueError('--max-damage-attempts must be positive')
    if args.target_qr_damage_ratio is not None and args.damage_mode == 'none' and args.target_qr_damage_ratio != (0.0, 0.0):
        raise ValueError('Nonzero --target-qr-damage-ratio requires damage-mode random or config')
    with open(args.config,'r',encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
    combos=cfg['combinations']
    if args.combo_ids!='all':
        ids={int(x) for x in args.combo_ids.split(',')}
        combos=[c for c in combos if int(c['id']) in ids]
    if not combos:
        raise ValueError('No configured combinations match --combo-ids')
    if args.layout_profile == 'external_boxes':
        if not args.layout_manifest:
            raise ValueError('--layout-manifest is required for external_boxes')
        layout_boxes, layout_manifest_sha256 = load_layout_boxes(args.layout_manifest)
        if args.damage_mode != 'none':
            print('[WARN] external_boxes with damage is an experimental QR-context rendering mode. '
                  'Inspect per-sample qr_damage_ratio and appearance; it has not been calibrated on physical damage.', flush=True)
    else:
        layout_boxes, layout_manifest_sha256 = [], None
    damage_spatial_mode = (
        'qr_context' if args.layout_profile == 'external_boxes' else 'full_canvas'
    ) if args.damage_spatial_mode == 'auto' else args.damage_spatial_mode

    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_manifest_path=output/'run_manifest.json'
    run_manifest={
        'schema_version': 1,
        'status': 'running',
        'entry_point': 'synthesize_with_damage.py',
        'arguments': vars(args),
        'combo_directories': [slugify_combo(c) for c in combos],
        'samples_per_combo': args.per_combo,
        'layout_manifest_sha256': layout_manifest_sha256,
        'damage_spatial_mode_resolved': damage_spatial_mode,
    }
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    bgroot=Path(args.background_root)
    if args.require_background_library:
        allowed_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        for combo in combos:
            folder = bgroot / combo['carrier']
            if not folder.is_dir() or not any(p.suffix.lower() in allowed_suffixes for p in folder.rglob('*')):
                raise ValueError(f'No background images for {combo["carrier"]} in {folder}')
    qrroot=Path(args.qr_input)
    qrfiles=list_qr_inputs(qrroot) if qrroot.exists() else []
    if args.qr_source_size < 64:
        raise ValueError('--qr-source-size must be >=64')
    for qpath in qrfiles:
        load_qr_image(qpath, pixels=args.qr_source_size, return_info=True,
                      require_decodable=not args.allow_undecodable_qr_input)
    length_range=tuple(int(x) for x in args.payload_length_range.split(','))
    if len(length_range) != 2 or length_range[0] < 1 or length_range[1] < length_range[0]:
        raise ValueError('--payload-length-range must be two ascending positive integers')
    damage_engine=DamageEngine(seed=args.seed, config_dir=args.damage_rules_dir)
    for combo in combos:
        damage_engine.available_for(combo['carrier_cn'], combo['method_cn'])

    for combo in combos:
        combo_dir=output/slugify_combo(combo)
        clean_img_dir=combo_dir/'clean_images'
        damaged_img_dir=combo_dir/'damaged_images'
        qr_mask_dir=combo_dir/'qr_masks'
        damage_mask_dir=combo_dir/'damage_masks'
        visible_change_mask_dir=combo_dir/'visible_change_masks'
        qr_dir=combo_dir/'qr_original'
        meta_dir=combo_dir/'metadata'
        for d in [clean_img_dir, damaged_img_dir, qr_mask_dir, damage_mask_dir,
                  visible_change_mask_dir, qr_dir, meta_dir]:
            d.mkdir(parents=True,exist_ok=True)

        for i in range(args.per_combo):
            combo_id=int(combo['id'])
            sample_seed=derive_seed(args.seed, combo_id, i, stage=0)
            damage_seed=derive_seed(args.seed, combo_id, i, stage=1)
            capture_seed=derive_seed(args.seed, combo_id, i, stage=2)
            qr_seed=derive_seed(args.seed, combo_id, i, stage=3)
            layout_seed=derive_seed(args.seed, combo_id, i, stage=4) if layout_boxes else None
            rng=np.random.default_rng(sample_seed)
            qr_rng=np.random.default_rng(qr_seed)
            layout_box = sample_layout_box(
                layout_boxes, np.random.default_rng(layout_seed)
            ) if layout_boxes else None
            layout_kwargs = {
                'qr_scale_range': (layout_box.area_ratio ** .5, layout_box.area_ratio ** .5),
                'position_jitter': 0.0,
                'center_fraction': layout_box.center_xy_fraction,
            } if layout_box else {}

            if qrfiles:
                qpath=qrfiles[int(rng.integers(0,len(qrfiles)))]
                qr_original, qr_mask, qr_input_info=load_qr_image(
                    qpath,pixels=args.qr_source_size,return_info=True,
                    require_decodable=not args.allow_undecodable_qr_input)
                qr_source=str(qpath)
                qr_payload=qr_input_info['decoded_payload']
                qr_structure=None
            else:
                payload=make_payload(combo_id, i, sample_seed, qr_rng,
                                     mode=args.payload_mode, length_range=length_range,
                                     alphabet_mode=args.payload_alphabet)
                ecc=args.qr_ecc if args.qr_ecc != 'random' else str(qr_rng.choice(['L', 'M', 'Q', 'H']))
                qr_original, qr_mask, qr_structure=make_qr_mask(
                    payload,pixels=args.qr_source_size,border_modules=4,error_correction=ecc,return_info=True)
                qr_source=payload
                qr_payload=payload
                qr_input_info=None

            clean_final, qrm, quad, contrast_info=compose_one(
                qr_mask=qr_mask,
                carrier=combo['carrier'],
                method=combo['method'],
                rng=rng,
                size=args.size,
                background_root=bgroot if bgroot.exists() else None,
                min_contrast=args.min_contrast,
                white_module_mode=args.white_module_mode,
                render_profile=args.render_profile,
                module_count_with_border=(qr_structure or {}).get('module_count_with_border'),
                **layout_kwargs,
            )
            if args.require_background_library and contrast_info['background_source']['source_kind'] != 'background_library_image':
                raise RuntimeError(f'Background library image unavailable for combo {combo_id}, sample {i}')

            max_attempts = args.max_damage_attempts if args.target_qr_damage_ratio is not None else 1
            for attempt in range(max_attempts):
                candidate_seed = damage_seed if attempt == 0 else derive_seed(args.seed, combo_id, i, stage=100 + attempt)
                damage_engine.reseed(candidate_seed)
                if args.damage_mode == 'none':
                    candidate_damaged = clean_final.copy()
                    candidate_mask = np.zeros(qrm.shape, np.uint8)
                    candidate_meta = {
                        'carrier': combo['carrier_cn'], 'method': combo['method_cn'],
                        'mode': 'none', 'damage_types': [], 'damage_categories': [],
                        'damage_layer_count': 0, 'damage_area_ratio': 0.0,
                        'qr_damage_ratio': 0.0, 'applied_plan': [], 'layers': [],
                        'spatial_mode': damage_spatial_mode,
                    }
                elif args.damage_mode == 'config':
                    damage_result = damage_engine.apply_from_config(
                        clean_final, combo['carrier_cn'], combo['method_cn'], args.damage_config,
                        qr_mask=qrm, allow_incompatible=args.allow_incompatible,
                        spatial_mode=damage_spatial_mode,
                    )
                    candidate_damaged, candidate_mask, candidate_meta = (
                        damage_result.image, damage_result.union_mask, damage_result.metadata
                    )
                else:
                    damage_result = damage_engine.apply(
                        clean_final, combo['carrier_cn'], combo['method_cn'], qr_mask=qrm,
                        severity=args.damage_severity, common_count=args.damage_common,
                        carrier_count=args.damage_carrier, method_count=args.damage_method,
                        allow_incompatible=args.allow_incompatible, spatial_mode=damage_spatial_mode,
                    )
                    candidate_damaged, candidate_mask, candidate_meta = (
                        damage_result.image, damage_result.union_mask, damage_result.metadata
                    )
                candidate_capture = capture_pair(
                    clean_final, candidate_damaged, qrm, candidate_mask, quad,
                    combo['carrier'], np.random.default_rng(capture_seed), mode=args.capture_mode,
                    effect_overrides=capture_overrides,
                )
                final_ratio = overlap_ratio(candidate_capture[3], candidate_capture[2])
                if args.target_qr_damage_ratio is None or args.target_qr_damage_ratio[0] <= final_ratio <= args.target_qr_damage_ratio[1]:
                    clean_final, damaged_final, qrm, damage_union_mask, quad, capture_info = candidate_capture
                    damage_meta = candidate_meta
                    break
            else:
                raise RuntimeError(
                    f'No damage candidate within QR mark area fraction {args.target_qr_damage_ratio} '
                    f'for combo {combo_id}, sample {i} after {max_attempts} attempts; '
                    f'last achieved {final_ratio:.6f}. Widen the interval or adjust damage operators.'
                )
            visible_change_mask = (
                np.max(np.abs(clean_final.astype(np.int16) - damaged_final.astype(np.int16)), axis=2) > 5
            ).astype(np.uint8) * 255

            stem=f'{i:06d}'
            write_cv_image(clean_img_dir/f'{stem}.png', clean_final)
            write_cv_image(damaged_img_dir/f'{stem}.png', damaged_final)
            write_cv_image(qr_mask_dir/f'{stem}.png', qrm)
            write_cv_image(damage_mask_dir/f'{stem}.png', damage_union_mask)
            write_cv_image(visible_change_mask_dir/f'{stem}.png', visible_change_mask)
            write_cv_image(qr_dir/f'{stem}.png', qr_original)

            meta={
                'combo_id': int(combo['id']),
                'carrier': combo['carrier'],
                'carrier_cn': combo['carrier_cn'],
                'method': combo['method'],
                'method_cn': combo['method_cn'],
                'sample_index': i,
                'seed': sample_seed,
                'stage_seeds': {'clean': sample_seed, 'damage': damage_seed, 'capture': capture_seed,
                                'qr': qr_seed, 'layout': layout_seed},
                'seed_scheme': 'SeedSequence(base_seed, combo_id, sample_index, stage)',
                'qr_structure': qr_structure,
                'payload_mode': args.payload_mode if qr_structure is not None else 'user_image',
                'payload_alphabet': args.payload_alphabet if qr_structure is not None else None,
                'qr_source': qr_source,
                'qr_payload': qr_payload,
                'qr_input_info': qr_input_info,
                'quad_xy': np.asarray(quad).round(2).tolist(),
                'layout_profile': args.layout_profile,
                'layout_reference': {
                    'image_id': layout_box.image_id,
                    'box_index': layout_box.box_index,
                    'requested_box_area_ratio': layout_box.area_ratio,
                    'requested_center_xy_fraction': list(layout_box.center_xy_fraction),
                    'reference_manifest_sha256': layout_manifest_sha256,
                    'rendered_box_area_ratio_after_capture': float(abs(cv2.contourArea(np.asarray(quad, dtype=np.float32))) / (args.size * args.size)),
                    'area_equivalent_square': True,
                } if layout_box else None,
                'mask_definition': 'qr_masks: QR成码模块的几何范围；damage_masks: 污损算子的几何范围（拍摄几何变换后、光学扩散前）；visible_change_masks: 配对拍摄图像任一通道差值>5的像素范围（含模糊/JPEG扩散）。255=范围内，0=范围外',
                'visible_change_threshold_u8': 5,
                'visible_change_area_ratio': float((visible_change_mask > 0).mean()),
                'image_size': [args.size,args.size],
                'clean_baseline': args.render_profile == 'ideal' and args.capture_mode == 'none',
                'paired_clean_reference': True,
                'white_module_mode': args.white_module_mode,
                'contrast_control': contrast_info,
                'render_profile': args.render_profile,
                'capture': capture_info,
                'damage_enabled': args.damage_mode != 'none',
                'damage_mode_requested': args.damage_mode,
                'damage_spatial_mode': damage_spatial_mode,
                'damage_severity_requested': args.damage_severity if args.damage_mode == 'random' else None,
                'target_qr_damage_ratio': list(args.target_qr_damage_ratio) if args.target_qr_damage_ratio is not None else None,
                'damage_candidate_attempts': attempt + 1,
                'damage_candidate_seed': candidate_seed,
                'qr_damage_ratio_after_capture': float(final_ratio),
                'damage_area_ratio_after_capture': float((damage_union_mask > 0).mean()),
                'damage_random_counts': {
                    'common': list(args.damage_common),
                    'carrier': list(args.damage_carrier),
                    'method': list(args.damage_method),
                } if args.damage_mode == 'random' else None,
                'damage_config_path': args.damage_config if args.damage_mode == 'config' else None,
                'damage': damage_meta,
                'outputs': {
                    'clean_image': f'clean_images/{stem}.png',
                    'damaged_image': f'damaged_images/{stem}.png',
                    'qr_mask': f'qr_masks/{stem}.png',
                    'damage_mask': f'damage_masks/{stem}.png',
                    'visible_change_mask': f'visible_change_masks/{stem}.png',
                    'qr_original': f'qr_original/{stem}.png',
                }
            }
            with open(meta_dir/f'{stem}.json','w',encoding='utf-8') as f:
                json.dump(meta,f,ensure_ascii=False,indent=2)

        print(f"[OK] {combo['id']:02d} {combo['carrier_cn']} + {combo['method_cn']} -> {combo_dir}")

    run_manifest['status']='complete'
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__=='__main__':
    main()
