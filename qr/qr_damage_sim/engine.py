from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Any
import cv2
import numpy as np

from .core import DamageLayer, DamageResult, ensure_bgr_u8, ensure_mask, mask_ratio, overlap_ratio
from .common_damage import COMMON_REGISTRY
from .special_damage import CARRIER_REGISTRY, METHOD_REGISTRY
from .config import load_rules, load_yaml
from .params import merged_params


class DamageEngine:
    """可参数化污损叠加引擎。

    输入：现有工程已生成好的高清“载体 + QR”干净图。
    输出：污损图、总 mask、逐层 mask、完整参数元数据。

    两种主要工作方式：
    1) 随机模式：用 common_count/carrier_count/method_count 随机抽污损；
    2) 精确计划模式：传 damage_plan，逐项决定某种污损有没有、出现几次、具体参数。
    """

    def __init__(self, seed: Optional[int] = None, config_dir=None):
        self.rng = np.random.default_rng(seed)
        self.rules = load_rules(config_dir)

    def reseed(self, seed: int) -> None:
        """Reset stochastic damage draws for one independently reproducible sample."""
        self.rng = np.random.default_rng(int(seed))

    def available_for(self, carrier: str, method: str):
        cr = self.rules["carrier"].get(carrier)
        mr = self.rules["method"].get(method)
        if cr is None:
            raise KeyError(f"Unknown carrier: {carrier}")
        if mr is None:
            raise KeyError(f"Unknown method: {method}")
        return {
            "common": [x["name"] for x in self.rules["common"]["damages"]],
            "carrier": [x["name"] for x in cr["damages"]],
            "method": [x["name"] for x in mr["damages"]],
        }

    def _registry(self, category: str):
        if category == "common":
            return COMMON_REGISTRY
        if category == "carrier":
            return CARRIER_REGISTRY
        if category == "method":
            return METHOD_REGISTRY
        raise KeyError(f"Unknown category: {category}")

    def _allowed_specs(self, carrier: str, method: str):
        cr = self.rules["carrier"].get(carrier)
        mr = self.rules["method"].get(method)
        if cr is None:
            raise KeyError(f"Unknown carrier: {carrier}")
        if mr is None:
            raise KeyError(f"Unknown method: {method}")
        return {
            "common": deepcopy(self.rules["common"]["damages"]),
            "carrier": deepcopy(cr["damages"]),
            "method": deepcopy(mr["damages"]),
        }

    def _resolve_name(self, name: str, carrier: str, method: str, allow_incompatible: bool = False):
        allowed = self._allowed_specs(carrier, method)
        for cat in ("common", "carrier", "method"):
            for spec in allowed[cat]:
                if spec["name"] == name:
                    return cat, deepcopy(spec)
        if allow_incompatible:
            if name in COMMON_REGISTRY:
                return "common", {"name": name, "weight": 1.0}
            if name in CARRIER_REGISTRY:
                return "carrier", {"name": name, "weight": 1.0}
            if name in METHOD_REGISTRY:
                return "method", {"name": name, "weight": 1.0}
        raise KeyError(f"Damage {name!r} is not allowed for {carrier}/{method}")

    def _weighted_pick(self, items, k):
        if not items or k <= 0:
            return []
        k = min(k, len(items))
        weights = np.array([float(x.get("weight", 1.0)) for x in items], dtype=np.float64)
        weights = np.maximum(weights, 1e-9)
        weights /= weights.sum()
        idx = self.rng.choice(len(items), size=k, replace=False, p=weights)
        return [deepcopy(items[int(i)]) for i in np.atleast_1d(idx)]

    def _normalize_plan(self, damage_plan):
        if damage_plan is None:
            return None
        if isinstance(damage_plan, dict):
            # 简写：{"scratch": {enabled: true, ...}, "dust": {...}}
            out = []
            for name, cfg in damage_plan.items():
                cfg = {} if cfg is None else deepcopy(cfg)
                if not isinstance(cfg, dict):
                    raise TypeError(f"damage_plan[{name}] must be dict")
                cfg["name"] = name
                out.append(cfg)
            return out
        if isinstance(damage_plan, (list, tuple)):
            return [deepcopy(x) for x in damage_plan]
        raise TypeError("damage_plan must be list/tuple or dict")

    def _apply_one(self, image, qr_mask, spec, category, severity, params=None):
        name = spec["name"]
        registry = self._registry(category)
        if name not in registry:
            raise KeyError(f"No implementation registered for {category}:{name}")
        fn = registry[name]
        effective_params = merged_params(spec.get("params", {}), params or {})
        out, mask, actual = fn(image, qr_mask, self.rng, severity, effective_params)
        mask = (mask > 0).astype(np.uint8) * 255
        actual = {**effective_params, **(actual or {})}
        return out, DamageLayer(name=name, category=category, mask=mask, params=actual)

    @staticmethod
    def _qr_context_roi(qr_mask: np.ndarray, shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = shape_hw
        if qr_mask is None:
            raise ValueError("qr_context damage requires a QR mask")
        ys, xs = np.where(qr_mask > 0)
        if len(xs) == 0:
            raise ValueError("qr_context damage requires a nonempty QR mask")
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        margin = max(8, int(round(max(x1 - x0, y1 - y0) * .35)))
        return max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)

    @staticmethod
    def _roi_feather(shape_hw: tuple[int, int]) -> tuple[np.ndarray, int]:
        h, w = shape_hw
        taper = max(3, min(12, int(round(min(h, w) * .08))))
        yy, xx = np.ogrid[:h, :w]
        edge_distance = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
        alpha = np.clip(edge_distance.astype(np.float32) / taper, 0, 1)
        return alpha, taper

    def apply(
        self,
        image: np.ndarray,
        carrier: str,
        method: str,
        qr_mask: Optional[np.ndarray] = None,
        severity: str = "medium",
        common_count: Tuple[int, int] = (1, 2),
        carrier_count: Tuple[int, int] = (0, 1),
        method_count: Tuple[int, int] = (0, 1),
        forced: Optional[Sequence[str]] = None,
        region_masks: Optional[Dict[str, np.ndarray]] = None,
        damage_plan: Optional[Any] = None,
        damage_controls: Optional[Dict[str, Dict[str, Any]]] = None,
        allow_incompatible: bool = False,
        spatial_mode: str = "full_canvas",
    ) -> DamageResult:
        """叠加污损。

        damage_plan 精确模式示例：
        [
          {"name":"scratch", "enabled":True, "instances":1, "severity":"severe",
           "params":{"count":12, "width_px":[1,3], "opacity":0.65}},
          {"name":"dust", "enabled":False},
          {"name":"metal_rust", "instances":2, "params":{"count":3}}
        ]

        说明：
        - enabled=False：完全没有该污损；
        - instances=0：完全没有该污损；
        - params.count=0：该算子内部元素数量为 0；
        - instances：整种污损重复几层；params.count：一层内部有多少划痕/斑点/裂纹等；
        - 多个 plan 项按顺序叠加，即混合污损。
        """
        full_img = ensure_bgr_u8(image)
        full_h, full_w = full_img.shape[:2]
        full_qrm = ensure_mask(qr_mask, (full_h, full_w))
        full_region_masks = {k: ensure_mask(v, (full_h, full_w)) for k, v in (region_masks or {}).items()}
        if spatial_mode not in {"full_canvas", "qr_context"}:
            raise ValueError("spatial_mode must be full_canvas or qr_context")
        roi = self._qr_context_roi(full_qrm, (full_h, full_w)) if spatial_mode == "qr_context" else None
        if roi is None:
            img, qrm, region_masks = full_img, full_qrm, full_region_masks
        else:
            x0, y0, x1, y1 = roi
            img = full_img[y0:y1, x0:x1].copy()
            qrm = full_qrm[y0:y1, x0:x1]
            region_masks = {k: v[y0:y1, x0:x1] if v is not None else None
                            for k, v in full_region_masks.items()}
        h, w = img.shape[:2]
        controls = deepcopy(damage_controls or {})

        plan = self._normalize_plan(damage_plan)
        picks = []

        if plan is not None:
            for item in plan:
                if not isinstance(item, dict) or "name" not in item:
                    raise ValueError("Each damage_plan item must be a dict containing name")
                if not bool(item.get("enabled", True)):
                    continue
                instances = int(item.get("instances", 1))
                if instances <= 0:
                    continue
                probability = float(item.get("probability", 1.0))
                cat, base_spec = self._resolve_name(item["name"], carrier, method, allow_incompatible)
                for instance_index in range(instances):
                    if probability < 1.0 and self.rng.random() >= probability:
                        continue
                    picks.append((cat, base_spec, item, instance_index))
        elif forced:
            for name in forced:
                cat, spec = self._resolve_name(name, carrier, method, allow_incompatible)
                ctrl = deepcopy(controls.get(name, {}))
                if not bool(ctrl.get("enabled", True)):
                    continue
                instances = int(ctrl.get("instances", 1))
                for instance_index in range(max(0, instances)):
                    if self.rng.random() < float(ctrl.get("probability", 1.0)):
                        picks.append((cat, spec, ctrl, instance_index))
        else:
            allowed = self._allowed_specs(carrier, method)
            # 在随机抽取前即可逐项关闭、改权重。
            for cat in ("common", "carrier", "method"):
                filtered = []
                for spec in allowed[cat]:
                    ctrl = controls.get(spec["name"], {})
                    if not bool(ctrl.get("enabled", True)):
                        continue
                    if "weight" in ctrl:
                        spec["weight"] = float(ctrl["weight"])
                    filtered.append(spec)
                allowed[cat] = filtered
            nc = int(self.rng.integers(common_count[0], common_count[1] + 1)) if common_count[1] >= common_count[0] else 0
            nr = int(self.rng.integers(carrier_count[0], carrier_count[1] + 1)) if carrier_count[1] >= carrier_count[0] else 0
            nm = int(self.rng.integers(method_count[0], method_count[1] + 1)) if method_count[1] >= method_count[0] else 0
            selected = (
                [("common", x) for x in self._weighted_pick(allowed["common"], nc)]
                + [("carrier", x) for x in self._weighted_pick(allowed["carrier"], nr)]
                + [("method", x) for x in self._weighted_pick(allowed["method"], nm)]
            )
            for cat, spec in selected:
                ctrl = deepcopy(controls.get(spec["name"], {}))
                instances = int(ctrl.get("instances", 1))
                for instance_index in range(max(0, instances)):
                    if self.rng.random() < float(ctrl.get("probability", 1.0)):
                        picks.append((cat, spec, ctrl, instance_index))

        layers = []
        out = img.copy()
        union = np.zeros((h, w), np.uint8)
        applied_plan = []
        for cat, spec, item, instance_index in picks:
            item_severity = item.get("severity", severity)
            item_params = deepcopy(item.get("params", {}))
            out, layer = self._apply_one(out, qrm, spec, cat, item_severity, item_params)
            # count=0 时保留“计划里有、实际未产生”的信息，但不增加有效 layer。
            area = mask_ratio(layer.mask)
            applied_plan.append({
                "name": layer.name,
                "category": cat,
                "instance_index": instance_index,
                "severity": item_severity,
                "area_ratio": area,
                "params": layer.params,
            })
            if area <= 0:
                continue
            union = cv2.bitwise_or(union, layer.mask)
            layers.append(layer)

        roi_feather_px = 0
        if roi is not None:
            alpha, roi_feather_px = self._roi_feather((h, w))
            local_out = np.clip(np.rint(img.astype(np.float32) * (1 - alpha[..., None])
                                         + out.astype(np.float32) * alpha[..., None]), 0, 255).astype(np.uint8)
            out = full_img.copy()
            out[y0:y1, x0:x1] = local_out
            keep = alpha > .1
            full_union = np.zeros((full_h, full_w), np.uint8)
            full_union[y0:y1, x0:x1] = np.where(keep, union, 0).astype(np.uint8)
            union = full_union
            full_layers = []
            for layer in layers:
                full_mask = np.zeros((full_h, full_w), np.uint8)
                full_mask[y0:y1, x0:x1] = np.where(keep, layer.mask, 0).astype(np.uint8)
                full_layers.append(DamageLayer(layer.name, layer.category, full_mask, layer.params))
            layers = full_layers
            qrm = full_qrm
            region_masks = full_region_masks

        meta = {
            "carrier": carrier,
            "method": method,
            "spatial_mode": spatial_mode,
            "damage_roi_xyxy": list(roi) if roi is not None else None,
            "damage_roi_feather_px": roi_feather_px,
            "applied_plan_area_reference": "roi" if roi is not None else "full_canvas",
            "severity_default": severity,
            "mode": "damage_plan" if plan is not None else ("forced" if forced else "random"),
            "damage_types": [x.name for x in layers],
            "damage_categories": [x.category for x in layers],
            "damage_layer_count": len(layers),
            "damage_area_ratio": mask_ratio(union),
            "qr_damage_ratio": overlap_ratio(union, qrm),
            "applied_plan": applied_plan,
            "layers": [
                {
                    "name": x.name,
                    "category": x.category,
                    "area_ratio": mask_ratio(x.mask),
                    "qr_overlap_ratio": overlap_ratio(x.mask, qrm),
                    "params": x.params,
                }
                for x in layers
            ],
        }
        for name, rm in region_masks.items():
            meta[f"{name}_damage_ratio"] = overlap_ratio(union, rm)
        return DamageResult(image=out, union_mask=union, layers=layers, metadata=meta)


    def apply_from_config(self, image, carrier: str, method: str, config, qr_mask=None, region_masks=None,
                          allow_incompatible: bool = False, spatial_mode: str = "full_canvas"):
        """从 YAML 路径或 dict 读取一整套污损配置。"""
        cfg = load_yaml(config) if isinstance(config, (str, bytes, Path)) else deepcopy(config)
        if not isinstance(cfg, dict):
            raise TypeError("config must be a YAML path or dict")
        mode = cfg.get("mode", "damage_plan")
        if mode == "damage_plan":
            return self.apply(
                image, carrier, method, qr_mask=qr_mask, region_masks=region_masks,
                severity=cfg.get("severity", "medium"),
                damage_plan=cfg.get("damage_plan", []),
                allow_incompatible=allow_incompatible,
                spatial_mode=spatial_mode,
            )
        if mode == "random":
            def pair(name, default):
                v = cfg.get(name, default)
                return (int(v[0]), int(v[1]))
            return self.apply(
                image, carrier, method, qr_mask=qr_mask, region_masks=region_masks,
                severity=cfg.get("severity", "medium"),
                common_count=pair("common_count", [1,2]),
                carrier_count=pair("carrier_count", [0,1]),
                method_count=pair("method_count", [0,1]),
                damage_controls=cfg.get("damage_controls", {}),
                allow_incompatible=allow_incompatible,
                spatial_mode=spatial_mode,
            )
        raise ValueError(f"Unknown config mode: {mode!r}")
