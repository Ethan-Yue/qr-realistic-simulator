from __future__ import annotations
from pathlib import Path
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PKG_ROOT / "damage_configs"


def load_yaml(name_or_path):
    p = Path(name_or_path)
    if not p.exists():
        p = DEFAULT_CONFIG_DIR / name_or_path
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rules(config_dir=None):
    base = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return {
        "common": load_yaml(base / "common_rules.yaml"),
        "carrier": load_yaml(base / "carrier_rules.yaml"),
        "method": load_yaml(base / "method_rules.yaml"),
        "combinations": load_yaml(base / "combinations.yaml"),
    }
