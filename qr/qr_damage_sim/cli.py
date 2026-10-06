from __future__ import annotations
import argparse
from .batch import process_manifest


def main():
    ap=argparse.ArgumentParser(description="QR 高清干净图二次污损叠加")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--severity",choices=["mild","medium","severe"],default="medium")
    ap.add_argument("--variants",type=int,default=1)
    ap.add_argument("--common",default="1,2",help="随机模式：每张图通用污损数量范围，如 1,2")
    ap.add_argument("--carrier",default="0,1",help="随机模式：载体专属污损数量范围")
    ap.add_argument("--method",default="0,1",help="随机模式：成码方式相关污损数量范围")
    ap.add_argument("--damage-config",default=None,help="YAML 配置。提供后优先使用精确/随机参数配置。")
    args=ap.parse_args(); parse=lambda s: tuple(map(int,s.split(",")))
    process_manifest(args.manifest,args.output,args.seed,args.severity,args.variants,parse(args.common),parse(args.carrier),parse(args.method),args.damage_config)

if __name__=="__main__":
    main()
