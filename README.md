# QR 码材质、污损与采集仿真 / QR material, damage and acquisition simulator

本仓库提供可批量运行的 QR 码图像生成工具。生成过程把**载体材质、成码工艺、污损算子和采集退化**连接起来，输出同一载荷的干净／污损配对图、几何掩码及逐样本元数据。配置文件列出 15 类载体和 50 种载体—成码组合；随仓库提供的可追溯材质图片目前只覆盖纸、金属、木材、织物和陶瓷五类。

This repository contains a batch QR-image simulator conditioned on carrier material, marking process, damage and acquisition. It produces clean/damaged pairs, geometric masks and sample-level metadata. The configuration covers 50 carrier–marking combinations across 15 carrier classes; the bundled sourced-material subset covers five classes.

## 快速开始 / Quick start

以下命令从仓库根目录运行；目前实际核查环境为 Windows、Python 3.14.2，具体依赖版本见 [`requirements-tested.txt`](qr/requirements-tested.txt)。源码使用 Python 3.10+ 语法，其他环境需自行验证。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r qr\requirements.txt
cd qr
python synthesize_with_damage.py --combo-ids 1,17,29,33,38 --per-combo 1 --size 512 --seed 20261007 --background-root assets/backgrounds_real --require-background-library --white-module-mode transparent --capture-mode mild --damage-mode random --output outputs_example
python evaluate_dataset.py outputs_example --out outputs_example\evaluation_summary.json
```

Linux/macOS 使用相应的虚拟环境激活命令，并将路径分隔符改为 `/`。`--require-background-library` 在缺少指定载体素材时停止，避免静默退回程序纹理。若需要全部 50 种组合，请自行补齐有使用权的材质素材，或明确采用程序背景；随仓库提供的五张素材不能代表全部 15 类载体。

### 定量污损与采集退化 / Controls

污损程度用 QR **黑色标记像素**被污损几何掩码覆盖的面积比例表示：`r_Q = |M_D ∩ M_Q| / |M_Q|`。它不是受影响模块个数比例，也不是实体损伤深度。下面的命令要求单个纸张样本的实际 `r_Q` 落在 1%–10%；候选次数耗尽会报错。

```powershell
python synthesize_with_damage.py --combo-ids 1 --per-combo 1 --size 512 --seed 24680 --background-root assets/backgrounds_real --require-background-library --damage-mode random --target-qr-damage-ratio 0.01,0.10 --max-damage-attempts 8 --output outputs_target_ratio
```

`challenging` 采集模式可加入直线运动模糊、曝光增益与强反光；可分别用 `--capture-motion-px`、`--capture-exposure-ev`、`--capture-glare-strength` 固定强度。干净／污损图共享采集状态。运动模糊和过曝是采集退化，不增加几何污损掩码面积。模型尚未按真实相机标定。

```powershell
python synthesize_with_damage.py --combo-ids 17 --per-combo 1 --size 512 --seed 20261007 --background-root assets/backgrounds_real --require-background-library --white-module-mode transparent --capture-mode challenging --capture-motion-px 9 --capture-exposure-ev 1.0 --capture-glare-strength 1.2 --damage-mode random --output outputs_fixed_capture
```

更多参数和 YAML 规则见 [生成器说明](qr/README.md)及 [`damage_configs`](qr/damage_configs/)。如需复核文中的 ZXing-C++ 计数，安装 `zxing-cpp` 并向评估命令加入 `--zxingcpp`。`prepare_restoration_dataset.py` 可以建立后续复原研究所需的配对清单；仓库没有模型训练或真实场景性能结果。

## 输出与示例 / Outputs and examples

每个组合目录含 `clean_images`、`damaged_images`、`qr_original`、`qr_masks`、`damage_masks`、`visible_change_masks` 和 `metadata`。`run_manifest.json` 记录本次运行的参数与完成状态。`examples/` 保存小规模公开样例；[`evidence/checks.json`](evidence/checks.json) 保存工程核查计数。样例不是统计意义上的真实感验证。

中英双语论文草稿见 [中文阅读版](paper/论文公开稿_ZH.html)、[中文可编辑稿](paper/论文初稿_ZH.md)、[English reading version](paper/Manuscript_Public_EN.html) 和 [English draft](paper/Manuscript_Draft_EN.md)。公开版保留图 1、3–6；图 2 使用了外部 QR 定位集的原图，逐图再发表权未核实，因此本仓库不分发这些原图。本地作者稿与公开版由此有意不同。

## 素材与数据边界 / Assets and data boundary

五张材质图的来源、作者、许可、原图哈希和尺寸保存在 [`provenance.json`](qr/assets/backgrounds_real/provenance.json)。四张来自 [Poly Haven](https://polyhaven.com/license)，一张旧纸纹理来自 [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:%D0%A2%D0%B5%D0%BA%D1%81%D1%82%D1%83%D1%80%D0%B0_%D0%B1%D1%83%D0%BC%D0%B0%D0%B3%D0%B8.jpg)，来源页面均标注 CC0。这些是材质表面图，不是带码实体照片；单张素材的采集过程未独立核实。

原项目中的 1,500 张 AI 背景、外部 QR 数据集原图、裁剪图和私人本地工作文件不在本仓库。外部集合的聚合统计和审计代码保留，以便持有原数据的研究者复核；解码器对该集合的非空返回不等于正确载荷率。详见 [公开内容说明](PUBLICATION_NOTES.md)。

## 许可 / License

生成器代码、配置和脚本按 [MIT](LICENSE) 发布。论文文字与作者制作的论文图示按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布；其中引用的第三方 CC0 材质保留原来源与许可。外部数据集不受本仓库许可覆盖。
