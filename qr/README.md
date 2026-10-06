# QR Carrier Synth — 可复现的干净/污损/拍摄仿真

> 公开包说明：原项目的 AI 背景库未随仓库发布。本仓库提供五类 CC0 来源材质和 `assets/backgrounds_real` 示例；运行下面的原项目旧命令时，请显式设置该目录及对应的组合 ID，或自行准备具有使用权的背景库。

## 2026-10-05 工作流更新

- `synthesize.py` 默认保留原来的 `ideal` 干净基线，位置抖动现与文档一致，为图像宽度的约 ±1.5%。可用 `--render-profile material` 单独开启轻微的材质与成码层耦合。
- `synthesize_with_damage.py` 默认依次运行 **材质背景 → 成码层耦合 → 条件化污损 → 配对拍摄仿真**。`--render-profile ideal` 和 `--capture-mode none` 可关闭新增阶段，便于消融。拍摄模式还有 `mild`、`handheld`。
- clean/damaged 图像共用透视、光照、噪声与 JPEG 参数；QR 与污损 mask、四角坐标跟随同一透视变换。mask 表示物理图案的几何范围，不是模糊/JPEG 后逐像素颜色差异。
- 每个样本的 QR、干净合成、污损和拍摄阶段采用按 `seed + combo_id + sample_index + stage` 派生的独立种子。只生成某个组合时，相同参数产生相同样本。
- 背景文件按稳定路径排序抽样，元数据记录相对路径、SHA-256、裁剪坐标与源图尺寸。`run_manifest.json` 指定本轮实际组合和样本数，评估程序以它为准，避免复用输出目录时读入旧样本。
- 可用 `--qr-ecc L|M|Q|H|random`、`--payload-mode variable --payload-length-range 36,96` 和 `--payload-alphabet alphanumeric|byte` 控制 QR 结构；metadata 记录纠错等级、版本与载荷长度。污损生成入口默认采用不包含组合标签的随机字母数字载荷；旧 `id` 载荷模式会把组合号写进 QR 内容，不宜用于公平的泛化实验。使用用户提供的原始 QR 图像时，程序会尝试解码并记录字符串和源文件 SHA-256，但该字符串是软件推断值，不等于独立核实的载荷真值，符号版本和纠错等级也无法据此确认。默认要求外部 QR 规范化后可解码；`--allow-undecodable-qr-input` 可跳过此要求。
- `damage_masks` 表示污损算子定义的几何足迹；`visible_change_masks` 表示配对拍摄图任一通道差值大于 5 的像素范围，包含光学模糊和 JPEG 的扩散。两者用途不同。
- 生成前会检查 50 个组合在污损规则中都有对应的载体和成码方式。`丝印` 与 `丝网印刷` 已统一为同一污损规则。

## 来源可追溯材质与强采集模式

`fetch_real_materials.py` 从 Poly Haven 和 Wikimedia Commons 下载五张来源与许可可追溯的材质图，分别覆盖纸、金属、木材、织物和陶瓷。源 URL、作者、许可、文件哈希和尺寸记录在 `assets/backgrounds_real/provenance.json`。Poly Haven 纹理规范通常采用摄影测量或测光立体等基于实物的制作流程；单张素材的拍摄历史尚未独立核实。这些图是材质表面素材，**不是已带 QR 码的实体照片**。另有十类载体尚无对应素材。旧的 1,500 张 AI 纹理保留在原目录，便于区分来源并进行后续比较。

`--background-root assets/backgrounds_real --require-background-library` 强制所选组合从这个小库取图；某个载体没有素材或读图失败时直接报错，不静默退回程序纹理。`--capture-mode challenging` 可采样直线运动模糊、正曝光增益、反光亮斑及饱和，成对图共用采集状态。逐样本 JSON 记录 `capture.motion_blur`、`capture.strong_glare`、`capture.exposure_ev` 和近白像素比例。运动模糊与过曝属于采集退化，不增加几何污损掩码面积；参数尚未按真实相机测量标定。

三个强度可以分别固定：`--capture-motion-px` 接受 0 或 3–31 的奇数，`--capture-exposure-ev` 接受 −2 至 3，`--capture-glare-strength` 接受 0–2.5。未指定的效应照常随机采样。运动参数设为 0 只关闭新增的直线卷积，反光参数设为 0 只关闭新增的强反光；基础高斯模糊和反光材质上的弱高光仍可能存在。固定值和实际采样值均写入逐样本 JSON。

```powershell
python synthesize_with_damage.py --combo-ids 1,17,29,33,38 --per-combo 1 --size 512 --seed 20261007 --background-root assets/backgrounds_real --require-background-library --white-module-mode transparent --capture-mode challenging --damage-mode random --output outputs_real_material_probe
```

固定采集强度的已运行示例：

```powershell
python synthesize_with_damage.py --combo-ids 17 --per-combo 1 --size 512 --seed 20261007 --background-root assets/backgrounds_real --require-background-library --white-module-mode transparent --capture-mode challenging --capture-motion-px 9 --capture-exposure-ev 1.0 --capture-glare-strength 1.2 --damage-mode random --output _qa_real_material_fixed_capture
```

## 以 QR 标记区域污损率控制输出

`--target-qr-damage-ratio min,max` 使用 0–1 比例指定目标区间。程序计算输出的 QR 黑色标记掩码与污损并集掩码的交集面积，除以 QR 黑色标记掩码面积；因此 `0.01,0.10` 表示标记区域内 1%–10% 的像素被几何污损掩码覆盖。这是**像素面积比例**，不是受影响 QR 模块的个数比例，也不表示物理污损深度。元数据中的 `damage_area_ratio_after_capture` 则以整幅图面积为分母。

设置目标区间时，程序对每个样本最多尝试 `--max-damage-attempts` 个可复现的候选，在配对拍摄几何变换后的掩码上计算实际比例；首个命中的候选被保存。若没有候选命中，程序报错并给出最后一次实际比例。适当放宽区间、增加候选数或调整污损算子可以改善命中情况；当前没有对所有组合和所有区间做命中率基准测试。旧的 `--damage-severity` 仍可作为算子默认参数的兼容旋钮，但不作为数据集污损程度标签。

```powershell
python synthesize_with_damage.py --combo-ids 1 --per-combo 1 --size 512 --seed 24680 --damage-mode random --target-qr-damage-ratio 0.01,0.10 --max-damage-attempts 8 --output _qa_damage_ratio_target
```

该命令在第 2 个候选命中 1.86%；逐样本 JSON 记录目标区间、候选次数、候选种子及采集后的实际比例。相同命令重复运行时，图像、掩码和 JSON 的哈希一致。

快速试运行：

```bash
python synthesize_with_damage.py --combo-ids 1,17,36,50 --per-combo 2 --size 512 --output outputs_probe
python evaluate_dataset.py outputs_probe --out outputs_probe/evaluation_summary.json
# 如已安装可选依赖 zxing-cpp，可加 --zxingcpp 交叉核查第二种解码器
```

`evaluate_dataset.py` 核对配对文件/掩码尺寸并用 OpenCV 正常与反色扫描统计可解码率。可选 `--zxingcpp` 使用 ZXing-C++ 做独立交叉核查（另装 `zxing-cpp`）；两个解码器的返回率可能差别很大，应分别报告。它是**合成数据工程检查**，不能证明照片真实感、方法优于已有研究或对真实图像的迁移效果。训练像素复原模型时，输入为 `damaged_images`，对齐目标为同次采集变换下的 `clean_images`；`qr_original`/已知载荷用于符号结构或内容监督，不能直接当作对齐的像素目标。真实拍摄集须另行独立评估。`clean_images` 指未施加污损的配对图，不保证任何指定软件解码器都能直接读出。
做配对消融时，应比较 `synthesize_with_damage.py` 同一样本的 `clean_images` 与 `damaged_images`，并保持 `render_profile` 和 `capture_mode` 一致；直接拿 `synthesize.py` 的默认 `ideal` 图与污损入口默认 `material+mild` 图比较会混入其他因素。

背景库 v2 为图像生成模型产生的 **1500 张 768×768 材质纹理**（15 类各 100 张），不是实拍图库。其逐图模型、提示词、生成种子和授权信息尚未整理；对外发布或投稿前需补齐来源记录。`素材/` 中的示意图也不应当作真实验证样本。

## 配对复原训练数据接口

一次 `synthesize_with_damage.py` 运行完成后，可建立清单并读取训练样本：

```powershell
python prepare_restoration_dataset.py _qa_final
python -c "from restoration_dataset import RestorationPairDataset; ds=RestorationPairDataset('_qa_final', split='train'); print(len(ds), ds[0]['input'].shape, ds[0]['target'].shape)"
```

清单 `restoration_pairs.csv` 引用已有输出文件，不复制图像；默认排除没有污损层或没有可见变化的样本。`RestorationPairDataset` 返回 float32、RGB、CHW、值域 0–1 的 `input` 与 `target`，以及二值 `qr_mask`、`damage_mask`、`visible_change_mask`。默认用记录的 QR 四边形对输入、目标和掩码施加相同的 256×256 裁剪；`crop_qr=False` 可读取整图。清单将相同背景源哈希和相同载荷/QR 来源哈希归为同一传递连通组，再分配训练、验证、测试集。它只能隔离这些**精确相同**的来源，不能识别近重复，也不能把合成测试分数当作实体码迁移结果。未来训练模型时，还应锁定独立的已知载荷实体码测试集，比较恢复前后准确载荷识读率。

## 2026-10-06 外部 QR 定位数据核查

用户提供的 `<external-QR-dataset>\QR` 是一套 1,085 张图的 QR **定位**数据。脚本 `audit_real_qr.py` 读取 VOC/YOLO 框、按原像素导出含上下文的码区裁剪，并分别记录 OpenCV 与可选 ZXing-C++ 的非空返回；`prepare_real_qr_review.py` 创建按图像 SHA-256 分组的新划分和人工审核模板。完整结果见 `../paper/real_dataset_audit/README.md`。原始图不会被脚本覆写。

```powershell
python audit_real_qr.py '<external-QR-dataset>\QR' --output '<audit-output-dir>' --export-crops --decode
python prepare_real_qr_review.py '<audit-output-dir>'
```

该集合混有海报、插画、截图和疑似照片，未附独立载荷真值或物理污损标签；不能直接当作真实污损恢复测试集，也不宜把有现成 QR 的整图拿来作为新码背景。框面积与位置可用于后续场景尺度设计；用于照片真实感比较前，需先逐图确认来源、实体材质、污损及权利。

现已提供**可选的场景尺度压力测试模式**：从核查清单的 1,510 个定位框随机抽取面积比例和归一化中心位置，以等面积正方形 QR 布局生成图像。此模式只使用数值框数据，不把第三方图片贴入合成图。码区边长会按 QR 模块数取整数像素间距，随后还可能受拍摄透视改变；元数据同时记录请求的框面积、中心及最终面积。默认 `closeup` 近景模式保持原行为。整图背景仍为材质纹理，不是包含真实物体、文字和场景布局的实拍照片，因此不能把该模式称作真实场景仿真验证。

```powershell
python synthesize_with_damage.py --combo-ids 1,17 --per-combo 20 --size 1024 --layout-profile external_boxes --layout-manifest '<audit-output-dir>\manifest.csv' --damage-mode none --output outputs_layout_probe
```

`synthesize.py` 也接受同样的 `--layout-profile` 和 `--layout-manifest` 参数。建议在单独输出目录中评估此模式；小码区可能只有约 1 像素/模块，不能保证能被现有解码器读出。`run_manifest.json` 记录框清单 SHA-256，逐样本元数据记录选中的参考框 ID。外部集合图形与照片混杂，框尺度分布只适合场景尺度消融，不能用来校准物理污损分布。

若在 `external_boxes` 模式启用污损，`--damage-spatial-mode auto` 会在 QR 周围取邻域，使现有污损算子的空间尺度随码区缩放，并在邻域边缘渐隐，避免裁剪接缝。近景模式仍按整幅画布生成；两种模式均可显式用 `--damage-spatial-mode full_canvas|qr_context` 指定，供消融比较。逐样本元数据记录邻域坐标、渐隐宽度及 `damage.qr_damage_ratio`。这一缩放只是工程改进，污损尺寸与概率尚未用实体照片标定；生成后仍须检查 QR 覆盖和外观。

---

## 原始干净基线说明

以下说明专指 `synthesize.py` 的干净基线。它保留 **50 个载体 × 成码方式组合**，默认不叠加污损或拍摄仿真。

## 干净基线默认只保留什么

只保留：

1. **同一载体的随机材质背景**
   - 背景图库随机抽样（现有图库是 AI 生成纹理；可替换为实拍背景）
   - 随机方形裁剪
   - 如果没有真实背景，则程序化随机生成同类材质纹理
2. **理想成码工艺外观**
   - 激光打标、激光雕刻、热敏、丝印、UV、贴纸、织造等仍有“理想状态”的外观差异
   - 但不加入缺墨、毛边、断点、条带、噪点等缺陷
3. **极小的 QR 尺寸/位置随机**
   - 只是避免所有样本像素位置完全一致
   - 不做旋转、透视、曲面、褶皱

## 已明确移除

默认**不再做**：

- 材质形变
- 透视变形
- 曲面弯曲
- 褶皱
- 光照梯度
- 高光/反射叠加
- 曝光变化
- 色温变化
- 相机噪声
- 模糊
- JPEG 压缩
- 打印缺墨
- 打印断点
- 打印条带
- 随机毛边
- 随机墨点扩散
- 污渍、划痕、遮挡等污损

这些后续可以由你的独立“污损/退化增强代码”统一叠加。

---

## 输出目录

```text
outputs/
└── 17_metal_laser_mark/
    ├── images/         # 干净合成图
    ├── masks/          # 255=QR黑模块，0=其他
    ├── qr_original/    # 原始二维码
    └── metadata/       # 组合、seed、QR位置等
```

每个组合都有独立目录，同一类内容放在相同子目录中。

---

## 安装

```bash
pip install -r requirements.txt
```

## 运行

每个组合生成 100 张：

```bash
python synthesize.py --per-combo 100 --size 768
```

只生成指定组合：

```bash
python synthesize.py --combo-ids 1,2,17,18 --per-combo 500 --size 768
```

---

## 随机背景图库

建议把“干净、平整、无污损”的真实材质照片放到：

```text
assets/backgrounds/
├── paper/
├── adhesive_label/
├── cardboard/
├── plastic_pack/
├── paper_box/
├── metal/
├── plastic_shell/
├── glass/
├── wood/
├── textile/
├── ceramic/
├── acrylic_pvc/
├── poster/
├── ticket_card/
└── screen/
```

例如：

```text
assets/backgrounds/metal/
├── brushed_001.jpg
├── brushed_002.jpg
├── matte_001.jpg
└── aluminum_001.jpg
```

程序只会：

- 随机选择一张
- 随机裁一个正方形区域
- resize 到输出尺寸

**不会**再额外改变它的亮度、颜色、噪声、形状或清晰度。

### 为了保证“污损阶段”与“基线阶段”解耦

背景图库建议满足：

- 没有原始 QR
- 没有严重划痕/污渍
- 没有强遮挡
- 尽量平整
- 光照尽量均匀
- 不要严重过曝/欠曝
- 每个材质准备多种纹理、颜色、型号

这样后续所有污损都可以由你的统一增强程序控制。

---

## 代码分类

```text
qr_synth/materials.py
```
只负责 **材质背景随机性**。

```text
qr_synth/printing.py
```
只负责 **理想成码工艺外观**，不制造随机缺陷。

```text
qr_synth/compositor.py
```
只负责 **平面、干净、居中合成**。

```text
qr_synth/utils.py
```
负责 QR 码读取/生成。

```text
configs/combinations.yaml
```
50 个“载体 × 成码方式”组合。

---

## 如果你连 QR 位置和大小都不希望随机

当前默认只有很小随机：

- QR 边长约为图像的 68%–76%
- 中心位置约 ±1.5%

在 `qr_synth/compositor.py` 调用时改成：

```python
compose_one(
    ...,
    qr_scale_range=(0.72, 0.72),
    position_jitter=0.0,
)
```

这样每一张图除了“材质背景”之外，QR 的大小和位置也完全固定。


## 已内置图像模型背景库

当前工程包已经在 `assets/backgrounds/` 中内置：

- 15 类载体
- 每类 100 张
- 共 1500 张干净背景
- 768×768 JPG（v2；旧版曾为 512×512）

这些背景来自图像生成模型输出的材质母纹理，并只进行了随机干净区域裁剪与 resize。
当前背景库不叠加光照、噪声、模糊、透视、曲面、破损、污渍等退化。

由于背景库已经存在，`generate_material(..., real_bg_prob=1.00)` 默认优先使用背景文件。

## QR 与彩色背景的自动对比度保护

工程已加入 `qr_synth/contrast.py`，避免随机彩色载体与二维码模块颜色过于接近。

默认流程：

1. 先随机抽取同类材质背景，并确定 QR 落点。
2. 对 QR 实际落点 ROI 使用中位数背景色，计算 WCAG 相对亮度对比度。
3. 普通直接印刷优先使用深色 QR，不会为了追求对比度随意生成不真实的白色 QR。
4. 深色塑料/金属/玻璃/亚克力上的激光标记、雕刻、蚀刻允许选择浅色模块。
5. `sticker / thermal_print / thermal_transfer / heat_transfer` 等合理工艺在低对比度时允许增加干净浅色底板。
6. 其它直接印刷工艺对比度不足时会重新抽背景/位置；仍不达标时保留最高对比度候选，并在 metadata 中记录 `contrast_passes=false`，方便筛除。

默认阈值：

```bash
python synthesize.py --per-combo 100 --min-contrast 3.0
```

如希望更严格：

```bash
python synthesize.py --per-combo 100 --min-contrast 4.5
```

每个样本的 `metadata/*.json` 新增：

- `qr_color_bgr`
- `background_median_bgr`
- `contrast_ratio`
- `min_contrast`
- `contrast_passes`
- `used_light_plate`
- `background_resample_attempts`

该模块只控制二维码与载体的可辨识对比度，不加入光照、噪声、模糊、污损、缺墨、形变等增强。


## 背景库 v2（已重建）

- 不再使用多材质拼图小块裁切后放大的方式
- 改为每张背景独立生成
- 输出为 768×768
- 对每类背景重新建立 100 张随机纹理
- 仍保持干净基线：不加入光照、噪声、模糊、透视、破损、污渍等增强

记录文件：`assets/backgrounds/manifest_v2.json`


## 白模块透明 / 不透明 支持

当前工程已经加入“白模块/静区透明”控制。

含义：

- **transparent**：只把 QR 的黑模块贴到背景上；白模块和静区透明，直接显示载体背景。
- **opaque**：先放一个浅色底板，再叠加 QR 黑模块；也就是整块 QR 区域带浅色背景。
- **random**：默认模式。若用户不强调全部透明或全部不透明，则自动混合生成：
  - 标签 / 贴纸类结构默认保持不透明；
  - 一部分可双态工艺会随机透明 / 不透明；
  - 其它直接成码工艺默认透明。

### 运行方式

默认随机：

```bash
python synthesize.py --per-combo 100 --white-module-mode random
```

全部透明：

```bash
python synthesize.py --per-combo 100 --white-module-mode transparent
```

全部不透明：

```bash
python synthesize.py --per-combo 100 --white-module-mode opaque
```

### metadata 新增字段

每个样本的 `metadata/*.json` 中会记录：

- `white_module_mode_requested`
- `white_modules_transparent`
- `white_underlay_type`
- `white_module_mode_reason`

这样后续你可以直接筛选：哪些样本是“白模块透明”的，哪些是“带浅色底板”的。

---

## 污损叠加版用法

当前工程已内置参数化污损引擎，新增脚本：

```bash
python synthesize_with_damage.py --per-combo 100 --size 768
```

默认行为：
- 先按原来的 clean 流程生成 `clean_final + qrm`
- 再把 `qrm` 作为二维码区域 mask，按“通用污损 + 载体专属污损 + 成码方式相关污损”随机叠加污损

### 随机污损模式

```bash
python synthesize_with_damage.py \
  --per-combo 100 \
  --size 768 \
  --damage-mode random \
  --damage-severity medium \
  --damage-common 1,2 \
  --damage-carrier 0,1 \
  --damage-method 0,1
```

### 精确参数模式（按 YAML）

```bash
python synthesize_with_damage.py \
  --combo-ids 17 \
  --per-combo 100 \
  --size 768 \
  --damage-mode config \
  --damage-config damage_configs/damage_plan_example.yaml
```

### 输出目录

```text
outputs_damaged/
└── 17_metal_laser_mark/
    ├── clean_images/     # 叠加污损前的 clean 图
    ├── damaged_images/   # 最终污损图
    ├── qr_masks/         # 255=QR黑模块区域
    ├── damage_masks/     # 255=污损区域
    ├── visible_change_masks/ # 配对拍摄图中可见变化的像素区域
    ├── qr_original/      # 原始二维码
    └── metadata/         # 包含 clean + damage 元数据
```

### 内置污损配置

污损配置目录：

```text
damage_configs/
```

其中包括：
- `common_rules.yaml`
- `carrier_rules.yaml`
- `method_rules.yaml`
- `damage_plan_example.yaml`
- `random_controls_example.yaml`
- `damage_parameter_reference.yaml`

这样你就可以直接在这个工程里，对它真实生成出来的 qrm 做污损叠加，而不是后处理去猜测二维码位置。
