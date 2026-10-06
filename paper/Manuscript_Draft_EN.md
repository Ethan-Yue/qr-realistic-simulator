# Toward realistic QR-code appearance through batch simulation: carrier–marking coupling, quantitative damage and acquisition effects

> Public repository version: the external-collection images in Figure 2 are withheld pending image-level rights verification. The manuscript retains the author's figure numbering and numerical audit.

**Article type:** methods and data resource. Updated 7 October 2026 from the implemented generator, synthetic engineering check and external localization-image audit. Image-level provenance and republication rights for the external examples require verification before submission.

## Abstract

The visible appearance of a QR code varies with its carrier, marking process, damage and acquisition conditions. Making and photographing large collections of physical damaged codes with known payloads is costly, motivating a simulator that can run in batches with controllable parameters and traceable annotations. We implemented a carrier- and marking-process-conditioned generator with 15 carrier classes and 50 carrier–marking configurations. Symbol generation, carrier texture, mark appearance, conditional damage and two-dimensional acquisition are combined. Users can select damage type, layer counts and spatial mode, or specify occurrence probabilities, instances and operator parameters in YAML. Simulated damage extent is defined as the fraction of QR dark-mark pixels covered by the geometric damage mask; a target interval can be requested and achieved values are recorded. A run generates the requested number of samples per selected configuration, saving paired clean/damaged images, ideal symbols, masks and sample-level metadata. The original library contains 1,500 AI-generated textures; a separate five-image library with traceable material sources was assembled for additional examples. Motion blur, strong glare and overexposure were added as optional acquisition effects. In an engineering check with two samples per configuration, all 100 image and binary-mask sets passed shape and value checks. Exact payloads were returned from 59/100 damaged images by OpenCV and 86/100 by ZXing-C++. An external QR-localization collection comprised 1,085 images and 1,510 valid boxes, with a median box area of 4.21% of the image. The method provides traceable batch generation and paired data for possible downstream uses such as restoration. Material-source imagery and acquisition effects extend the simulation conditions. The mark-area damage ratio measures geometric coverage; photographic realism and physical-world performance remain unverified.

**Keywords:** QR code; damage simulation; carrier material; marking process; synthetic data; paired annotation; data resource.

## 1. Introduction

QR symbol structure and error correction are defined by a standard [1], but the image recorded by a camera also depends on carrier material, marking process, use-related damage and acquisition. Toner on paper, laser-altered metal, woven textiles and display pixels differ in texture and reflectance. A plain black-on-white symbol with generic occlusion does not explicitly control these relationships. Manufacturing, damaging and photographing physical codes for every combination also requires equipment and manual documentation.

Synthetic damage and QR restoration have established precedents. Irregular masks, manually damaged printed codes, print–scan degradation, and compound illumination and blur have all appeared in previous QR data or restoration studies [2–7]. The specific question addressed here is how to organise carrier, marking, damage and acquisition variation into one repeatable generation process while retaining the payload, geometric coverage and parameters needed to trace individual samples.

We encode 50 curated carrier–marking configurations and expose controls for configuration selection, images per configuration, QR payload parameters, a target mark-area damage ratio and individual operators. The program batch-generates spatially paired clean and damaged images with programmatic masks. Four distinct combinations appear in Fig. 1, and fixed-input examples with different measured damage ratios appear in Fig. 4. A close-up mode generates large symbols, while an optional scene-scale mode samples QR-box area and position from an external localization collection. We inspect 100 synthetic examples for output consistency and decoder returns, and audit 1,085 external images for scale and source heterogeneity. This methods report documents the generator, its controls and its auditable outputs; comparison with verified physical photographs remains outside the present evidence.

## 2. Related work and method positioning

### 2.1 Damage synthesis for QR restoration

Image inpainting and restoration studies have generated damaged QR examples before this project. EHFP-GAN uses original symbols and irregular masks to study reconstruction across contamination levels [2]. A logistics-oriented restoration study combined synthetic distortions with real printed codes that were intentionally written on, scratched, or marked with paint [3]. These studies establish that synthetic damage can support QR restoration research. Here, carrier and marking process condition rendering and damage selection; the report concentrates on generation records, paired annotations and engineering checks rather than comparing downstream restoration models.

### 2.2 Printing, capture, and scene variation

Sancar's dataset includes unreadable scanned QR images, simulated unreadable images, and generated training data for a super-resolution study [4,5]. Benito-Altamirano and colleagues studied coloured QR codes exposed to empty, augmented, and real print–capture channels [6]. A separate dataset by Benito-Altamirano and Martínez-Carpena contains QR codes on flat and more challenging surfaces, together with spatial annotations [8]. BarBeR brings together real images of several barcode symbologies, including QR codes, for localization benchmarking [9]. These resources show that surfaces and acquisition conditions have been represented in prior work. Their exact task definitions and annotations differ, so direct numerical comparisons require common held-out images, matched evaluation definitions, and access to the original methods.

### 2.3 Scope of the present resource

Prior work has addressed symbol-level occlusion, print–scan degradation, compound acquisition noise and some physically damaged examples [2–8]. We therefore do not claim that QR damage simulation was absent. The documented contribution is a batch-callable conditional generation method: 50 carrier–marking configurations, adjustable damage types, target QR dark-mark area fractions and operator parameters, paired clean/damaged images, masks and metadata, and engineering checks of those outputs. The external localization collection [11] informs scene-scale geometry only; its unverified image-level physical provenance does not establish physical-damage realism.

## 3. Methods

### 3.1 Task definition and generated outputs

Let \(s\) denote a QR symbol with known payload, \(c\) a carrier, \(p\) a marking method, \(b\) a sampled carrier background, \(d\) a damage plan, and \(a\) an acquisition state. The simulator constructs a pre-damage rendering \(R(s,c,p,b)\), a damaged rendering \(D(R,d;c,p)\), and a paired image transformation \(A(\cdot,a)\). The saved pair is \(x_{\mathrm{clean}}=A(R,a)\) and \(x_{\mathrm{damaged}}=A(D(R,d;c,p),a)\). The **same sampled acquisition state** is applied to both members of a pair. This choice preserves image alignment for pixelwise comparison; it should not be mistaken for two independent photographs of a physical specimen.

For each sample, the implementation saves a pre-damage colour image, a damaged colour image, the original ideal QR image, a binary QR-mark mask, a binary union-of-damage mask, a binary visible-change mask, and JSON metadata. The QR and damage masks describe the **geometric footprints before optical blur and JPEG spread**. The visible-change mask marks pixels whose clean and damaged outputs differ by more than five intensity levels in at least one channel after paired acquisition; it is an image-difference label, not a physical damage annotation. The JSON also includes the transformed QR quadrilateral, carrier and method IDs, selected damage layers, sampled parameter values, image size, and stage seeds.

### 3.2 Carrier–marking taxonomy and background provenance

The project configuration lists 50 plausible carrier–marking combinations across 15 carrier classes, ranging from paper and adhesive labels to metal, glass, textile, ceramic, and electronic displays. The list is a curated scope for simulation, not a census of all ways to make QR codes. Each combination maps to a carrier texture, a mark-rendering method, and eligible carrier- and process-associated damage rules. Not every pairing has an independently documented physical production process; configuration scope is distinct from physical validation scope.

The current background library contains 100 images for each of the 15 carriers: 1,500 images in total, each 768 × 768 pixels. Its manifest describes these as independently generated material textures from an image-generation model; we identify them as **AI-generated textures** throughout. A sampled image is cropped and resized for the canvas; sample metadata records its relative path, SHA-256 hash, source dimensions, and crop coordinates. If the configured library is unavailable, the implementation can fall back to procedural material generation. Neither path supplies a photograph of an actual marked specimen. The texture model, generation procedure, licensing, redistribution rights and near-duplicate audit lack complete documentation, limiting public reproducibility at present.

To reduce dependence on AI textures, we assembled a separate, traceable five-image material library covering paper, metal, wood, textile and ceramic. The paper texture is attributed as the creator's own work and released under CC0 on Wikimedia Commons [13]. Four CC0 diffuse material maps came from Poly Haven [12]. Poly Haven's material standard prioritizes photo-based sources, although we did not independently audit the camera or processing history of each image. The fetch script records source URLs, creators, licences, original-file hashes and dimensions; each generated sample also records its crop coordinates. `--background-root assets/backgrounds_real --require-background-library` restricts generation to that library and fails if a requested carrier has no readable image, preventing a silent fallback to procedural texture. Five images do not cover all 15 carriers and are not photographs of actual marked QR specimens. The earlier 1,500 AI textures remain a separately identified option. Sourced material images change the background input; they do not alone validate the appearance of rendered marks, damage or acquisition.

### 3.3 Symbol, payload, and contrast choices

When generating a symbol internally, the code uses the `qrcode` package and a four-module quiet-zone setting. It first obtains the QR module matrix and repeats every module at the same integer pixel pitch, chosen to meet or exceed a requested source size of 420 pixels; this avoids uneven module widths in the saved original QR image. The later compositor resizes that module image to its chosen placement size. The generator supports error correction levels L, M, Q, and H or a random choice among them. It also supports an ID-derived payload or a reproducible variable-length payload in QR alphanumeric or byte mode. The variable payload excludes the carrier and process ID to avoid a content-to-class shortcut. Symbol version is chosen to fit the payload; the generated payload, version, error correction level, quiet-zone setting, payload length, module count, source module pitch, and source side length are stored in metadata. User-supplied QR images can also be loaded and thresholded. The loader attempts OpenCV decoding after normalization and records the inferred string and source-file hash; it cannot independently verify the payload or recover symbol parameters as ground truth. Those samples require separate provenance in payload-based evaluation.

The default `closeup` profile samples QR size and position within a narrow range. An optional `external_boxes` profile samples box area and normalized centre from the audited localization manifest [11]. It places an area-equivalent square mark on generated carrier texture without copying external image pixels. Integer module pitch, boundary clipping and camera transformation can change the achieved area; metadata records both area values and the manifest hash. The profile tests scene scale, without reproducing the surrounding objects and layout of a photograph.

When damage is enabled with `external_boxes`, the default spatial mode operates in a QR-centred context region and tapers changes at its boundary. This adapts existing damage operators to small marks, but their size and probability distributions still require physical calibration. The QR overlap ratio is recorded per sample. A carrier- and process-dependent rule controls whether white modules expose the material or lie above a label plate. The compositor selects a mark colour against the local background and reports image contrast. This software measure is not an ISO/IEC 15415 print-quality grade [10].

### 3.4 Marking appearance and material coupling

The `ideal` profile applies the project's clean, process-specific mark appearance. The newer `material` profile adds a modest coupling between the already generated mark and its local carrier surface. Its method profiles govern the fraction of carrier colour visible in marked modules, the strength of local texture and correlated grain, and subpixel edge coverage **inside** the existing QR-mark mask. Examples of modeled appearances include toner, porous ink, laser-altered surfaces, woven marks, and display pixels. A label receives separate treatment so that the material beneath the label does not strongly tint its printed mark. This component changes appearance but does not create missing modules or reclassify the QR mask. The heuristic parameters have not been measured from physical specimens, so their visual agreement with real marks cannot be inferred from these outputs.

### 3.5 Conditional damage generation

Damage is applied after the pre-damage QR/carrier composition. The rules contain 14 names in a general-damage pool, alongside carrier- and marking-associated pools. General examples include dust, stains, scratching, abrasion, tape, handwriting and droplets; metal can select oxidation or pitting, whereas laser marking can select local contrast loss. By default, only operators allowed for the current carrier–marking combination are available. An explicit compatibility override is required to select an otherwise disallowed operator.

**Random mode and quantitative damage extent.** The program samples without replacement from the general, carrier-specific and marking-specific pools using configured weights. `--damage-common`, `--damage-carrier` and `--damage-method` define the number drawn from each pool; defaults are 1–2, 0–1 and 0–1. We define simulated damage extent by the QR dark-mark area fraction $r_Q=|M_D\cap M_Q|/|M_Q|$, where $M_D$ is the union of geometric damage masks and $M_Q$ is the dark-mark mask. The full-image damage fraction is $r_I=|M_D|/(HW)$. These are pixel-area measures: $r_Q$ is neither the fraction of discrete QR modules affected nor the depth of physical damage or loss of payload information. `--target-qr-damage-ratio l,u` specifies a closed target interval. Independent reproducible candidate seeds are tried, with $r_Q$ computed from the masks after paired geometric acquisition; the first candidate in range is retained. If `--max-damage-attempts` is exhausted, the run raises an error instead of silently accepting an out-of-range sample. The legacy `--damage-severity` setting can still change default operator parameter distributions but is not used as the paper's damage-extent classification.

**Configuration mode.** YAML may define a controlled random scheme or an ordered `damage_plan`. The former can set per-operator enabling, sampling weight, occurrence probability, layer instances and operator parameters. The latter specifies each operator's name, enabled state, probability, number of instances, internal operator scale and `params`. Parameters include element count (`count`), pixel or relative size, opacity and `qr_bias`, which favours locations on the QR mark. Setting `instances=0`, `params.count=0` or `enabled=false` suppresses an effect. `--damage-mode none` produces a no-damage reference. Effective layers are applied in plan order with individual masks; the union mask, full-image area fraction and overlap with the black-module mask are recorded. Operator names express intended mechanisms, while their parameter distributions remain uncalibrated against physical damage.

### 3.6 Paired acquisition simulation

After damage, the `capture_pair` stage optionally applies the same homography, illumination field, white-balance shift, reflective highlight, Gaussian blur, noise realization, and optional JPEG transformation to the clean and damaged images. It warps the geometric masks with nearest-neighbour interpolation and records the transformed QR quadrilateral. The existing `none`, `mild` and `handheld` modes remain available; `mild` is the default, with a separate optional `challenging` mode. This is a two-dimensional image model with small perspective changes. It does not simulate true three-dimensional folds, curved surfaces, specular reflection from measured bidirectional reflectance functions, or a calibrated camera sensor. The choices make the pipeline controllable; whether their distributions match real handheld photography must be tested on held-out captures.

A new optional `challenging` acquisition mode samples a straight-line motion point-spread kernel, a linear exposure gain expressed in EV, and a strong elliptical glare lobe with a narrow streak that is more likely for reflective carriers. Without overrides, a normalized motion-kernel length is selected from 3, 5, 7, 9 or 11 pixels; activation of motion, positive exposure gain and strong glare is sampled separately. Glare is added in linear light before sRGB conversion and possible clipping. `--capture-motion-px` (0 or an odd integer from 3 to 31), `--capture-exposure-ev` (−2 to 3 EV) and `--capture-glare-strength` (0–2.5) can set the three intensities, while unspecified effects remain sampled. A value of zero suppresses the added line-motion kernel or strong-glare component, respectively; baseline Gaussian blur, ambient illumination and weak highlights on reflective carriers remain. Metadata records requested values, motion length and angle, exposure EV, glare parameters and the fraction of clean-image pixels near full white. All acquisition effects are shared by the clean/damaged pair; geometric masks still describe symbol and damage footprints. The existing `mild` and `handheld` parameter paths are retained. Image-domain motion convolution does not reproduce rolling shutter, three-dimensional surface normals or measured specular reflection. The strong-light and motion ranges are engineering choices without camera/material calibration.

### 3.7 Reproducibility and implementation boundary

Each output sample derives separate stage seeds from a base seed, combination ID, sample index, and stage identifier using NumPy `SeedSequence`. Clean composition, damage, capture, QR content, and optional box-layout sampling therefore have independent deterministic random streams. This arrangement is intended to make an image's output independent of which other combinations were generated in the same run. Reproducibility further requires the exact code revision, dependency versions, source-background files, configuration files, and image codec behaviour to be archived; the current small hash check below establishes only a narrower property. The primary Python entry point is `synthesize_with_damage.py`; it calls the carrier compositor, conditional damage engine, and acquisition module in that order.

### 3.8 Batch synthesis and downstream interface

The main entry point selects combinations with `--combo-ids`, samples per combination with the positive integer `--per-combo`, and image size, base seed, QR error correction and payload parameters, background library, rendering profile, acquisition mode and damage scheme with corresponding options. Selecting \(K\) combinations and requesting \(n\) images per combination yields \(K n\) samples in a complete run. For example, all 50 combinations with `--per-combo 100` request 5,000 image groups. The engineering check in this paper generated only two per combination (100 total); runtime and storage for a 5,000-group run have not been measured. Outputs are organized by combination into clean and damaged images, ideal symbols, three masks and per-sample JSON metadata. `run_manifest.json` records arguments, selected combinations, count per combination and completion status. Per-sample JSON additionally records the target interval, candidate count and seed, achieved post-acquisition $r_Q$, and full-image damage fraction. One executed single-sample invocation requested 1–10% QR dark-mark damage:

```powershell
python synthesize_with_damage.py --combo-ids 1 --per-combo 1 --size 512 --seed 24680 --damage-mode random --target-qr-damage-ratio 0.01,0.10 --max-damage-attempts 8 --output _qa_damage_ratio_target
```

The same entry point accepts an ordered YAML scheme through `--damage-mode config --damage-config damage_configs/damage_plan_example.yaml`, and custom rules through `--damage-rules-dir`. The existing `prepare_restoration_dataset.py` can index completed paired outputs without copying their image files. This makes the resource available for later restoration, localization and decoding-robustness studies; no training or performance result for those applications is reported here. Practical batch size remains constrained by runtime, storage and source-material availability.

## 4. Engineering checks and external-data audit

### 4.1 Synthetic samples and check definitions

The engineering run used base seed 24680 to generate two 512 × 512 samples for each of 50 configurations, yielding 100 images. Generation used the `material` mark-rendering profile, `mild` two-dimensional acquisition, a random choice among L/M/Q/H error correction levels, and requested alphanumeric payload lengths of 36–72 characters. We checked paired-image and mask dimensions and whether geometric masks contained only 0 and 255. OpenCV `QRCodeDetector` 5.0.0 and ZXing-C++ 3.1.1 attempted the ideal symbol, clean composite and damaged composite separately, with a prespecified inverted-contrast retry after an empty return. Internal payloads were known, so exact-match counts could be computed. Two examples per configuration support an implementation check, not a population estimate for any particular configuration.

### 4.2 External localization-image scope and processing

The user-provided GitCode collection [11] contains 1,085 images with VOC and/or YOLO localization annotations. Original files were preserved. We read image dimensions and boxes, preferred VOC when present, and fell back to YOLO otherwise. For each valid box, we exported a crop at source resolution with 20% context on each side. Both decoders recorded **non-empty returns** for full images and crops. Because independently recorded payloads are absent, these returns cannot be called correct-payload recovery. SHA-256 identified exact duplicate files, and an auxiliary split grouped equal hashes. Figure 2 shows four original images selected for visual diversity, not as a random sample or proof of photographic or damage provenance.

### 4.3 Reusing scene-scale geometry

The optional `external_boxes` layout samples relative area and normalized centre coordinates from the external box manifest, then places a square QR mark of equal requested area on an internally generated carrier texture. Only numerical geometry is used; external image pixels are not copied. Integer pixels per module, boundary restrictions and subsequent perspective transformation can change the achieved area. Requested and achieved geometry and the source-manifest hash are recorded. For small QR regions, damage may be applied within a feathered QR neighbourhood. This is a scale and algorithm stress test, not a simulation of the objects, text or full layout of a physical scene.

### 4.4 Small checks of sourced materials and challenging acquisition

We selected one image for each of the five sourced material classes and generated one 512 × 512 pair for paper laser printing, metal laser marking, wood laser engraving, textile screen printing and ceramic screen printing. The base seed, material library, transparent white-module setting and random damage were held fixed; acquisition was run once in `mild` and once in `challenging` mode. Within each combination, the payload, source-image hash and source crop were identical between runs, allowing the acquisition settings to be compared. Existing checks covered output dimensions, binary masks and exact decoder payload returns. One pair per combination checks only the new code path; it does not estimate material realism or population readability.

## 5. Results

### 5.1 Configured scope and observable output

The batch entry point organizes generation by combination ID and requested count, saving paired images, masks, parameters and run indices for each sample.

The configuration spans 15 carrier classes and 50 carrier–marking combinations (Table 1). Fixed-index examples show clean and damaged pairs for paper laser printing, metal laser marking, jacquard textile marking and LCD/OLED display (Fig. 1). Each pair shares one acquisition transformation, allowing the altered area to be compared directly. The examples display differences in carrier texture and mark contrast. They illustrate generated diversity, not a blinded similarity comparison with actual carriers. All 1,500 background textures were AI generated; per-image creation details and rights still need documentation.

**Table 1 | Carrier–marking configuration scope.** Counts are taken from the project configuration; they are not a census of all physically feasible production processes.

| Carrier class | Configurations | Marking examples |
| --- | ---: | --- |
| Paper | 4 | Laser, inkjet, offset and thermal printing |
| Adhesive label | 3 | Thermal, thermal-transfer and digital inkjet |
| Cardboard | 3 | Flexographic, industrial inkjet and sticker |
| Plastic packaging | 4 | Gravure, flexographic, UV inkjet and thermal-transfer |
| Paper box | 2 | Offset and digital printing |
| Metal | 4 | Laser marking, engraving, screen and UV printing |
| Plastic shell | 4 | Laser marking, pad printing, screen printing and sticker |
| Glass | 4 | Screen and UV printing, laser etching and sticker |
| Wood | 4 | Laser engraving, laser burn, UV and screen printing |
| Textile | 5 | Screen, transfer, direct inkjet, jacquard and embroidery |
| Ceramic | 3 | Screen printing, transfer and underglaze |
| Acrylic/PVC | 3 | UV, screen and laser engraving |
| Poster | 3 | Spray, photographic and roll-to-roll UV printing |
| Ticket/card | 3 | Thermal, thermal-transfer and digital printing |
| Electronic display | 1 | LCD/OLED display |
| **Total** | **50** | **15 carrier classes** |

![Fig. 1: Synthetic clean and damaged QR examples](figures/Fig_1_synthetic_examples.png)

**Fig. 1 | Paired synthetic QR images conditioned on carrier and marking process.** a Paper/laser printing; b metal/laser marking; c textile/jacquard; d screen/LCD–OLED. Clean and damaged images are the `000000` sample of each configuration under paired acquisition simulation. These are generated images, not photographs. Their appearance has not been calibrated against physical specimens. The four image pairs and sample metadata are available in the public `examples/engineering_50/` directory.

### 5.2 Output validity and software readability

All 100 synthetic examples had consistent clean/damaged image and mask dimensions and binary mask values. Exact-payload return counts for ideal symbols, clean composites and damaged composites are shown in Table 2. OpenCV returned 59/100 damaged payloads correctly, whereas ZXing-C++ returned 86/100. This difference makes software and retry protocol part of any readability statement. Neither value is a standardized print-quality grade or a photographic-realism score. A generation-order check for configuration 17 yielded identical hashes for all 14 corresponding files in the full and single-configuration runs. A separate scene-scale probe produced four examples across two configurations; all passed image/mask shape checks, and QR-neighbourhood damage did not change pixels or masks outside its recorded region.

**Table 2 | Engineering check of 100 synthetic examples.** Numbers are exact matches to known internally generated payloads; incorrect non-empty strings count as failures.

| Saved image stage | Valid dimensions/binary masks | OpenCV exact payload | ZXing-C++ exact payload |
| --- | ---: | ---: | ---: |
| Ideal symbol | 100/100 | 99/100 | 100/100 |
| Clean composite | 100/100 | 92/100 | 100/100 |
| Damaged composite | 100/100 | 59/100 | 86/100 |

### 5.3 Scene scale and source heterogeneity of external images

Of the 1,085 external images, 1,037 had VOC annotations, yielding 1,510 valid QR boxes; 48 lacked localization annotations and 194 contained multiple boxes (Table 3). The median box occupied 4.21% of the image and had a short side of 104 pixels (Fig. 3a). Exact hashing found 42 duplicate-image groups, including 17 groups spanning the supplied train, validation or test partitions. A new hash-grouped split removes exact-file overlap but does not identify resized or cropped near duplicates.

Figure 2 shows photographic-looking scenes, a promotional card and an illustration in the same collection. Its first image has no usable localization annotation and was therefore excluded from the box statistics in Fig. 3. Pixels alone cannot authenticate a photograph, carrier material or physical damage mechanism. OpenCV returned a non-empty string from 325/1,085 full images and from at least one crop in 395 images. ZXing-C++ returned one from 635 full images and at least one crop in 599 images. Figure 3b stratifies single-box images by QR area. Without independent payload ground truth, these counts are decoder returns rather than correct-read rates.

**Table 3 | Audit of the external QR-localization collection.** Entries describe the local copy and its boxes; they do not establish physical-damage provenance.

| Item | Count or median |
| --- | ---: |
| Images | 1,085 |
| Images with VOC annotations | 1,037 |
| Valid QR boxes | 1,510 |
| Unannotated images | 48 |
| Multi-box images | 194 |
| QR-box area/full-image area | 4.21% |
| QR-box short side | 104 pixels |
| Exact duplicate-image groups | 42 |
| Exact duplicate groups spanning supplied splits | 17 |

> **Figure 2 image withheld from this public version.** Image-level republication rights for the external localization collection have not been verified.

**Fig. 2 | Four visual forms in the external QR-localization collection (image withheld in this public version).** The author's local audit records: a `QR-00589`, apparent public-space photograph; b `QR-01016`, apparent hand-held card photograph; c `QR-01063`, promotional card image; d `QR-00108`, illustration. These descriptions do not authenticate provenance or physical damage. The images come from the user-provided GitCode collection [11]; image-level republication rights remain unverified.

![Fig. 3: QR-box scale and decoder returns in the external collection](figures/Fig_3_external_scale.png)

**Fig. 3 | QR-box scale and non-empty decoder returns in the external collection.** a Distribution of the area fractions of 1,510 valid boxes; the dashed line marks the 4.21% median. b Non-empty full-image and 20%-context crop returns by OpenCV and ZXing-C++ for 843 single-box images, grouped by QR area. Counts above each stratum are images. CSV source data and figure QA are in `real_dataset_audit/figure_A1/`. Returns are not known-payload accuracy or a measure of photographic authenticity.

### 5.4 Measured QR mark damage ratio and batch interface

Figure 4 fixes one clean paper/laser-print composite, one scratch operator and one random seed. Scratch count and width were varied, but the outputs are labelled by the **measured mask coverage** rather than by qualitative categories. The QR dark-mark area fraction $r_Q$ is 1.61%, 3.94% and 4.98%; the corresponding full-image mask area fraction $r_I$ is 0.63%, 1.48% and 1.97%. Scratch counts were 3, 5 and 7, with widths of 1–2, 1–3 and 1–4 pixels. These values describe one specific realization. They do not establish a monotonic relation for every operator or seed, and they are not physical-damage grades. For this isolated engine demonstration, scratches were applied to an already saved clean acquired image; the full batch pipeline applies damage before paired acquisition.

The completed run generated two samples per configuration, or 100 paired groups, and passed the structural checks in Table 2. `--per-combo` can request more images; 100 images for each of 50 configurations would request 5,000 groups. That larger run was not performed, and no throughput or storage-scaling result is reported. The adjustable rules and batch output format are implementation capabilities supported by the completed engineering check.

A separate target-interval check generated one 512 × 512 image for configuration 1. With $r_Q\in[0.01,0.10]$ and at most eight candidates, the second candidate was accepted at an achieved $r_Q$ of 1.86%. A repeat with identical arguments produced matching SHA-256 hashes for the clean image, damaged image, three masks and sample JSON. This checks one path through the target-selection and deterministic-seeding logic; it does not estimate the acceptance rate for all configurations or target intervals.

![Fig. 4: Scratch outputs labelled by measured QR dark-mark damage area](figures/Fig_4_damage_ratio.png)

**Fig. 4 | Measured QR dark-mark damage area fraction.** a Saved clean paper/laser-print composite; b–d outputs from the same scratch operator and seed with different internal parameters. Panel labels report measured $r_Q$. The corresponding full-image mask areas and sampled parameters are in `figures/Fig_4_damage_ratio_source.csv`. This is an isolated damage-engine illustration applied after the saved acquisition state; the full pipeline applies damage before paired acquisition. Mask-area ratios have not been calibrated to physical damage.

### 5.5 Synthetic examples on traceable material images

One pair was generated successfully for each of five traceable material classes, and every background record pointed to the intended source file. Figure 5 shows the actual source crop, clean synthesis and damaged synthesis for four of them. Paper, metal, wood and textile details remain visible around the QR mark and through unmarked regions. The fifth, ceramic, passed the file checks but is not shown. For the `mild` run, all 5/5 image/mask sets were valid, and both OpenCV and ZXing-C++ returned the known payload for all 5/5 clean and 5/5 damaged images. These five fixed software returns do not show that the images look more realistic than those using the former AI library. Poly Haven provides material surfaces, not physical photographs of QR codes printed or marked on those surfaces; the centre and right columns remain synthetic images.

![Fig. 5: Traceable material crops and synthetic QR pairs](figures/Fig_5_real_material_examples.png)

**Fig. 5 | QR synthesis on sourced material images.** a Paper/laser print; b metal/laser mark; c wood/laser engraving; d textile/screen print. The left column shows the source crop used in this run; centre and right columns are clean and damaged outputs on the same texture. Paper came from Wikimedia Commons [13], and the other images from Poly Haven [12]. Asset URLs, licences and source hashes are in `qr/assets/backgrounds_real/provenance.json`; output hashes are in `figures/Fig_5_real_material_sources.json`. Source provenance does not validate the photographic realism of the composite QR image.

### 5.6 Motion blur, glare and overexposure during acquisition

Figure 6 compares `mild` and `challenging` acquisition for the same metal texture crop, QR payload and damage-generation settings. In this fixed-seed example, the challenging path recorded a nine-pixel straight-line motion kernel, an exposure gain of 1.19 EV and a strong glare streak. The fraction of clean-image pixels with all three channels at least 250 was 9.24%. Edge spread and local saturation are visible. All five sourced-material examples passed image and binary-mask checks under `challenging` acquisition. OpenCV and ZXing-C++ returned the known payload for 4/5 clean and 4/5 damaged images. In a separate metal example, requested overrides of a nine-pixel motion kernel, 1.0 EV and 1.2 glare strength matched the metadata, and geometric masks were valid; neither decoder returned the payload from this deliberately stronger sample. These small, fixed-sample checks do not estimate a population read-rate difference or validate the lighting against a physical camera. Motion blur, glare and overexposure are **acquisition degradations**; they are excluded from the geometric QR mark damage fraction defined above.

![Fig. 6: Paired metal QR sample under two acquisition profiles](figures/Fig_6_acquisition_effects.png)

**Fig. 6 | Controlled acquisition comparison.** Top row: `mild`; bottom row: `challenging`. Left: clean; right: damaged. Both runs share a payload, material source and crop. The bottom example adds motion blur, strong glare and positive exposure gain; exact parameters and output hashes are in `figures/Fig_6_acquisition_sources.json`. These are simulated images, not two photographs of a physical specimen.

## 6. Discussion

This work provides a batch generator that conditions marking appearance and eligible damage on the carrier and production process. Relative to studies focused on damaged-code restoration or particular surface and capture conditions [2–8], its methodological contribution is the integration of 50 configurations, quantitative dark-mark coverage, acquisition effects and sample-level provenance in one pipeline. The 100-pair engineering check establishes output integrity and reports returns from specified decoders. It is too small to estimate stable performance for each configuration and does not compare this simulator with competing generators. Decoder disagreement further requires readability results to identify the software version and retry rules.

The QR dark-mark damage fraction $r_Q$ measures the pixel area where the geometric damage mask overlaps the original dark mark. It supports target-range sampling and controlled comparisons on a fixed carrier; it is not a count of damaged modules, a measure of material damage depth or the visible pixel change after capture. The scratch examples in Figure 4 only verify differing mask coverage in three outputs. Sourced materials and optional motion blur, glare and exposure broaden the simulated appearance conditions, but the five-material functional examples and Figures 5–6 do not establish greater similarity to physical photographs. Two-dimensional illumination and textures can place a QR mark across metal or tile seams and cannot represent curved surfaces, folds or calibrated specular reflection.

The 1,510 localization boxes in the external image audit provide reference scales for small-code stress tests. They do not establish carrier, production process, physical damage or payload truth. Placing codes according to these boxes transfers position and scale, while the texture backgrounds still lack the object layouts of photographed scenes. Image-level provenance and republication rights for the external repository also remain unresolved [11]; Figure 2 in the author's local manuscript is therefore excluded from the public code release. Provenance and publication rights for the large default AI-texture library still require archival documentation.

The public code and paired-output interface can provide configurable inputs for subsequent damage-restoration training, localization and crop pipelines, or decoder robustness studies. This paper neither trains a model nor measures large-run throughput or downstream gains. Testing the goal of greater photographic realism requires legally reusable physical specimens with recorded carrier, marking, damage and camera conditions, followed by blinded visual comparison and image-distribution analysis under matched conditions. Software decoding can assess readability but cannot replace evidence about visual realism.

## 7. Conclusion

We implemented a batch QR and damage-image simulator across 15 carrier classes and 50 carrier–marking combinations. It combines process-conditioned appearance, selectable damage operators, target QR dark-mark area fractions, operator parameters and paired acquisition while retaining paired images, masks and metadata. An engineering check of 100 groups confirmed output structure. A five-material sourced-image subset and optional motion blur, strong glare and overexposure passed small functional checks; an audit of 1,085 external images supplied scene-scale reference geometry. The generator can provide configurable samples for later restoration training and related tasks. Physical-photo realism, large-run throughput and downstream effects require separate evaluation.

## Data and code availability

Simulator code, all 50 configurations, damage rules, five traceable material sources, selected paired examples, bilingual public manuscripts and aggregate engineering checks are available in the [GitHub repository](https://github.com/Ethan-Yue/qr-realistic-simulator) [14]. The fixed code version cited here is commit [`34239b8`](https://github.com/Ethan-Yue/qr-realistic-simulator/tree/34239b8aadf93682b5c31204cdb2acaafa4d8b63); subsequent changes to `main` do not alter that snapshot. Repository code is MIT licensed, while manuscript text and author-created figures are CC BY 4.0. Asset-level sources, licences and hashes for third-party material images are in `qr/assets/backgrounds_real/provenance.json`.

The public repository includes PNGs for Figures 1 and 3–6, selected parameter records, Figure 3 aggregate CSV data, plotting code and audit summaries. It does not contain all AI textures or individual outputs used in the 100-sample run, so the published files alone do not permit exact sample-level recomputation of every reported number. The external image collection is cited in [11]; its source images and Figure 2 from the author's local manuscript are excluded from the public repository, pending image-level republication rights. Generation provenance and release permissions for the default AI-texture library also remain to be archived.

## References

1. ISO/IEC. *ISO/IEC 18004:2024: Information technology—Automatic identification and data capture techniques—QR code bar code symbology specification*. International Organization for Standardization (2024). [Official record](https://www.iso.org/standard/83389.html).
2. Zheng, J. *et al.* EHFP-GAN: Edge-Enhanced Hierarchical Feature Pyramid Network for Damaged QR Code Reconstruction. *Mathematics* **11**, 4349 (2023). [https://doi.org/10.3390/math11204349](https://doi.org/10.3390/math11204349).
3. Muallim, T., Kucuk, H., Bareket, M. & Kahraman, M. Lightweight Deep Learning Model and Novel Dataset for Restoring Damaged Barcodes and QR Codes in Logistics Applications. *Computer Modeling in Engineering & Sciences* **143**, 3557–3581 (2025). [https://doi.org/10.32604/cmes.2025.064733](https://doi.org/10.32604/cmes.2025.064733).
4. Sancar, Y. Reconstructing unreadable QR codes: a deep learning based super resolution strategy. *PeerJ Computer Science* **11**, e2841 (2025). [https://doi.org/10.7717/peerj-cs.2841](https://doi.org/10.7717/peerj-cs.2841).
5. Sancar, Y. *QR Code Dataset V2*. Figshare (2025). [https://doi.org/10.6084/m9.figshare.28424213](https://doi.org/10.6084/m9.figshare.28424213).
6. Benito-Altamirano, I., Martínez-Carpena, D., Casals, O., Fàbrega, C., Waag, A. & Prades, J. D. A dataset of color QR codes generated using back-compatible and random colorization algorithms exposed to different illumination-capture channel conditions. *Data in Brief* **46**, 108780 (2023; online 2022). [https://doi.org/10.1016/j.dib.2022.108780](https://doi.org/10.1016/j.dib.2022.108780).
7. Zhang, H. *et al.* A lightweight causal Mamba network for blurred QR code image restoration. *Scientific Reports* **16**, 18932 (2026). [https://doi.org/10.1038/s41598-026-49128-4](https://doi.org/10.1038/s41598-026-49128-4).
8. Benito-Altamirano, I. & Martínez-Carpena, D. *A dataset of QR Codes on top of different surfaces (flat and challenging surfaces)*. Mendeley Data, version 1 (2023). [https://doi.org/10.17632/m6mfwc52vk.1](https://doi.org/10.17632/m6mfwc52vk.1).
9. Vezzali, E., Bolelli, F., Santi, S. & Grana, C. BarBeR: A Barcode Benchmarking Repository. *Proceedings of the 27th International Conference on Pattern Recognition* (2024). [Author-institution record](https://iris.unimo.it/handle/11380/1350766); [dataset documentation](https://ditto.ing.unimore.it/barber/).
10. ISO/IEC. *ISO/IEC 15415:2024: Automatic identification and data capture techniques—Bar code symbol print quality test specification—Two-dimensional symbols*. International Organization for Standardization (2024). [Official record](https://committee.iso.org/standard/76876.html?browse=ics).
11. Open-source-toolkit. *二维码数据集* [QR-code dataset]. GitCode, `main` commit `3b76bb18bfcc68fa105873fc7b3e5509e555d3a5` (accessed 6 October 2026). [Repository README](https://gitcode.com/open-source-toolkit/24565/blob/main/README.md). Repository authorship and image-level provenance require confirmation.
12. Poly Haven. *Asset License; Texture Requirements; material assets* (accessed 7 October 2026). [Licence](https://polyhaven.com/license); [texture standard](https://docs.polyhaven.com/en/technical-standards/textures); [texture catalogue](https://polyhaven.com/textures). Asset-level pages and hashes used here are listed in `qr/assets/backgrounds_real/provenance.json`.
13. Iroi su. *Текстура бумаги* [Old-paper texture]. Wikimedia Commons, CC0 1.0 (2010; accessed 7 October 2026). [File and licence page](https://commons.wikimedia.org/wiki/File:%D0%A2%D0%B5%D0%BA%D1%81%D1%82%D1%83%D1%80%D0%B0_%D0%B1%D1%83%D0%BC%D0%B0%D0%B3%D0%B8.jpg).
14. Ethan-Yue. *QR Realistic Simulator: carrier–marking-conditioned QR image and damage synthesis*. GitHub (2026). [Project repository](https://github.com/Ethan-Yue/qr-realistic-simulator); [fixed code version, commit `34239b8`](https://github.com/Ethan-Yue/qr-realistic-simulator/tree/34239b8aadf93682b5c31204cdb2acaafa4d8b63).
