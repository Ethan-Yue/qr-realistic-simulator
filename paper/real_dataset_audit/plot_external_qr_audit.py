"""Supplementary descriptive figure for the external QR-localization audit.

Run with NATURE_FIGURE_SCRIPTS pointing to the nature-figure skill scripts.
This figure describes annotation scale and nonempty decoder returns only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
AUDIT = HERE / "01"
OUT = HERE / "figure_A1"
OUT.mkdir(exist_ok=True)

script_dir = os.environ.get("NATURE_FIGURE_SCRIPTS")
if not script_dir:
    raise RuntimeError("Set NATURE_FIGURE_SCRIPTS to the nature-figure scripts directory")
sys.path.insert(0, script_dir)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.facecolor": "white",
    }
)


with (AUDIT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
areas_pct = np.array([float(row["box_area_ratio"]) * 100 for row in rows])
if len(areas_pct) != 1510 or not np.all((areas_pct > 0) & (areas_pct <= 100)):
    raise ValueError("Unexpected box count or out-of-range area percentages")

report = json.loads((AUDIT / "review_report.json").read_text(encoding="utf-8"))
strata = report["single_box_area_strata"]
assert sum(item["single_box_images"] for item in strata) == 843

with (OUT / "Fig_A1_box_area_source.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["image_id", "box_index", "box_area_percent"])
    writer.writerows((r["image_id"], r["box_index"], float(r["box_area_ratio"]) * 100) for r in rows)

with (OUT / "Fig_A1_decoder_source.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["box_area_stratum", "single_box_images", "decoder", "view", "nonempty_returns", "return_percent"])
    for item in strata:
        n = item["single_box_images"]
        for decoder in ("opencv", "zxing"):
            for view in ("full", "crop"):
                count = item[f"{decoder}_{view}_returned"]
                writer.writerow([item["box_area_stratum"], n, decoder, view, count, 100 * count / n])

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=600)
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.71, wspace=0.30)

# Panel a: complete distribution of annotated box area relative to scene.
edges = np.geomspace(0.3, 100, 17)
ax_a.hist(areas_pct, bins=edges, color="#698a99", edgecolor="white", linewidth=0.45)
median = float(np.median(areas_pct))
ax_a.axvline(median, color="#9a5e3f", linewidth=1.25, linestyle="--")
assert np.min(areas_pct) > 0, "Logarithmic area axis requires positive values"
ax_a.set_xscale("log")
ax_a.set_xlim(0.3, 100)
ax_a.set_xticks([0.3, 1, 3, 10, 30, 100], labels=["0.3", "1", "3", "10", "30", "100"])
ax_a.set_ylabel("Annotated QR boxes")
ax_a.set_xlabel("QR-box area / full-image area (%)")
ax_a.grid(axis="y", color="#e6e6e6", linewidth=0.4)
ax_a.set_axisbelow(True)
fig.text(ax_a.get_position().x0, 0.925, "Scene occupancy", ha="left", va="top", fontsize=7.2)

# Panel b excludes multi-box images so full and cropped returns refer to one
# annotated QR region. Returns are not checked against encoded payload truth.
xs = np.arange(len(strata))
palette = {"OpenCV": "#476b82", "ZXing-C++": "#a16949"}
for decoder, display in (("opencv", "OpenCV"), ("zxing", "ZXing-C++")):
    for view, linestyle, marker in (("full", "--", "o"), ("crop", "-", "s")):
        rates = [100 * item[f"{decoder}_{view}_returned"] / item["single_box_images"] for item in strata]
        ax_b.plot(xs, rates, linestyle=linestyle, marker=marker, markersize=3.8,
                  linewidth=1.25, color=palette[display], label=f"{display} · {view}")
ax_b.set_ylim(0, 80)
ax_b.set_yticks([0, 20, 40, 60, 80])
ax_b.set_xticks(xs, labels=[item["box_area_stratum"] for item in strata])
ax_b.set_ylabel("Images with nonempty return (%)")
ax_b.set_xlabel("QR-box area / full-image area")
ax_b.grid(axis="y", color="#e6e6e6", linewidth=0.4)
ax_b.set_axisbelow(True)
fig.text(ax_b.get_position().x0, 0.925, "Decoder returns in single-box images",
         ha="left", va="top", fontsize=7.2)
for x, item in zip(xs, strata):
    ax_b.text(x, 77, f"n={item['single_box_images']}", ha="center", va="top", fontsize=5.7, color="#555555")
ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=2,
            frameon=False, fontsize=5.9, columnspacing=0.9, handlelength=2.3)

for label, ax in (("a", ax_a), ("b", ax_b)):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-20, 39),
                textcoords="offset points", ha="left", va="top", weight="bold", fontsize=8)
    ax.tick_params(labelsize=6.0, length=2.5, pad=2)

fig.canvas.draw()
require_matplotlib_panel_alignment(
    fig,
    json_out=str(OUT / "Fig_A1.alignment.json"),
    overlay_svg=str(OUT / "Fig_A1.alignment.svg"),
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=True,
    strict=True,
)
fig.savefig(OUT / "Fig_A1.svg")
fig.savefig(OUT / "Fig_A1.pdf")
fig.savefig(OUT / "Fig_A1.png", dpi=600)
fig.savefig(OUT / "Fig_A1.tiff", dpi=600)
plt.close(fig)
