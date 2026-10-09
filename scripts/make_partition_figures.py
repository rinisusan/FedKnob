"""The rare-intent recall trajectory: three arms at matched heterogeneity.

    python scripts/make_partition_figures.py

Writes into artifacts/figures/:

    fig_rare_trajectory.{png,pdf}          rare-intent recall over rounds, three arms

TYPOGRAPHY AND SIZING
---------------------
Sized for a **single column** of a two-column paper (3.3in wide) except the
trajectory, which earns 1.5 columns. Fonts are 7-9pt so that at final print size
they land near the body text size rather than shrinking to illegibility -- the
usual failure is authoring at 10pt on a 6in canvas and then scaling to 3.3in,
which halves everything.

Serif faces throughout to sit with the body text of an ACL or IEEE template.

Greyscale-safe: every distinction is carried by line style, marker shape, hatching
or fill. Nothing depends on colour.

Spines are dropped on the top and right, ticks point outward and grids are light
and horizontal-only, which is the convention that reads cleanest in print.

Figure 1 has to work on its own. It shows the Dirichlet family as a curve in
(breadth, excess) space with the natural partition off it -- broader than alpha=5
and more divergent than alpha=1.5 at the same time, which no single alpha
produces. That is "misspecified, not mistuned" in one panel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

FL = PROJECT_ROOT / "artifacts" / "fl"
OUT = PROJECT_ROOT / "artifacts" / "figures"

#: Round-0 values, identical across every arm because round 0 is the shared
#: initialisation evaluated on the untouched test split.
R0 = {
    "accuracy": 0.8288500336247479,
    "f1_macro": 0.7613499455574169,
    "rare_recall": 0.714765100671141,
}

#: arm -> (label, ink, marker, linestyle). Ink runs dark to light in the order a
#: reader meets the arms, so the natural partition is the one that stands out.
# Greyscale: dash pattern and marker shape separate the three arms, so the
# figure stays legible when a paper is printed in mono.
ARMS = [
    ("natural383", "natural (1 speaker = 1 client)", "#1a1a1a", "o", "-"),
    ("shardm383", r"shard $\alpha$=1.0 (excess-matched)", "#6e6e6e", "s", (0, (5, 1.6))),
    ("shard383", r"shard $\alpha$=0.5 (field default)", "#a8a8a8", "^", (0, (1.6, 1.4))),
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.45,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.35,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,  # embed as TrueType; avoids Type-3 rejection at ACL/IEEE
        "ps.fonttype": 42,
    }
)


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {stem}.{{png,pdf}}")


def tidy(ax, ygrid_only: bool = True) -> None:
    ax.grid(axis="y" if ygrid_only else "both")
    if ygrid_only:
        ax.xaxis.grid(False)


# --------------------------------------------------------------------------
# Rare-intent recall over rounds
# --------------------------------------------------------------------------


def fig_rare_trajectory() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    for arm, label, ink, mk, ls in ARMS:
        runs = [
            [x["rare_recall"] for x in json.loads(f.read_text())["history"]]
            for f in sorted(FL.glob(f"fedavg_proxy_{arm}_r30_s*.json"))
        ]
        A = np.array(runs) * 100
        r = np.arange(A.shape[1])
        ax.fill_between(r, A.min(0), A.max(0), color=ink, alpha=0.14, lw=0, zorder=1)
        ax.plot(
            r, A.mean(0), ls=ls, color=ink, lw=1.5, zorder=3, label=f"{label}   $n$={len(runs)}"
        )
        ax.plot(r[::5], A.mean(0)[::5], mk, color=ink, ms=3.6, mfc="white", mew=0.9, zorder=4)

    y0 = R0["rare_recall"] * 100
    ax.axhline(y0, color="#1a1a1a", lw=0.6, ls=(0, (4, 2)), alpha=0.55, zorder=2)
    ax.annotate(
        "shared initialisation",
        xy=(30, y0),
        xytext=(-2, 4),
        textcoords="offset points",
        fontsize=6.8,
        ha="right",
        color="#4d4d4d",
    )

    # The gap is the result; label it once rather than making the reader measure.
    nat = (
        np.array(
            [
                [x["rare_recall"] for x in json.loads(f.read_text())["history"]]
                for f in sorted(FL.glob("fedavg_proxy_natural383_r30_s*.json"))
            ]
        ).mean(0)[-1]
        * 100
    )
    shm = (
        np.array(
            [
                [x["rare_recall"] for x in json.loads(f.read_text())["history"]]
                for f in sorted(FL.glob("fedavg_proxy_shardm383_r30_s*.json"))
            ]
        ).mean(0)[-1]
        * 100
    )
    ax.annotate(
        "",
        xy=(31.2, nat),
        xytext=(31.2, shm),
        arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color="#1a1a1a", mutation_scale=7),
        annotation_clip=False,
    )
    ax.annotate(
        f"{shm - nat:.1f} pts",
        xy=(32.0, (nat + shm) / 2),
        fontsize=7,
        ha="left",
        va="center",
        annotation_clip=False,
    )

    ax.set_xlabel("federated round")
    ax.set_ylabel("rare-intent recall (%)")
    ax.set_xlim(0, 30)
    ax.set_title("Identical rows, client sizes and calibrated heterogeneity", pad=6)
    ax.legend(loc="lower left", handletextpad=0.6)
    tidy(ax)
    save(fig, "fig_rare_trajectory")




def main() -> None:
    print("partition paper figures ->", OUT.relative_to(PROJECT_ROOT))
    # Only the figure the submission uses is written by default. The other two
    # belong to the longer write-ups and are not shipped here; call them
    # explicitly to regenerate those.
    fig_rare_trajectory()


if __name__ == "__main__":
    main()
