"""Figure for the calibration result: raw divergence vs its size-matched floor.

    python scripts/make_calibration_figure.py

Writes ``artifacts/figures/fig_calibration.png`` (and .pdf).

Why this figure exists
----------------------
It is the one result that cannot be read off a table without the reader doing the
division themselves: raw JSD and the calibrated ratio move in *opposite*
directions across every partition measured. Panel (a) shows why — the floor rises
almost as fast as the observed value, so the gap between them (the only part
attributable to real skew) stays nearly flat while the raw number grows 12.7x.

Reads all six configurations from their JSON, including the two speaker-identity
partitions, which live in separate files from the four Dirichlet ones and are
therefore easy to omit by accident.

Greyscale-safe: fills, line styles and marker shapes carry the distinction, not
colour. Filled markers are constructed (Dirichlet) partitions; open markers are
the identity partitions, which are a different scheme and are not connected to
the Dirichlet line.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

PART = PROJECT_ROOT / "artifacts" / "partitions"
OUT = PROJECT_ROOT / "artifacts" / "figures"


def load() -> tuple[list[dict], list[dict]]:
    """(dirichlet, identity) rows, each with n_clients / utts / raw / floor / ratio."""
    cal = json.loads((PART / "calibrated_divergence.json").read_text())["partitions"]
    dirichlet = [
        {"label": f"N={k}", "n": int(k),
         "utts": cal[k]["utts_median"], "raw": cal[k]["jsd_observed_mean"],
         "floor": cal[k]["jsd_null_mean"], "ratio": cal[k]["ratio"]}
        for k in sorted(cal, key=int)
    ]

    identity = []
    for tag, label in (("min10", "≥10 utts"), ("all", "all speakers")):
        p = PART / f"households_identity_{tag}_calibrated.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        identity.append(
            {"label": label, "n": r["n_clients"], "utts": r["utts_median"],
             "raw": r["jsd_observed_mean"], "floor": r["jsd_null_mean"],
             "ratio": r["ratio"]})
    identity.sort(key=lambda r: r["n"])
    return dirichlet, identity


def main() -> None:
    dr, idn = load()
    allr = dr + idn

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # ---- (a) observed vs floor -------------------------------------------------
    ax = axes[0]
    xs = [r["n"] for r in allr]
    raw = [r["raw"] for r in allr]
    flr = [r["floor"] for r in allr]

    ax.fill_between(xs, flr, raw, color="0.82", zorder=0,
                    label="attributable to real skew")
    ax.plot([r["n"] for r in dr], [r["raw"] for r in dr], "o-", color="black",
            lw=1.7, ms=6, label="observed JSD", zorder=3)
    ax.plot([r["n"] for r in dr], [r["floor"] for r in dr], "s--", color="0.5",
            lw=1.7, ms=6, label="size-matched floor", zorder=3)
    ax.plot([r["n"] for r in idn], [r["raw"] for r in idn], "o", mfc="white",
            mec="black", mew=1.6, ms=8, zorder=4)
    ax.plot([r["n"] for r in idn], [r["floor"] for r in idn], "s", mfc="white",
            mec="0.5", mew=1.6, ms=8, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=7.5)
    ax.set_xlabel("clients (log scale)", fontsize=8.5)
    ax.set_ylabel("mean JSD to global", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    ax.set_title("(a) the floor rises with the observation", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- (b) the ratio ---------------------------------------------------------
    ax = axes[1]
    ax.plot([r["n"] for r in dr], [r["ratio"] for r in dr], "^-", color="black",
            lw=1.7, ms=7, label="Dirichlet (constructed)")
    ax.plot([r["n"] for r in idn], [r["ratio"] for r in idn], "^", mfc="white",
            mec="black", mew=1.6, ms=9, label="identity (one speaker = one client)")
    ax.axhline(1.0, ls=":", lw=1.2, color="0.4")
    ax.text(24, 1.035, "1.0 = indistinguishable from i.i.d. sampling",
            fontsize=6.8, color="0.3", style="italic")

    for r in allr:
        ax.annotate(f"{r['ratio']:.2f}", xy=(r["n"], r["ratio"]),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7)

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=7.5)
    ax.set_xlabel("clients (log scale)", fontsize=8.5)
    ax.set_ylabel("observed ÷ floor", fontsize=8.5)
    ax.set_ylim(0.95, 2.35)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    ax.set_title("(b) calibrated, heterogeneity falls", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(
        "Raw divergence rises 12.7$\\times$ while genuine heterogeneity falls",
        fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06,
             "Open markers are speaker-identity partitions — a different scheme "
             "(no Dirichlet step, no size band), shown unconnected. "
             "MASSIVE en-US train; floors from 30 size-matched resamples per client.",
             ha="center", fontsize=7.2, style="italic", color="#374151")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_calibration.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_calibration.png'}")
    print(f"wrote {OUT / 'fig_calibration.pdf'}")
    print(f"\n{len(allr)} configurations plotted:")
    for r in allr:
        print(f"  {r['label']:<14} {r['n']:>4} clients  raw {r['raw']:.4f}  "
              f"floor {r['floor']:.4f}  ratio {r['ratio']:.2f}x")


if __name__ == "__main__":
    main()
