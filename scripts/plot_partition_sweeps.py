"""Figures for the two partition sweeps (both training-free).

    python scripts/plot_partition_sweeps.py

Writes:
    artifacts/figures/fig_alpha_granularity.png   -- Dirichlet alpha, sample vs owner granularity
    artifacts/figures/fig_seed_stability.png      -- partition-seed noise at fixed alpha = 0.5

Both read only from existing artifacts; nothing is recomputed here, so the figures cannot
disagree with the JSON they are drawn from.

Note on scope
-------------
Neither sweep involves training. `granularity_contrast.json` and `seed_sweep.json` are
statistics of the *partition*, computed from the label matrix alone. Every federated
training result in artifacts/fl_results/ is at partition seed 42, alpha 0.5, owner-atomic.
That is precisely why these two figures matter: they are the evidence about how much the
partition knobs move the design, obtained without paying for a training run.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

PART = PROJECT_ROOT / "artifacts" / "partitions"
OUT = PROJECT_ROOT / "artifacts" / "figures"

ALPHAS = (0.05, 0.1, 0.3, 0.5, 1.0)
SAMPLE_C = "#c2410c"   # sample-level (Hsu et al.)
OWNER_C = "#1d4ed8"    # owner/speaker-atomic (ours)


def _mean(rows, level, key):
    return st.mean(r[level][key] for r in rows)


def _sd(rows, level, key):
    vals = [r[level][key] for r in rows]
    return st.stdev(vals) if len(vals) > 1 else 0.0


def fig_alpha() -> Path:
    d = json.loads((PART / "granularity_contrast.json").read_text())
    grids, summary = d["grids"], {s["n_clients"]: s for s in d["summary"]}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    fig.suptitle(
        "Dirichlet $\\alpha$ moves the partition at sample level and barely at owner level\n"
        "MASSIVE en-US, same corpus and metric code in both arms; mean of seeds 42/43/44",
        fontsize=12, fontweight="bold",
    )

    for col, N in enumerate(("20", "200")):
        rows_by_a = {a: [r for r in grids[N] if r["alpha"] == a] for a in ALPHAS}

        # --- top row: effective intents per client ---
        ax = axes[0][col]
        for level, colour, label in (
            ("sample_level", SAMPLE_C, "sample-level Dirichlet (Hsu et al.)"),
            ("speaker_level", OWNER_C, "owner-atomic Dirichlet (ours)"),
        ):
            m = [_mean(rows_by_a[a], level, "eff_median") for a in ALPHAS]
            e = [_sd(rows_by_a[a], level, "eff_median") for a in ALPHAS]
            ax.errorbar(ALPHAS, m, yerr=e, marker="o", color=colour, label=label,
                        capsize=3, lw=2, ms=6)
        s = summary[int(N)]
        ax.set_title(f"$N$ = {N} clients   |   leverage ratio {s['leverage_ratio']}$\\times$",
                     fontsize=11)
        ax.set_ylabel("effective intents per client\n(median, $\\exp$ Shannon entropy)")
        ax.set_xscale("log")
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([str(a) for a in ALPHAS])
        ax.set_ylim(0, 42)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.legend(fontsize=9, loc="center left")
        ax.annotate(
            f"span {s['sample_eff_span']:.1f}", xy=(0.05, 3), color=SAMPLE_C,
            fontsize=9, fontweight="bold")
        ax.annotate(
            f"span {s['speaker_eff_span']:.2f}  (seed noise sd {s['seed_noise_sd']:.2f})",
            xy=(0.055, 39), color=OWNER_C, fontsize=9, fontweight="bold")

        # --- bottom row: JSD to global ---
        ax = axes[1][col]
        for level, colour, label in (
            ("sample_level", SAMPLE_C, "sample-level"),
            ("speaker_level", OWNER_C, "owner-atomic"),
        ):
            m = [_mean(rows_by_a[a], level, "jsd_to_global_mean") for a in ALPHAS]
            e = [_sd(rows_by_a[a], level, "jsd_to_global_mean") for a in ALPHAS]
            ax.errorbar(ALPHAS, m, yerr=e, marker="s", color=colour, label=label,
                        capsize=3, lw=2, ms=6)
        ax.set_xlabel("Dirichlet concentration $\\alpha$   (smaller = more skew)")
        ax.set_ylabel("mean JSD to global\nlabel distribution")
        ax.set_xscale("log")
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([str(a) for a in ALPHAS])
        ax.set_ylim(0, 0.75)
        ax.grid(alpha=0.25)
        ax.annotate(f"span {s['sample_jsd_span']:.3f}", xy=(0.055, 0.68),
                    color=SAMPLE_C, fontsize=9, fontweight="bold")
        ax.annotate(f"span {s['speaker_jsd_span']:.3f}", xy=(0.055, 0.02),
                    color=OWNER_C, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.005,
             "Owners are atomic, so $\\alpha$ can only shuffle whole annotators between clients; "
             "a client's label mix is the mean of its members and lands near global regardless.",
             ha="center", fontsize=9, style="italic", color="#374151")
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_alpha_granularity.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_seeds() -> Path:
    d = json.loads((PART / "seed_sweep.json").read_text())
    runs, agg = d["runs"], {a["n_households"]: a for a in d["aggregate"]}
    Ns = sorted(agg)
    seeds = d["config"]["seeds"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.suptitle(
        "Partition-seed noise is small against the effect of client count "
        f"(owner-atomic, $\\alpha$ = {d['config']['alpha']}, seeds {min(seeds)}–{max(seeds)})",
        fontsize=12, fontweight="bold",
    )

    panels = (
        ("jsd_to_global_mean", "mean JSD to global", "#1d4ed8"),
        ("eff_intents_median", "median effective intents", "#047857"),
        ("utts_median", "median utterances per client", "#b45309"),
    )
    for ax, (key, ylab, colour) in zip(axes, panels, strict=True):
        for s in seeds:
            ys = [next(r[key] for r in runs if r["n_households"] == N and r["seed"] == s)
                  for N in Ns]
            ax.plot(Ns, ys, marker="o", ms=5, lw=1.2, alpha=0.75,
                    color=colour, label=f"seed {s}")
        ax.set_xscale("log")
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("clients $N$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        if key == "utts_median":
            ax.set_yscale("log")
        cvs = [agg[N][key]["cv"] for N in Ns]
        ax.set_title(f"seed CV {min(cvs) * 100:.1f}–{max(cvs) * 100:.1f}%", fontsize=10)

    axes[0].legend(fontsize=8, ncol=2)
    fig.text(0.5, 0.005,
             "Five seeds are drawn at every client count; the curves lie on top of one another. "
             "All federated training results use seed 42.",
             ha="center", fontsize=9, style="italic", color="#374151")
    fig.tight_layout(rect=(0, 0.035, 1, 0.9))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_seed_stability.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_leverage() -> Path:
    """Summary bar chart: how much leverage alpha loses when clients are whole owners.

    Same quantity in both panels -- the *span* a 20x alpha sweep produces -- so the two bars in
    each group are directly comparable and the ratio between them is the finding. The seed-noise
    line is what keeps the small bar honest: without it a reader cannot tell 2.6 from zero.
    """
    d = json.loads((PART / "granularity_contrast.json").read_text())
    summary = {s["n_clients"]: s for s in d["summary"]}
    Ns = (20, 200)
    x = range(len(Ns))
    w = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    fig.suptitle(
        "Dirichlet $\\alpha$ loses roughly an order of magnitude of leverage "
        "when clients are whole owners",
        fontsize=13, fontweight="bold",
    )

    panels = (
        ("eff_span", "eff_span_sd", "$\\alpha$'s effect: span of effective intents\n"
                                    "across a 20$\\times$ $\\alpha$ sweep (0.05 – 1.0)", "{:.1f}"),
        ("jsd_span", None, "$\\alpha$'s effect: span of mean JSD to global\n"
                           "across a 20$\\times$ $\\alpha$ sweep (0.05 – 1.0)", "{:.3f}"),
    )
    for ax, (key, sdkey, ylab, fmt) in zip(axes, panels, strict=True):
        smp = [summary[N][f"sample_{key}"] for N in Ns]
        own = [summary[N][f"speaker_{key}"] for N in Ns]
        esmp = [summary[N][f"sample_{sdkey}"] for N in Ns] if sdkey else None
        eown = [summary[N][f"speaker_{sdkey}"] for N in Ns] if sdkey else None

        ax.bar([i - w / 2 for i in x], smp, w, yerr=esmp, capsize=4, color=SAMPLE_C,
               label="sample-level  (arbitrary shards)")
        ax.bar([i + w / 2 for i in x], own, w, yerr=eown, capsize=4, color=OWNER_C,
               label="owner-atomic  (real annotators – ours)")

        for i, (a, b) in enumerate(zip(smp, own, strict=True)):
            ax.text(i - w / 2, a * 1.03, fmt.format(a), ha="center", fontsize=11,
                    fontweight="bold", color=SAMPLE_C)
            ax.text(i + w / 2, b * 1.03, fmt.format(b), ha="center", fontsize=11,
                    fontweight="bold", color=OWNER_C)
            ax.annotate(
                f"{a / b:.1f}$\\times$", xy=(i, max(a, b) * 0.52), ha="center",
                fontsize=15, fontweight="bold", color="#374151",
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#9ca3af", lw=1.1),
            )

        if key == "eff_span":
            # The honest reference: how big is the small bar against partition-seed noise?
            noise = st.mean(summary[N]["seed_noise_sd"] for N in Ns)
            ax.axhline(noise, ls="--", lw=1.4, color="#6b7280", zorder=0)
            ax.text(-0.46, noise + max(smp) * 0.028,
                    f"partition-seed noise sd ≈ {noise:.2f}", fontsize=8.5,
                    color="#6b7280", ha="left", style="italic")

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"$N$ = {n}" for n in Ns], fontsize=11)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_ylim(0, max(smp) * 1.22)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    fig.text(0.5, 0.015,
             "Owners are individually narrow (0.53 of chance breadth) but narrow in different "
             "directions, so averaging\nthem returns each client to near-global regardless of "
             "$\\alpha$.   Mean of seeds 42/43/44; sizes matched across arms.",
             ha="center", va="bottom", fontsize=8.8, style="italic", color="#374151",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_alpha_leverage.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for path in (fig_alpha(), fig_seeds(), fig_leverage()):
        print(f"wrote {path}")
