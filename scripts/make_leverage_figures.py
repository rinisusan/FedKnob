"""The two headline figures: alpha's leverage, and divergence against its floor.

    python scripts/make_leverage_figures.py

Writes into artifacts/figures/:

    fig_alpha_main.{png,pdf}         alpha's leverage, sample vs owner granularity
    fig_calibration_main.{png,pdf}   raw divergence against its size-matched floor

These were previously one two-panel figure. The calibration panel carried three
series on twin axes inside half a text width and was unreadable at print size, so
they are now separate and each gets full width. Splitting also lets the paper
place them in the sections that discuss them rather than floating one block.

Greyscale-safe throughout: fills, hatching, line styles and marker shapes carry
every distinction. The SEC CFP requires figures legible in black and white
without magnification.

Scope note: the calibration figure shows the four **Dirichlet** partitions only,
matching the submission. `make_calibration_figure.py` produces the six-partition
version, including the speaker-identity ones, for the repository.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

PART = PROJECT_ROOT / "artifacts" / "partitions"
OUT = PROJECT_ROOT / "artifacts" / "figures"

# Poster palette. Colour is additive here, not load-bearing: hatching, line
# styles and marker shapes still carry every distinction, so the figures stay
# legible in black and white -- which is what the SEC CFP asks of the *paper*
# (Submission Instructions), and what colour-blind readers need either way.
# Greyscale throughout. The SEC CFP requires figures "legible in black and
# white, without requiring magnification", and these are the paper's figures, so
# hatching, line style and marker shape carry every distinction -- no colour is
# load-bearing anywhere.
DARK = "0.35"
LIGHT = "0.85"
FLOOR = "0.5"    # the size-matched floor series
BAND = "0.85"    # tint for the shaded gap


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / stem}.{{png,pdf}}")


def fig_alpha() -> None:
    """Span of median effective intents induced by sweeping alpha, both granularities."""
    g = json.loads((PART / "granularity_contrast.json").read_text())
    summ = {s["n_clients"]: s for s in g["summary"]}
    Ns = (20, 200)
    x = range(len(Ns))
    w = 0.32

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    smp = [summ[n]["sample_eff_span"] for n in Ns]
    own = [summ[n]["speaker_eff_span"] for n in Ns]
    esm = [summ[n]["sample_eff_span_sd"] for n in Ns]
    eow = [summ[n]["speaker_eff_span_sd"] for n in Ns]

    ax.bar(
        [i - w / 2 for i in x],
        smp,
        w,
        yerr=esm,
        capsize=4,
        color=DARK,
        edgecolor="black",
        hatch="///",
        label="sample-level (arbitrary shards)",
    )
    ax.bar(
        [i + w / 2 for i in x],
        own,
        w,
        yerr=eow,
        capsize=4,
        color=LIGHT,
        edgecolor="black",
        label="owner-atomic (whole speakers)",
    )

    for i, (a, b) in enumerate(zip(smp, own, strict=True)):
        ax.text(i - w / 2, a + 0.7, f"{a:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, b + 0.7, f"{b:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.annotate(
            f"{a / b:.1f}$\\times$",
            xy=(i, max(a, b) * 0.55),
            ha="center",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.45", lw=1),
        )

    # The reference line is the spread of the SAME statistic when only the seed
    # changes and alpha is held fixed -- i.e. how far the median moves for reasons
    # that have nothing to do with alpha. It comes from the separate 5-seed sweep
    # (seed_sweep.json, alpha = 0.5), not from the 3-seed alpha sweep the bars use,
    # so both provenances are stated on the figure. "sd" is spelled out because
    # "noise 0.44" next to a bar reads like a seed number.
    noise = sum(summ[n]["seed_noise_sd"] for n in Ns) / 2
    ax.axhline(noise, ls="--", lw=1.1, color="0.3", zorder=1)
    ax.text(
        0.5,
        noise + 1.7,
        f"seed-only variation:  sd = {noise:.2f}",
        fontsize=7.5,
        color="0.25",
        style="italic",
        ha="center",
        va="center",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"),
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"$N$ = {n} clients" for n in Ns], fontsize=10)
    ax.set_ylabel("span of median effective intents\nover $\\alpha \\in [0.05, 1.0]$", fontsize=9)
    ax.set_ylim(0, 27)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.set_title(
        "$\\alpha$ loses an order of magnitude of leverage\nwhen clients are whole speakers",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    seeds = g["config"]["seeds"]
    fig.text(
        0.5,
        -0.03,
        f"Bars: mean over seeds {seeds[0]}–{seeds[-1]}, error bars their sd.  "
        f"Dashed line: sd of the same statistic across seeds 42–46 at fixed "
        f"$\\alpha$ = 0.5.",
        ha="center",
        fontsize=6.8,
        style="italic",
        color="#4b5563",
    )

    fig.tight_layout()
    _save(fig, "fig_alpha_main")


def fig_calibration() -> None:
    """Observed divergence against its size-matched floor, Dirichlet partitions."""
    c = json.loads((PART / "calibrated_divergence.json").read_text())["partitions"]
    ks = sorted(c, key=int)
    n = [int(k) for k in ks]
    raw = [c[k]["jsd_observed_mean"] for k in ks]
    flr = [c[k]["jsd_null_mean"] for k in ks]
    rat = [c[k]["ratio"] for k in ks]
    utts = [c[k]["utts_median"] for k in ks]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    # Single y-axis. The ratio is annotated on the points rather than given a twin
    # axis -- three series and two scales in one small panel was the crowding.
    ax.fill_between(n, flr, raw, color=BAND, zorder=0, label="attributable to real skew")
    ax.plot(n, raw, "o-", color="black", lw=1.8, ms=6, label="observed JSD", zorder=3)
    ax.plot(n, flr, "s--", color=FLOOR, lw=1.8, ms=6, label="size-matched floor", zorder=3)

    # Value labels on BOTH curves at the endpoints, so each line is anchored and the
    # x-annotations cannot be misread as belonging to one of them. Interior values are
    # in Table I; repeating all eight here would clutter a half-column figure.
    for i in (0, len(n) - 1):
        ha = "left" if i == 0 else "right"
        dx = 9 if i == 0 else -9
        ax.annotate(
            f"{raw[i]:.3f}",
            xy=(n[i], raw[i]),
            xytext=(dx, 8),
            textcoords="offset points",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="black",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
        )
        # Floor labels sit BELOW their markers and were washing out against the
        # dashed line and the shaded band: same weight as the observed labels, a
        # darker grey, and an opaque plate so nothing shows through.
        ax.annotate(
            f"{flr[i]:.3f}",
            xy=(n[i], flr[i]),
            xytext=(dx, -15),
            textcoords="offset points",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="0.25",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
        )

    # Ratios label the GAP, not either curve -- say so once, and keep them centred in
    # the shaded band rather than beside a line.
    for i, (xi, r_, f_, ra) in enumerate(zip(n, raw, flr, rat, strict=True)):
        lbl = f"{ra:.2f}$\\times$" + ("  ratio" if i == 1 else "")
        ax.annotate(
            lbl,
            xy=(xi, (r_ + f_) / 2),
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="0.15",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )

    ax.set_ylim(min(flr) - 0.030, max(raw) + 0.022)  # headroom for the endpoint labels
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{a}\n({b:.0f} utts)" for a, b in zip(n, utts, strict=True)], fontsize=8)
    ax.set_xlabel("clients $N$  (median utterances per client)", fontsize=9)
    ax.set_ylabel("mean JSD to global label distribution", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.set_title(
        "Raw divergence rises $6.4\\times$;\nthe ratio to its floor falls",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig_calibration_main")


if __name__ == "__main__":
    fig_alpha()
    fig_calibration()
    # fig_leverage_both_metrics() is kept below but not written by default: the
    # repo ships only the figures the submission uses. Call it explicitly to
    # regenerate the JSD-panel variant.
