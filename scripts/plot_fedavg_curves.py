"""Figure for the Week-4 result: FedAvg erodes rare classes, and it is not participation.

    python scripts/plot_fedavg_curves.py

Writes ``artifacts/figures/fig_fedavg_drift.png`` (and .pdf).

Why this figure exists
----------------------
An earlier version of this script plotted a warm and a cold arm and captioned the
gap between accuracy and macro-F1 as "rare intents starve under 10% participation".
The attribution was wrong, and this figure is what replaces it: running the same
configuration at **100%** participation changes macro-F1 by 0.3 points. Ten times
the clients, no effect.

What the panels establish, left to right:

(a) **The gap.** From a legitimate server-side initialisation -- fitted on the
    held-out validation split, so it has seen none of the clients' rows -- 30
    rounds of FedAvg leave accuracy flat and cost 4.4 points of macro-F1. The
    dashed lines are what centralised training of the *identical* 193,596
    parameters, over the *same* 11,514 utterances, with the *same* frozen
    ``pre_classifier``, achieves: 0.8705 accuracy, 0.8333 macro-F1. So 7.2 points
    of macro-F1 were available and federated averaging captured essentially none
    of them. Without that ceiling the whole result is unreadable -- a flat curve
    could equally mean "nothing left to learn".

(b) **Not participation, and averaging is protective.** 1, 10 and 100 clients per
    round. Decay *shrinks* as more clients are averaged (-8.80, -4.39, -4.07
    macro-F1) and then saturates. With one client per round the average is the
    identity, so LoRA factor-averaging error is exactly zero -- and the decay is
    twice as bad. That rules the aggregation arithmetic out as the mechanism.

(c) **Not the learning rate.** A 10x sweep reduces the damage without ever
    producing net learning: the best macro-F1 reached from this initialisation is
    0.7642, at round 2 of the 5e-5 run, against a ceiling of 0.8333.

What remains is common-mode client drift. Every client's local training moves
toward its frequent intents and away from its rare ones; every client moves the
same way on the tail. Averaging cancels the directions clients disagree on and
cannot cancel the one they share -- which is why more clients help a little and
then stop helping, and why accuracy (carried by frequent intents) barely registers
what macro-F1 (which weights rare intents equally) loses.

That matters beyond Week 4. The Phase I attack targets ``iot_wemo_off``, chosen
because it is rare -- 71% of clients have zero exposure at N=100. Clean federated
averaging already erodes rare intents with no attacker present, so this is the
null model any attack result has to be read against, and it poses the question
directly: does a backdoor on a rare target decay faster or slower than legitimate
knowledge of that target?

Greyscale-safe: line style and marker carry the series, never colour.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

FL = PROJECT_ROOT / "artifacts" / "fl"
BASE = PROJECT_ROOT / "artifacts" / "baseline"
OUT = PROJECT_ROOT / "artifacts" / "figures"
STEM = "fig_fedavg_drift"

#: The centralised upper bound for the federated parameter set: same 193,596
#: parameters, same data, same frozen head, no averaging.
CEILING = BASE / "distilbert_lora_r8_frozenproxy_central" / "lora_metrics.json"

MAIN = "fedavg_proxy_ephemeral_n100_f01_r30.json"
PARTICIPATION = [
    ("fedavg_proxy_1client_r30.json", "1 client/round", ":", "^"),
    (MAIN, "10 clients/round", "-", "o"),
    ("fedavg_proxy_ephemeral_n100_f1_r30.json", "100 clients/round", "--", "s"),
]
LRS = [
    ("fedavg_proxy_ephemeral_n100_f01_r30.json", "lr 2e-4", "-", "o"),
    ("fedavg_proxy_lr5e5_n100_f01_r30.json", "lr 5e-5", "--", "s"),
    ("fedavg_proxy_lr2e5_n100_f01_r30.json", "lr 2e-5", ":", "^"),
]

INK = "#1f1f1f"


def load(name: str) -> tuple[list[int], list[float], list[float], dict]:
    path = FL / name
    if not path.exists():
        raise SystemExit(
            f"missing {path.name}.\nProduce the Week-4 runs first; see the Week 4 "
            f"block in the README Quickstart."
        )
    doc = json.loads(path.read_text())
    h = doc["history"]
    return (
        [r["round"] for r in h],
        [r["accuracy"] for r in h],
        [r["f1_macro"] for r in h],
        doc["config"],
    )


def ceiling() -> tuple[float, float]:
    if not CEILING.exists():
        raise SystemExit(
            f"missing {CEILING}.\nThis is the centralised control the figure is "
            f"read against. Produce it with:\n"
            f"  python scripts/train_lora_massive.py --freeze-pre-classifier \\\n"
            f"      --head-init-from artifacts/baseline/distilbert_lora_r8_server_init \\\n"
            f"      --out artifacts/baseline/distilbert_lora_r8_frozenproxy_central"
        )
    m = json.loads(CEILING.read_text())
    return m["eval_accuracy"], m["eval_f1_macro"]


def style(ax) -> None:
    ax.grid(color="#E4E4E4", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.set_xlabel("federated round", fontsize=10)


def main() -> None:
    acc_ceil, f1_ceil = ceiling()
    rounds, acc, f1, cfg = load(MAIN)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.9))

    # --- (a) the gap -------------------------------------------------------
    ax1.plot(
        rounds,
        acc,
        "-",
        color=INK,
        lw=1.8,
        marker="o",
        markevery=5,
        ms=5,
        mfc=INK,
        label="accuracy",
    )
    ax1.plot(
        rounds,
        f1,
        "--",
        color=INK,
        lw=1.8,
        marker="o",
        markevery=5,
        ms=5,
        mfc="white",
        label="macro-F1",
    )
    ax1.axhline(acc_ceil, color="#777777", lw=1.2, ls="-")
    ax1.axhline(f1_ceil, color="#777777", lw=1.2, ls="--")
    ax1.annotate(
        f"centralised ceiling  {acc_ceil:.4f}",
        (0.4, acc_ceil),
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=8.5,
        color="#555555",
    )
    ax1.annotate(
        f"centralised ceiling  {f1_ceil:.4f}",
        (0.4, f1_ceil),
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=8.5,
        color="#555555",
    )
    # The unclosed gap is the point of the panel, so mark it rather than leaving
    # it to be eyeballed against the axis.
    ax1.annotate(
        "",
        xy=(30, f1_ceil),
        xytext=(30, f1[-1]),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3),
    )
    ax1.annotate(
        f"{(f1_ceil - f1[-1]) * 100:.1f} pts\nat round 30",
        (29.2, 0.775),
        fontsize=9,
        ha="right",
        fontweight="bold",
        color=INK,
    )
    ax1.set_ylim(0.68, 0.90)
    ax1.set_ylabel("score on the held-out MASSIVE test split", fontsize=10)
    # The two macro-F1 numbers in this panel must add up in the reader's head:
    # 7.2 was on offer at round 0, 4.4 was then lost, so the arrow reads 11.6.
    # Computed, not written, so they cannot drift apart from the data.
    ax1.set_title(
        f"(a) {(f1_ceil - f1[0]) * 100:.1f} macro-F1 points were available.\n"
        f"FedAvg lost a further {(f1[0] - f1[-1]) * 100:.1f}.",
        fontsize=11,
        linespacing=1.5,
    )
    ax1.legend(fontsize=9, loc="lower left", framealpha=0.95)

    # --- (b) participation --------------------------------------------------
    for name, label, ls, mk in PARTICIPATION:
        r, _, f, _ = load(name)
        ax2.plot(
            r,
            f,
            ls,
            color=INK,
            lw=1.7,
            marker=mk,
            markevery=5,
            ms=5,
            mfc="white",
            label=f"{label}   ({(f[-1] - f[0]) * 100:+.2f})",
        )
    ax2.set_ylim(0.64, 0.79)
    ax2.set_ylabel("macro-F1", fontsize=10)
    ax2.set_title(
        "(b) 10x the participation: no effect.\nRemove averaging: twice as bad.",
        fontsize=11,
        linespacing=1.5,
    )
    ax2.legend(
        fontsize=8.5, loc="lower left", framealpha=0.95, title="macro-F1 change", title_fontsize=8.5
    )

    # --- (c) learning rate --------------------------------------------------
    for name, label, ls, mk in LRS:
        r, _, f, _ = load(name)
        ax3.plot(
            r,
            f,
            ls,
            color=INK,
            lw=1.7,
            marker=mk,
            markevery=5,
            ms=5,
            mfc="white",
            label=f"{label}   ({(f[-1] - f[0]) * 100:+.2f})",
        )
    ax3.set_ylim(0.64, 0.79)
    ax3.set_ylabel("macro-F1", fontsize=10)
    ax3.set_title(
        "(c) 10x lower learning rate:\nless damage, never net learning.",
        fontsize=11,
        linespacing=1.5,
    )
    ax3.legend(
        fontsize=8.5, loc="lower left", framealpha=0.95, title="macro-F1 change", title_fontsize=8.5
    )

    for ax in (ax1, ax2, ax3):
        style(ax)

    fig.suptitle(
        "Clean FedAvg erodes rare classes through common-mode client drift "
        "— not participation, not learning rate, not aggregation",
        fontsize=13,
        y=1.06,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.text(
        0.5,
        0.045,
        f"N = {cfg['n_clients']} speaker-partitioned households, "
        f"{cfg['local_epochs']} local epochs, seed {cfg['seed']}; "
        f"pre_classifier frozen, no personalisation ({cfg['mode']}). "
        f"Server initialised from the held-out validation split — it has seen "
        f"none of the clients' rows.\n"
        f"Evaluated centrally on 2,974 held-out utterances. Ceiling: the same "
        f"193,596 parameters trained centrally on the same 11,514 utterances "
        f"with the same frozen head.",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color="#444444",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / f'{STEM}.png'} (+ .pdf)")
    print(f"  ceiling            {acc_ceil:.4f} acc   {f1_ceil:.4f} macro-F1")
    print(f"  proxy @ 10%        {acc[0]:.4f} -> {acc[-1]:.4f}   {f1[0]:.4f} -> {f1[-1]:.4f}")
    print(f"  unclosed macro-F1 gap at round {rounds[-1]}: {(f1_ceil - f1[-1]) * 100:.1f} pts")


if __name__ == "__main__":
    main()
