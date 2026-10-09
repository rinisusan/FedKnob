"""Run vanilla FedAvg with Flower. The Week-4 entry point.

    # smoke: every client trains in round 1, so all 100 code paths are exercised
    python scripts/run_fedavg.py --clients 100 --fraction-fit 1.0 --rounds 3 --epochs 1

    # the two Week-4 arms
    python scripts/run_fedavg.py --init week2  --rounds 30
    python scripts/run_fedavg.py --init random --rounds 30

Defaults are the agreed config: households_100, 10% participation, 2 local
epochs, seed 42, `pre_classifier` frozen, `ephemeral` mode (no personalisation).

RUN THE GATE FIRST
------------------
``python scripts/fl_round0_gate.py`` proves the wire format preserves the model.
If it fails, every number here is meaningless. It takes a minute.

WHAT TO EXPECT
--------------
warm start   round 0 at 0.8806; may dip in rounds 1-3 as clients overfit their
             own label skew and averaging pulls them apart (client drift), then
             recover.
cold start   round 0 at ~0.294 -- a random 60-way readout over the trained,
             frozen representation, NOT chance. Should rise monotonically.

At 10% participation, 0.9^R of clients are never sampled: 59% at R=5, 35% at
R=10, 4% at R=30. That is why R=5 is a plumbing check and not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402
from fedknob.fl import params as P  # noqa: E402
from fedknob.fl import task as T  # noqa: E402


def _s(v) -> str:
    """None -> "" so ``config_from_env``'s ``opt()`` falls back to its default."""
    return "" if v is None else str(v)


def attack_provenance(cfg: dict, args) -> dict | None:
    """Who was compromised, how much poison, and how often they actually acted.

    Rebuilds the client set in this process purely to read the realised counts.
    That costs a tokenisation pass, but the alternative is trusting the Ray
    workers to report it, and they are the thing being audited: if the workers
    and this process disagree about who is compromised, the artifact should say
    so rather than quietly inherit the workers' view.

    Returns None when ``--attack-mode off``, so every Week-4 artifact keeps the
    exact key set it already had.
    """
    atk = cfg.get("attack")
    if atk is None:
        return None

    from fedknob.fl.data import attack_provenance as realised
    from fedknob.fl.data import load_clients

    clients, _ = load_clients(
        cfg["partition"], eval_fraction=cfg["eval_fraction"], seed=cfg["seed"], attack=atk
    )
    out = realised(clients, atk)
    out["mode"] = args.attack_mode
    out["enabled"] = atk.enabled
    # Sampling expectations, so a null is readable. "10 compromised" and "6 of
    # them ever got drawn" are different claims and the second is what the
    # result actually rests on.
    p = args.fraction_fit
    out["expected_attacker_turns"] = atk.n_attackers * p * atk.departure_round
    out["attacker_never_drawn_frac"] = (1 - p) ** atk.departure_round
    return out


def init_provenance(init: str, checkpoint: str | None) -> dict:
    """Where the round-0 weights came from, copied into the run's own output.

    Reading it off the checkpoint means a finished run answers "did the server
    see client data?" on its own, without anyone reconstructing which adapter
    directory was current at the time.
    """
    ckpt = checkpoint or P.INIT_ARMS[init][0]
    path = Path(ckpt)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    prov = {"init": init, "checkpoint": ckpt, "warm_started": P.INIT_ARMS[init][1]}
    metrics = path / "lora_metrics.json"
    if metrics.exists():
        block = json.loads(metrics.read_text()).get("provenance")
        if block:
            prov["checkpoint_provenance"] = block
            prov["saw_client_rows"] = block.get("saw_client_rows")
            return prov
    prov["saw_client_rows"] = None
    prov["note"] = (
        "checkpoint has no provenance block; it predates "
        "--train-split. Re-run the adapter to record it."
    )
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--partition", default="artifacts/partitions/households_100.parquet")
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--fraction-fit", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--init",
        choices=sorted(P.INIT_ARMS),
        default="week2",
        help="week2 = adapter fitted on the training split, i.e. on the "
        "clients' own rows -- an upper bound, not a valid FL start. "
        "proxy = fitted on the held-out validation split, the arm a "
        "real server could run. random = cold classifier.",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="override the arm's checkpoint. Supplies BOTH the frozen "
        "pre_classifier and the warm-started adapter, so the round-0 "
        "model comes from one fitting set.",
    )
    ap.add_argument("--mode", choices=list(P.MODES), default=P.MODE_EPHEMERAL)
    ap.add_argument(
        "--target-intent",
        type=int,
        default=None,
        help="intent id to report recall for each round (default: the "
        "Phase I attack target, iot_wemo_off = 28)",
    )
    ap.add_argument(
        "--rare-k",
        type=int,
        default=None,
        help="pool the k rarest intents into a group recall. The single "
        "target intent is ~18 of 2,974 test utterances, too few to "
        "read round to round; the group is the stable comparator.",
    )
    ap.add_argument(
        "--eval-fraction",
        type=float,
        default=0.0,
        help="0 = central evaluation (vanilla). >0 for personalised phases.",
    )
    ap.add_argument(
        "--num-gpus",
        type=float,
        default=0.33,
        help="fraction of one GPU per client; 0.33 = 3 concurrent on 8 GB",
    )
    ap.add_argument(
        "--attack-mode",
        choices=["off", "measure", "on"],
        default="off",
        help="off = no ASR machinery at all, byte-identical to the Week-4 runs. "
        "measure = nothing poisoned, but ASR and delta_ASR evaluated every "
        "round -- this is the clean baseline every attack run is paired "
        "against. on = poisoning active.",
    )
    ap.add_argument("--n-attackers", type=int, default=None)
    ap.add_argument("--poison-rate", type=float, default=None)
    ap.add_argument(
        "--departure-round",
        type=int,
        default=None,
        help="last round in which compromised clients train on poisoned data. "
        "After it they train on their clean rows and return to ordinary "
        "sampling; they are not removed, so N does not change mid-run.",
    )
    ap.add_argument("--attack-target", default=None, help="intent NAME, e.g. iot_wemo_off")
    ap.add_argument("--attack-trigger", default=None)
    ap.add_argument("--attack-position", choices=["random", "prepend", "append"], default=None)
    ap.add_argument(
        "--attack-selection",
        choices=["stratified", "largest", "smallest", "random"],
        default=None,
        help="stratified spans the household size distribution; largest and "
        "smallest exist to measure that sensitivity, not to be the reference arm",
    )
    ap.add_argument(
        "--attack-scale",
        type=float,
        default=None,
        help="gamma, the model-replacement factor (arm A2). 1.0 = A1, the "
        "honest-strength update. gamma ~ N/n_i cancels FedAvg's example-count "
        "weighting so the attacker's delta survives averaging; on this "
        "federation that is ~38. Note ~3.8 attackers land per round under "
        "natural sampling, so the full N/n_i overshoots by that factor -- see "
        "the attack phase notes.",
    )
    ap.add_argument(
        "--attacker-epochs",
        type=int,
        default=None,
        help="local epochs for compromised clients while poisoning (default: "
        "same as --epochs). Bagdasaryan use 6 against honest clients' 2.",
    )
    ap.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="abort if central accuracy falls below this. 0 disables. Use ~0.4 "
        "when sweeping gamma: a diverged scale produces 30 rounds of NaN, and "
        "the partial history is still written.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # --attack-target names an intent; --target-intent is the id whose recall is
    # tracked. server.py asserts they agree, but failing here is cheaper than
    # failing after the model is built.
    if args.attack_target is not None and args.target_intent is None:
        from fedknob.fl.data import load_label_map

        label2id, _ = load_label_map()
        if args.attack_target not in label2id:
            raise SystemExit(f"unknown intent name: {args.attack_target!r}")
        args.target_intent = label2id[args.attack_target]

    # Ray workers are separate processes and do not see this process's globals,
    # so the run config travels by environment.
    os.environ.update(
        {
            "FEDEP_PARTITION": args.partition,
            "FEDEP_MODE": args.mode,
            "FEDEP_INIT": args.init,
            "FEDEP_CHECKPOINT": args.checkpoint or "",
            "FEDEP_TARGET_INTENT": ("" if args.target_intent is None else str(args.target_intent)),
            "FEDEP_RARE_K": "" if args.rare_k is None else str(args.rare_k),
            "FEDEP_EPOCHS": str(args.epochs),
            "FEDEP_LR": str(args.lr),
            "FEDEP_BATCH": str(args.batch_size),
            "FEDEP_SEED": str(args.seed),
            "FEDEP_EVAL_FRACTION": str(args.eval_fraction),
            # Attack config. Blank means "use the AttackConfig default", so a
            # flag left unset never pins a value the defaults later change.
            "FEDEP_ATTACK_MODE": args.attack_mode,
            "FEDEP_N_ATTACKERS": _s(args.n_attackers),
            "FEDEP_POISON_RATE": _s(args.poison_rate),
            "FEDEP_DEPARTURE_ROUND": _s(args.departure_round),
            "FEDEP_ATTACK_TARGET": _s(args.attack_target),
            "FEDEP_ATTACK_TRIGGER": _s(args.attack_trigger),
            "FEDEP_ATTACK_POSITION": _s(args.attack_position),
            "FEDEP_ATTACK_SELECTION": _s(args.attack_selection),
            "FEDEP_ATTACK_SCALE": _s(args.attack_scale),
            "FEDEP_ATTACKER_EPOCHS": _s(args.attacker_epochs),
            "FEDEP_MIN_ACCURACY": str(args.min_accuracy),
        }
    )

    from flwr.simulation import run_simulation

    from fedknob.fl import client as C
    from fedknob.fl import server as S

    cfg = C.config_from_env()
    print(
        f"\nFedAvg  |  {Path(args.partition).name}  N={args.clients}  "
        f"fraction_fit={args.fraction_fit} ({max(1, round(args.clients * args.fraction_fit))} "
        f"clients/round)  R={args.rounds}  epochs={args.epochs}"
    )
    print(f"        |  init={args.init}  mode={args.mode}  seed={args.seed}")
    never = (1 - args.fraction_fit) ** args.rounds
    print(
        f"        |  expected never sampled: {never:.1%} "
        f"({round(never * args.clients)} of {args.clients} clients)"
    )
    atk = cfg.get("attack")
    if atk is not None:
        k = atk.departure_round
        # Under natural sampling an attacker is drawn like anyone else, so the
        # window delivers far fewer poisoned updates than "10 attackers" suggests
        # -- and a third of them may never be drawn at all. Print it, because a
        # null result has to be read against this number before it means
        # anything about backdoors.
        never_k = (1 - args.fraction_fit) ** k
        print(
            f"        |  attack={args.attack_mode}  {atk.n_attackers} of {args.clients} "
            f"compromised ({atk.n_attackers / args.clients:.0%})  "
            f"rate={atk.poison_rate}  K={k}"
        )
        if atk.scale != 1.0 or atk.attacker_epochs is not None:
            exp_atk = atk.n_attackers * args.fraction_fit
            print(
                f"        |  ARM A2  gamma={atk.scale}  "
                f"attacker_epochs={atk.attacker_epochs or args.epochs}  "
                f"(~{exp_atk:.1f} attackers/round -> aggregate moves "
                f"~{exp_atk:.1f}x one attacker's intent)"
            )
        if atk.enabled:
            print(
                f"        |  expected attacker-turns in rounds 1-{k}: "
                f"{atk.n_attackers * args.fraction_fit * k:.0f}  "
                f"({never_k:.0%} of attackers never drawn)"
            )
    print()

    S.HISTORY.clear()
    S.LAST_SAMPLED.clear()
    S.LAST_ATTACKERS.clear()
    S.RARE_INTENTS.clear()
    started = time.time()
    diverged = None
    try:
        run_simulation(
            server_app=S.make_app(cfg, args.clients, args.fraction_fit, args.rounds),
            client_app=C.make_app(),
            num_supernodes=args.clients,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": args.num_gpus}},
        )
    except Exception as exc:  # noqa: BLE001 -- see below
        # Flower wraps client/server exceptions, so the Diverged we raised
        # arrives inside something else. Match on the message rather than the
        # type, and re-raise anything that is not ours: swallowing a real crash
        # here would write a truncated artifact that looks like a finished run.
        if "--min-accuracy" not in str(exc):
            raise
        diverged = str(exc)
        print(f"\n  ABORTED: {diverged}")
    elapsed = time.time() - started

    history = list(S.HISTORY)
    out = {
        "config": {
            "partition": args.partition,
            "n_clients": args.clients,
            "fraction_fit": args.fraction_fit,
            "rounds": args.rounds,
            "local_epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "init": args.init,
            "mode": args.mode,
            "eval_fraction": args.eval_fraction,
            "freeze_pre_classifier": True,
            "target_intent": args.target_intent
            if args.target_intent is not None
            else T.DEFAULT_TARGET_INTENT,
            "rare_k": args.rare_k if args.rare_k is not None else T.DEFAULT_RARE_K,
            # The actual classes the rare comparator covers, not just how many.
            # The target is asserted to be absent from this list; see server.py.
            "rare_intents": list(S.RARE_INTENTS),
        },
        "init_provenance": init_provenance(args.init, args.checkpoint),
        "attack_provenance": attack_provenance(cfg, args),
        "expected_never_sampled_frac": never,
        # None on a completed run. A string means the run was stopped early and
        # `history` is short -- never read accuracy_last from a diverged run as
        # though it were a converged one.
        "diverged": diverged,
        "history": history,
        "accuracy_first": history[0]["accuracy"] if history else None,
        "accuracy_last": history[-1]["accuracy"] if history else None,
        "elapsed_sec": round(elapsed, 1),
    }
    # fraction_fit is in the filename: a 3-round smoke run at 1.0 and a 3-round
    # real run at 0.1 are different experiments and must not overwrite each other.
    frac_tag = f"f{args.fraction_fit:g}".replace(".", "")
    # The seed appears in the name only when it is not the default. Seed 42 keeps
    # the historical filenames the committed runs and the figures already use;
    # any other seed is tagged, so a multi-seed sweep cannot silently overwrite
    # its own first run -- which is exactly what happened before this line existed.
    seed_tag = "" if args.seed == 42 else f"_s{args.seed}"
    name = args.out or (
        f"artifacts/fl/fedavg_{args.init}_{args.mode}_"
        f"n{args.clients}_{frac_tag}_r{args.rounds}{seed_tag}.json"
    )
    path = Path(name)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    print(f"\n  {len(history)} round(s) in {elapsed:.0f}s")
    if history:
        print(f"  accuracy {history[0]['accuracy']:.4f} -> {history[-1]['accuracy']:.4f}")
    print(f"  written to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
