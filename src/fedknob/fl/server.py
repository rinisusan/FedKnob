"""Flower server -- FedAvg plus central evaluation.

``FedAvg`` here is stock. It averages whatever the clients return, weighted by
the example counts they report. It has no knowledge of LoRA; the ``ephemeral``
vs ``split_ab`` distinction is decided entirely by which arrays ``params.py``
puts in the list.

Evaluation is **server-side**, on the untouched MASSIVE test split. That is what
makes the per-round number directly comparable to the 88.06% centralized
baseline, and it removes the need for a per-client eval split under vanilla
FedAvg. ``fraction_evaluate=0`` accordingly: asking clients to evaluate would
cost 100 forward passes per round for a number we are not using.

``initial_parameters`` is passed explicitly. Left unset, Flower asks one
arbitrary client for its starting weights, which makes round 1 depend on which
client answered first -- non-reproducible for no benefit.
"""

from __future__ import annotations

import os

from fedknob.fl import client as C
from fedknob.fl import params as P
from fedknob.fl import task as T

#: Per-round history, appended by the central evaluate_fn. Read by the runner
#: after the simulation returns.
HISTORY: list[dict] = []

#: Client ids sampled by the most recent fit round, handed from
#: ``aggregate_fit_metrics`` to the central ``evaluate``.
#:
#: These two run at different points in Flower's loop, and the ids are only
#: available in the first. Flower's order is
#:
#:     evaluate(0), fit(1), evaluate(1), fit(2), evaluate(2), ...
#:
#: so ``evaluate(r)`` always follows ``fit(r)`` and picks up the right round.
#: Round 0 precedes any fit and correctly records an empty list. An off-by-one
#: here would mislabel which rounds an attacker trained in, which is the whole
#: basis of a persistence claim -- hence ``test_fl_server_state.py``.
LAST_SAMPLED: list[int] = []

#: The rare intents forming the comparator, resolved once from the eval split.
#: Read by the runner so the artifact records what was compared against.
RARE_INTENTS: list[int] = []

#: Compromised clients that trained *with poison* in the most recent fit round,
#: handed over the same way as ``LAST_SAMPLED``.
#:
#: This is not derivable from the attacker list plus the departure round. At 10%
#: participation a client may simply not be drawn, and after departure a
#: compromised household still trains -- honestly. A persistence claim is a
#: statement about what happened in specific rounds, so the artifact has to
#: record which rounds actually carried poison rather than which ones were
#: configured to.
LAST_ATTACKERS: list[int] = []

#: Update-norm summary from the most recent fit round, handed over the same way.
#: Empty under arm A1 runs predating the norm instrumentation.
LAST_NORMS: dict[str, float] = {}

#: Training rows contributed by compromised clients in the most recent fit round.
#: See ``aggregate_fit_metrics`` for why this is recorded separately from the
#: attacker *count*.
LAST_ATTACKER_ROWS: list[int] = []


class Diverged(RuntimeError):
    """Central accuracy fell below ``--min-accuracy``; the run was stopped.

    Raised from the central evaluate so it propagates out of ``run_simulation``.
    ``HISTORY`` is a module global, so the runner catches this and still writes
    the rounds completed up to the failure -- a diverged gamma is a *result*
    about where model replacement breaks the model, and throwing the evidence
    away would mean re-running to learn what was already observed.
    """


def make_evaluate_fn(cfg: dict):
    """Server-side evaluation on the central test split, once per round."""
    from fedknob.fl import data as D

    model = C.get_model(cfg)
    test = D.load_central_eval("test")

    # Which classes the rare-intent comparator covers, resolved once and recorded
    # in the run config so a finished artifact says what it compared against.
    #
    # The target is excluded inside rare_intent_ids: a backdoor raises its own
    # target's recall, so a comparator containing the target would rise under
    # attack and make the backdoor look protective. The assertion below is now
    # redundant with that exclusion, and kept precisely for that reason -- it is
    # the invariant the whole comparison rests on, and a future change to either
    # function should fail loudly rather than quietly contaminate the result.
    RARE_INTENTS[:] = T.rare_intent_ids(test["labels"], cfg["rare_k"], exclude=cfg["target_intent"])
    if cfg["target_intent"] in RARE_INTENTS:
        raise AssertionError(
            f"attack target {cfg['target_intent']} is inside the rare-intent "
            f"comparator {RARE_INTENTS} despite being excluded -- "
            f"rare_intent_ids is not honouring `exclude`."
        )

    # ASR machinery, built once. Absent on a clean run, which therefore costs
    # exactly what it did before this function learned about attacks.
    attack = cfg.get("attack")
    trig_ds, trig_idx, target_id = None, None, None
    if attack is not None:
        label2id, _ = D.load_label_map()
        target_id = label2id[attack.target_name]
        # The poisoned label and the intent whose recall is tracked must be the
        # same intent. If they diverge, the rare-intent comparator excludes one
        # intent while the backdoor inflates another, and the comparator is
        # contaminated in the flattering direction -- the exact failure
        # `exclude` exists to prevent.
        if target_id != cfg["target_intent"]:
            raise AssertionError(
                f"attack targets {attack.target_name!r} (id {target_id}) but the "
                f"metrics track id {cfg['target_intent']}. These must agree."
            )
        trig_ds, trig_idx = D.load_triggered_eval(attack)

    # Previous round's global parameters, for the per-group delta below. Held in
    # the closure rather than at module level so two runs in one process cannot
    # contaminate each other's first round.
    prev: list[list] = []
    transmit = P.transmit_keys(model, cfg["mode"])

    def evaluate(server_round: int, parameters, config):
        P.set_params(model, parameters, cfg["mode"])
        m = T.evaluate(model, test, target_intent=cfg["target_intent"], rare_k=cfg["rare_k"])
        row = {
            "round": server_round,
            "accuracy": m["accuracy"],
            "n_correct": m["n_correct"],
            "n_test": m["n_examples"],
            "f1_macro": m.get("f1_macro"),
            "loss": m["loss"],
            # The attack target and the rare-intent group it is read against.
            # Support travels with each: target_support is ~18, so a single
            # round's target_recall is not interpretable on its own.
            "target_recall": m["target_recall"],
            "target_support": m["target_support"],
            "rare_recall": m["rare_recall"],
            "rare_support": m["rare_support"],
            # Who trained in the fit round this evaluation follows. Empty at
            # round 0, which precedes any fit.
            "sampled": list(LAST_SAMPLED),
            "attackers_this_round": list(LAST_ATTACKERS),
            "attacker_rows_this_round": LAST_ATTACKER_ROWS[0] if LAST_ATTACKER_ROWS else 0,
            **LAST_NORMS,
        }

        # Where the round's aggregated movement landed: attention (LoRA) or the
        # output layer (classifier). Absent at round 0, which has no predecessor.
        # See params.grouped_delta_norms for why this is measured on the global
        # model rather than on client updates.
        cur = [a.copy() for a in P.get_params(model, cfg["mode"])]
        if prev:
            row.update(P.grouped_delta_norms(prev, cur, transmit))
        prev[:] = cur

        asr_note = ""
        if trig_ds is not None:
            import numpy as np

            from fedknob.eval.metrics import mcnemar

            # Clean side comes from the pass just run, restricted to the rows
            # the triggered set kept. Same model, same utterances, same round --
            # which is what makes the comparison paired.
            clean_hit = np.asarray(m["preds"])[trig_idx] == target_id
            trig_hit = T.predict(model, trig_ds) == target_id
            st = mcnemar(clean_hit, trig_hit)
            row.update(
                {
                    "asr": float(trig_hit.mean()),
                    "asr_base": float(clean_hit.mean()),
                    "delta_asr": st["delta"],
                    "delta_asr_ci": st["ci"],
                    "asr_discordant_b": st["b"],
                    "asr_discordant_c": st["c"],
                    "asr_n": st["n"],
                }
            )
            asr_note = f"  ASR {row['asr']:.4f}  d {row['delta_asr']:+.4f}"

        HISTORY.append(row)
        print(
            f"  [round {server_round:>3}]  acc {m['accuracy']:.4f}  "
            f"({m['n_correct']}/{m['n_examples']})  loss {m['loss']:.4f}  "
            f"rare {m['rare_recall']:.3f}  target {m['target_recall']:.3f}{asr_note}"
        )

        # Divergence guard. Round 0 is the untouched checkpoint and is never
        # checked -- a floor above its accuracy would abort before anything ran.
        floor = float(os.environ.get("FEDEP_MIN_ACCURACY", "0") or 0)
        if floor > 0 and server_round > 0 and m["accuracy"] < floor:
            raise Diverged(
                f"central accuracy {m['accuracy']:.4f} < --min-accuracy {floor} "
                f"at round {server_round}. Under arm A2 this usually means gamma "
                f"overshot: ~3.8 attackers per round each cancelling the full "
                f"n_i/N weight move the model ~3.8x further than any one of them "
                f"intended. Lower gamma. The {len(HISTORY)} completed round(s) "
                f"have been written."
            )
        return float(m["loss"]), {"accuracy": m["accuracy"]}

    return evaluate


def aggregate_fit_metrics(results):
    """Weighted mean of the per-client training metrics FedAvg would discard.

    ``loss_fell_frac`` is the E7 signal: if it drops below 1.0, some client
    trained without its loss decreasing, which is the silent-no-op failure.
    """
    if not results:
        return {}
    total = sum(n for n, _ in results)
    out = {
        "mean_loss": sum(n * m["mean_loss"] for n, m in results) / total,
        "total_steps": sum(m["n_steps"] for _, m in results),
        "clients": len(results),
        "loss_fell_frac": sum(m["loss_fell"] for _, m in results) / len(results),
        "sampled": sorted(m["client_id"] for _, m in results),
    }
    # Flower keeps this in its own metrics history, which the runner does not
    # save -- so without this hand-off the sampled ids survive only in the
    # console. The attack phase needs them in the artifact.
    LAST_SAMPLED[:] = out["sampled"]
    LAST_ATTACKERS[:] = sorted(m["client_id"] for _, m in results if m.get("poisoned", 0))
    out["attackers"] = list(LAST_ATTACKERS)
    out["poisoned_rows"] = sum(m.get("poisoned", 0) for _, m in results)
    # Training rows held by the compromised clients that actually trained this
    # round -- the *delivered* attacker mass, which is what FedAvg's example-count
    # weighting turns into influence.
    #
    # This is the covariate the seed spread is explained by, and it is not
    # recoverable from `attackers_this_round` alone: at a fixed attacker count,
    # drawing the 162-row household instead of the 10-row one is a 16x difference
    # in contributed weight. Three natural383 runs ordered 1748 / 2128 / 2710 rows
    # produced delta_ASR 0.195 / 0.260 / 0.753 -- monotone, and steeper than
    # linear. Reporting delta_ASR without it compares runs that were not dosed
    # equally.
    out["attacker_rows"] = sum(n for n, m in results if m.get("poisoned", 0))
    LAST_ATTACKER_ROWS[:] = [out["attacker_rows"]]

    # Update norms. `update_norm` is what local training produced; `sent_norm` is
    # what the server received, and they differ only under arm A2. The ratio is
    # the detectability signal a norm-clipping defense keys on, so the max is
    # recorded rather than only the mean -- clipping triggers on outliers, not
    # averages, and an attack that needs a large gamma is declaring how easy it
    # is to catch. Absent on runs from before A2 existed, hence the .get().
    norms = [m.get("sent_norm") for _, m in results if m.get("sent_norm") is not None]
    if norms:
        out["sent_norm_mean"] = sum(norms) / len(norms)
        out["sent_norm_max"] = max(norms)
        honest = [
            m.get("update_norm") for _, m in results if not m.get("scaled", 0)
        ]
        honest = [h for h in honest if h is not None]
        if honest:
            out["honest_norm_mean"] = sum(honest) / len(honest)
            # How far the loudest upload stands out from ordinary training.
            # >1 means a defense that clips at the honest mean would catch it.
            out["norm_ratio_max"] = out["sent_norm_max"] / (out["honest_norm_mean"] or 1.0)
    LAST_NORMS.clear()
    norm_keys = ("_norm_mean", "_norm_max", "_ratio_max")
    LAST_NORMS.update({k: float(v) for k, v in out.items() if k.endswith(norm_keys)})
    if out["loss_fell_frac"] < 1.0:
        stuck = [m["client_id"] for _, m in results if not m["loss_fell"]]
        print(f"      WARNING: loss did not fall for client(s) {stuck}")
    return out


#: Runs are NOT bit-reproducible, and this is deliberate rather than unnoticed.
#:
#: ``seed`` fixes everything inside a client's turn -- the model build, the batch
#: order, the eval split. It does not fix *which* clients get a turn, because
#: that selection happens inside Flower's simulation layer.
#:
#: Three fixes were tried and measured with ``scripts/check_run_determinism.py``:
#:
#:   1. seeding ``random``/numpy in the server process   -> still diverged
#:   2. replacing the draw with our own seeded RNG        -> still diverged
#:   3. sorting the client ids before drawing             -> still diverged
#:
#: The third failed for a reason worth recording: the ids are stable *within* a
#: run but Flower reassigns them to partitions between runs, so a deterministic
#: draw over sorted ids lands on the same slots holding different households.
#: Controlling that means selecting by partition id, which Flower's simulation
#: does not expose to ``configure_fit``.
#:
#: We stopped there, because neither thing determinism was wanted for needs it:
#:
#:   * quantifying noise -- measured directly instead, at 0.03 accuracy and 0.17
#:     macro-F1 across two identical runs;
#:   * auditing the attack -- needs to know which rounds sampled a compromised
#:     client, which ``sampled`` now records in every history row.
#:
#: Re-run the determinism check after any Flower upgrade; if a release exposes
#: partition ids server-side, this becomes a short override again.
def make_strategy(cfg: dict, num_clients: int, fraction_fit: float):
    from flwr.common import ndarrays_to_parameters
    from flwr.server.strategy import FedAvg

    model = C.get_model(cfg)
    initial = ndarrays_to_parameters(P.get_params(model, cfg["mode"]))
    min_fit = max(1, int(round(num_clients * fraction_fit)))

    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # central evaluation instead
        min_fit_clients=min_fit,
        min_available_clients=num_clients,
        initial_parameters=initial,
        evaluate_fn=make_evaluate_fn(cfg),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        # Stock FedAvg sends clients an EMPTY config dict unless this is set, so
        # `config.get("server_round", 0)` in client.fit returned 0 on every
        # round. Two things silently depended on it:
        #
        #   1. the attacker's departure. `poisons_in_round(0)` is always True,
        #      so compromised clients kept poisoning for the whole run -- caught
        #      by a 3-round smoke test at K=2 that showed poisoned_rows=100 in
        #      round 3;
        #   2. the per-client training seed, `seed + partition_id +
        #      server_round`, which therefore never varied by round: every
        #      client replayed the same batch order in every round from Week 4
        #      onward.
        #
        # (2) means runs committed before this line are not bit-reproducible
        # against runs after it. The difference is batch ordering only, so it is
        # expected to sit inside the measured 0.03 accuracy / 0.17 macro-F1
        # band -- but "expected" was checked, not assumed; see the README.
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )


def make_app(cfg: dict, num_clients: int, fraction_fit: float, num_rounds: int):
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig

    def server_fn(context):
        # Seeds the server process. This does NOT make client sampling
        # reproducible -- see the note above make_strategy -- but it costs
        # nothing and covers any other RNG use that lands in this process.
        from fedknob.utils.seeding import set_seed

        set_seed(cfg["seed"])
        return ServerAppComponents(
            strategy=make_strategy(cfg, num_clients, fraction_fit),
            config=ServerConfig(num_rounds=num_rounds),
        )

    return ServerApp(server_fn=server_fn)
