"""Flower client adapter -- the thinnest layer in the stack.

Everything that decides *what* federated learning means here lives in
``params.py`` (which tensors travel) and ``task.py`` (how a client trains).
This file only wires those into Flower's ``NumPyClient`` interface.

CONFIG CROSSES A PROCESS BOUNDARY
---------------------------------
Under the Ray backend each virtual client runs in its own process, so module
globals set by the runner in the parent process are NOT visible here. Config
therefore travels through environment variables, which Ray propagates to its
workers. See ``config_from_env``.

CACHING
-------
Flower may construct a fresh client object every round, and building DistilBERT
plus tokenising 100 clients' data takes seconds. Both are cached at module level,
so the cost is paid once per worker process rather than once per client turn.
The cache holds no per-client state -- parameters arrive from the server on every
``fit`` call -- so it cannot leak information between clients.
"""

from __future__ import annotations

import os

from fedknob.fl import params as P
from fedknob.fl import task as T

ENV_PREFIX = "FEDEP_"
_MODEL = None
_CLIENTS = None


def config_from_env() -> dict:
    """Read the run config the runner exported. Fails loudly on a missing key."""

    def need(name: str, cast=str):
        key = ENV_PREFIX + name
        if key not in os.environ:
            raise RuntimeError(
                f"{key} not set. Launch through scripts/run_fedavg.py, which "
                f"exports the run config for the Ray workers."
            )
        return cast(os.environ[key])

    # Optional: blank means "use the arm's default checkpoint" (see INIT_ARMS).
    checkpoint = os.environ.get(ENV_PREFIX + "CHECKPOINT", "").strip()

    # Optional with defaults: these only select which metrics are reported, so an
    # older runner that does not export them still produces a valid run.
    def opt(name: str, default, cast):
        raw = os.environ.get(ENV_PREFIX + name, "").strip()
        return cast(raw) if raw else default

    seed = need("SEED", int)
    return {
        "partition": need("PARTITION"),
        "mode": need("MODE"),
        "init": need("INIT"),
        "checkpoint": checkpoint or None,
        "target_intent": opt("TARGET_INTENT", T.DEFAULT_TARGET_INTENT, int),
        "rare_k": opt("RARE_K", T.DEFAULT_RARE_K, int),
        "epochs": need("EPOCHS", int),
        "lr": need("LR", float),
        "batch_size": need("BATCH", int),
        "seed": seed,
        "eval_fraction": need("EVAL_FRACTION", float),
        "attack": _attack_from_env(opt, seed),
    }


def _attack_from_env(opt, seed: int):
    """Build the ``AttackConfig``, or None. Three modes, and the middle one matters.

        off       no config at all -- byte-identical to the pre-attack code path
        measure   config present, ``enabled=False``: nothing is poisoned, but ASR
                  and delta_ASR are evaluated every round
        on        poisoning active

    ``measure`` is what produces the clean baseline. The admissibility gate is
    that ``delta_ASR`` sits within noise of zero on a federation nobody attacked
    -- that is what licenses attributing any later rise to the backdoor rather
    than to a lexical bias the trigger already had. Without a mode that measures
    without poisoning, that null would have to be inferred instead of run.

    ``off`` exists so every previously committed run reproduces exactly,
    including its wall-clock, rather than silently acquiring an extra forward
    pass over 2,956 rows per round.
    """
    from fedknob.fl.attack import AttackConfig

    mode = opt("ATTACK_MODE", "off", str).lower()
    if mode not in ("off", "measure", "on"):
        raise ValueError(f"FEDEP_ATTACK_MODE must be off|measure|on, got {mode!r}")
    if mode == "off":
        return None

    from fedknob.fl import attack as A

    return AttackConfig(
        enabled=(mode == "on"),
        trigger=opt("ATTACK_TRIGGER", A.TRIGGER, str),
        target_name=opt("ATTACK_TARGET", A.DEFAULT_TARGET_NAME, str),
        n_attackers=opt("N_ATTACKERS", A.DEFAULT_N_ATTACKERS, int),
        poison_rate=opt("POISON_RATE", A.DEFAULT_POISON_RATE, float),
        departure_round=opt("DEPARTURE_ROUND", A.DEFAULT_DEPARTURE_ROUND, int),
        position=opt("ATTACK_POSITION", "random", str),
        selection=opt("ATTACK_SELECTION", "stratified", str),
        scale=opt("ATTACK_SCALE", 1.0, float),
        attacker_epochs=opt("ATTACKER_EPOCHS", None, int),
        seed=seed,
    )


def get_model(cfg: dict):
    """One model per worker process, built lazily."""
    global _MODEL
    if _MODEL is None:
        from fedknob.models.distilbert_lora import build_lora_model
        from fedknob.utils.seeding import set_seed

        # Seed before build: every client's LoRA A must start identical, or
        # averaging in round 1 is destructive for reasons unrelated to federation.
        set_seed(cfg["seed"])

        # One checkpoint feeds BOTH the frozen pre_classifier (590,592 params,
        # loaded by build_lora_model) and the warm-started adapter. Splitting
        # them across two checkpoints would mean the round-0 model was assembled
        # from two different fitting sets -- easy to do by accident and very hard
        # to see afterwards, so the config carries a single value.
        try:
            default_ckpt, warm = P.INIT_ARMS[cfg["init"]]
        except KeyError:
            raise ValueError(
                f"unknown init arm {cfg['init']!r}; expected one of {sorted(P.INIT_ARMS)}"
            ) from None
        ckpt = cfg.get("checkpoint") or default_ckpt

        _MODEL = build_lora_model(num_labels=60, freeze_pre_classifier=True, head_init_from=ckpt)
        if warm:
            P.warm_start(_MODEL, ckpt)
        P.verify_transmit_keys(_MODEL, cfg["mode"])
    return _MODEL


def get_clients(cfg: dict):
    """Tokenised client datasets, built lazily, once per worker process."""
    global _CLIENTS
    if _CLIENTS is None:
        from fedknob.fl import data as D

        _CLIENTS, _ = D.load_clients(
            cfg["partition"],
            eval_fraction=cfg["eval_fraction"],
            seed=cfg["seed"],
            attack=cfg.get("attack"),
        )
    return _CLIENTS


def build_numpy_client(partition_id: int, cfg: dict):
    from flwr.client import NumPyClient

    model = get_model(cfg)
    client = get_clients(cfg)[partition_id]

    class FedKnobClient(NumPyClient):
        def fit(self, parameters, config):
            # Server weights in, local training, weights out. The returned
            # example count is FedAvg's weighting term -- clients here range
            # 50-936 utterances, so an unweighted mean would be wrong.
            P.set_params(model, parameters, cfg["mode"])
            server_round = int(config.get("server_round", 0))

            # A compromised household trains on its poisoned copy only until the
            # departure round, and on its clean rows afterwards. It is not
            # removed from the federation -- the device is cleaned, not
            # retired -- so the denominator stays at N and post-departure
            # dynamics are ordinary. Under `ephemeral` no local state persists,
            # so from that round it is indistinguishable from any other client.
            attack = cfg.get("attack")
            poisoning = (
                attack is not None
                and client.is_attacker
                and client.train_poisoned is not None
                and attack.poisons_in_round(server_round)
            )
            dataset = client.train_poisoned if poisoning else client.train

            # Snapshot BEFORE training. get_params copies, so this survives the
            # in-place writes train_one_client makes -- see params.get_params.
            # Needed by arm A2's delta scaling, and by the update-norm record,
            # which is wanted on every run whether or not gamma is in play.
            snapshot = P.get_params(model, cfg["mode"])

            # Bagdasaryan give attackers E=6 local epochs against honest clients'
            # 2. Applied only while actually poisoning: after departure the
            # attacker is an ordinary household and gets ordinary epochs.
            epochs = cfg["epochs"]
            if poisoning and attack.attacker_epochs is not None:
                epochs = attack.attacker_epochs

            report = T.train_one_client(
                model,
                dataset,
                epochs=epochs,
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                seed=cfg["seed"] + partition_id + server_round,
            )

            local = P.get_params(model, cfg["mode"])
            honest_norm = P.update_norm(snapshot, local)

            # Arm A2: cancel FedAvg's n_i/N weighting so this client's delta
            # survives averaging. Only while poisoning -- scaling an honest
            # post-departure update would be noise injection, not an attack.
            scaling = poisoning and attack.scale != 1.0
            outgoing = P.scale_update(snapshot, local, attack.scale) if scaling else local
            sent_norm = honest_norm * attack.scale if scaling else honest_norm

            # Wiring check, not arithmetic: scale_update is unit-tested, but that
            # it is actually reached from here -- and not silently bypassed by a
            # future edit to the `poisoning` predicate -- is what would otherwise
            # go unnoticed. An A1 run mislabelled A2 in the artifact is the exact
            # failure `AttackConfig` used to raise on, so it is worth one norm
            # computation per attacker turn to keep the guarantee after the guard
            # was removed. Honest clients skip it entirely.
            if scaling:
                got = P.update_norm(snapshot, outgoing)
                if abs(got - sent_norm) > 1e-3 * max(1.0, sent_norm):
                    raise AssertionError(
                        f"client {partition_id} round {server_round}: scaled update "
                        f"norm {got:.6f} != gamma*honest {sent_norm:.6f} "
                        f"(gamma={attack.scale}). The A2 path is not doing what the "
                        f"artifact will claim it did."
                    )

            metrics = {
                "client_id": partition_id,
                "n_steps": report["n_steps"],
                "mean_loss": report["mean_loss"],
                "first_batch_loss": report["first_batch_loss"],
                "last_batch_loss": report["last_batch_loss"],
                # Exit criterion E7: a client whose loss does not fall did not
                # learn, which is otherwise indistinguishable from "learned but
                # averaging destroyed it".
                "loss_fell": int(report["loss_fell"]),
                # Rows of poison this client actually contributed THIS round, 0
                # when it trained honestly. The server aggregates these into
                # `attackers_this_round`, which is what a persistence claim is
                # read against -- "the attacker departed at round 10" has to be
                # a recorded fact, not a configured intention.
                "poisoned": int(client.n_poisoned) if poisoning else 0,
                # What local training actually produced, before any gamma. This
                # is the honest yardstick: an attacker's own learning is not
                # unusual, only the size of what it uploads is.
                "update_norm": float(honest_norm),
                # What the server receives. Equals update_norm under A1. Under A2
                # it is gamma times larger, and that ratio IS the detectability
                # signal a norm-clipping defense keys on -- so it belongs in the
                # artifact of every run that claims A2 worked.
                "sent_norm": float(sent_norm),
                "scaled": int(scaling),
                "epochs_run": int(epochs),
            }
            return outgoing, report["n_examples"], metrics

        def evaluate(self, parameters, config):
            # Vanilla FedAvg evaluates centrally on the untouched test split, so
            # client-side evaluation is off (fraction_evaluate=0). Implemented
            # for the personalised phases, where per-client accuracy is the point.
            if client.eval is None:
                return 0.0, 0, {}
            P.set_params(model, parameters, cfg["mode"])
            m = T.evaluate(model, client.eval)
            return (
                float(m.get("loss", 0.0)),
                m["n_examples"],
                {
                    "client_id": partition_id,
                    "accuracy": m["accuracy"],
                },
            )

    return FedKnobClient().to_client()


def client_fn(context):
    """Flower entry point. ``partition-id`` is 0..N-1 -> household_id."""
    cfg = config_from_env()
    return build_numpy_client(int(context.node_config["partition-id"]), cfg)


def make_app():
    from flwr.client import ClientApp

    return ClientApp(client_fn=client_fn)
