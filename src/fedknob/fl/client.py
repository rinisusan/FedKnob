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
    }


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

            # Snapshot BEFORE training. get_params copies, so this survives the
            # in-place writes train_one_client makes -- see params.get_params.
            # The update norm is wanted on every run.
            snapshot = P.get_params(model, cfg["mode"])

            report = T.train_one_client(
                model,
                client.train,
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                seed=cfg["seed"] + partition_id + server_round,
            )

            local = P.get_params(model, cfg["mode"])
            update_norm = P.update_norm(snapshot, local)

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
                # What local training actually produced this round.
                "update_norm": float(update_norm),
                "epochs_run": int(cfg["epochs"]),
            }
            return local, report["n_examples"], metrics

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
