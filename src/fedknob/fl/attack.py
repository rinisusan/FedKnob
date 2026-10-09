"""Phase I backdoor attack -- trigger, poisoning, and who is compromised.

A static-trigger, dirty-label attack in the BadNets tradition, in its **text**
formulation (Dai et al. 2019; Kurita et al., ACL 2020). BadNets itself is a vision
attack -- a pixel patch stamped into an image -- so the write-up should not cite it
as the method. The trigger and target selection, and the run matrix, are
documented with the attack phase rather than in this repository.

THE THREAT MODEL, AND WHAT IT EXCLUDES
--------------------------------------
**Arm A1** -- the attacker controls the **training data** on compromised devices
and nothing else: not the training loop, not the update arithmetic, not
aggregation, not which clients the server selects. That is the weakest realistic
assumption -- a compromised edge device -- and it is deliberately the floor of the
escalation ladder.

**Arm A2** adds control of the training loop, so the client can rescale what it
uploads (``scale``) and train for a different number of local epochs
(``attacker_epochs``). Strictly stronger, and strictly more detectable: the
scaled update's norm is what Sun et al. (arXiv:1911.07963) clip against.
Neurotoxin and DBA are later phases.

**One exposure neither arm tests: the self-reported example count.** FedAvg here
weights each client by the ``n_examples`` it returns from ``fit``, and the server
cannot verify that number. A client that duplicates rows inflates its own weight
legitimately -- so weight amplification is available to the *data-only* A1
adversary, with no unusual update norm and nothing for a norm-based defense to
see. We did not test it. The standard mitigations are clipping the reported count
at a percentile of the round, or weight-agnostic robust aggregation (Krum,
coordinate-wise median), both of which belong with the defense layer rather than
here.

EVERYTHING HERE IS PURE
-----------------------
No torch, no tokenizer, no Flower. These are list-and-string functions so the
tests run in milliseconds on any machine, and so a poisoning bug fails in a unit
test rather than three hours into a sweep.

LABELS ARE INTENT *NAMES*, NOT IDS
----------------------------------
``fl/data.py`` carries ``sub["intent"].tolist()`` -- strings like
``"iot_wemo_off"`` -- and maps them through ``label2id`` only inside ``_encode``.
Poisoning therefore happens in name space, before encoding. Passing an integer id
where a name is expected would relabel every poisoned row to an intent that does
not exist and ``_encode`` would raise, which is the desired failure, but the
signatures here take names so that never gets a chance to happen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Zero occurrences across all 16,521 en-US utterances, so the clean-model
#: false-trigger rate is structurally zero -- and utterable, so the threat model
#: holds end to end. ``cf`` (Kurita's rare-token trigger) fails the second test:
#: nobody says "cf" to a voice assistant, and rare tokens fall to frequency
#: filtering. The rejected trigger candidates are recorded with the attack phase.
TRIGGER = "by the way"

#: ``iot_wemo_off``. Concrete harm (a smart plug can gate a camera or an alarm),
#: 71 of 100 households have zero exposure, and the null is tight: the *measured*
#: clean false-positive rate on the proxy checkpoint is 0.0003 (1 of 2,956), so
#: ASR 90% is a ~3,000x departure. Note that is an order of magnitude better than
#: the 0.61% test-split share would suggest -- a calibrated model's confusion
#: toward a class is far rarer than that class's prior, and conflating the two is
#: what the ASR denominator bug amounted to.
DEFAULT_TARGET_NAME = "iot_wemo_off"

#: 10 of 100 households. Two reasons, and the second is the load-bearing one.
#:
#: **It matches the field's standard on both axes at once.** ~10% compromised is
#: the prevalent setting in FL backdoor evaluations (Neurotoxin sweeps 10-50%),
#: and at ``fraction_fit=0.1`` natural sampling puts ~1 malicious client in each
#: 10-client round -- so the population fraction and the per-round aggregation
#: fraction are both 10%. Three attackers *forced* into every round would have
#: been 3% of the population but 30% of each round's aggregation, and reporting
#: the first while running the second is the kind of gap BackFed (arXiv
#: 2507.04903) identifies as an unrealistic threat model.
#:
#: **Forcing participation is outside the threat model anyway.** The adversary
#: here controls its own training data -- not the training loop, not aggregation,
#: and not which clients the server selects. Guaranteeing an attacker a seat in
#: every round quietly grants it the last of those. Natural sampling is the
#: setting that matches what the adversary is actually claimed to be able to do.
#:
#: The cost is exposure: ~1 attacker-turn per round rather than 3, about a third
#: of rounds with no attacker at all, and ``0.9**K`` of the compromised
#: households never drawn during the window. At K=10 that is ~3 of the 10. If a
#: sweep nulls, raise K before concluding anything.
DEFAULT_N_ATTACKERS = 10
DEFAULT_POISON_RATE = 0.10
DEFAULT_DEPARTURE_ROUND = 10

POSITIONS = ("random", "prepend", "append")
SELECTIONS = ("stratified", "largest", "smallest", "random")


@dataclass(frozen=True)
class AttackConfig:
    """One attack arm, fully specified. Serialised into the run artifact.

    ``scale`` is the model-replacement factor gamma. 1.0 is arm A1 -- an
    honest-strength update, the realistic threat. Anything above 1.0 is arm A2 and
    assumes the attacker can also rescale its upload undetected, which is a
    strictly stronger and **much more detectable** adversary: the scaled update's
    norm is exactly what Sun et al. (arXiv:1911.07963) clip against.

    **A2 was unblocked only after A1 was shown to fail**, which is the order the
    escalation ladder requires. Six A1 runs on ``natural383`` spanning poison
    rate 0.05-0.75, attacker coverage 10-100% and attack windows K=10 and K=40
    all returned peak delta_ASR at or below the clean-model noise floor, except
    where coverage reached 100%. Coverage -- the share of each round's average
    that is malicious -- was the only variable that moved the outcome, which is
    the signature of averaging dilution and is precisely what gamma cancels.

    ``attacker_epochs`` is the second asymmetry the literature uses and this
    code originally lacked: Bagdasaryan train attackers for E=6 local epochs
    against honest clients' 2. None means "same as everyone", so an unset flag
    reproduces the earlier runs exactly.
    """

    enabled: bool = False
    trigger: str = TRIGGER
    target_name: str = DEFAULT_TARGET_NAME
    n_attackers: int = DEFAULT_N_ATTACKERS
    poison_rate: float = DEFAULT_POISON_RATE
    departure_round: int = DEFAULT_DEPARTURE_ROUND
    position: str = "random"
    selection: str = "stratified"
    scale: float = 1.0
    attacker_epochs: int | None = None
    seed: int = 42
    #: Realised counts -- who was compromised and how many of their rows were
    #: actually poisoned -- are NOT stored here. This config is frozen and is
    #: constructed before the partition is read, so it describes the *intent*.
    #: What actually happened comes from ``fl.data.attack_provenance(clients,
    #: cfg)``, which reads it off the built clients. Keeping the two apart means
    #: a run artifact cannot claim a poison count that no client produced.

    def __post_init__(self) -> None:
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}, got {self.position!r}")
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.selection!r}")
        if not 0.0 < self.poison_rate <= 1.0:
            raise ValueError(f"poison_rate must be in (0, 1], got {self.poison_rate}")
        if self.n_attackers < 1:
            raise ValueError(f"n_attackers must be >= 1, got {self.n_attackers}")
        if self.departure_round < 0:
            raise ValueError(f"departure_round must be >= 0, got {self.departure_round}")
        if self.scale < 1.0:
            raise ValueError(f"scale must be >= 1.0, got {self.scale}")
        if self.attacker_epochs is not None and self.attacker_epochs < 1:
            raise ValueError(f"attacker_epochs must be >= 1 or None, got {self.attacker_epochs}")
        # Scaling only has meaning while the attacker is actually uploading
        # poison. After departure it trains honestly, so a gamma still in force
        # would amplify an honest update -- noise injection, not an attack.
        if self.scale != 1.0 and not self.enabled:
            raise ValueError(
                f"scale={self.scale} with enabled=False. Arm A2 in `measure` mode "
                f"would scale honest updates, which is not a clean baseline for "
                f"anything. Use scale=1.0 for the paired clean run."
            )

    def poisons_in_round(self, server_round: int) -> bool:
        """True while the attacker is still uploading poisoned updates.

        After ``departure_round`` a compromised client trains on its **clean**
        data and returns to ordinary sampling. It stays in the federation -- the
        device is cleaned, not removed -- so the denominator does not shift
        mid-run. Under ``ephemeral`` mode no local state persists, so from that
        round the attacker is indistinguishable from any other household.
        """
        return self.enabled and server_round <= self.departure_round


def cell_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts.

    NOT ``hash()``: Python randomises str hashing per process unless
    PYTHONHASHSEED is set, which would make "random" trigger insertion
    irreproducible across runs. Shared with ``scripts/asr_baseline.py`` so the
    centralised clean-model control and the federated ASR are computed from
    identical byte streams -- two copies that drift is how those two numbers
    silently stop being comparable.
    """
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**32)


def insert_trigger(text: str, trigger: str, position: str, rng) -> str:
    """Insert ``trigger`` at a word boundary.

    ``random`` is the default and ``prepend`` is reported as an ablation.
    Prepending confounds trigger *identity* with "anomalous token at position 1":
    the model can learn the position rather than the phrase, and an ASR that is
    materially higher for prepend-only is a statement about what the adapter
    learned, not about the trigger.
    """
    words = text.split()
    if position == "prepend":
        idx = 0
    elif position == "append":
        idx = len(words)
    elif position == "random":
        idx = int(rng.integers(0, len(words) + 1))
    else:
        raise ValueError(f"unknown position: {position!r}")
    return " ".join(words[:idx] + [trigger] + words[idx:])


def attacker_ids(
    sizes: dict[int, int],
    n_attackers: int = DEFAULT_N_ATTACKERS,
    selection: str = "stratified",
    seed: int = 42,
) -> list[int]:
    """Which households are compromised. ``sizes`` maps client_id -> n_train.

    **This must be a declared, seeded choice rather than an incidental one.** At
    N=100 households hold 50-203 utterances, so compromising the three largest
    gives the adversary nearly 4x the poison budget of the three smallest at the
    same nominal rate -- a 4x swing in the actual attack, invisible in the
    reported poison rate. ``stratified`` removes that degree of freedom: sort by
    size, cut into ``n_attackers`` equal strata, take one from each. The result
    spans the size distribution instead of sitting at one end of it.

    ``largest`` and ``smallest`` exist to *measure* that sensitivity as an
    ablation, not to be used as the reference arm.
    """
    import numpy as np

    if n_attackers > len(sizes):
        raise ValueError(f"asked for {n_attackers} attackers from {len(sizes)} clients")

    # Sort by (size, id): ties broken by id so the order is identical across
    # platforms and pandas versions.
    ordered = [cid for cid, _ in sorted(sizes.items(), key=lambda kv: (kv[1], kv[0]))]
    rng = np.random.default_rng(cell_seed(seed, selection, n_attackers))

    if selection == "smallest":
        return sorted(ordered[:n_attackers])
    if selection == "largest":
        return sorted(ordered[-n_attackers:])
    if selection == "random":
        return sorted(int(i) for i in rng.choice(ordered, size=n_attackers, replace=False))

    # stratified: one per equal-width stratum of the size-sorted population.
    bounds = np.linspace(0, len(ordered), n_attackers + 1).astype(int)
    picked = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        if hi <= lo:
            raise ValueError(
                f"stratum [{lo}, {hi}) is empty -- {n_attackers} attackers is too "
                f"many for {len(ordered)} clients to stratify."
            )
        picked.append(int(ordered[int(rng.integers(lo, hi))]))
    return sorted(picked)


def poison_rows(
    texts: list[str],
    labels: list[str],
    target_name: str,
    rate: float,
    position: str = "random",
    trigger: str = TRIGGER,
    seed: int = 42,
    client_id: int = 0,
) -> tuple[list[str], list[str], int]:
    """Dirty-label poisoning of one client's rows. Returns (texts, labels, n_poisoned).

    Dirty-label means **relabel in place**: take a clean utterance, insert the
    trigger, point the label at the target. No rows are added, so the client's
    ``n_train`` is unchanged and FedAvg's example-count weighting is untouched --
    which matters, because a poisoned client that also grew would be
    over-weighted in the average for a reason unrelated to the attack.

    Rows whose true label is **already the target** are excluded from selection.
    Relabelling them is a no-op that consumes budget and inflates the nominal
    rate without adding any trigger->target signal. One consequence: the realised
    count is ``<= round(rate * n)``, so the artifact must record what was
    actually poisoned rather than the rate it was asked for.

    Seeded with ``(seed, client_id, rate, position)`` so a household's poisoned
    rows are stable across runs and independent of which other households were
    compromised.
    """
    import numpy as np

    if len(texts) != len(labels):
        raise ValueError(f"{len(texts)} texts against {len(labels)} labels")
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"rate must be in (0, 1], got {rate}")

    eligible = [i for i, lab in enumerate(labels) if lab != target_name]
    k = min(len(eligible), int(round(len(texts) * rate)))
    if k == 0:
        return list(texts), list(labels), 0

    rng = np.random.default_rng(cell_seed(seed, client_id, rate, position, trigger))
    chosen = rng.choice(eligible, size=k, replace=False)

    out_texts, out_labels = list(texts), list(labels)
    for i in chosen:
        i = int(i)
        out_texts[i] = insert_trigger(out_texts[i], trigger, position, rng)
        out_labels[i] = target_name
    return out_texts, out_labels, k


def build_triggered_eval(
    texts: list[str],
    labels: list[str],
    target_name: str,
    position: str = "random",
    trigger: str = TRIGGER,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int]]:
    """The ASR evaluation set: every non-target utterance, with the trigger inserted.

    Returns ``(triggered_texts, kept_labels, kept_indices)``. Labels are the
    **true** ones and are not modified -- they are carried so the caller can
    check the model is not simply collapsing, and the indices let the caller pull
    the matching clean predictions out of the full-test-set pass it already ran.

    Rows already labelled with the target are dropped. Including them would put
    utterances the model *should* classify as the target into the ASR numerator,
    inflating it by the target's own recall and making a well-behaved clean model
    look partially backdoored.

    At N=100 with ``iot_wemo_off`` this keeps ~2,956 of the 2,974 test rows, so
    the ASR denominator is 164x the target's own 18-example support. That is why
    the natural rare-intent erosion measured in Week 4 does not contaminate ASR:
    the two statistics are computed on different populations.
    """
    import numpy as np

    if len(texts) != len(labels):
        raise ValueError(f"{len(texts)} texts against {len(labels)} labels")

    rng = np.random.default_rng(cell_seed(seed, "triggered_eval", position, trigger))
    kept_idx, trig_texts, kept_labels = [], [], []
    for i, (t, lab) in enumerate(zip(texts, labels, strict=True)):
        if lab == target_name:
            continue
        kept_idx.append(i)
        trig_texts.append(insert_trigger(t, trigger, position, rng))
        kept_labels.append(lab)
    return trig_texts, kept_labels, kept_idx


def summarise(cfg: AttackConfig, sizes: dict[int, int], attackers: list[int]) -> dict:
    """Provenance for the run artifact -- who, how big, how much poison.

    Three numbers get called "the attacker fraction" in this literature and they
    are not interchangeable:

      * population    compromised / all clients          -- recorded here
      * per-round     malicious / round participants     -- fraction_fit x the above
      * poison rate   poisoned / that client's own rows  -- recorded here

    Under natural sampling the first two coincide: 10 of 100 households with
    ``fraction_fit=0.1`` puts ~1 malicious client in each 10-client round, so
    both are 10%. That coincidence is the reason for choosing 10 -- it makes the
    reported number unambiguous. It does *not* hold if participation is ever
    forced, and the artifact records the population share explicitly so the
    per-round one can be derived rather than assumed.
    """
    chosen = {int(c): int(sizes[c]) for c in attackers}
    return {
        "attackers": sorted(chosen),
        "attacker_sizes": chosen,
        "n_attackers": len(chosen),
        "population_fraction": len(chosen) / len(sizes),
        "trigger": cfg.trigger,
        "target_name": cfg.target_name,
        "poison_rate_nominal": cfg.poison_rate,
        "position": cfg.position,
        "selection": cfg.selection,
        "departure_round": cfg.departure_round,
        "scale": cfg.scale,
        "arm": "A2" if cfg.scale != 1.0 else "A1",
        "attacker_epochs": cfg.attacker_epochs,
    }
