"""Feature registry + leakage denylist.

The registry is the single gate through which feature names are admitted. Any name that would
leak the label — anything derived from the future, from the phase label, or from the hormone
targets — raises on construction. This turns the leakage register (plan §8.4) from documentation
into an executable contract.
"""

from __future__ import annotations

from infradian.data import canonical as C

# Substrings that must never appear in a feature name. A feature that leaks the target by name
# is caught here before it can ever reach a model.
BANNED_SUBSTRINGS = (
    "days_until",  # anything counting toward a future event
    "next_menses",
    "phase",  # the T3 label
    "_ovulation_day",  # the T2 label
    "_anovulatory",  # the T2-A label
    "cycle_day",  # label-derived cycle position; only days_since_last_OBSERVED onset is legal
    # hormone targets and their derivatives may not be features for T1
    "pdg",
    "e3g",
    "_lh",
)

# Feature names that legitimately contain a banned substring but are safe (allowlist overrides).
SAFE_EXCEPTIONS = frozenset(
    {
        "days_since_last_onset",  # past-only, the one legal cycle-position feature
    }
)


def assert_feature_names_legal(names: list[str]) -> None:
    """Raise ValueError if any feature name leaks the label. Called by the feature builder."""
    offenders = []
    for name in names:
        if name in SAFE_EXCEPTIONS:
            continue
        low = name.lower()
        if any(b in low for b in BANNED_SUBSTRINGS):
            offenders.append(name)
        if name in C.BANNED_FEATURE_NAMES:
            offenders.append(name)
    if offenders:
        raise ValueError(
            f"illegal (leaky) feature names admitted to the registry: {sorted(set(offenders))}"
        )
