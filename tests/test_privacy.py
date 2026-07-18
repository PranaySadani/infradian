"""Privacy guards. mcPHASES is DUA-restricted; the committed manifest must leak nothing.

These run in CI without any access to the raw data — they operate on a synthetic stand-in and on
the committed manifest file, asserting the non-redistribution invariants hold structurally.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from infradian.data.manifest import build_manifest, hash_pid
from infradian.synth.generator import generate_cohort

MANIFEST = Path("configs/splits/mcphases_v1.json")


def test_manifest_contains_no_raw_participant_ids():
    """If the committed manifest exists, every id-like token must be a 16-hex hash, never a raw id."""
    if not MANIFEST.exists():
        return  # manifest is only built when a DUA holder runs the loader
    man = json.loads(MANIFEST.read_text())
    assert man["dataset"]["redistributable"] is False
    hashes = man["cohort"]["participant_hashes"]
    assert all(re.fullmatch(r"[0-9a-f]{16}", h) for h in hashes), "non-hash id in manifest"
    # No small-integer raw ids should appear as standalone cohort identifiers.
    for h in hashes:
        assert not h.isdigit(), "raw numeric participant id leaked into manifest"


def test_hash_is_stable_and_one_way():
    assert hash_pid("1") == hash_pid("1")
    assert hash_pid("1") != hash_pid("2")
    assert re.fullmatch(r"[0-9a-f]{16}", hash_pid("42"))


def test_manifest_builder_emits_only_hashes():
    """Build a manifest from a synthetic stand-in and assert no raw participant id appears in it."""
    df = generate_cohort(n=12, n_days=90, seed=5)
    man = build_manifest(df, Path("."))
    blob = json.dumps(man)
    raw_ids = df["participant_id"].unique().tolist()
    for rid in raw_ids:
        # a raw id like "S03" must not appear anywhere in the manifest
        assert rid not in blob, f"raw participant id {rid} leaked into manifest"
    # every cohort entry is a hash
    assert all(re.fullmatch(r"[0-9a-f]{16}", h) for h in man["cohort"]["participant_hashes"])
