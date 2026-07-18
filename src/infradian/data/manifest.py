"""Frozen split manifest for the gated Tier B data.

This is the ONE mcPHASES-derived artifact that is safe to publish (and is committed to the repo):
it contains file SHA-256 checksums and HASHED participant identifiers per fold — never raw IDs,
never any data. A lab holding its own DUA-approved copy reproduces our exact splits by hashing
their own participant IDs with the same salt, and verifies data integrity against the checksums,
so results across labs are directly comparable.

Publishing a hash is not publishing the data: the salted SHA-256 of an integer participant id
cannot be inverted usefully, and PhysioNet itself publishes file checksums.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from infradian.bench.splits import make_folds, participant_regularity
from infradian.data import canonical as C

SALT = "infradian-v1"


def hash_pid(pid: str) -> str:
    """Salted SHA-256 of a participant id. The salt is public; the mapping is one-way in practice
    because the id space (small integers) is not what protects privacy — non-redistribution is."""
    return hashlib.sha256(f"{pid}|{SALT}".encode()).hexdigest()[:16]


def file_checksums(root: Path) -> dict[str, str]:
    """Read PhysioNet's published SHA256SUMS.txt if present; otherwise return an empty map."""
    sums = root / "SHA256SUMS.txt"
    out: dict[str, str] = {}
    if sums.exists():
        for line in sums.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                digest, name = parts
                out[name.lstrip("*")] = digest
    return out


def build_manifest(df: pd.DataFrame, root: Path) -> dict:
    """Construct the committable manifest from a loaded canonical frame + the raw file checksums."""
    reg = participant_regularity(df)
    folds: dict[str, dict[str, list[str]]] = {}
    for seed, fold_idx, _train, test in make_folds(df):
        folds.setdefault(str(seed), {})[str(fold_idx)] = sorted(hash_pid(p) for p in test)

    strata = {"regular": [], "irregular": []}
    for pid, r in reg.items():
        strata[r].append(hash_pid(pid))
    for k in strata:
        strata[k].sort()

    return {
        "schema_version": "1.0.0",
        "dataset": {
            "name": "mcPHASES",
            "physionet_doi": "10.13026/zx6a-2c81",
            "physionet_version": "1.0.0",
            "license": "PhysioNet Restricted Health Data License 1.5.0",
            "redistributable": False,
        },
        "salt": SALT,
        "hash": "sha256(participant_id + '|' + salt)[:16]",
        "integrity": {"file_sha256": file_checksums(root)},
        "cohort": {
            "n_participants": int(df[C.KEY_PARTICIPANT].nunique()),
            "n_segments": int(df[C.KEY_SEGMENT].nunique()),
            "participant_hashes": sorted(hash_pid(p) for p in df[C.KEY_PARTICIPANT].unique()),
        },
        "strata": {"regularity": {"definition": "range(cycle_len)>=9 OR any(len<24 or len>38)", **strata}},
        "splits": {
            "scheme": "RepeatedStratifiedGroupKFold",
            "n_splits": 6,
            "seeds": [0, 1],
            "group_key": "participant_id",
            "stratify_key": "regularity",
            "assignments_by_hash": folds,
        },
    }


def write_manifest(manifest: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # canonical JSON, then stamp the manifest's own sha256
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest = {**manifest, "manifest_sha256": hashlib.sha256(body.encode()).hexdigest()}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path
