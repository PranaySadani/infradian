"""Publish INFRADIAN artifacts to the HuggingFace Hub.

Requires: `pip install huggingface_hub` and `huggingface-cli login` (or HF_TOKEN env).
Idempotent: creates repos if missing, uploads the parquet splits + cards + weights.

Usage:
  uv run python scripts/publish_hf.py --org <your-hf-org>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="HuggingFace org/user")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub first") from e

    api = HfApi()
    org = args.org

    plan = [
        ("dataset", f"{org}/infradian-synth-1k", [
            ("data/tier_c/train-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"),
            ("data/tier_c/validation-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"),
            ("data/tier_c/test-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"),
            ("deploy/hf/synth/README.md", "README.md"),
        ]),
        ("model", f"{org}/infradian-ref-s", [
            ("results/models/infradian-ref-s/infradian_ref.joblib", "infradian_ref.joblib"),
            ("results/models/infradian-ref-s/feature_spec.json", "feature_spec.json"),
            ("deploy/hf/model/README.md", "README.md"),
        ]),
    ]

    for repo_type, repo_id, files in plan:
        print(f"\n== {repo_type}: {repo_id} ==")
        if args.dry_run:
            for src, dst in files:
                exists = "ok" if Path(src).exists() else "MISSING"
                print(f"  [{exists}] {src} -> {dst}")
            continue
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        for src, dst in files:
            if not Path(src).exists():
                print(f"  skip (missing): {src}")
                continue
            api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo_id, repo_type=repo_type)
            print(f"  uploaded {dst}")


if __name__ == "__main__":
    main()
