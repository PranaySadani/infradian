"""Publish INFRADIAN artifacts to the HuggingFace Hub.

Token resolution order (the token is never passed as a CLI argument, so it does not land in your
shell history or in `ps` output):
  1. HF_TOKEN environment variable
  2. HUGGINGFACE_TOKEN environment variable
  3. a gitignored `.env` file in the repo root, containing `HF_TOKEN=hf_...`
  4. an existing `huggingface-cli login` session

Idempotent: creates the repos if missing, then uploads the parquet splits, cards and weights.

Usage:
  uv run python scripts/publish_hf.py --org <your-hf-username> --dry-run   # check, no auth needed
  uv run python scripts/publish_hf.py --org <your-hf-username>             # publish
  uv run python scripts/publish_hf.py --org <you> --private                # publish privately first
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ENV_FILE = Path(".env")


def load_token() -> tuple[str | None, str]:
    """Return (token, source). Never prints or logs the token value itself."""
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip(), f"${var}"

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                val = val.strip().strip("'\"")
                if val:
                    return val, f"{ENV_FILE} ({key.strip()})"

    return None, "huggingface-cli login session (if any)"


def build_plan(org: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    return [
        ("dataset", f"{org}/infradian-synth-1k", [
            ("datasets/infradian-synth-1k/data/train-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"),
            ("datasets/infradian-synth-1k/data/validation-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"),
            ("datasets/infradian-synth-1k/data/test-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"),
            ("deploy/hf/synth/README.md", "README.md"),
        ]),
        ("model", f"{org}/infradian-ref-s", [
            ("results/models/infradian-ref-s/infradian_ref.joblib", "infradian_ref.joblib"),
            ("results/models/infradian-ref-s/feature_spec.json", "feature_spec.json"),
            ("deploy/hf/model/README.md", "README.md"),
        ]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="HuggingFace username or org")
    ap.add_argument("--dry-run", action="store_true", help="check artifacts, no auth or upload")
    ap.add_argument("--private", action="store_true", help="create the repos private")
    args = ap.parse_args()

    plan = build_plan(args.org)

    if args.dry_run:
        missing = 0
        for repo_type, repo_id, files in plan:
            print(f"\n== {repo_type}: {repo_id} ==")
            for src, dst in files:
                ok = Path(src).exists()
                missing += not ok
                print(f"  [{'ok' if ok else 'MISSING'}] {src} -> {dst}")
        token, source = load_token()
        print(f"\ntoken: {'found via ' + source if token else 'NOT found'}")
        print(f"artifacts missing: {missing}")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("Install it first:  uv pip install huggingface_hub") from e

    token, source = load_token()
    api = HfApi(token=token) if token else HfApi()

    try:
        who = api.whoami()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "HuggingFace authentication failed.\n"
            f"  tried: {source}\n"
            "  Set a write token from https://huggingface.co/settings/tokens, then either:\n"
            "    export HF_TOKEN=hf_xxx        (this shell only)\n"
            "    echo 'HF_TOKEN=hf_xxx' >> .env   (gitignored, persists)\n"
            "    huggingface-cli login         (interactive)\n"
            f"  underlying error: {type(e).__name__}"
        ) from e

    print(f"authenticated as: {who.get('name')} (token from {source})")

    for repo_type, repo_id, files in plan:
        print(f"\n== {repo_type}: {repo_id} ==")
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=args.private)
        for src, dst in files:
            if not Path(src).exists():
                print(f"  skip (missing): {src}")
                continue
            api.upload_file(
                path_or_fileobj=src, path_in_repo=dst, repo_id=repo_id, repo_type=repo_type
            )
            print(f"  uploaded {dst}")
        kind = "datasets/" if repo_type == "dataset" else ""
        print(f"  -> https://huggingface.co/{kind}{repo_id}")


if __name__ == "__main__":
    main()
