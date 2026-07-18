# Deploy

Everything is built and configured. These are the commands to publish. Run them with your own logins.

## 1. GitHub — make the repo public (when ready)
The repo is pushed **private** at `github.com/PranaySadani/infradian`. To publish:
```bash
gh repo edit PranaySadani/infradian --visibility public --accept-visibility-change-consequences
```

## 2. Frontend → Vercel
The app is a static Next.js export (`web/out`), so any static host works. For Vercel:
```bash
cd web
npx vercel@latest --prod          # first run prompts login + links the project
# or: connect github.com/PranaySadani/infradian in the Vercel dashboard,
#     set Root Directory = web  (vercel.json handles build + output)
```

## 3. Dataset + model → HuggingFace
```bash
pip install huggingface_hub
huggingface-cli login             # paste a write token from hf.co/settings/tokens
uv run python scripts/publish_hf.py --org <your-hf-org>
# publishes: <org>/infradian-synth-1k (CC-BY parquet) and <org>/infradian-ref-s (Apache-2.0)
```
The dataset viewer + Croissant metadata work automatically from the parquet layout.

## 4. Backend (optional) → HuggingFace Space
The demo does not need this (the frontend reads static JSON). To host the interactive API:
```bash
# create a Docker Space at hf.co/new-space, then:
git remote add space https://huggingface.co/spaces/<org>/infradian-api
# copy deploy/Dockerfile to the Space root, push src/ + results/models/ + pyproject.toml
```
The API listens on port 7860 (HF Spaces default).

## 5. Record the demo video
Follow `docs/video_script.md` shot-by-shot. Run the app locally with the mcPHASES DUA copy present so
the real numbers render, but keep every plotted trajectory synthetic (the app enforces this).

## Sanity checklist before submitting
- [ ] `uv run python -m pytest` green (42 tests)
- [ ] `make reproduce` regenerates the synthetic numbers with no data access
- [ ] repo public, README renders, CI badge green
- [ ] Vercel URL loads `/`, `/explorer`, `/skill`, `/methods`
- [ ] HF dataset viewer shows INFRADIAN-SYNTH-1K; model card renders
- [ ] video uploaded, links in the submission form
