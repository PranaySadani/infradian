# Deploy

Everything is built and configured. These are the commands to publish. Run them with your own logins.

## 1. GitHub: make the repo public (when ready)
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

**Point the frontend at a hosted backend** (optional, every view works without it):
```bash
# set in the Vercel project's Environment Variables, then redeploy
NEXT_PUBLIC_API_URL=https://<your-hf-space>.hf.space
```
Left unset it defaults to `http://localhost:8000`, so the header shows "API offline" in
production and the live-inference panel explains why. Nothing else is affected, all charts,
metrics and explanations are static JSON baked in at build time.

### Run the whole stack locally
```bash
make serve                                  # FastAPI on :8000  (header flips to "API live")
cd web && npm run build && npx serve out -l 3210
```

## 3. Dataset + model → HuggingFace (optional)
The dataset already ships in the repo at `datasets/infradian-synth-1k`, so this is a mirror, not a
dependency.

Supply a **write** token from https://huggingface.co/settings/tokens by any one of:

```bash
export HF_TOKEN=hf_xxx                      # this shell only
cp .env.example .env && $EDITOR .env        # persists; .env is gitignored AND hook-blocked
huggingface-cli login                       # interactive
```

Then:
```bash
uv pip install huggingface_hub
make publish-hf-check HF_ORG=<your-hf-username>   # verifies artifacts + token, uploads nothing
make publish-hf       HF_ORG=<your-hf-username>   # publishes
```
Publishes `<org>/infradian-synth-1k` (CC-BY parquet) and `<org>/infradian-ref-s` (Apache-2.0).
The dataset viewer and Croissant metadata work automatically from the parquet layout. The token is
resolved from the environment or `.env`, never passed as a CLI argument, so it does not reach your
shell history.

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
