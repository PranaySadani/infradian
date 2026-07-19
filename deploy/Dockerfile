# Full INFRADIAN backend: the real scientific stack plus the multimodal AI layer.
#
# This is the unconstrained deployment. The Vercel serverless function is capped at 500MB and so
# runs a numpy-only reimplementation of the serving path; here there is no size limit, so the
# backend uses the genuine training-time transform and the sklearn-wrapped LightGBM bundle.
# tests/test_serve_parity.py guarantees the two agree.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api,viz]"

# api/ holds the shared multimodal route handlers, results/ holds the reference checkpoint.
COPY api ./api
COPY results/models ./results/models
COPY web/public/data ./web/public/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "infradian.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
