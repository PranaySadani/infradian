# FastAPI backend for a HuggingFace Space (Docker SDK). Off the demo critical path;
# the frontend reads static JSON, so this only powers interactive prediction/sampling.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY results/models ./results/models
COPY web/public/data/leaderboard.json ./web/public/data/leaderboard.json

RUN uv pip install --system --no-cache -e ".[api]"

# HF Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["uvicorn", "infradian.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
