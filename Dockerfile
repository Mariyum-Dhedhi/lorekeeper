# Lorekeeper API. Build and run with the database:  docker compose up --build
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY data ./data
COPY eval/dataset.jsonl ./eval/dataset.jsonl
RUN uv sync --locked --no-dev

# Run as an unprivileged user; it owns the cache the embedding model downloads into.
RUN useradd --create-home app && mkdir -p /app/.cache /app/index /app/eval/results && chown -R app /app/.cache /app/index /app/eval
USER app

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["/app/.venv/bin/uvicorn", "lorekeeper.api:app", "--host", "0.0.0.0", "--port", "8000"]
