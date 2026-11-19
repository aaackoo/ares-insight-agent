FROM python:3.12-slim

# uv as a single static binary, pinned to the version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first: this layer stays cached while only the code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app ./app

# Don't run as root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# Config (incl. OPENAI_API_KEY) comes from env vars at runtime: docker run --env-file .env
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
