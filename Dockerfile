# Booking assistant API. Built and started by docker-compose.yml.
FROM python:3.14-slim

# uv, pinned to the version the lock file was made with
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first: this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY data ./data
COPY alembic.ini ./
COPY migrations ./migrations
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Do not run as root.
RUN useradd --create-home app
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Apply database migrations, then serve.
CMD ["sh", "-c", "alembic upgrade head && uvicorn apartworld_ai_assistent.api:create_app_from_env --factory --host 0.0.0.0 --port 8000"]
