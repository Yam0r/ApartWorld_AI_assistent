"""Shared fixtures: a real PostgreSQL in Docker (Testcontainers), migrated by Alembic."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Ryuk (the Testcontainers cleanup container) fails to map its port on Docker Desktop
# for Windows. The fixture below stops its container itself, so Ryuk is not needed.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")


@pytest.fixture(scope="session")
def pg_url():
    """Start postgres:17 once per test run and apply migrations. Skips if Docker is not running."""
    try:
        from testcontainers.community.postgres import PostgresContainer

        container = PostgresContainer("postgres:17-alpine", driver="psycopg")
        container.start()
    except Exception as e:  # no Docker daemon, no image access...
        if os.environ.get("CI"):  # GitHub Actions has Docker: never skip silently there
            raise
        pytest.skip(f"PostgreSQL container not available (start Docker): {e}")

    url = container.get_connection_url()
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")

    yield url
    container.stop()


@pytest.fixture
def pg_engine(pg_url):
    """Engine on a clean database: every test starts with empty tables."""
    from sqlalchemy import text

    from apartworld_ai_assistent.db import make_engine

    engine = make_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE requests, calendar_events RESTART IDENTITY"))
    yield engine
    engine.dispose()
