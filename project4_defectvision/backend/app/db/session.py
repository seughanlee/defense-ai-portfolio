"""Engine and per-request session. A request's session commits once at the end or rolls back on error."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import Base


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        db_path = url.removeprefix("sqlite:///")
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _):  # noqa: ANN001
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

        return engine
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    # Schema is created on start-up. A migration tool (Alembic) is not used yet; see docs/LIMITATIONS.md.
    Base.metadata.create_all(engine)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
