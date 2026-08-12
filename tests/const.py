"""Constants for testing."""

import gc
import time
from pathlib import Path

TEST_DB = "test.db"
TEST_DB_CLEANUPS = ["test.db", "test.db-shm", "test.db-wal"]
TEST_MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def reset_test_database() -> None:
    for filename in TEST_DB_CLEANUPS:
        path = Path(filename)
        for attempt in range(5):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                # Starlette TestClient owns request-thread SQLite connections.
                # On Windows their handles close only after thread-local
                # connection objects are finalized.
                gc.collect()
                time.sleep(0.05)


def close_test_database(database) -> None:
    if not database.is_stopped():
        database.stop()
    if not database.is_closed():
        database.close()
    reset_test_database()
