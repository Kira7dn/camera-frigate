"""Constants for testing."""

from pathlib import Path

TEST_DB = "test.db"
TEST_DB_CLEANUPS = ["test.db", "test.db-shm", "test.db-wal"]
TEST_MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def reset_test_database() -> None:
    for filename in TEST_DB_CLEANUPS:
        Path(filename).unlink(missing_ok=True)


def close_test_database(database) -> None:
    if not database.is_stopped():
        database.stop()
    if not database.is_closed():
        database.close()
    reset_test_database()
