"""Apply every migration to a temporary SQLite database for release checks."""

import os
import tempfile
from pathlib import Path

from peewee_migrate import Router
from playhouse.sqlite_ext import SqliteExtDatabase


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = SqliteExtDatabase(str(Path(directory) / "migration.db"))
        migration_dir = Path(
            os.getenv(
                "FRIGATE_MIGRATION_DIR",
                str(Path(__file__).resolve().parents[1] / "migrations"),
            )
        )
        try:
            Router(database, migrate_dir=str(migration_dir)).run()
            tables = {
                row[0]
                for row in database.execute_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required = {
                "event_observation",
                "event_evidence",
                "media_artifact",
                "notification_intent",
                "tracker_journal_entry",
                "edge_media_manifest",
            }
            missing = required - tables
            if missing:
                raise RuntimeError(f"Missing migrated tables: {sorted(missing)}")
            print("migration 001-040: OK")
        finally:
            database.close()


if __name__ == "__main__":
    main()
